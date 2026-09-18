"""Bank statement extraction and non-LLM transaction classification."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pymupdf as fitz
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


DATE_START = re.compile(r"^\s*(\d{2}[/-]\d{2}[/-]\d{4}|\d{4}-\d{2}-\d{2}|\d{2}\s+[A-Za-z]{3}\s+\d{4})\b")
MONEY = re.compile(r"^(?:[-+]?\(?[₹$]?\s*[\d,]+(?:\.\d{1,2})?\)?\s*(?:CR|DR)?|[-–])$", re.I)
ACCOUNT = re.compile(r"(?:account\s*(?:no\.?|number|#)|a/c\s*(?:no\.?)?)\s*[:#-]?\s*([Xx*\d][Xx*\d\s-]{5,})", re.I)
IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
HOLDER = re.compile(r"(?:account\s*holder|customer\s*name|name)\s*[:\-]\s*(.+)", re.I)

RULES = [
    ("Salary", r"\b(salary|payroll|wages)\b"),
    ("Cash withdrawal", r"\b(atm|cash withdrawal|cash wdl)\b"),
    ("Groceries", r"\b(grocery|supermarket|freshmart|bigbasket)\b"),
    ("Food & dining", r"\b(restaurant|cafe|swiggy|zomato|dining)\b"),
    ("Utilities", r"\b(electricity|water bill|gas bill|utility|broadband|internet bill)\b"),
    ("Shopping", r"\b(amazon|flipkart|retail|shopping|fashion)\b"),
    ("Transport", r"\b(uber|ola|fuel|petrol|metro|taxi)\b"),
    ("Rent", r"\b(rent|landlord|lease)\b"),
    ("Fees & charges", r"\b(bank charge|service fee|annual (?:card )?fee|penalty|gst charge)\b"),
    ("Interest", r"\b(interest|int\. paid)\b"),
    ("Healthcare", r"\b(pharmacy|hospital|clinic|medical|chemist|doctor)\b"),
    # Payment rails alone do not imply a person-to-person transfer.
    ("Transfer", r"\b(?:fund transfer|transfer|(?:upi|neft|imps|rtgs)\s+(?:to|from)\b)"),
]
DEFAULT_MIN_CONFIDENCE = 0.40


def normalize_description(description: str) -> str:
    """Remove channel labels and long references, retaining merchant words."""
    text = str(description).upper()
    text = re.sub(r"\b(?:POS|CARD|ONLINE|TXN|UPI|IMPS|NEFT|RTGS)\b", " ", text)
    text = re.sub(r"\b\d{4,}\b", " ", text)
    text = re.sub(r"[^A-Z0-9& ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class ExtractionResult:
    transactions: pd.DataFrame
    account: dict[str, str]
    page_types: list[str]
    warnings: list[str]


def parse_money(value: str) -> float | None:
    value = str(value).strip()
    if value in {"", "-", "–"}:
        return None
    negative = value.startswith("-") or (value.startswith("(") and ")" in value) or value.upper().endswith("DR")
    cleaned = re.sub(r"[₹$,()\s]|CR|DR", "", value, flags=re.I).lstrip("+-")
    try:
        amount = Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid amount: {value}") from exc
    return float(-amount if negative else amount)


def extract_account(text: str) -> dict[str, str]:
    match_account = ACCOUNT.search(text)
    match_holder = HOLDER.search(text)
    match_ifsc = IFSC.search(text.upper())
    return {
        "account_holder": match_holder.group(1).strip() if match_holder else "",
        "account_number": re.sub(r"\s", "", match_account.group(1)).strip("-") if match_account else "",
        "ifsc": match_ifsc.group(0) if match_ifsc else "",
    }


def _ocr_page(page, tesseract_cmd: str | None = None) -> str:
    import pytesseract

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(image, config="--psm 6 -c preserve_interword_spaces=1")
    except (pytesseract.TesseractNotFoundError, FileNotFoundError) as exc:
        raise RuntimeError("Scanned PDF detected. Install Tesseract OCR and add it to PATH, or provide its executable path.") from exc


def _parse_date(value: str):
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y"):
        try:
            return pd.to_datetime(value, format=fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _split_row(line: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*\|\s*|\s{2,}", line.strip()) if part.strip()]


def _parse_transaction(line: str, layout: str) -> dict | None:
    match = DATE_START.match(line)
    if not match:
        return None
    date = _parse_date(match.group(1))
    if date is None:
        return None
    rest = line[match.end():].strip(" |\t")
    parts = _split_row(rest)
    minimum = 4 if layout == "split" else 3
    if len(parts) < minimum:
        # OCR commonly collapses column spacing to a single space.
        parts = rest.split()
    if layout == "split" and len(parts) >= 4:
        description = " ".join(parts[:-3])
        debit, credit, balance = (parse_money(p) for p in parts[-3:])
    elif layout == "signed" and len(parts) >= 3:
        description = " ".join(parts[:-2])
        signed, balance = (parse_money(p) for p in parts[-2:])
        debit = abs(signed) if signed is not None and signed < 0 else None
        credit = signed if signed is not None and signed > 0 else None
    else:
        return None
    if not description or balance is None or (debit is None and credit is None):
        return None
    return {"date": date, "description": description, "debit": debit, "credit": credit, "balance": balance}


def process_pdf(source: str | Path | bytes, tesseract_cmd: str | None = None) -> ExtractionResult:
    """Extract statement rows, preserving warnings for anything needing review."""
    document = fitz.open(stream=source, filetype="pdf") if isinstance(source, bytes) else fitz.open(str(source))
    page_types, warnings, records = [], [], []
    account = {"account_holder": "", "account_number": "", "ifsc": ""}
    layout = "split"
    try:
        for page_number, page in enumerate(document, 1):
            text = page.get_text(sort=True)
            has_rows = any(DATE_START.match(line) for line in text.splitlines())
            needs_ocr = len(text.strip()) < 30 or (not has_rows and bool(page.get_images()))
            page_type = "image/OCR" if needs_ocr else "text"
            if page_type == "image/OCR":
                text = _ocr_page(page, tesseract_cmd)
                warnings.append(f"Page {page_number}: OCR was used; verify extracted values against the image.")
            page_types.append(page_type)
            found = extract_account(text)
            for key, value in found.items():
                if value and not account[key]:
                    account[key] = value
            for raw_line in text.splitlines():
                line = raw_line.strip()
                lower = line.lower()
                if "date" in lower and "balance" in lower:
                    layout = "signed" if "amount" in lower and "debit" not in lower else "split"
                    continue
                if not DATE_START.match(line):
                    if records and line.startswith(("- ", "  ")):
                        records[-1]["description"] += " " + line.strip("- ")
                    continue
                try:
                    row = _parse_transaction(line, layout)
                except ValueError:
                    row = None
                if row is None:
                    warnings.append(f"Page {page_number}: could not parse transaction line: {line[:110]}")
                    continue
                row["page"] = page_number
                records.append(row)
    finally:
        document.close()
    columns = ["date", "description", "debit", "credit", "balance", "page"]
    frame = pd.DataFrame(records, columns=columns)
    if frame.empty:
        warnings.append("No transactions found. Check layout and OCR output.")
    else:
        frame["balance_check"] = ""
        for i in range(1, len(frame)):
            prior = Decimal(str(frame.at[i - 1, "balance"]))
            debit = Decimal(str(frame.at[i, "debit"])) if pd.notna(frame.at[i, "debit"]) else Decimal(0)
            credit = Decimal(str(frame.at[i, "credit"])) if pd.notna(frame.at[i, "credit"]) else Decimal(0)
            current = Decimal(str(frame.at[i, "balance"]))
            if abs(prior - debit + credit - current) > Decimal("0.01"):
                frame.at[i, "balance_check"] = "REVIEW"
                warnings.append(f"Row {i + 1}: balance does not reconcile with prior transaction.")
    for key, value in account.items():
        if not value:
            warnings.append(f"Account field missing: {key}.")
        frame[key] = value
    return ExtractionResult(frame, account, page_types, warnings)


def train_classifier(labeled: pd.DataFrame):
    needed = {"description", "category"}
    if not needed.issubset(labeled.columns):
        raise ValueError(f"Training data must contain {sorted(needed)}")
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, preprocessor=normalize_description),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    model.fit(labeled["description"].fillna(""), labeled["category"])
    return model


def classify_transactions(frame: pd.DataFrame, model=None, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> pd.DataFrame:
    result = frame.copy()
    categories, methods, confidences = [], [], []
    for description in result["description"].fillna(""):
        match = next((category for category, pattern in RULES if re.search(pattern, description, re.I)), None)
        if match:
            category, method, confidence = match, "rule", 1.0
        elif model is not None:
            probabilities = model.predict_proba([description])[0]
            index = probabilities.argmax()
            confidence = float(probabilities[index])
            category = str(model.classes_[index]) if confidence >= min_confidence else "Review"
            method = "ml" if category != "Review" else "low confidence"
        else:
            category, method, confidence = "Review", "no matching rule", 0.0
        categories.append(category)
        methods.append(method)
        confidences.append(round(confidence, 3))
    result["category"] = categories
    result["classification_method"] = methods
    result["category_confidence"] = confidences
    return result


def export_data(frame: pd.DataFrame, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".csv":
        frame.to_csv(destination, index=False)
    elif destination.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(destination, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="Transactions", index=False)
            sheet = writer.sheets["Transactions"]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                width = min(max(len(str(cell.value or "")) for cell in column) + 2, 48)
                sheet.column_dimensions[column[0].column_letter].width = width
    else:
        raise ValueError("Output must end in .csv or .xlsx")
    return destination
