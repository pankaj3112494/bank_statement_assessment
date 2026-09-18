"""Upload and review bank statements in a local Streamlit UI."""

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from bank_statement import DEFAULT_MIN_CONFIDENCE, classify_transactions, process_pdf, train_classifier


st.set_page_config(page_title="Statement Processor", layout="wide")
st.title("Bank statement processor")
st.caption("Non-LLM extraction and classification prototype. Review all OCR and balance warnings.")

training = pd.read_csv(Path(__file__).parent / "data" / "transactions_labeled.csv")
model = train_classifier(training)
tesseract_path = st.text_input("Tesseract executable path (optional, for scanned PDFs)", value="")
min_confidence = st.slider(
    "Minimum ML confidence for automatic category",
    min_value=0.20,
    max_value=0.90,
    value=DEFAULT_MIN_CONFIDENCE,
    step=0.05,
    help="Lower values classify more transactions automatically; higher values send more to review. Rule matches are unaffected.",
)
uploads = st.file_uploader("Upload bank statement PDFs", type="pdf", accept_multiple_files=True)

if uploads:
    for upload in uploads:
        st.subheader(upload.name)
        try:
            extraction = process_pdf(upload.getvalue(), tesseract_path or None)
        except Exception as exc:
            st.error(f"Could not process {upload.name}: {exc}")
            continue
        account = extraction.account.copy()
        number = account.get("account_number", "")
        if number:
            account["account_number"] = "*" * max(0, len(number) - 4) + number[-4:]
        st.write("Account details", account)
        st.write("Page types", ", ".join(extraction.page_types))
        if extraction.warnings:
            for warning in extraction.warnings:
                st.warning(warning)
        if extraction.transactions.empty:
            continue
        classified = classify_transactions(extraction.transactions, model, min_confidence=min_confidence)
        edited = st.data_editor(
            classified,
            key=f"editor-{upload.name}",
            hide_index=True,
            num_rows="fixed",
            column_config={"category": st.column_config.TextColumn("Category (editable)")},
        )
        st.caption("Edit a category before downloading if it needs correction. Exported account numbers are unmasked.")
        csv_bytes = edited.to_csv(index=False).encode("utf-8-sig")
        xlsx_buffer = BytesIO()
        with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
            edited.to_excel(writer, sheet_name="Transactions", index=False)
            sheet = writer.sheets["Transactions"]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
        stem = Path(upload.name).stem
        left, right = st.columns(2)
        left.download_button("Download CSV", csv_bytes, file_name=f"{stem}_processed.csv", mime="text/csv", key=f"csv-{upload.name}")
        right.download_button("Download Excel", xlsx_buffer.getvalue(), file_name=f"{stem}_processed.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"xlsx-{upload.name}")
