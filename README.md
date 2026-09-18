# Bank Statement Processing and Transaction Categorization

A non-LLM prototype that extracts account and transaction data from PDF bank statements, categorizes transactions, flags uncertain results, and exports CSV or Excel. The project includes a runnable analysis notebook and a Streamlit upload interface.

## Submission contents

| File | Purpose |
| --- | --- |
| `bank_statement_assessment.ipynb` | Step-by-step Pandas audit, cleaning, EDA, train/test split, TF-IDF and logistic regression pipeline, evaluation, PDF examples, and export |
| `bank_statement.py` | PDF extraction, OCR routing, validation, classification, and export functions |
| `app.py` | Upload, review, category editing, and downloads |
| `data/transactions_labeled.csv` | Synthetic labeled transaction descriptions |
| `data/challenge_transactions.csv` | Hand-written classification regression examples |
| `data/sample_statement_*.pdf` | Generated text and image-only statement examples |
| `outputs/` | Example CSV and Excel results from the notebook |
| `test_pipeline.py` | Focused parser and classification checks |

All sample data and account details are synthetic. No real customer statement is included.

## Run locally

Use Python 3.11 or newer. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Extract the submission ZIP into a normal folder before opening its files. Open `bank_statement_assessment.ipynb` in JupyterLab or VS Code, select the environment where the requirements were installed, and run all cells. Keep the notebook, `bank_statement.py`, and `data/` together. If Jupyter starts in another folder, set `PROJECT_DIR` in the first code cell to the extracted project folder. To use the upload interface:

```powershell
streamlit run app.py
```

To run the checks:

```powershell
python -m unittest -v test_pipeline.py
```

Image-only PDFs require the separate **Tesseract OCR executable** on `PATH`. The app also accepts its executable path. Text PDFs work without Tesseract. OCR values must be checked against the source image.

## Approach and results

The parser detects text and image pages, handles separate debit/credit or signed-amount tables, extracts holder/account/IFSC fields, and checks balance continuity. Category rules handle clear descriptions; character n-gram TF-IDF with logistic regression handles unmatched descriptions. Low-confidence model predictions are marked `Review`. The app lets a reviewer correct categories before download.

The notebook shows the Pandas checks and cleaning, category and text distributions, a merchant-group train/test split, explicit TF-IDF and logistic regression training, individual predictions, and evaluation. It holds out complete merchant phrases, including all four generated channel variants, to avoid a near-duplicate row split. The majority-class baseline is **8.3% accuracy**. On this **synthetic** 96-row holdout, the ML-only results are **62.5% accuracy**, **0.606 macro F1**, and **0.894 macro one-vs-rest ROC AUC**. The notebook includes precision, recall, weighted metrics, a confusion matrix, a ROC curve, and per-category results. ROC AUC measures probability ranking across thresholds; it is not the accuracy of the final category choice. With the 0.40 review cutoff, accuracy among automatically labeled ML rows is **91.7%** at **50% coverage**. The combined rule + ML check is 100% because every held-out phrase contains a known rule term; that figure is not a real-world accuracy estimate. The two generated text PDFs yield 14 rows with no balance warnings. The hand-written 34-row regression set also passes in the notebook.

## Limitations

Bank layouts and OCR quality vary. The parser supports the demonstrated layouts and flags rows it cannot parse; it is not a universal bank adapter. The included labels and statements are generated examples. Before deployment, evaluate against consented, annotated statements from the target banks, calibrate confidence on real labels, validate OCR and field-level extraction, and add access control and retention rules for financial data.
