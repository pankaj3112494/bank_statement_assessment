"""Focused regression checks for the synthetic end-to-end examples."""

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from bank_statement import classify_transactions, normalize_description, parse_money, process_pdf, train_classifier


DATA = Path(__file__).parent / "data"


class PipelineTests(unittest.TestCase):
    def test_two_statement_layouts(self):
        split = process_pdf(DATA / "sample_statement_split.pdf")
        signed = process_pdf(DATA / "sample_statement_signed.pdf")
        self.assertEqual(len(split.transactions), 8)
        self.assertEqual(len(signed.transactions), 6)
        self.assertEqual(split.account["ifsc"], "DEMO0123456")
        self.assertEqual(split.warnings, [])
        self.assertEqual(signed.warnings, [])
        self.assertEqual(split.transactions.iloc[-1]["balance"], 42950.0)
        self.assertEqual(signed.transactions.iloc[-1]["balance"], 41780.0)

    def test_scanned_pdf_detection_and_ocr_path(self):
        with patch("bank_statement._ocr_page", return_value="\n".join([
            "Account Holder: Asha Rao", "Account Number: 123456789012", "IFSC: DEMO0123456",
            "Date  Description  Debit  Credit  Balance",
            "01/04/2026 Grocery 100.00 - 900.00",
        ])) as ocr:
            result = process_pdf(DATA / "sample_statement_scanned.pdf")
        self.assertTrue(ocr.called)
        self.assertEqual(result.page_types, ["image/OCR"])
        self.assertEqual(len(result.transactions), 1)
        self.assertTrue(any("OCR was used" in warning for warning in result.warnings))

    def test_classification_and_review(self):
        model = train_classifier(pd.read_csv(DATA / "transactions_labeled.csv"))
        rows = pd.DataFrame({"description": [
            "ATM cash withdrawal",
            "UPI Freshmart grocery",
            "UPI to Rohan",
            "UPI ABCD 92831",
            "unfamiliar XYZ transaction",
        ]})
        result = classify_transactions(rows, model)
        self.assertEqual(result.loc[0, "category"], "Cash withdrawal")
        self.assertEqual(result.loc[0, "classification_method"], "rule")
        self.assertEqual(result.loc[1, "category"], "Groceries")
        self.assertEqual(result.loc[2, "category"], "Transfer")
        self.assertEqual(result.loc[3, "category"], "Review")
        self.assertEqual(result.loc[4, "category"], "Review")

    def test_amount_formats(self):
        self.assertEqual(parse_money("(1,250.50)"), -1250.5)
        self.assertEqual(parse_money("1,250.50 CR"), 1250.5)
        self.assertIsNone(parse_money("-"))

    def test_description_preparation(self):
        self.assertEqual(normalize_description("UPI/482912 Freshmart grocery"), "FRESHMART GROCERY")


if __name__ == "__main__":
    unittest.main()
