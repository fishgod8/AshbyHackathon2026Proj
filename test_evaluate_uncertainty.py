import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluate_uncertainty import read_predictions


class ReadPredictionsTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "evaluation.csv"

    def write_csv(self, fieldnames, row):
        with self.path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def valid_row():
        return {
            "image_id": "image-1",
            "true_label": "InSitu",
            "predicted_label": "InSitu",
            "confidence": "0.85",
            "score_Benign": "0.05",
            "score_InSitu": "0.85",
            "score_Invasive": "0.07",
            "score_Normal": "0.03",
        }

    def test_accepts_prediction_csv_with_per_class_score_columns(self):
        row = self.valid_row()
        self.write_csv(row.keys(), row)
        predictions = read_predictions(self.path)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].image_id, "image-1")
        self.assertEqual(predictions[0].confidence, 0.85)

    def test_preserves_compatibility_with_required_columns_only(self):
        row = {
            key: value
            for key, value in self.valid_row().items()
            if not key.startswith("score_")
        }
        self.write_csv(row.keys(), row)
        self.assertEqual(len(read_predictions(self.path)), 1)

    def test_rejects_missing_required_column(self):
        row = self.valid_row()
        row.pop("true_label")
        self.write_csv(row.keys(), row)
        with self.assertRaisesRegex(ValueError, "missing required.*true_label"):
            read_predictions(self.path)

    def test_rejects_incomplete_score_columns(self):
        row = self.valid_row()
        row.pop("score_Normal")
        self.write_csv(row.keys(), row)
        with self.assertRaisesRegex(ValueError, "incomplete score columns"):
            read_predictions(self.path)

    def test_rejects_malformed_scores(self):
        for invalid_score in ("not-a-number", "nan", "-0.01", "1.01", ""):
            with self.subTest(score=invalid_score):
                row = self.valid_row()
                row["score_Benign"] = invalid_score
                self.write_csv(row.keys(), row)
                with self.assertRaisesRegex(ValueError, "score_Benign"):
                    read_predictions(self.path)

    def test_rejects_scores_that_do_not_sum_to_one(self):
        row = self.valid_row()
        row["score_Normal"] = "0.13"
        self.write_csv(row.keys(), row)
        with self.assertRaisesRegex(ValueError, "sum approximately to 1"):
            read_predictions(self.path)

    def test_rejects_confidence_that_does_not_match_scores(self):
        row = self.valid_row()
        row["confidence"] = "0.80"
        self.write_csv(row.keys(), row)
        with self.assertRaisesRegex(ValueError, "confidence does not match"):
            read_predictions(self.path)

    def test_rejects_predicted_label_that_is_not_argmax(self):
        row = self.valid_row()
        row["predicted_label"] = "Normal"
        self.write_csv(row.keys(), row)
        with self.assertRaisesRegex(ValueError, "predicted_label does not match"):
            read_predictions(self.path)


class EvaluatorCliTests(unittest.TestCase):
    def run_cli(self, purpose):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "results.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    "evaluate_uncertainty.py",
                    "synthetic_uncertainty_example.csv",
                    "--output-csv",
                    str(output_path),
                    "--purpose",
                    purpose,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            rows = []
            if output_path.exists():
                with output_path.open(encoding="utf-8", newline="") as csv_file:
                    rows = list(csv.DictReader(csv_file))
            return completed, rows

    def test_cli_records_validation_purpose(self):
        completed, rows = self.run_cli("validation")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(rows)
        self.assertEqual({row["evaluation_purpose"] for row in rows}, {"validation"})
        self.assertEqual({row["data_status"] for row in rows}, {"PROVIDED_DATA"})

    def test_cli_warns_for_test_purpose(self):
        completed, rows = self.run_cli("test")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("do not use them to select thresholds", completed.stderr)
        self.assertEqual({row["evaluation_purpose"] for row in rows}, {"test"})

    def test_cli_requires_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "evaluate_uncertainty.py",
                    "synthetic_uncertainty_example.csv",
                    "--output-csv",
                    str(Path(directory) / "results.csv"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("one of the arguments --purpose --simulated is required", completed.stderr)

    def test_legacy_simulated_flag_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "results.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    "evaluate_uncertainty.py",
                    "synthetic_uncertainty_example.csv",
                    "--output-csv",
                    str(output_path),
                    "--simulated",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            with output_path.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual({row["evaluation_purpose"] for row in rows}, {"simulated"})
        self.assertEqual({row["data_status"] for row in rows}, {"SIMULATED"})


if __name__ == "__main__":
    unittest.main()
