import csv
import tempfile
import unittest
from pathlib import Path

from evaluate_uncertainty import read_predictions


class ReadPredictionsTests(unittest.TestCase):
    def test_accepts_prediction_csv_with_per_class_score_columns(self):
        fieldnames = (
            "image_id",
            "true_label",
            "predicted_label",
            "confidence",
            "score_Benign",
            "score_InSitu",
            "score_Invasive",
            "score_Normal",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.csv"
            with path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "image_id": "image-1",
                        "true_label": "InSitu",
                        "predicted_label": "InSitu",
                        "confidence": "0.85",
                        "score_Benign": "0.05",
                        "score_InSitu": "0.85",
                        "score_Invasive": "0.07",
                        "score_Normal": "0.03",
                    }
                )

            predictions = read_predictions(path)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].image_id, "image-1")
        self.assertEqual(predictions[0].confidence, 0.85)


if __name__ == "__main__":
    unittest.main()
