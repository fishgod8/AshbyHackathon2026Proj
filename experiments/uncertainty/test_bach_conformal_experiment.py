"""Tests for the BACH experiment using small synthetic embeddings only."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.uncertainty.run_bach_conformal_experiment import (
    CLASS_NAMES,
    EmbeddingRecord,
    final_split,
    policy_metrics,
    run_experiment,
    select_calibration_ids,
)


class SplitTests(unittest.TestCase):
    @staticmethod
    def records():
        records = []
        for class_index, label in enumerate(CLASS_NAMES):
            for index in range(5):
                records.append(
                    EmbeddingRecord(
                        f"{label}-{index}", label, "train", [float(class_index)]
                    )
                )
            records.append(
                EmbeddingRecord(f"{label}-val", label, "val", [float(class_index)])
            )
            records.append(
                EmbeddingRecord(f"{label}-test", label, "test", [float(class_index)])
            )
        return records

    def test_calibration_selection_is_deterministic_and_stratified(self):
        records = self.records()
        first = select_calibration_ids(records, per_class=2, salt="fixed")
        second = select_calibration_ids(list(reversed(records)), per_class=2, salt="fixed")

        self.assertEqual(first, second)
        for label in CLASS_NAMES:
            self.assertEqual(
                sum(image_id.startswith(f"{label}-") for image_id in first), 2
            )

    def test_calibration_comes_only_from_original_train(self):
        records = self.records()
        calibration_ids = select_calibration_ids(records, per_class=1)
        for record in records:
            split = final_split(record, calibration_ids)
            if split == "calibration":
                self.assertEqual(record.original_split, "train")
            elif record.original_split in ("val", "test"):
                self.assertEqual(split, record.original_split)


class PolicyMetricTests(unittest.TestCase):
    def test_policy_metric_calculations(self):
        metrics = policy_metrics(
            "example",
            ["A", "B", "C", "D"],
            ["A", "A", "C", "A"],
            [False, True, False, False],
        )

        self.assertEqual(metrics["review_count"], 1)
        self.assertEqual(metrics["review_rate"], 0.25)
        self.assertEqual(metrics["captured_errors"], 1)
        self.assertEqual(metrics["error_capture_rate"], 0.5)
        self.assertEqual(metrics["reviewed_error_rate"], 1.0)
        self.assertAlmostEqual(metrics["unreviewed_error_rate"], 1 / 3)

    def test_zero_denominators_are_reported_as_none(self):
        metrics = policy_metrics("example", ["A"], ["A"], [False])
        self.assertIsNone(metrics["error_capture_rate"])
        self.assertIsNone(metrics["reviewed_error_rate"])
        self.assertEqual(metrics["unreviewed_error_rate"], 0.0)


class ExperimentOutputTests(unittest.TestCase):
    def write_synthetic_embeddings(self, path):
        class_count = len(CLASS_NAMES)
        with path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=("image_id", "label", "split", "embedding"),
            )
            writer.writeheader()
            for class_index, label in enumerate(CLASS_NAMES):
                base = [0.0] * class_count
                base[class_index] = 2.0
                for split, count in (("train", 4), ("val", 2), ("test", 2)):
                    for index in range(count):
                        vector = list(base)
                        vector[(class_index + 1) % class_count] = index * 0.01
                        writer.writerow(
                            {
                                "image_id": f"{label}-{split}-{index}",
                                "label": label,
                                "split": split,
                                "embedding": vector,
                            }
                        )

    def test_end_to_end_output_schemas_and_split_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "embeddings.csv"
            outputs = root / "outputs"
            self.write_synthetic_embeddings(embeddings)

            summary = run_experiment(
                embeddings,
                outputs,
                calibration_per_class=1,
                expected_dimension=4,
                train_per_class=4,
                validation_per_class=2,
                test_per_class=2,
            )

            expected_files = {
                "split_manifest.csv",
                "calibration_predictions.csv",
                "calibration_artifact.json",
                "test_predictions.csv",
                "policy_comparison.csv",
                "experiment_summary.json",
            }
            self.assertEqual(
                {path.name for path in outputs.iterdir()}, expected_files
            )

            with (outputs / "split_manifest.csv").open(
                encoding="utf-8", newline=""
            ) as input_file:
                manifest = list(csv.DictReader(input_file))
            calibration_ids = {
                row["image_id"] for row in manifest if row["final_split"] == "calibration"
            }
            test_ids = {
                row["image_id"] for row in manifest if row["final_split"] == "test"
            }
            self.assertEqual(len(calibration_ids), 4)
            self.assertTrue(calibration_ids.isdisjoint(test_ids))

            with (outputs / "calibration_predictions.csv").open(
                encoding="utf-8", newline=""
            ) as input_file:
                calibration_rows = list(csv.DictReader(input_file))
            expected_probability_columns = {
                f"score_{class_name}" for class_name in CLASS_NAMES
            }
            self.assertTrue(
                expected_probability_columns.issubset(calibration_rows[0])
            )
            self.assertEqual(
                {row["image_id"] for row in calibration_rows}, calibration_ids
            )

            with (outputs / "policy_comparison.csv").open(
                encoding="utf-8", newline=""
            ) as input_file:
                comparison = list(csv.DictReader(input_file))
            self.assertEqual(
                [row["policy"] for row in comparison],
                ["confidence_only", "conformal_only", "combined_or"],
            )
            self.assertIn("empirical_coverage", comparison[0])
            self.assertIn("average_prediction_set_size", comparison[0])
            self.assertIn("singleton_rate", comparison[0])

            with (outputs / "calibration_artifact.json").open(
                encoding="utf-8"
            ) as input_file:
                artifact = json.load(input_file)
            self.assertEqual(artifact["class_names"], list(CLASS_NAMES))
            self.assertEqual(artifact["calibration_sample_count"], 4)
            self.assertEqual(summary["experiment_status"], "exploratory")


if __name__ == "__main__":
    unittest.main()
