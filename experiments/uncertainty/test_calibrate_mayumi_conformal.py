import csv
import tempfile
import unittest
from pathlib import Path

from experiments.uncertainty.calibrate_mayumi_conformal import build_artifact, evaluate_artifact
from model_identity import inspect_checkpoint, verify_calibration_binding


class ExactClassifierCalibrationTests(unittest.TestCase):
    def write_rows(self, path, ids):
        classes = inspect_checkpoint("bach/classifier.pkl")["class_order"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "true_label", *[f"score_{x}" for x in classes]])
            writer.writeheader()
            for index, image_id in enumerate(ids):
                true = classes[index % len(classes)]
                scores = {name: .1 for name in classes}
                scores[true] = .7
                writer.writerow({"image_id": image_id, "true_label": true,
                                 **{f"score_{name}": scores[name] for name in classes}})

    def test_artifact_binds_exact_checkpoint_and_disjoint_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "cal.csv"
            evaluation = root / "eval.csv"
            self.write_rows(calibration, [f"c{i}" for i in range(8)])
            self.write_rows(evaluation, [f"e{i}" for i in range(4)])
            artifact = build_artifact("bach/classifier.pkl", calibration, alpha=.2,
                                      dataset_name="independent", dataset_split="calibration",
                                      independence_note="not used for fitting or selection")
            verify_calibration_binding(artifact, "bach/classifier.pkl")
            result = evaluate_artifact(artifact, evaluation)
            self.assertEqual(result["evaluation_sample_count"], 4)

    def test_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same.csv"
            self.write_rows(path, [f"x{i}" for i in range(8)])
            artifact = build_artifact("bach/classifier.pkl", path, alpha=.2,
                                      dataset_name="independent", dataset_split="calibration",
                                      independence_note="independent")
            with self.assertRaisesRegex(ValueError, "overlap"):
                evaluate_artifact(artifact, path)


if __name__ == "__main__":
    unittest.main()
