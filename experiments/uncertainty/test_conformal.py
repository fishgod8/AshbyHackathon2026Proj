"""Unit tests using synthetic, hand-checkable probabilities only."""

import math
import unittest

from experiments.uncertainty.conformal import (
    METHOD_NAME,
    assess_conformal_prediction,
    calibrate_conformal,
    predict_set,
)


class CalibrateConformalTests(unittest.TestCase):
    def test_nonconformity_scores_label_mapping_and_corrected_quantile(self):
        # True-class probabilities are 0.9, 0.8, 0.7, 0.6, giving sorted
        # nonconformity scores 0.1, 0.2, 0.3, 0.4. The corrected rank is
        # ceil((4 + 1) * (1 - 0.4)) = 3, so the quantile is 0.3.
        artifact = calibrate_conformal(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.4, 0.6],
            ],
            ["A", "B", "A", "B"],
            ["A", "B"],
            alpha=0.4,
        )

        self.assertEqual(
            artifact,
            {
                "method": METHOD_NAME,
                "alpha": 0.4,
                "quantile": 0.30000000000000004,
                "class_names": ["A", "B"],
                "calibration_sample_count": 4,
            },
        )

    def test_infinite_quantile_when_rank_is_n_plus_one(self):
        artifact = calibrate_conformal(
            [[0.8, 0.2], [0.3, 0.7]],
            ["A", "B"],
            ["A", "B"],
            alpha=0.1,
        )
        self.assertTrue(math.isinf(artifact["quantile"]))

    def test_rejects_invalid_alpha(self):
        for alpha in (0, 1, -0.1, 1.1, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    calibrate_conformal([[0.5, 0.5]], ["A"], ["A", "B"], alpha)

    def test_rejects_empty_or_incorrectly_dimensioned_calibration_data(self):
        invalid_cases = (
            ([], []),
            ([0.5, 0.5], ["A", "B"]),
            ([[0.5, 0.3, 0.2]], ["A"]),
        )
        for probabilities, labels in invalid_cases:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    calibrate_conformal(probabilities, labels, ["A", "B"])

    def test_rejects_invalid_probabilities_and_row_sums(self):
        invalid_rows = (
            [float("nan"), float("nan")],
            [float("inf"), 0.0],
            [-0.1, 1.1],
            [0.4, 0.4],
            [True, 0.0],
        )
        for row in invalid_rows:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "calibration row 1"):
                    calibrate_conformal([row], ["A"], ["A", "B"])

    def test_rejects_missing_and_unknown_labels(self):
        with self.assertRaisesRegex(ValueError, "length must match"):
            calibrate_conformal([[0.8, 0.2]], [], ["A", "B"])
        with self.assertRaisesRegex(ValueError, "not present in class_names"):
            calibrate_conformal([[0.8, 0.2]], ["C"], ["A", "B"])
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            calibrate_conformal([[0.8, 0.2]], [""], ["A", "B"])

    def test_rejects_duplicate_or_invalid_class_names(self):
        for class_names in ([], ["A", "A"], ["A", ""], ["A", 2]):
            with self.subTest(class_names=class_names):
                with self.assertRaisesRegex(ValueError, "class_names"):
                    calibrate_conformal([[0.8, 0.2]], ["A"], class_names)


class PredictSetTests(unittest.TestCase):
    def test_singleton_multi_class_and_empty_sets(self):
        probabilities = [0.7, 0.2, 0.1]
        class_names = ["A", "B", "C"]
        self.assertEqual(predict_set(probabilities, class_names, 0.35), ["A"])
        self.assertEqual(predict_set(probabilities, class_names, 0.85), ["A", "B"])
        self.assertEqual(predict_set(probabilities, class_names, 0.05), [])

    def test_infinite_quantile_includes_all_classes_in_input_order(self):
        self.assertEqual(
            predict_set([0.2, 0.5, 0.3], ["B", "A", "C"], float("inf")),
            ["B", "A", "C"],
        )

    def test_rejects_invalid_prediction_inputs(self):
        invalid_probabilities = (
            [0.5],
            [0.6, 0.6],
            [float("nan"), 1.0],
            [float("inf"), 0.0],
            [-0.1, 1.1],
        )
        for probabilities in invalid_probabilities:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    predict_set(probabilities, ["A", "B"], 0.5)


class AssessConformalPredictionTests(unittest.TestCase):
    @staticmethod
    def artifact(quantile):
        return {
            "method": METHOD_NAME,
            "alpha": 0.1,
            "quantile": quantile,
            "class_names": ["A", "B", "C"],
            "calibration_sample_count": 20,
        }

    @staticmethod
    def prediction():
        return {
            "predicted_class": "A",
            "scores": {"C": 0.1, "A": 0.7, "B": 0.2},
            "embedding": [1.0, 2.0],
        }

    def test_singleton_does_not_trigger_review(self):
        result = assess_conformal_prediction(self.prediction(), self.artifact(0.35))
        self.assertEqual(result["prediction_set"], ["A"])
        self.assertEqual(result["set_size"], 1)
        self.assertFalse(result["needs_review"])

    def test_empty_and_multi_class_sets_trigger_review(self):
        for quantile, expected_set in ((0.05, []), (0.85, ["A", "B"])):
            with self.subTest(quantile=quantile):
                result = assess_conformal_prediction(
                    self.prediction(), self.artifact(quantile)
                )
                self.assertEqual(result["prediction_set"], expected_set)
                self.assertTrue(result["needs_review"])

    def test_predicted_class_is_optional(self):
        prediction = self.prediction()
        prediction.pop("predicted_class")
        result = assess_conformal_prediction(prediction, self.artifact(0.35))
        self.assertIsNone(result["predicted_class"])

    def test_rejects_mismatched_score_keys(self):
        prediction = self.prediction()
        prediction["scores"] = {"A": 0.7, "B": 0.2, "D": 0.1}
        with self.assertRaisesRegex(ValueError, "score keys must match"):
            assess_conformal_prediction(prediction, self.artifact(0.35))

    def test_output_order_uses_calibration_class_order(self):
        artifact = self.artifact(0.85)
        artifact["class_names"] = ["B", "A", "C"]
        result = assess_conformal_prediction(self.prediction(), artifact)
        self.assertEqual(result["prediction_set"], ["B", "A"])


if __name__ == "__main__":
    unittest.main()
