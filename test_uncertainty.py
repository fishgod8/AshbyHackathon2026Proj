import unittest

from uncertainty import assess_prediction, assess_probabilities, assess_uncertainty


class AssessUncertaintyTests(unittest.TestCase):
    def test_preserves_existing_interface(self):
        self.assertEqual(assess_uncertainty([0.1, 0.8, 0.1], 0.7), (0.8, False))

    def test_invalid_probabilities(self):
        invalid_probabilities = (
            [],
            [-0.1, 1.1],
            [0.2, 0.9],
            [float("nan"), 1.0],
            [True, 0.0],
        )
        for probabilities in invalid_probabilities:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    assess_uncertainty(probabilities, 0.7)

    def test_invalid_thresholds(self):
        for threshold in (-0.01, 1.01, float("inf"), float("nan"), True, "0.7"):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    assess_uncertainty([0.8, 0.2], threshold)


class AssessPredictionTests(unittest.TestCase):
    def prediction(self, scores=None):
        return {
            "predicted_class": "InSitu",
            "scores": scores or {
                "Benign": 0.05,
                "InSitu": 0.85,
                "Invasive": 0.07,
                "Normal": 0.03,
            },
            "embedding": [1.0, 2.0],
        }

    def test_high_confidence_prediction(self):
        self.assertEqual(
            assess_prediction(self.prediction(), 0.7),
            {
                "predicted_class": "InSitu",
                "confidence": 0.85,
                "needs_review": False,
                "threshold": 0.7,
            },
        )

    def test_low_confidence_prediction_without_embedding(self):
        prediction = self.prediction(
            {"Benign": 0.2, "InSitu": 0.4, "Invasive": 0.3, "Normal": 0.1}
        )
        prediction.pop("embedding")
        self.assertTrue(assess_prediction(prediction, 0.7)["needs_review"])

    def test_confidence_exactly_at_threshold_does_not_need_review(self):
        prediction = self.prediction(
            {"Benign": 0.1, "InSitu": 0.7, "Invasive": 0.1, "Normal": 0.1}
        )
        self.assertFalse(assess_prediction(prediction, 0.7)["needs_review"])

    def test_missing_or_empty_scores(self):
        for prediction in (
            {"predicted_class": "InSitu"},
            {"predicted_class": "InSitu", "scores": {}},
            {"predicted_class": "InSitu", "scores": []},
        ):
            with self.subTest(prediction=prediction):
                with self.assertRaisesRegex(ValueError, "scores"):
                    assess_prediction(prediction, 0.7)

    def test_missing_or_empty_predicted_class(self):
        for predicted_class in (None, "", "   "):
            prediction = {"scores": {"InSitu": 1.0}}
            if predicted_class is not None:
                prediction["predicted_class"] = predicted_class
            with self.subTest(predicted_class=predicted_class):
                with self.assertRaisesRegex(ValueError, "predicted_class"):
                    assess_prediction(prediction, 0.7)

    def test_prediction_must_be_dictionary(self):
        with self.assertRaisesRegex(ValueError, "dictionary"):
            assess_prediction(None, 0.7)

    def test_reuses_probability_and_threshold_validation(self):
        invalid_cases = (
            (self.prediction({"InSitu": 1.1, "Normal": -0.1}), 0.7),
            (self.prediction({"InSitu": 0.8, "Normal": 0.1}), 0.7),
            (self.prediction({"InSitu": 1.0}), 1.1),
        )
        for prediction, threshold in invalid_cases:
            with self.subTest(prediction=prediction, threshold=threshold):
                with self.assertRaises(ValueError):
                    assess_prediction(prediction, threshold)


class FullMetricTests(unittest.TestCase):
    def test_entropy_margin_and_label_mapping(self):
        result = assess_probabilities({"B": 0.1, "A": 0.7, "C": 0.2}, 0.7)
        self.assertEqual(result["predicted_class"], "A")
        self.assertAlmostEqual(result["margin"], 0.5)
        self.assertAlmostEqual(result["max_probability_uncertainty"], 0.3)
        self.assertFalse(result["needs_review"])  # boundary uses confidence < threshold

    def test_zero_probability_entropy_is_safe(self):
        result = assess_probabilities({"A": 1.0, "B": 0.0})
        self.assertEqual(result["entropy"], 0.0)
        self.assertEqual(result["normalized_entropy"], 0.0)

    def test_entropy_and_margin_review_reasons(self):
        result = assess_probabilities(
            {"A": .26, "B": .25, "C": .25, "D": .24},
            confidence_threshold=0,
            entropy_threshold=.9,
            margin_threshold=.1,
        )
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["review_reasons"],
                         ["High predictive entropy", "Small top-two margin"])

    def test_optional_threshold_boundaries_are_inclusive(self):
        base = assess_probabilities({"A": .7, "B": .3})
        result = assess_probabilities(
            {"A": .7, "B": .3}, confidence_threshold=0,
            entropy_threshold=base["normalized_entropy"],
            margin_threshold=base["margin"],
        )
        self.assertEqual(
            result["review_reasons"],
            ["High predictive entropy", "Small top-two margin"],
        )

    def test_expected_class_mapping(self):
        with self.assertRaisesRegex(ValueError, "expected classes"):
            assess_probabilities(
                {"A": .5, "B": .5}, expected_labels=("A", "C")
            )


if __name__ == "__main__":
    unittest.main()
