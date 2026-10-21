import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Ashley_uncertainty.analysis_pipeline import analyze_image, analyze_prediction, rank_for_review
from bach.predict_image import load_classifier
from Ashley_uncertainty.model_identity import inspect_checkpoint


class UnifiedAnalysisTests(unittest.TestCase):
    def prediction(self):
        return {
            "predicted_class": "Benign",
            "scores": {"Normal": .05, "Benign": .64, "InSitu": .26, "Invasive": .05},
            "embedding": [1.0, 2.0],
        }

    def test_schema_and_metric_values(self):
        result = analyze_prediction(self.prediction(), image_id="sample.tif", confidence_threshold=.70)
        self.assertEqual(result["image_id"], "sample.tif")
        self.assertEqual(result["predicted_class"], "Benign")
        self.assertAlmostEqual(result["confidence"], .64)
        self.assertAlmostEqual(result["uncertainty"]["max_probability_uncertainty"], .36)
        self.assertAlmostEqual(result["uncertainty"]["margin"], .38)
        self.assertGreater(result["uncertainty"]["entropy"], 0)
        self.assertEqual(result["conformal_status"], "unavailable")
        self.assertIsNone(result["prediction_set"])
        self.assertFalse(result["conformal"]["available"])
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["embedding"], [1.0, 2.0])

    def test_rejects_mismatched_label(self):
        prediction = self.prediction()
        prediction["predicted_class"] = "Normal"
        with self.assertRaisesRegex(ValueError, "does not match"):
            analyze_prediction(prediction)

    def test_injectable_predictor(self):
        result = analyze_image("unused.jpg", predictor=lambda path: self.prediction())
        self.assertEqual(result["image_id"], "unused.jpg")
        self.assertEqual(result["embedding"], [1.0, 2.0])
        json.dumps(result)

    def test_rejects_wrong_bach_class_mapping(self):
        prediction = self.prediction()
        prediction["scores"] = {"Normal": .05, "Benign": .64, "InSitu": .26, "Other": .05}
        prediction["predicted_class"] = "Benign"
        with self.assertRaisesRegex(ValueError, "expected classes"):
            analyze_prediction(prediction)

    def test_retrieval_embedding_is_512d_and_json_serializable(self):
        prediction = self.prediction()
        prediction["embedding"] = [0.0] * 512
        result = analyze_prediction(prediction)
        self.assertEqual(len(result["embedding"]), 512)
        json.dumps(result)


class ReviewQueueTests(unittest.TestCase):
    @staticmethod
    def result(image_id, confidence, entropy, margin, needs_review):
        return {
            "image_id": image_id,
            "predicted_class": "Benign",
            "confidence": confidence,
            "probabilities": {"Benign": confidence},
            "uncertainty": {
                "max_probability_uncertainty": 1 - confidence,
                "entropy": entropy,
                "normalized_entropy": entropy,
                "margin": margin,
            },
            "needs_review": needs_review,
            "review_reasons": ["Low model confidence"] if needs_review else [],
            "embedding": None,
        }

    def test_default_queue_groups_flags_then_confidence_and_ties_by_id(self):
        rows = [
            self.result("z.tif", .8, .2, .7, False),
            self.result("b.tif", .6, .8, .1, True),
            self.result("a.tif", .6, .7, .2, True),
            self.result("c.tif", .7, .4, .4, False),
        ]
        ranked = rank_for_review(rows)
        self.assertEqual([row["image_id"] for row in ranked],
                         ["a.tif", "b.tif", "c.tif", "z.tif"])
        self.assertEqual([row["review_rank"] for row in ranked], [1, 2, 3, 4])
        self.assertNotIn("review_rank", rows[0])

    def test_optional_sorting_does_not_change_review_flags(self):
        rows = [
            self.result("a.tif", .6, .7, .2, True),
            self.result("b.tif", .5, .8, .3, True),
            self.result("c.tif", .8, .1, .7, False),
        ]
        entropy = rank_for_review(rows, sort_by="normalized_entropy")
        margin = rank_for_review(rows, sort_by="margin")
        self.assertEqual([row["image_id"] for row in entropy], ["b.tif", "a.tif", "c.tif"])
        self.assertEqual([row["image_id"] for row in margin], ["a.tif", "b.tif", "c.tif"])
        self.assertEqual({row["image_id"]: row["needs_review"] for row in entropy},
                         {row["image_id"]: row["needs_review"] for row in rows})

    def test_queue_is_json_serializable_and_rejects_bad_sort(self):
        rows = [self.result("a.tif", .6, .7, .2, True)]
        json.dumps(rank_for_review(rows))
        with self.assertRaisesRegex(ValueError, "sort_by"):
            rank_for_review(rows, sort_by="hybrid")

class ClassifierArtifactTests(unittest.TestCase):
    def test_real_checkpoint_metadata_and_dimension(self):
        checkpoint = str(Path("bach/classifier.pkl").resolve())
        classifier = load_classifier(checkpoint)
        self.assertEqual([str(value) for value in classifier.classes_],
                         ["Benign", "InSitu", "Invasive", "Normal"])
        self.assertEqual(classifier.n_features_in_, 512)
        self.assertEqual(
            inspect_checkpoint(checkpoint)["classifier_sha256"],
            "a3d3115fc3e1a88b4720ba85d083602202b86e315fcfd8f0abd381cc62e15885",
        )

    def test_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_classifier(str(Path(directory) / "missing.pkl"))

    def test_cpu_fallback(self):
        from bach import predict_image as module
        with patch.object(module.torch.cuda, "is_available", return_value=False):
            self.assertEqual(module._default_device(), "cpu")


if __name__ == "__main__":
    unittest.main()
