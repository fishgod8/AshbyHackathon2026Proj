"""Hand-checkable tests for Ashley's reference-support scoring."""

import unittest

import numpy as np

from ashley_review.reference_support import (
    K_NEIGHBORS,
    build_reference_index,
    score_queries,
)


def test_config() -> dict[str, object]:
    return {
        "model": "test-model",
        "checkpoint": "test-checkpoint",
        "checkpoint_sha256": "test-hash",
        "preprocessing": {"size": 1},
        "extraction": {"output_dtype": "float32"},
        "software_versions": {"numpy": "test"},
    }


def unit_vector(degrees: float) -> list[float]:
    radians = np.deg2rad(degrees)
    return [float(np.cos(radians)), float(np.sin(radians))] + [0.0] * 1022


class ReferenceSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"image_id": "query", "row_index": "0", "label": "A", "split": "reference"},
            {"image_id": "near30", "row_index": "1", "label": "B", "split": "reference"},
            {"image_id": "near60", "row_index": "2", "label": "C", "split": "reference"},
            {"image_id": "near90", "row_index": "3", "label": "D", "split": "reference"},
            {"image_id": "near120", "row_index": "4", "label": "E", "split": "reference"},
            {"image_id": "near180", "row_index": "5", "label": "F", "split": "reference"},
            {"image_id": "ignored", "row_index": "6", "label": "G", "split": "test"},
        ]
        self.embeddings = np.asarray(
            [unit_vector(angle) for angle in (0, 30, 60, 90, 120, 180, 1)],
            dtype=np.float32,
        )
        self.index = build_reference_index(self.embeddings, self.rows, test_config())

    def test_distance_order_and_fifth_neighbor_score(self) -> None:
        scores = score_queries(
            self.index, ["query"], np.asarray([unit_vector(0)], dtype=np.float32), test_config()
        )
        score = scores[0]
        self.assertEqual(
            [neighbor.image_id for neighbor in score.neighbors],
            ["near30", "near60", "near90", "near120", "near180"],
        )
        expected = [1 - np.cos(np.deg2rad(angle)) for angle in (30, 60, 90, 120, 180)]
        np.testing.assert_allclose(
            [neighbor.distance for neighbor in score.neighbors], expected, atol=1e-7
        )
        self.assertAlmostEqual(score.reference_distance, 2.0)
        self.assertEqual(len(score.neighbors), K_NEIGHBORS)

    def test_matching_image_id_excludes_self_and_returns_neighbor_labels(self) -> None:
        score = score_queries(
            self.index, ["query"], np.asarray([unit_vector(0)], dtype=np.float32), test_config()
        )[0]
        self.assertNotIn("query", [neighbor.image_id for neighbor in score.neighbors])
        self.assertEqual([neighbor.label for neighbor in score.neighbors], ["B", "C", "D", "E", "F"])
        self.assertNotIn("ignored", [neighbor.image_id for neighbor in score.neighbors])

    def test_rejects_nonfinite_and_zero_norm_vectors(self) -> None:
        nonfinite = self.embeddings.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_reference_index(nonfinite, self.rows, test_config())

        zero = self.embeddings.copy()
        zero[0] = 0
        with self.assertRaisesRegex(ValueError, "zero or invalid-norm"):
            build_reference_index(zero, self.rows, test_config())

        with self.assertRaisesRegex(ValueError, "zero or invalid-norm"):
            score_queries(
                self.index,
                ["zero-query"],
                np.zeros((1, 1024), dtype=np.float32),
                test_config(),
            )

    def test_requires_five_eligible_references_and_matching_config(self) -> None:
        short_manifest = [dict(row, split="validation") for row in self.rows]
        for row in short_manifest[:5]:
            row["split"] = "reference"
        short_index = build_reference_index(self.embeddings, short_manifest, test_config())
        with self.assertRaisesRegex(ValueError, "after self-exclusion"):
            score_queries(
                short_index,
                [short_manifest[0]["image_id"]],
                self.embeddings[:1],
                test_config(),
            )

        mismatched = test_config()
        mismatched["preprocessing"] = {"size": 2}
        with self.assertRaisesRegex(ValueError, "settings do not match"):
            score_queries(
                self.index,
                ["external-query"],
                np.asarray([unit_vector(0)], dtype=np.float32),
                mismatched,
            )

    def test_requires_at_least_five_reference_images(self) -> None:
        too_few = [dict(row, split="validation") for row in self.rows]
        for row in too_few[:4]:
            row["split"] = "reference"
        with self.assertRaisesRegex(ValueError, "at least 5"):
            build_reference_index(self.embeddings, too_few, test_config())


if __name__ == "__main__":
    unittest.main()