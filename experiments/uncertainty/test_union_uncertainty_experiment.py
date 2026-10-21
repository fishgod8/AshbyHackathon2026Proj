import unittest

from experiments.uncertainty.run_union_uncertainty_experiment import metrics, selected_for_base_k


class UnionStrategyTests(unittest.TestCase):
    def test_union_deduplicates_and_vote_sets(self):
        rows = [{"image_id": value} for value in "abcd"]
        ranked = {
            "confidence_only": rows,
            "entropy_only": [rows[1], rows[0], rows[2], rows[3]],
            "margin_only": [rows[1], rows[2], rows[0], rows[3]],
        }
        result = selected_for_base_k(ranked, 2)
        self.assertEqual(result["union_all_three"], {"a", "b", "c"})
        self.assertEqual(result["two_of_three"], {"a", "b"})
        self.assertEqual(result["intersection_all_three"], {"b"})

    def test_metric_counts(self):
        result = metrics({"a", "b", "c"}, {"b", "d"}, 5)
        self.assertEqual(result["selected_count"], 3)
        self.assertEqual(result["incorrect_selected"], 1)
        self.assertEqual(result["correct_selected"], 2)
        self.assertEqual(result["error_recall"], .5)
        self.assertAlmostEqual(result["review_precision"], 1/3)
        self.assertEqual(result["missed_error_ids"], ["d"])


if __name__ == "__main__":
    unittest.main()
