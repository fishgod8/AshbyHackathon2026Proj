import unittest

from experiments.uncertainty.run_carcinoma_priority_experiment import (
    build_rankings, enrich, selection_metrics, selection_curves, voting_selection,
)


class CarcinomaPriorityTests(unittest.TestCase):
    def rows(self):
        raw = []
        examples = [
            ("x", "InSitu", "Benign", [.55, .20, .20, .05]),
            ("y", "Normal", "Normal", [.05, .05, .05, .85]),
            ("z", "Invasive", "Invasive", [.05, .05, .85, .05]),
        ]
        classes = ("Benign", "InSitu", "Invasive", "Normal")
        for image_id, true, predicted, values in examples:
            ordered = sorted(values, reverse=True)
            raw.append({"image_id": image_id, "true_label": true, "predicted_label": predicted,
                        "confidence": max(values), "normalized_entropy": .5,
                        "margin": ordered[0] - ordered[1], "is_error": true != predicted,
                        **{f"score_{name}": value for name, value in zip(classes, values)}})
        return enrich(raw)

    def test_binary_probabilities_and_false_negative_definition(self):
        rows = self.rows()
        self.assertAlmostEqual(rows[0]["p_carcinoma"], .4)
        self.assertAlmostEqual(rows[0]["p_noncarcinoma"], .6)
        self.assertTrue(rows[0]["is_carcinoma_false_negative"])
        self.assertFalse(rows[1]["is_carcinoma_false_negative"])

    def test_rankings_do_not_require_true_labels_for_sort_keys(self):
        rows = self.rows()
        first = build_rankings(rows)
        for row in rows:
            row["true_label"] = "changed-after-ranking-definition"
        second = build_rankings(rows)
        for strategy in first:
            self.assertEqual([r["image_id"] for r in first[strategy]],
                             [r["image_id"] for r in second[strategy]])

    def test_selection_metrics_distinguish_target_outcomes(self):
        rows = self.rows()
        result = selection_metrics(rows[:1], rows)
        self.assertEqual(result["carcinoma_false_negative_recall"], 1.0)
        self.assertEqual(result["overall_error_recall"], 1.0)
        self.assertEqual(result["true_carcinoma_recall"], .5)

    def test_union_and_two_of_three_are_set_selections(self):
        rows = self.rows()
        rankings = build_rankings(rows)
        base = {name: rankings[name] for name in ("confidence", "four_class_entropy", "top_two_margin")}
        union = voting_selection(base, 1, 1)
        two_vote = voting_selection(base, 1, 2)
        self.assertGreaterEqual(len(union), len(two_vote))
        self.assertEqual(len({row["image_id"] for row in union}), len(union))

    def test_selection_curves_report_actual_set_sizes(self):
        rows = self.rows()
        curves = selection_curves(rows, build_rankings(rows))
        self.assertIn("union_top_k", curves)
        self.assertIn("two_of_three", curves)
        self.assertEqual(curves["confidence"][0]["selected"], build_rankings(rows)["confidence"][:1])


if __name__ == "__main__":
    unittest.main()
