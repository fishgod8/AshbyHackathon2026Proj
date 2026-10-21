import unittest

from Ashley_uncertainty.evaluate_bach_uncertainty import read_rows


class BachEvaluationReproducibilityTests(unittest.TestCase):
    def test_saved_real_evaluation_is_reproducible(self):
        rows = read_rows("bach/evaluation.csv")
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(row["is_error"] for row in rows), 4)
        self.assertEqual(
            {row["predicted_label"] for row in rows},
            {"Benign", "InSitu", "Invasive", "Normal"},
        )


if __name__ == "__main__":
    unittest.main()
