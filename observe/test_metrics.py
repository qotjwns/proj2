"""Check selective-risk math against hand calculations and exhaustive ties."""

import itertools
import unittest

import numpy as np

from observe.metrics import confidence_scores, evaluate_logits, expected_risk_curve


class SelectiveMetricsTests(unittest.TestCase):
    def test_distinct_scores_and_discrete_aurc(self):
        risk, order = expected_risk_curve([0.2, 0.9, 0.5, 0.1], [1, 0, 1, 0])
        np.testing.assert_array_equal(order, [1, 2, 0, 3])
        np.testing.assert_allclose(risk, [0, 1 / 2, 2 / 3, 1 / 2])
        self.assertAlmostEqual(risk.mean(), 5 / 12)

    def test_ties_equal_exhaustive_uniform_permutations(self):
        scores = [3, 2, 2, 2, 1]
        errors = np.array([1, 0, 1, 0, 1])
        risk, _ = expected_risk_curve(scores, errors)
        exhaustive = [
            np.cumsum(errors[[0, *middle, 4]]) / np.arange(1, 6)
            for middle in itertools.permutations([1, 2, 3])
        ]
        np.testing.assert_allclose(risk, np.mean(exhaustive, axis=0))
        for permutation in itertools.permutations(range(5)):
            permutation = list(permutation)
            shuffled_risk, _ = expected_risk_curve(np.array(scores)[permutation], errors[permutation])
            np.testing.assert_allclose(shuffled_risk, risk)

    def test_all_tied_scores_are_random_rejection(self):
        risk, _ = expected_risk_curve(np.ones(10), [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(risk, 0.3)

    def test_scores_exclude_drainage_from_prediction_msp_and_margin(self):
        logits = np.array([[2.0, 1.0] + [0.0] * 8 + [100.0]])
        prediction, scores = confidence_scores(logits, 10)
        np.testing.assert_array_equal(prediction, [0])
        self.assertAlmostEqual(scores["msp"][0], np.exp(2) / (np.exp(2) + np.exp(1) + 8))
        self.assertEqual(scores["margin"][0], 1.0)
        self.assertAlmostEqual(scores["drainage"][0], 0.0)
        moved = np.concatenate([logits[:, -1:], logits[:, :-1]], axis=1)
        moved_prediction, moved_scores = confidence_scores(moved, 0)
        np.testing.assert_array_equal(moved_prediction, prediction)
        for key in scores:
            np.testing.assert_allclose(moved_scores[key], scores[key])

    def test_fixed_accuracy_coverage_boundary_and_aurc(self):
        # Every score ties; 3/100 errors remain 3% in expectation at every k.
        logits = np.zeros((100, 11))
        labels = np.r_[np.ones(3, dtype=int), np.zeros(97, dtype=int)]
        rows, curves, _ = evaluate_logits(logits, labels, 10)
        for row in rows:
            self.assertEqual(row["accuracy"], 0.97)
            for percentage in (90, 95, 99):
                self.assertAlmostEqual(row[f"risk_at_{percentage}"], 0.03)
            self.assertAlmostEqual(row["aurc"], 0.03)
        for key in ("drainage", "msp", "margin"):
            self.assertAlmostEqual(curves[key][-1], 0.03)

    def test_invalid_inputs_fail_instead_of_producing_metrics(self):
        with self.assertRaises(ValueError):
            confidence_scores(np.zeros((2, 10)), 10)
        with self.assertRaises(ValueError):
            confidence_scores(np.full((2, 11), np.nan), 10)
        with self.assertRaises(ValueError):
            evaluate_logits(np.zeros((2, 11)), [0, 10], 10)
        with self.assertRaises(ValueError):
            expected_risk_curve([0.5, np.inf], [0, 1])


if __name__ == "__main__":
    unittest.main()
