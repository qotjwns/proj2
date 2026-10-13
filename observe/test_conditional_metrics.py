import itertools
import unittest

import numpy as np

from observe.conditional_metrics import conditional_comparison, quantile_bins, tail_weights


class ConditionalMetricsTests(unittest.TestCase):
    def test_control_ties_are_never_split(self):
        values = np.repeat([0.1, 0.5, 0.9], [3, 5, 4])
        bins = quantile_bins(values, 10)
        for value in np.unique(values):
            self.assertEqual(len(np.unique(bins[values == value])), 1)
        self.assertEqual(len(np.unique(quantile_bins(np.ones(8), 10))), 1)

    def test_boundary_ties_match_all_permutations(self):
        scores = np.array([1, 2, 2, 2, 3, 4])
        errors = np.array([1, 0, 1, 0, 0, 1])
        weights, _ = tail_weights(scores, 2)
        expected = np.mean([errors[[0, *p]][:2].sum() for p in itertools.permutations([1, 2, 3])])
        self.assertAlmostEqual(weights.sum(), 2)
        self.assertAlmostEqual(weights @ errors, expected)
        high, _ = tail_weights(scores, 4, high=True)
        np.testing.assert_allclose(weights + high, 1)

    def test_added_score_perfectly_separates_errors_within_bins(self):
        control = np.repeat([0.2, 0.8], 8)
        target = np.tile(np.arange(8), 2)
        errors = np.tile([1, 1, 1, 1, 0, 0, 0, 0], 2)
        result, membership = conditional_comparison(control, target, errors, bins=2, repetitions=20)
        self.assertEqual(result["summary"]["actual_bins"], 2)
        self.assertEqual(result["summary"]["low_risk"], 1)
        self.assertEqual(result["summary"]["high_risk"], 0)
        self.assertEqual(result["summary"]["gap"], 1)
        np.testing.assert_allclose(membership["low_weight"].sum(), 4)
        np.testing.assert_allclose(membership["high_weight"].sum(), 4)

    def test_constant_target_cannot_separate_errors(self):
        control = np.arange(16)
        result, _ = conditional_comparison(control, np.ones(16), np.arange(16) % 3 == 0,
                                           bins=2, repetitions=30)
        self.assertEqual(result["summary"]["gap"], 0)
        np.testing.assert_allclose(result["summary"]["gap_ci"], [0, 0])
        self.assertTrue(all(not row["target_has_variation"] for row in result["bins"]))

    def test_membership_does_not_depend_on_errors(self):
        control = np.arange(20)
        target = np.tile([2, 1, 1, 3, 4], 4)
        first, memberships = conditional_comparison(control, target, control % 2, bins=2, repetitions=20)
        second, flipped = conditional_comparison(control, target, 1 - control % 2, bins=2, repetitions=20)
        for key in memberships:
            np.testing.assert_array_equal(memberships[key], flipped[key])
        self.assertAlmostEqual(first["summary"]["gap"], -second["summary"]["gap"])


if __name__ == "__main__":
    unittest.main()
