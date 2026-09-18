"""Numerical and protocol checks, runnable with unittest (no pytest needed)."""

import pickle
import unittest

import numpy as np
from sklearn.isotonic import IsotonicRegression

from gmcm25b.features import build_meta_features, eesm_db, sequence_statistics
from gmcm25b.models import (
    ModelConfig, best_blend_weight, fit_model, isotonic_weights, make_splits,
    nearest_labels, search_parameters,
)


class EESMTests(unittest.TestCase):
    def test_constant_channel_matches_db_appendix_not_linear_eesm(self):
        X = np.full((2, 122), 10.0)
        np.testing.assert_allclose(eesm_db(X, 2, 4), [5, 5])

    def test_stable_formula_and_two_clipping_levels(self):
        X = np.array([[1.0, 10.0, 100.0], [1e200, 1e200, 1e200]])
        expected = -2 * np.log(np.mean(np.exp(-np.array([0, 10, 20]) / 3)))
        self.assertAlmostEqual(eesm_db(X[:1], 2, 3)[0], expected)
        self.assertEqual(eesm_db(X[1:], 2, .1, winsor=None)[0], 300)
        self.assertEqual(eesm_db(X[1:], 2, .1)[0], 200)
        self.assertTrue(np.isfinite(eesm_db(np.zeros((1, 122)), 6, .1)).all())

    def test_feature_count_segments_and_inference_categories(self):
        X = np.array([[1.0] * 40 + [10.0] * 41 + [100.0] * 41])
        features = build_meta_features(X, [5], [-95], 2, 2, ["unknown"], ["a", "b"])
        self.assertEqual(sequence_statistics(X).shape[1], 23)
        self.assertEqual(features.shape[1], 33)
        self.assertEqual(features.columns[0], "eff_db")
        np.testing.assert_allclose(features[["eff_seg1", "eff_seg2", "eff_seg3"]], [[0, 10, 20]])
        np.testing.assert_array_equal(features[["file_a", "file_b"]], [[0, 0]])
        constant = sequence_statistics(np.ones((2, 122)))
        self.assertTrue(np.isfinite(constant.to_numpy()).all())


class ModelProtocolTests(unittest.TestCase):
    def test_inverse_frequency_weights_and_disjoint_group_folds(self):
        y = np.array([1, 1, 1, 2, 2, 3.0])
        weights = isotonic_weights(y)
        self.assertAlmostEqual(weights.mean(), 1)
        sums = [weights[y == value].sum() for value in np.unique(y)]
        np.testing.assert_allclose(sums, [2, 2, 2])
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        folds = make_splits(y, groups)
        self.assertEqual(len(folds), 3)
        for train, valid in folds:
            self.assertFalse(set(groups[train]) & set(groups[valid]))

    def test_appendix_weight_difference_from_true_mse_minimum(self):
        iso = np.zeros(4)
        g = np.array([1., 2., 3., 4.])
        y = g * .2
        faithful = best_blend_weight(y, iso, g, "variance")
        corrected = best_blend_weight(y, iso, g, "mean_square")
        self.assertAlmostEqual(corrected, .2)
        self.assertEqual(faithful, 1.)
        self.assertLess(np.mean((corrected * g - y)**2), np.mean((faithful * g - y)**2))
        self.assertEqual(best_blend_weight(y, iso, np.ones(4)), .5)

    def test_alpha_scaling_reuse_has_same_scores_without_clipping(self):
        rng = np.random.RandomState(5)
        X = np.exp(rng.normal(1, .3, (60, 122)))
        y = np.round(X.mean(axis=1), 1)
        groups = np.repeat(["a", "b", "c"], 20)
        config = ModelConfig(alpha_min=1, alpha_max=1.2, alpha_step=.2,
                             beta_min=3, beta_max=3, beta_logspace=False, refine_tries=0)
        _, _, ledger, _ = search_parameters(X, y, groups, config)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(ledger.iloc[1]["evaluation"], "positive_alpha_scaling_without_clipping")
        self.assertEqual(ledger.iloc[0]["cv_mse"], ledger.iloc[1]["cv_mse"])
        # Independent direct fits verify that cached scale invariance is valid.
        eff = eesm_db(X, 1.2, 3)
        weights = isotonic_weights(y)
        fold_errors = []
        for train, valid in make_splits(y, groups):
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                eff[train], y[train], sample_weight=weights[train])
            fold_errors.append(np.mean((y[valid] - iso.predict(eff[valid]))**2))
        self.assertAlmostEqual(ledger.iloc[1]["cv_mse"], np.mean(fold_errors), places=12)

    def test_fixed_fit_pickle_roundtrip_and_observed_label_outputs(self):
        rng = np.random.RandomState(6)
        X = np.exp(rng.normal(2, .5, (90, 122)))
        y = np.repeat([10., 20., 30.], 30)
        groups = np.tile(np.repeat(["a", "b", "c"], 10), 3)
        noise = np.full(90, -95.)
        config = ModelConfig(hgbr_candidates=({"max_iter": 12, "max_leaf_nodes": 5,
                                               "learning_rate": .1, "l2_regularization": 0},))
        model, metrics, search = fit_model(X, y, groups, noise, groups, config, fixed_parameters=(2, 4))
        copy = pickle.loads(pickle.dumps(model))
        p1 = model.predict(X[:5], noise[:5], np.repeat("unknown", 5))
        p2 = copy.predict(X[:5], noise[:5], np.repeat("unknown", 5))
        np.testing.assert_allclose(p1, p2)
        self.assertTrue(set(p1.label).issubset(set(y)))
        self.assertEqual(search.iloc[0].phase, "fixed_parameters")
        self.assertIn("training_fit", metrics)
        self.assertIn("iso_oof_at_selected_parameters", metrics)
        np.testing.assert_equal(nearest_labels([15, 16, -5, 100], [10, 20, 30]), [10, 20, 10, 30])


if __name__ == "__main__":
    unittest.main()
