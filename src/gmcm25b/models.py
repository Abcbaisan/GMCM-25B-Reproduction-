"""Reconstructed EESM → isotonic → HGBR model for question 1.

Appendix code is followed where the main text and appendix differ. The source
paper omits its main program, so this module is an explicit reconstruction, not
the original authors' executable release. Search/OOF scores and training fit
scores are named separately; none is an independent full-pipeline test score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from .features import apply_eesm_alpha, build_meta_features, eesm_db, eesm_unit_alpha, validate_sinr


HGBR_CANDIDATES = (
    {"max_leaf_nodes": 31, "learning_rate": 0.08, "max_iter": 300, "l2_regularization": 0.01},
    {"max_leaf_nodes": 63, "learning_rate": 0.06, "max_iter": 400, "l2_regularization": 0.02},
    {"max_leaf_nodes": 31, "learning_rate": 0.10, "max_iter": 250, "l2_regularization": 0.00},
)


@dataclass
class ModelConfig:
    protocol: str = "paper_appendix"
    alpha_min: float = 1.0
    alpha_max: float = 6.0
    alpha_step: float = 0.2
    beta_min: float = 0.1
    beta_max: float = 20.0
    beta_step: float = 0.5
    beta_logspace: bool = True
    n_splits: int = 5
    balance_isotonic: bool = True
    winsor_eff_db: tuple[float, float] | None = (-200.0, 200.0)
    random_state: int = 42
    refine_tries: int = 30
    # Optional short candidate list is useful for smoke tests, not paper claims.
    hgbr_candidates: tuple[dict, ...] = field(default_factory=lambda: tuple(dict(v) for v in HGBR_CANDIDATES))


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    mse = float(mean_squared_error(y, prediction))
    return {
        "n": int(len(y)), "mse": mse, "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)) if len(y) > 1 else float("nan"),
    }


def isotonic_weights(y: np.ndarray) -> np.ndarray:
    """Appendix: inverse frequency of labels rounded to three decimal places."""
    _, inverse = np.unique(np.round(np.asarray(y, dtype=float), 3), return_inverse=True)
    weights = 1.0 / (np.bincount(inverse)[inverse].astype(float) + 1e-12)
    return weights * (len(weights) / weights.sum())


def make_splits(y: np.ndarray, groups: np.ndarray | None, n_splits: int = 5):
    """GroupKFold first; reproduce the appendix's seeded KFold fallback."""
    if len(y) < 2 or n_splits < 2:
        raise ValueError("At least two samples and two requested folds are needed")
    if groups is not None:
        groups = np.asarray(groups, dtype=str)
        if groups.shape != np.asarray(y).shape:
            raise ValueError("groups and y must have identical one-dimensional shapes")
        n_groups = len(np.unique(groups))
        if n_groups >= 2:
            return list(GroupKFold(n_splits=min(n_splits, n_groups)).split(np.zeros(len(y)), y, groups))
    folds = min(n_splits, max(2, len(y) // 10), len(y))
    return list(KFold(n_splits=folds, shuffle=True, random_state=42).split(np.zeros(len(y)), y))


def best_blend_weight(
    y: np.ndarray, iso_prediction: np.ndarray, hgbr_prediction: np.ndarray,
    denominator: str = "variance",
) -> float:
    """Weight on HGBR; ``variance`` is appendix, ``mean_square`` minimizes MSE.

    The appendix uses var(d) without centering the numerator. This is generally
    not the MSE minimizer; retain it faithfully and expose the correction for
    auditing. The appendix returns 0.5 for variance below 1e-12.
    """
    delta = np.asarray(hgbr_prediction) - np.asarray(iso_prediction)
    if denominator == "variance":
        normalizer = np.var(delta)
    elif denominator == "mean_square":
        normalizer = np.mean(delta**2)
    else:
        raise ValueError("denominator must be 'variance' or 'mean_square'")
    if normalizer < 1e-12:
        return 0.5
    return float(np.clip(np.mean((np.asarray(y) - iso_prediction) * delta) / normalizer, 0, 1))


def nearest_labels(prediction: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Round to observed training labels; ties choose the lower label."""
    labels = np.sort(np.unique(np.asarray(labels, dtype=float)))
    prediction = np.asarray(prediction, dtype=float)
    upper = np.clip(np.searchsorted(labels, prediction), 0, len(labels) - 1)
    lower = np.maximum(upper - 1, 0)
    choose_lower = np.abs(prediction - labels[lower]) <= np.abs(prediction - labels[upper])
    return np.where(choose_lower, labels[lower], labels[upper])


def _iso_cv(eff, y, splits, weights, fold_local_weights=False):
    prediction = np.empty(len(y), dtype=float)
    fold_mse, fold_r2 = [], []
    for train, valid in splits:
        sw = isotonic_weights(y[train]) if fold_local_weights else weights[train]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(eff[train], y[train], sample_weight=sw)
        prediction[valid] = np.clip(iso.predict(eff[valid]), y.min(), y.max())
        fold_mse.append(mean_squared_error(y[valid], prediction[valid]))
        fold_r2.append(r2_score(y[valid], prediction[valid]) if len(valid) > 1 else np.nan)
    return float(np.mean(fold_mse)), float(np.nanmean(fold_r2)), prediction


def search_parameters(X, y, groups, config: ModelConfig, progress: Callable[[str], None] | None = None):
    """Full appendix grid and 11-seed local search, with auditable reuse records.

    For a fixed beta and *no clipping*, positive alpha rescales every x value
    by a common factor. Weighted isotonic fit and linear interpolation therefore
    produce the same predictions. Only this analytically equivalent case is
    reused; clipped cases are evaluated separately. Every grid point is logged.
    Alpha may exceed six during local refinement, as in the appendix.
    """
    splits = make_splits(y, groups, config.n_splits)
    weights = isotonic_weights(y) if config.balance_isotonic else np.ones(len(y))
    alphas = np.arange(config.alpha_min, config.alpha_max + 1e-9, config.alpha_step)
    betas = np.arange(config.beta_min, config.beta_max + 1e-9, config.beta_step)
    if config.beta_logspace:
        betas = np.unique(np.concatenate((
            np.logspace(np.log10(max(config.beta_min, 1e-2)), np.log10(config.beta_max), 40), betas,
        )))
    records: list[dict] = []
    units: dict[float, np.ndarray] = {}
    unclipped_scores: dict[float, tuple[float, float, float]] = {}
    exact_scores: dict[tuple[float, float], tuple[float, float]] = {}
    best = {"alpha": None, "beta": None, "cv_mse": np.inf, "cv_r2": -np.inf}

    def evaluate(alpha, beta, phase):
        alpha, beta = float(alpha), float(beta)
        key = (alpha, beta)
        if beta not in units:
            units[beta] = eesm_unit_alpha(X, beta)
        unit = units[beta]
        eff = apply_eesm_alpha(unit, alpha, config.winsor_eff_db)
        no_clip = bool(np.all(eff == alpha * unit))
        reuse_alpha = None
        if key in exact_scores:
            mse, r2 = exact_scores[key]
            reuse = "identical_parameters"
        elif no_clip and beta in unclipped_scores:
            mse, r2, reuse_alpha = unclipped_scores[beta]
            reuse = "positive_alpha_scaling_without_clipping"
        else:
            mse, r2, _ = _iso_cv(eff, y, splits, weights)
            reuse = "evaluated"
            if no_clip:
                unclipped_scores[beta] = (mse, r2, alpha)
        exact_scores[key] = mse, r2
        result = {"phase": phase, "alpha": alpha, "beta": beta,
                  "cv_mse": mse, "cv_r2": r2, "evaluation": reuse,
                  "reuse_alpha": reuse_alpha, "clipped_count": int(np.sum(eff != alpha * unit))}
        records.append(result)
        return result

    def better(candidate, incumbent):
        return candidate["cv_mse"] + 1e-12 < incumbent["cv_mse"] or (
            abs(candidate["cv_mse"] - incumbent["cv_mse"]) < 1e-9
            and candidate["cv_r2"] > incumbent["cv_r2"]
        )

    for alpha in alphas:
        if alpha < 1:
            continue
        for beta in betas:
            candidate = evaluate(alpha, beta, "coarse_grid")
            if better(candidate, best):
                best = candidate
        if progress:
            progress(f"EESM coarse search: alpha={alpha:.2f}; best MSE={best['cv_mse']:.6f}")
    if best["alpha"] is None:
        raise ValueError("Parameter grid is empty")
    leaders = sorted(records, key=lambda r: (r["cv_mse"], -r["cv_r2"], r["alpha"], r["beta"]))[:6]
    a0, b0 = best["alpha"], best["beta"]
    seeds = [(a0, b0)] + [(r["alpha"], r["beta"]) for r in leaders]
    seeds += [(max(1.0, a0 + da), max(config.beta_min, min(config.beta_max, b0 + db)))
              for da in (-0.2, 0.2) for db in (-0.5, 0.5)]
    rng = np.random.RandomState(config.random_state)
    best_local = {"alpha": None, "beta": None, "cv_mse": np.inf, "cv_r2": -np.inf}
    for seed_index, (alpha, beta) in enumerate(seeds):
        for _ in range(config.refine_tries):
            a = max(1.0, alpha + rng.uniform(-0.3, 0.3))
            b = max(config.beta_min, min(config.beta_max, beta * rng.uniform(0.8, 1.25)))
            candidate = evaluate(a, b, "local_refinement")
            candidate["seed_index"] = seed_index
            if better(candidate, best_local):
                best_local = candidate
        if progress and config.refine_tries:
            progress(f"EESM local search: seed {seed_index + 1}/{len(seeds)}")
    # Appendix requires a strictly better MSE to replace the coarse winner.
    if best_local["cv_mse"] + 1e-12 < best["cv_mse"]:
        best = best_local
    return float(best["alpha"]), float(best["beta"]), pd.DataFrame(records), dict(best)


@dataclass
class QuestionOneModel:
    alpha: float
    beta: float
    iso: IsotonicRegression
    hgbr: HistGradientBoostingRegressor
    blend_weight: float
    blend_weight_mse_audit: float
    labels: np.ndarray
    file_categories: tuple[str, ...]
    feature_columns: tuple[str, ...]
    config: ModelConfig
    iso_oof_: np.ndarray
    training_predictions_: pd.DataFrame

    def predict(self, X, noise_dbm, files) -> pd.DataFrame:
        eff = eesm_db(X, self.alpha, self.beta, winsor=self.config.winsor_eff_db)
        features = build_meta_features(X, eff, noise_dbm, self.alpha, self.beta, files, self.file_categories)
        if tuple(features.columns) != self.feature_columns:
            raise ValueError("Prediction features differ from training schema")
        iso = np.clip(self.iso.predict(eff), self.labels.min(), self.labels.max())
        hgbr = self.hgbr.predict(features)
        blend = np.clip(self.blend_weight * hgbr + (1 - self.blend_weight) * iso,
                        self.labels.min(), self.labels.max())
        audit = np.clip(self.blend_weight_mse_audit * hgbr + (1 - self.blend_weight_mse_audit) * iso,
                        self.labels.min(), self.labels.max())
        return pd.DataFrame({"eff_db": eff, "iso": iso, "hgbr": hgbr, "blend": blend,
                             "label": nearest_labels(blend, self.labels), "blend_mse_audit": audit})


def fit_model(
    X, y, groups, noise_dbm, files, config: ModelConfig | None = None,
    fixed_parameters: tuple[float, float] | None = None,
    progress: Callable[[str], None] | None = None,
):
    """Fit and return ``(pickleable_model, metrics_dict, parameter_search_df)``.

    Fixed parameters skip search, enabling independently specified parameter
    device-holdout experiments. Only labels passed into this call are used.
    HGBR candidate selection is by training R², as in appendix p. 65, and its
    early-stopping validation is random rather than device-grouped. A fixed
    random_state is an explicit reproducibility addition to the appendix.
    """
    config = ModelConfig() if config is None else config
    if config.protocol != "paper_appendix":
        raise ValueError("Only the explicitly reconstructed paper_appendix protocol is implemented")
    X = validate_sinr(X)
    y = np.asarray(y, dtype=float)
    files = np.asarray(files, dtype=str)
    noise_dbm = np.asarray(noise_dbm, dtype=float)
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError("Finite one-dimensional labels must align with X")
    if files.shape != y.shape or noise_dbm.shape != y.shape or not np.isfinite(noise_dbm).all():
        raise ValueError("Finite noise and file identifiers must align with X")
    if groups is not None:
        groups = np.asarray(groups, dtype=str)
    splits = make_splits(y, groups, config.n_splits)
    weights = isotonic_weights(y) if config.balance_isotonic else np.ones(len(y))
    if fixed_parameters is None:
        alpha, beta, search, selected = search_parameters(X, y, groups, config, progress)
    else:
        alpha, beta = map(float, fixed_parameters)
        selected = {"phase": "fixed_parameters", "alpha": alpha, "beta": beta}
        search = pd.DataFrame([selected])
    eff = eesm_db(X, alpha, beta, config.winsor_eff_db)
    iso = IsotonicRegression(out_of_bounds="clip").fit(eff, y, sample_weight=weights)
    _, _, iso_oof = _iso_cv(eff, y, splits, weights, fold_local_weights=config.balance_isotonic)
    categories = tuple(np.unique(files))
    features = build_meta_features(X, eff, noise_dbm, alpha, beta, files, categories)
    best_hgbr, best_cfg, best_r2 = None, None, -np.inf
    hgbr_records = []
    for index, candidate in enumerate(config.hgbr_candidates):
        hgbr = HistGradientBoostingRegressor(
            loss="squared_error", early_stopping=True, validation_fraction=0.1,
            monotonic_cst=[1] + [0] * (features.shape[1] - 1),
            random_state=config.random_state, **candidate,
        )
        hgbr.fit(features, y)
        train_score = float(r2_score(y, hgbr.predict(features)))
        hgbr_records.append({**candidate, "train_r2": train_score, "n_iter": int(hgbr.n_iter_)})
        if train_score > best_r2:
            best_hgbr, best_cfg, best_r2 = hgbr, dict(candidate), train_score
        if progress:
            progress(f"HGBR candidate {index + 1}/{len(config.hgbr_candidates)}: train R2={train_score:.6f}")
    if best_hgbr is None:
        raise ValueError("At least one HGBR candidate is required")
    hgbr_prediction = best_hgbr.predict(features)
    weight = best_blend_weight(y, iso_oof, hgbr_prediction, "variance")
    weight_audit = best_blend_weight(y, iso_oof, hgbr_prediction, "mean_square")
    model = QuestionOneModel(alpha, beta, iso, best_hgbr, weight, weight_audit,
                             np.unique(y), categories, tuple(features.columns), config,
                             iso_oof, pd.DataFrame())
    prediction = model.predict(X, noise_dbm, files)
    model.training_predictions_ = prediction.assign(y_true=y, iso_oof=iso_oof)
    metrics = {
        "protocol": config.protocol, "n_train": int(len(y)), "n_features": features.shape[1],
        "alpha": alpha, "beta": beta, "parameter_selection": selected,
        "splitter": "GroupKFold" if groups is not None and len(np.unique(groups)) >= 2 else "KFold",
        "n_folds": len(splits), "n_groups": None if groups is None else int(len(np.unique(groups))),
        "training_fit": {name: regression_metrics(y, prediction[name])
                         for name in ("iso", "hgbr", "blend", "blend_mse_audit")},
        "iso_oof_at_selected_parameters": regression_metrics(y, iso_oof),
        "iso_oof_scope": "ISO-only OOF; parameters may have been selected on the same folds; not an independent full-pipeline score",
        "hgbr_selection_rule": "largest training R2 among three appendix configurations",
        "hgbr_candidates": hgbr_records, "hgbr_selected": best_cfg,
        "blend_weight_hgbr_appendix_variance": weight,
        "blend_weight_hgbr_mse_audit": weight_audit,
        "blend_weight_data": "ISO OOF predictions and HGBR in-sample predictions",
        "reconstruction_additions": ["fixed HGBR random_state", "stable training feature schema", "vectorized arithmetic", "full parameter search ledger"],
    }
    return model, metrics, search
