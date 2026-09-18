"""Question 1 features reconstructed from the paper's Appendix 2, pp. 57–64.

The appendix applies EESM to dB inputs. This differs from conventional linear
SINR EESM and is kept deliberately for the ``paper_appendix`` protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_sinr(X: np.ndarray) -> np.ndarray:
    """Require a finite sample-by-subcarrier matrix; allow zero before clipping."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be a two-dimensional matrix with subcarriers")
    if not np.isfinite(X).all() or (X < 0).any():
        raise ValueError("SINR must be finite and nonnegative")
    return X


def eesm_unit_alpha(X: np.ndarray, beta: float) -> np.ndarray:
    """Unclipped appendix EESM for alpha=1, cached during the parameter search."""
    X = validate_sinr(X)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    gamma_db = np.clip(10.0 * np.log10(np.clip(X, 1e-300, None)), -200, 200)
    z = -gamma_db / max(float(beta), 1e-6)
    zmax = np.max(z, axis=1)
    exp_mean = np.mean(np.exp(np.clip(z - zmax[:, None], -745, 709)), axis=1)
    return -(zmax + np.log(np.maximum(exp_mean, 1e-300)))


def apply_eesm_alpha(
    unit_alpha: np.ndarray,
    alpha: float,
    winsor: tuple[float, float] | None = (-200.0, 200.0),
) -> np.ndarray:
    """Apply alpha, the appendix's raw ±300 limit, then global winsorization."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    result = np.clip(float(alpha) * np.asarray(unit_alpha), -300.0, 300.0)
    return result if winsor is None else np.clip(result, *winsor)


def eesm_db(
    X: np.ndarray,
    alpha: float,
    beta: float,
    winsor: tuple[float, float] | None = (-200.0, 200.0),
) -> np.ndarray:
    """Vectorized dB-EESM: -alpha*log(mean(exp(-gamma_db/beta))).

    Global ISO/meta inputs use ``winsor=(-200, 200)``. Segment EESM uses
    ``winsor=None`` and therefore retains the appendix's ±300 raw clipping.
    """
    return apply_eesm_alpha(eesm_unit_alpha(X, beta), alpha, winsor)


def sequence_statistics(X: np.ndarray) -> pd.DataFrame:
    """Exactly the 23 base statistics listed on appendix pp. 62–63."""
    X = np.clip(validate_sinr(X), 1e-12, None)
    db = 10.0 * np.log10(X)
    result: dict[str, np.ndarray] = {}
    for prefix, values in (("lin", X), ("db", db)):
        result[f"{prefix}_mean"] = np.mean(values, axis=1)
        result[f"{prefix}_std"] = np.std(values, axis=1)
        result[f"{prefix}_min"] = np.min(values, axis=1)
        result[f"{prefix}_max"] = np.max(values, axis=1)
        for percentile, value in zip(
            (10, 25, 50, 75, 90), np.percentile(values, (10, 25, 50, 75, 90), axis=1)
        ):
            result[f"{prefix}_p{percentile}"] = value
        if prefix == "lin":
            # scipy.stats skew/kurtosis defaults: biased moments, Fisher kurtosis.
            centered = values - values.mean(axis=1, keepdims=True)
            variance = np.mean(centered**2, axis=1)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                result["lin_skew"] = np.mean(centered**3, axis=1) / variance**1.5
                result["lin_kurt"] = np.mean(centered**4, axis=1) / variance**2 - 3
    for threshold in (1, 3, 10):
        result[f"share_above_{threshold}"] = np.mean(X > threshold, axis=1)
    return pd.DataFrame(result).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_meta_features(
    X: np.ndarray,
    eff_db: np.ndarray,
    noise_dbm: np.ndarray,
    alpha: float,
    beta: float,
    files: np.ndarray,
    file_categories: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Build 23 statistics + 8 physical features + training file indicators.

    Subcarriers split at K//3 and 2*K//3: 122 becomes 40/41/41. Categories
    supplied from training are reused at inference; unknown files have zeros.
    ``eff_db`` is first, so HGBR's positive monotonic constraint is well defined.
    """
    X = validate_sinr(X)
    if X.shape[1] < 3:
        raise ValueError("At least three subcarriers are needed for segment features")
    eff_db = np.asarray(eff_db, dtype=float)
    noise_dbm = np.asarray(noise_dbm, dtype=float)
    files = np.asarray(files, dtype=str)
    if any(v.shape != (len(X),) for v in (eff_db, noise_dbm, files)):
        raise ValueError("eff_db, noise_dbm and files must align with rows of X")
    result = sequence_statistics(X)
    cuts = (0, X.shape[1] // 3, 2 * X.shape[1] // 3, X.shape[1])
    segment = np.column_stack([
        eesm_db(X[:, start:end], alpha, beta, winsor=None)
        for start, end in zip(cuts[:-1], cuts[1:])
    ])
    result["eff_db"] = eff_db
    result["phi_nl"] = np.logaddexp(0, eff_db * np.log(10.0) / 10.0)
    for index in range(3):
        result[f"eff_seg{index + 1}"] = segment[:, index]
    result["eff_seg_range"] = np.ptp(segment, axis=1)
    result["eff_seg_var"] = np.var(segment, axis=1)
    result["noise_dbm"] = noise_dbm
    categories = np.unique(files) if file_categories is None else file_categories
    for category in categories:
        result[f"file_{category}"] = (files == str(category)).astype(float)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result[["eff_db"] + [name for name in result.columns if name != "eff_db"]]
