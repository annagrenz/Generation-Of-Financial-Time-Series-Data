"""
Stylised-fact statistics for a matrix of return paths, shape (N, T).

Every statistic is computed per path and averaged (temporal statistics), or on
the pooled returns (distributional statistics). The same functions are applied
to real windows, generated paths and baselines, so all are measured identically.

Facts (Cont 2001; Chakraborti et al. 2011):
  heavy tails            excess kurtosis, VaR / ES at 1 % and 5 %, Hill tail index of |r|
  no linear ACF          ACF of r, lags 1..max_lag
  volatility clustering  ACF of |r| and r^2, lags 1..max_lag
  leverage effect        corr(r_t, |r_{t+k}|), k = 1..lev_lags
  gain/loss asymmetry    skewness; P(R > th | |R| > th) over a threshold grid
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample ACF at lags 1..max_lag (standard biased estimator, denominator sum of squares)."""
    x = np.asarray(x, np.float64) - np.mean(x)
    v = float(x @ x)
    n = len(x)
    if v < 1e-30:
        return np.zeros(max_lag)
    return np.array([float(x[:n - k] @ x[k:]) / v for k in range(1, max_lag + 1)])


def leverage(r: np.ndarray, max_lag: int) -> np.ndarray:
    """corr(r_t, |r_{t+k}|) for k = 1..max_lag, using full-sample means and SDs."""
    r = np.asarray(r, np.float64)
    a = np.abs(r)
    rc, ac = r - r.mean(), a - a.mean()
    d = r.std() * a.std() * len(r)
    if d < 1e-30:
        return np.zeros(max_lag)
    n = len(r)
    return np.array([float(rc[:n - k] @ ac[k:]) / d for k in range(1, max_lag + 1)])


def gain_loss_curve(pooled: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    out = []
    for th in thresholds:
        m = np.abs(pooled) > th
        out.append(float((pooled[m] > 0).mean()) if m.any() else np.nan)
    return np.array(out)


def hill_tail_index(pooled: np.ndarray, tail_frac: float = 0.05) -> float:
    """Hill estimator of the tail index of |r| using the largest tail_frac observations."""
    a = np.sort(np.abs(pooled))[::-1]
    k = max(int(len(a) * tail_frac), 2)
    top, thr = a[:k], a[k]
    if thr <= 0:
        return np.nan
    return float(1.0 / np.mean(np.log(top / thr)))


def var_es(pooled: np.ndarray, level: float) -> tuple[float, float]:
    """Left-tail VaR and ES of returns at `level` (reported as positive losses)."""
    q = np.quantile(pooled, level)
    return float(-q), float(-pooled[pooled <= q].mean())


def stylised_stats(paths: np.ndarray, max_lag: int = 20, lev_lags: int = 10,
                   thresholds: np.ndarray | None = None) -> dict:
    paths = np.asarray(paths, np.float64)
    if paths.ndim == 1:
        paths = paths[None]
    pooled = paths.ravel()
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.03, 31)
    var1, es1 = var_es(pooled, 0.01)
    var5, es5 = var_es(pooled, 0.05)
    return {
        "mean": float(pooled.mean()), "std": float(pooled.std()),
        "skew": float(sps.skew(pooled)), "excess_kurtosis": float(sps.kurtosis(pooled)),
        "var_1": var1, "es_1": es1, "var_5": var5, "es_5": es5,
        "hill_alpha": hill_tail_index(pooled),
        "acf_r": np.mean([acf(p, max_lag) for p in paths], axis=0),
        "acf_abs": np.mean([acf(np.abs(p), max_lag) for p in paths], axis=0),
        "acf_sq": np.mean([acf(p ** 2, max_lag) for p in paths], axis=0),
        "leverage": np.mean([leverage(p, lev_lags) for p in paths], axis=0),
        "gain_loss": gain_loss_curve(pooled, thresholds),
        "thresholds": thresholds,
    }


def _rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def discrepancies(gen: dict, real: dict) -> dict:
    """Scalar discrepancy per stylised fact between generated and real statistics (0 = identical)."""
    return {
        "tails_var1": abs(gen["var_1"] - real["var_1"]),
        "tails_es1": abs(gen["es_1"] - real["es_1"]),
        "tails_var5": abs(gen["var_5"] - real["var_5"]),
        "tails_es5": abs(gen["es_5"] - real["es_5"]),
        "tails_hill": abs(gen["hill_alpha"] - real["hill_alpha"]),
        "acf_r_rmse": _rmse(gen["acf_r"], real["acf_r"]),
        "acf_abs_rmse": _rmse(gen["acf_abs"], real["acf_abs"]),
        "acf_sq_rmse": _rmse(gen["acf_sq"], real["acf_sq"]),
        "leverage_rmse": _rmse(gen["leverage"], real["leverage"]),
        "skew": abs(gen["skew"] - real["skew"]),
        "gain_loss_rmse": _rmse(gen["gain_loss"], real["gain_loss"]),
        "std_ratio_log": abs(np.log(gen["std"] / real["std"])),
    }
