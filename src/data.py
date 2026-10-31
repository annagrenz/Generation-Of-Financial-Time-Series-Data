"""
Single source of truth for the S&P 500 data used by every model.

Responsibilities
----------------
1. Load the price file and verify its SHA-256 checksum, so that local runs,
   Kaggle runs and analysis runs provably use byte-identical data.
2. Define the rolling-origin (expanding-window) folds used in the study.
3. Build model-specific training windows that lie entirely inside a period,
   so no window can leak test data into training.

Conventions
-----------
* Log return r_t = log(P_t) - log(P_{t-1}). Returns are computed *within* a
  period, so the first return of a period uses only prices from that period.
* Time is measured in calendar days / 365 (Chung & Sester, 2025, Sec. 5.1).
  The same convention is used for training and for generation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "spx_20231229.csv"

# SHA-256 of data/spx_20231229.csv as distributed with the Chung & Sester
# repository (identical to the file in the thesis GitHub repository).
DATA_SHA256 = "93d9412d561c8d73b8e38c479043a4d27a65559c9e9a5a090d6503d251dde6aa"

DAYS_PER_YEAR = 365.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def file_sha256(path: Path | str = DATA_PATH) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prices(path: Path | str = DATA_PATH,
                start: str | None = None,
                end: str | None = None,
                verify: bool = True) -> pd.Series:
    """Return the closing-price series (DatetimeIndex) between start and end, inclusive."""
    if verify:
        digest = file_sha256(path)
        if digest != DATA_SHA256:
            raise ValueError(
                f"Data checksum mismatch for {path}: got {digest}, expected {DATA_SHA256}. "
                "The data file differs from the one used in the study.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df.iloc[:, 0].astype(np.float64).dropna()
    if start is not None or end is not None:
        s = s.loc[start:end]
    return s


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns indexed by the date of the *later* price."""
    lp = np.log(prices.values)
    return pd.Series(np.diff(lp), index=prices.index[1:], name="log_return")


def calendar_time_vector(index: pd.DatetimeIndex) -> np.ndarray:
    """Cumulative time in years (calendar days / 365), starting at 0."""
    t = np.zeros(len(index), dtype=np.float64)
    if len(index) > 1:
        days = np.diff(index.values).astype("timedelta64[D]").astype(np.float64)
        t[1:] = np.cumsum(days / DAYS_PER_YEAR)
    return t


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold. All dates are inclusive."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    val_fraction: float = 0.10   # last share of the training period held out in pilot mode

    def train_prices(self, pilot: bool = False, path=DATA_PATH) -> pd.Series:
        """Training prices. In pilot mode the validation slice is excluded."""
        p = load_prices(path, self.train_start, self.train_end)
        if pilot:
            n_val = int(round(len(p) * self.val_fraction))
            p = p.iloc[: len(p) - n_val]
        return p

    def val_prices(self, path=DATA_PATH) -> pd.Series:
        """Validation slice = last `val_fraction` of the training period (pilot decisions only)."""
        p = load_prices(path, self.train_start, self.train_end)
        n_val = int(round(len(p) * self.val_fraction))
        return p.iloc[len(p) - n_val:]

    def test_prices(self, path=DATA_PATH) -> pd.Series:
        return load_prices(path, self.test_start, self.test_end)


# Expanding-window folds with one-year test blocks (2014-2023): ten folds, so that a paired
# sign-flip permutation test across folds can reach p = 2/2^10 ~ 0.002 (with five folds the
# smallest attainable two-sided p-value would be 2/32 = 0.0625 > 0.05).
FOLDS: dict[int, Fold] = {
    k: Fold(k, "1995-01-01", f"{year - 1}-12-31", f"{year}-01-01", f"{year}-12-31")
    for k, year in enumerate(range(2014, 2024), start=1)
}

# Anchor split of Chung & Sester (2025, Sec. 5.1): train to 2018-09-18, test to the end of the data.
# Used for the pilot, the capacity check, the seed-variance estimate and the replication of the
# original setup; it is NOT one of the ten folds entering the between-model tests.
ANCHOR = Fold(0, "1995-01-01", "2018-09-18", "2018-09-19", "2023-12-28")
FOLDS_AND_ANCHOR = {0: ANCHOR, **FOLDS}


def get_fold(fold_id: int) -> Fold:
    """Fold 0 is the Chung & Sester anchor split; folds 1-10 have test years 2014-2023."""
    if fold_id not in FOLDS_AND_ANCHOR:
        raise KeyError(f"Unknown fold {fold_id}; available: {sorted(FOLDS_AND_ANCHOR)}")
    return FOLDS_AND_ANCHOR[fold_id]


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def log_price_windows(prices: pd.Series, sample_len: int, stride: int) -> np.ndarray:
    """
    Windows of (time, log price), both re-based to 0 at the window start.
    Shape (N, sample_len, 2). Used by the MMD-Signature model.
    """
    lp = np.log(prices.values)
    t = calendar_time_vector(prices.index)
    out = []
    for s in range(0, len(prices) - sample_len + 1, stride):
        e = s + sample_len
        out.append(np.stack([t[s:e] - t[s], lp[s:e] - lp[s]], axis=-1))
    if not out:
        raise ValueError("Period too short for the requested window length.")
    return np.asarray(out, dtype=np.float32)


def return_windows(returns: np.ndarray, sample_len: int, stride: int) -> np.ndarray:
    """Sliding windows of log returns, shape (N, sample_len). Used by the diffusion model."""
    r = np.asarray(returns, dtype=np.float64)
    out = [r[s:s + sample_len] for s in range(0, len(r) - sample_len + 1, stride)]
    if not out:
        raise ValueError("Series too short for the requested window length.")
    return np.asarray(out, dtype=np.float64)
