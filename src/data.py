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

def all_log_returns(path=DATA_PATH) -> pd.DataFrame:
    """
    Log returns of the full series, computed once in chronological order and indexed by the
    date of the later price, with the calendar time step dt (years) of each return.
    A return belongs to the period that contains its end date, so the first return of a test
    period uses the last close before it (known at that time; not leakage).
    """
    p = load_prices(path)
    r = log_returns(p)
    dt = np.diff(p.index.values).astype("timedelta64[D]").astype(np.float64) / DAYS_PER_YEAR
    return pd.DataFrame({"r": r.values, "dt": dt}, index=r.index)


@dataclass(frozen=True)
class Fold:
    """A train/test split. All dates are inclusive. For the development split the 'test'
    period is the validation period."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    role: str = "evaluation"     # "development" | "evaluation" | "replication"

    def train_prices(self, path=DATA_PATH) -> pd.Series:
        return load_prices(path, self.train_start, self.train_end)

    def test_prices(self, path=DATA_PATH) -> pd.Series:
        return load_prices(path, self.test_start, self.test_end)

    def train_returns(self, path=DATA_PATH) -> pd.DataFrame:
        return all_log_returns(path).loc[self.train_start:self.train_end]

    def test_returns(self, path=DATA_PATH) -> pd.DataFrame:
        return all_log_returns(path).loc[self.test_start:self.test_end]


# Development split: every data-dependent decision (preprocessing choices, training budget,
# checkpoint rule, metric definitions, seed count) is made on data up to 2013-12-31 only,
# i.e. before the first evaluation year. Train 1995-2011, validate 2012-2013.
DEVELOPMENT = Fold(0, "1995-01-01", "2011-12-31", "2012-01-01", "2013-12-31", role="development")

# Evaluation folds: expanding training window, one-year test block (2014-2023).
FOLDS: dict[int, Fold] = {
    k: Fold(k, "1995-01-01", f"{year - 1}-12-31", f"{year}-01-01", f"{year}-12-31")
    for k, year in enumerate(range(2014, 2024), start=1)
}

# Replication of the Chung & Sester (2025, Sec. 5.1) split: train to 2018-09-18. Reported
# separately; never used for development decisions and not part of the between-model analysis.
REPLICATION = Fold(99, "1995-01-01", "2018-09-18", "2018-09-19", "2023-12-28", role="replication")

ALL_SPLITS: dict[int, Fold] = {0: DEVELOPMENT, **FOLDS, 99: REPLICATION}


def get_fold(fold_id: int) -> Fold:
    """0 = development split, 1-10 = evaluation folds (test years 2014-2023), 99 = replication."""
    if fold_id not in ALL_SPLITS:
        raise KeyError(f"Unknown split {fold_id}; available: {sorted(ALL_SPLITS)}")
    return ALL_SPLITS[fold_id]


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
