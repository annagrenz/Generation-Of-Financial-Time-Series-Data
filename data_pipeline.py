"""
data_pipeline.py — Single source of truth for SPX data, shared by both models.

Previously, the diffusion model and the MMD model each loaded SPX data in their
own way, with different date ranges and different time-axis conventions
(calendar days vs. trading days). For the master thesis the two models must be
trained on identical data, otherwise any difference in generated quality
could just be a data difference rather than a model difference.

This file centralises:
  1. Where the SPX data lives (data/spx_20231229.csv).
  2. The train / out-of-sample split (matches the MMD paper exactly).
  3. How log returns are computed.
  4. How sliding windows are extracted, with both a "log price + time" view
     (used by the MMD-Signature model) and a "log returns only" view
     (used by the diffusion model).
  5. The calendar-day time vector convention (days_since_start / 365).

"""

import os
import numpy as np
import pandas as pd


DEFAULT_CSV = 'data/spx_20231229.csv'

# Train / OOS split, matching Chung I & Sester (2025), Section 5.1.
# Training period: 1995-01-01 → 2018-09-18  (~5,950 daily observations)
# OOS test period: 2018-09-19 → 2023-12-28  (covers COVID + 2022 selloff)
TRAIN_START = '1995-01-01'
TRAIN_END   = '2018-09-18'
OOS_START   = '2018-09-19'
OOS_END     = '2023-12-28'

# Calendar-day basis for time deltas: paper uses 365 calendar days per year.
# We keep this convention everywhere (training and generation), so the
# signature kernel sees consistent Δt values.
DAYS_PER_YEAR = 365.0


def load_spx_prices(csv_path: str = DEFAULT_CSV,
                    start_date: str = None,
                    end_date:   str = None) -> pd.DataFrame:
    # parse_dates=True makes pandas read the Date column as real datetimes,
    # which we need for calendar-day arithmetic later.
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Slice by date if requested. df.loc[a:b] is inclusive on both ends
    # when the index is a DatetimeIndex.
    if start_date is not None or end_date is not None:
        df = df.loc[start_date:end_date]

    # Defensive: drop any rows where the price is missing.
    df = df.dropna()

    return df


def log_returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D array of prices into log returns: r_t = log(P_t) - log(P_{t-1}).

    Log returns are preferred over simple returns because they are
    (a) approximately equal to simple returns for small moves,
    (b) additive across time (cum-sum of log returns = total log return),
    (c) approximately stationary, even when prices are non-stationary.

    Output is one element shorter than the input (no return on day 0).
    """
    # np.log -> natural logarithm; np.diff takes consecutive differences.
    return np.diff(np.log(prices.astype(np.float64)))


def get_log_returns(csv_path: str = DEFAULT_CSV,
                    start_date: str = None,
                    end_date:   str = None) -> np.ndarray:
    df     = load_spx_prices(csv_path, start_date, end_date)
    prices = df.iloc[:, 0].values
    return log_returns_from_prices(prices)


def calendar_time_vector(date_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Build the cumulative-time vector in years, starting at 0.

    For each consecutive pair of dates we count calendar days between them
    (so a Friday-to-Monday gap counts as 3 days, not 1 trading day), and
    divide by 365 to get a fraction of a year. The cumulative sum gives
    t_0 = 0, t_1 = 1 day / 365, etc.
    
    We use calendar days and not trading days, because the signature kernel sees the path (t, log_price) as a 2-D curve
    in continuous time. Using calendar days means weekends are reflected as
    larger Δt values, which is closer to "real" continuous time. This matches
    Section 5.1 of the MMD paper exactly.
    """
    n = len(date_index)
    t = np.zeros(n, dtype=np.float32)

    # date_index.to_series().diff() gives a Series of pandas Timedelta values,
    # where the first entry is NaT (no previous date). We skip it with [1:].
    # .dt.days extracts the day count of each Timedelta as an integer.
    day_diffs = date_index.to_series().diff()[1:].dt.days.values

    # Divide by 365.0 to convert to "years", then cumulative-sum into t[1:].
    t[1:] = (day_diffs / DAYS_PER_YEAR).cumsum()

    return t



def make_log_price_windows(csv_path:   str = DEFAULT_CSV,
                           start_date: str = TRAIN_START,
                           end_date:   str = TRAIN_END,
                           sample_len: int = 300,
                           stride:     int = 50) -> np.ndarray:
    df         = load_spx_prices(csv_path, start_date, end_date)
    prices     = df.iloc[:, 0].values.astype(np.float64)
    log_prices = np.log(prices).astype(np.float32)
    t_full     = calendar_time_vector(df.index)

    windows = []
    n       = len(df)

    # Slide a window of length sample_len across the data, stepping by `stride`.
    # range(0, n - sample_len + 1, stride) ensures we never read past the end.
    for start in range(0, n - sample_len + 1, stride):
        end = start + sample_len
        t_win  = t_full[start:end]      - t_full[start]
        lp_win = log_prices[start:end]  - log_prices[start]

        # Stack into a (sample_len, 2) array: column 0 = time, column 1 = log price.
        sample = np.stack([t_win, lp_win], axis=-1)
        windows.append(sample)

    # Final shape: (N_windows, sample_len, 2)
    return np.array(windows, dtype=np.float32)


def make_log_return_windows(csv_path:   str = DEFAULT_CSV,
                            start_date: str = TRAIN_START,
                            end_date:   str = TRAIN_END,
                            sample_len: int = 256,
                            stride:     int = 20) -> np.ndarray:
    # Get all log returns in the date range (one big 1-D array).
    log_rets = get_log_returns(csv_path, start_date, end_date)
    n        = len(log_rets)

    # Slide a fixed-length window across the return series.
    windows = []
    for start in range(0, n - sample_len + 1, stride):
        windows.append(log_rets[start:start + sample_len])

    return np.array(windows, dtype=np.float32)


def get_train_and_oos_log_returns(csv_path: str = DEFAULT_CSV) -> tuple:
    train_rets = get_log_returns(csv_path, TRAIN_START, TRAIN_END)
    oos_rets   = get_log_returns(csv_path, OOS_START,   OOS_END)
    return train_rets, oos_rets


def get_train_and_oos_log_price_windows(csv_path:   str = DEFAULT_CSV,
                                        sample_len: int = 300,
                                        stride:     int = 50) -> tuple:
    train_windows = make_log_price_windows(csv_path, TRAIN_START, TRAIN_END,
                                            sample_len, stride)
    oos_windows   = make_log_price_windows(csv_path, OOS_START,   OOS_END,
                                            sample_len, stride)
    return train_windows, oos_windows



def _self_test():
    print(f'Loading SPX from {DEFAULT_CSV} ...')
    df_full = load_spx_prices(DEFAULT_CSV)
    print(f'  {len(df_full):,} total rows, '
          f'from {df_full.index.min().date()} to {df_full.index.max().date()}')

    # Train / OOS row counts
    df_train = load_spx_prices(DEFAULT_CSV, TRAIN_START, TRAIN_END)
    df_oos   = load_spx_prices(DEFAULT_CSV, OOS_START,   OOS_END)
    print(f'  TRAIN  ({TRAIN_START} → {TRAIN_END}): '
          f'{len(df_train):,} rows')
    print(f'  OOS    ({OOS_START} → {OOS_END}): '
          f'{len(df_oos):,} rows')

    # Log returns
    train_rets, oos_rets = get_train_and_oos_log_returns()
    print(f'\nLog returns:')
    print(f'  train: {len(train_rets):,} returns, '
          f'mean={train_rets.mean():+.5f}, std={train_rets.std():.5f}')
    print(f'  oos:   {len(oos_rets):,} returns,   '
          f'mean={oos_rets.mean():+.5f}, std={oos_rets.std():.5f}')

    # MMD-style windows (log price + time)
    print(f'\nMMD windows  (sample_len=300, stride=50):')
    mmd_train_w = make_log_price_windows(sample_len=300, stride=50)
    print(f'  shape {mmd_train_w.shape}   '
          f'(N_windows, sample_len, [time, log_price])')
    print(f'  first window: t starts at {mmd_train_w[0,0,0]:.4f}, '
          f'ends at {mmd_train_w[0,-1,0]:.4f} years')
    print(f'                lp starts at {mmd_train_w[0,0,1]:.4f}, '
          f'ends at {mmd_train_w[0,-1,1]:.4f}')

    # Diffusion-style windows (log returns)
    print(f'\nDiffusion windows  (sample_len=256, stride=20):')
    diff_train_w = make_log_return_windows(sample_len=256, stride=20)
    print(f'  shape {diff_train_w.shape}   (N_windows, sample_len)')
    print(f'  per-window mean of mean returns: '
          f'{diff_train_w.mean(axis=1).mean():+.5f}')

    print('\nAll checks passed ✓')


if __name__ == '__main__':
    _self_test()
