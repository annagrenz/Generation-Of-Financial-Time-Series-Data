import numpy as np
import pandas as pd
import pytest

from src import data


def test_checksum_matches():
    assert data.file_sha256() == data.DATA_SHA256


def test_checksum_mismatch_raises(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("Date,spx\n2000-01-03,1\n")
    with pytest.raises(ValueError):
        data.load_prices(p)


@pytest.mark.parametrize("k", sorted(data.FOLDS_AND_ANCHOR))
def test_fold_has_no_leakage(k):
    f = data.get_fold(k)
    tr, te = f.train_prices(), f.test_prices()
    assert tr.index.max() < te.index.min()
    assert len(data.log_returns(te)) >= 240    # one trading year (2023 ends on 2023-12-28)
    # pilot training excludes exactly the validation slice
    trp, val = f.train_prices(pilot=True), f.val_prices()
    assert trp.index.max() < val.index.min()
    assert len(trp) + len(val) == len(tr)


def test_folds_are_expanding_and_test_blocks_disjoint():
    folds = [data.FOLDS[k] for k in sorted(data.FOLDS)]
    for a, b in zip(folds, folds[1:]):
        assert pd.Timestamp(a.train_end) < pd.Timestamp(b.train_end)
        assert pd.Timestamp(a.test_end) < pd.Timestamp(b.test_start)


def test_ten_folds_allow_significance_at_5_percent():
    assert len(data.FOLDS) == 10
    assert 2 / 2 ** len(data.FOLDS) < 0.05 / 12      # survives Holm over 12 discrepancies


def test_anchor_matches_chung_sester_split():
    f = data.get_fold(0)
    assert (f.train_start, f.train_end, f.test_start) == ("1995-01-01", "2018-09-18", "2018-09-19")
    assert len(data.log_returns(f.train_prices())) == 5970     # value reported in the thesis


def test_calendar_time_vector():
    idx = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])   # Fri, Mon, Tue
    t = data.calendar_time_vector(idx)
    np.testing.assert_allclose(t, [0, 3 / 365, 4 / 365])


def test_windows_stay_inside_period():
    p = data.get_fold(0).train_prices()
    w = data.log_price_windows(p, 300, 50)
    assert w.shape[1:] == (300, 2)
    assert np.all(w[:, 0, :] == 0)                         # rebased time and log price
    n_expected = (len(p) - 300) // 50 + 1
    assert len(w) == n_expected
    r = data.log_returns(p).values
    rw = data.return_windows(r, 256, 20)
    np.testing.assert_allclose(rw[-1], r[(len(r) - 256) // 20 * 20:][:256])
