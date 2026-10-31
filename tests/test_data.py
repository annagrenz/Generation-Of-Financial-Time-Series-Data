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


@pytest.mark.parametrize("k", sorted(data.ALL_SPLITS))
def test_split_has_no_leakage(k):
    f = data.get_fold(k)
    tr, te = f.train_returns(), f.test_returns()
    assert tr.index.max() < te.index.min()
    assert len(te) >= 249                      # a full trading year (2023 data end on 2023-12-28)


def test_test_year_includes_its_first_return():
    """The first return of a test year uses the previous year's last close."""
    f = data.get_fold(1)                       # test year 2014
    te = f.test_returns()
    p = data.load_prices()
    first = te.index[0]
    prev_close = p.loc[:"2013-12-31"].iloc[-1]
    assert np.isclose(te["r"].iloc[0], np.log(p.loc[first] / prev_close))
    assert len(te) == len(f.test_prices())     # one return per trading day of the year


def test_development_split_ends_before_first_test_year():
    dev = data.get_fold(0)
    assert pd.Timestamp(dev.test_end) < pd.Timestamp(data.FOLDS[1].test_start)
    assert dev.role == "development"


def test_folds_are_expanding_and_test_blocks_disjoint():
    folds = [data.FOLDS[k] for k in sorted(data.FOLDS)]
    assert len(folds) == 10
    for a, b in zip(folds, folds[1:]):
        assert pd.Timestamp(a.train_end) < pd.Timestamp(b.train_end)
        assert pd.Timestamp(a.test_end) < pd.Timestamp(b.test_start)


def test_replication_matches_chung_sester_split():
    f = data.get_fold(99)
    assert (f.train_start, f.train_end, f.test_start) == ("1995-01-01", "2018-09-18", "2018-09-19")
    assert len(f.train_returns()) == 5970      # value reported in the thesis draft


def test_calendar_time_vector():
    idx = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])   # Fri, Mon, Tue
    t = data.calendar_time_vector(idx)
    np.testing.assert_allclose(t, [0, 3 / 365, 4 / 365])


def test_windows_stay_inside_period():
    p = data.get_fold(99).train_prices()
    w = data.log_price_windows(p, 300, 50)
    assert w.shape[1:] == (300, 2)
    assert np.all(w[:, 0, :] == 0)                         # rebased time and log price
    n_expected = (len(p) - 300) // 50 + 1
    assert len(w) == n_expected
    r = data.log_returns(p).values
    rw = data.return_windows(r, 256, 20)
    np.testing.assert_allclose(rw[-1], r[(len(r) - 256) // 20 * 20:][:256])
