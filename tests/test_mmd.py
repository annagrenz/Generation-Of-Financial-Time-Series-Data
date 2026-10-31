import numpy as np
import pytest
import tensorflow as tf

from src import data
from src.models import mmd_sig
from src.models.mmd_noise import build_ma_noise_sampler

CFG = {"sample_len": 300, "hist_len": 50, "noise_dim": 4, "hidden_size": 64, "n_lstm_layers": 1,
       "use_ma_noise": True, "n_levels": 10, "static_kernel": "rq", "kernel_sigma": 0.1, "kernel_alpha": 1.0}


@pytest.fixture(scope="module")
def sampler():
    p = data.get_fold(0).train_prices()
    r = data.log_returns(p).values
    dt = np.diff(data.calendar_time_vector(p.index))
    s, fit = build_ma_noise_sampler(r, dt, noise_dim=4, p=20, seed=0)
    return s, fit, p


def _mean_acf_sq(z):     # z: (n_paths, T); mean ACF of z^2 at lags 1..5
    x = z ** 2 - (z ** 2).mean(axis=1, keepdims=True)
    v = (x * x).sum(axis=1)
    return np.mean([((x[:, :-k] * x[:, k:]).sum(axis=1) / v).mean() for k in range(1, 6)])


def test_ma_noise_keeps_volatility_clustering(sampler):
    s, fit, _ = sampler
    assert fit["ma_res"].scale == 1.0
    z = s.sample(300, 64)
    assert z.shape == (64, 299, 4)
    paths = z.transpose(0, 2, 1).reshape(-1, 299)
    iid = np.random.default_rng(0).standard_normal(paths.shape)
    assert _mean_acf_sq(paths) > 0.05 > _mean_acf_sq(iid)     # old reshape gave ~0.003


def test_generator_keeps_history_and_generate_returns_excludes_it(sampler):
    s, _, p = sampler
    g = mmd_sig.build_generator(CFG)
    w = data.log_price_windows(p, 300, 50)[:8]
    noise = s.sample(300, 8)
    path = g((tf.constant(noise), tf.constant(w[:, :, :1]), tf.constant(w[:, :50, 1:]))).numpy()
    np.testing.assert_allclose(path[:, :50, 0], w[:, :50, 1], atol=1e-7)

    r = mmd_sig.generate_returns(g, w, s, CFG, n_paths=8, seed=0)
    assert r.shape == (8, 250)
    # generated returns must not reproduce the real burn-in returns
    real_hist_r = np.diff(w[:, :50, 1], axis=1)
    assert not np.allclose(r[:, :49], real_hist_r)


def test_lead_lag_shape_and_values():
    x = tf.constant(np.arange(5, dtype=np.float32).reshape(1, 5, 1))
    t = tf.constant(np.linspace(0, 1, 5, dtype=np.float32).reshape(1, 5, 1))
    ll = mmd_sig.lead_lag(x, t, [1]).numpy()[0]
    assert ll.shape == (6, 3)
    np.testing.assert_allclose(ll[:, 1], [0, 1, 2, 3, 4, 4])
    np.testing.assert_allclose(ll[:, 2], [0, 0, 1, 2, 3, 4])


def test_mmd_is_near_zero_for_identical_samples_and_positive_for_shifted():
    rng = np.random.default_rng(0)
    k = mmd_sig.build_kernel(CFG)
    t = np.linspace(0, 1, 30)[None, :, None].repeat(32, 0)
    X = np.concatenate([t, np.cumsum(rng.normal(0, .01, (32, 30, 1)), 1)], -1).astype(np.float32)
    Y = np.concatenate([t, np.cumsum(rng.normal(0, .01, (32, 30, 1)), 1)], -1).astype(np.float32)
    Z = np.concatenate([t, np.cumsum(rng.normal(.02, .03, (32, 30, 1)), 1)], -1).astype(np.float32)
    same = float(mmd_sig.mmd2_unbiased(tf.constant(X), tf.constant(Y), k))
    diff = float(mmd_sig.mmd2_unbiased(tf.constant(X), tf.constant(Z), k))
    assert abs(same) < diff
