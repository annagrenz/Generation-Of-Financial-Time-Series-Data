import numpy as np

from src.metrics import stylised as st


def test_acf_of_ar1():
    rng = np.random.default_rng(0)
    phi, n = 0.6, 200_000
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    np.testing.assert_allclose(st.acf(x, 3), [phi, phi ** 2, phi ** 3], atol=0.01)


def test_acf_white_noise_near_zero():
    x = np.random.default_rng(1).standard_normal(100_000)
    assert np.all(np.abs(st.acf(x, 10)) < 5 / np.sqrt(len(x)))


def test_leverage_sign():
    """Negative returns followed by large |r| -> negative leverage at lag 1."""
    rng = np.random.default_rng(2)
    r = rng.standard_normal(100_000)
    r[1:] *= np.exp(-0.5 * r[:-1])            # scale tomorrow up after a loss today
    assert st.leverage(r, 1)[0] < -0.05


def test_gain_loss_symmetric_gaussian():
    x = np.random.default_rng(3).standard_normal(200_000) * 0.01
    gl = st.gain_loss_curve(x, np.array([0.0, 0.01, 0.02]))
    np.testing.assert_allclose(gl, 0.5, atol=0.02)


def test_hill_recovers_pareto_index():
    a = 3.0
    x = np.random.default_rng(4).pareto(a, 200_000) + 1
    assert abs(st.hill_tail_index(x, 0.05) - a) < 0.2


def test_discrepancies_zero_for_identical():
    x = np.random.default_rng(5).standard_t(4, (50, 250)) * 0.01
    s = st.stylised_stats(x)
    d = st.discrepancies(s, s)
    assert all(v == 0 or np.isnan(v) for v in d.values())
