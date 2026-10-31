"""
Structured MA(p) noise for the MMD-Signature generator (Chung & Sester, 2025, Sec. 4.3).

Pipeline (fitted on training returns only):
  1. annualise log returns by the calendar time step:  a_t = r_t / dt_t
  2. z-score:                                            u_t = (a_t - mean) / std
  3. Lambert W x Gaussian transform (Goerg 2015):        g_t = W(u_t)
  4. fit the variance model  sigma_t^2 = omega + sum_{i=1..p} beta_i g_{t-i}^2
     (arch's ARCH(p) with zero mean, called "MA(p)" by Chung & Sester)
  5. simulate noise paths z_t = sigma_t * eps_t, eps_t ~ N(0, 1)

Bug fixed relative to the earlier thesis code
---------------------------------------------
`arch` returns simulated residuals with shape (simulations, horizon) for the
single forecast origin. The earlier code reshaped this array directly to
(horizon, noise_dim, batch), which interleaves values from different
simulations and time points and destroys the volatility clustering of each
noise path (measured mean ACF of z^2 over lags 1-5: 0.003, same as i.i.d.
noise, versus 0.096 with the correct layout). Here each simulation is kept
intact as one time series: (n_sim, horizon) -> (batch, noise_dim, horizon)
-> (batch, horizon, noise_dim).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import arch
from arch.univariate import Normal

from src.utils.gaussianize import Gaussianize

MA_ORDER = 20


def gaussianize_returns(log_returns: np.ndarray, dt_years: np.ndarray):
    annualised = np.asarray(log_returns, np.float64) / np.asarray(dt_years, np.float64)
    mean, std = float(annualised.mean()), float(annualised.std())
    u = (annualised - mean) / std
    lambert = Gaussianize()
    lambert.fit(u)
    g = lambert.transform(u).ravel().astype(np.float64)
    return g, {"mean": mean, "std": std, "lambert": lambert}


def fit_ma_model(g: np.ndarray, p: int = MA_ORDER):
    model = arch.arch_model(pd.Series(g), mean="Zero", p=p, q=0, rescale=False)
    return model.fit(update_freq=0, disp="off")


class MANoiseSampler:
    """Draws noise tensors of shape (batch, sample_len - 1, noise_dim)."""

    def __init__(self, history: np.ndarray, fitted_res, noise_dim: int = 4,
                 p: int = MA_ORDER, seed: int = 0):
        self.history = np.asarray(history, np.float64)
        self.res = fitted_res
        self.noise_dim = noise_dim
        self.p = p
        self.rs = np.random.RandomState(seed)

    def sample(self, sample_len: int, batch_size: int) -> np.ndarray:
        horizon = sample_len - 1
        n_sim = self.noise_dim * batch_size
        # Variance recursion starts from the last p observed (Gaussianised) values.
        anchor = pd.Series(self.history[-self.p:])
        model = arch.arch_model(anchor, mean="Zero", p=self.p, q=0, rescale=False)
        model.distribution = Normal(seed=self.rs)
        fc = model.forecast(params=self.res.params, horizon=horizon,
                            method="simulation", simulations=n_sim)
        sims = np.asarray(fc.simulations.residuals)[-1]          # (n_sim, horizon)
        assert sims.shape == (n_sim, horizon), sims.shape
        sims = sims.reshape(batch_size, self.noise_dim, horizon)  # each row = one intact path
        return sims.transpose(0, 2, 1).astype(np.float32)        # (batch, horizon, noise_dim)


def build_ma_noise_sampler(log_returns: np.ndarray, dt_years: np.ndarray,
                           noise_dim: int = 4, p: int = MA_ORDER, seed: int = 0):
    g, stats = gaussianize_returns(log_returns, dt_years)
    res = fit_ma_model(g, p=p)
    stats["ma_res"] = res
    return MANoiseSampler(g, res, noise_dim=noise_dim, p=p, seed=seed), stats
