

import numpy as np
import pandas as pd

import arch
from arch.univariate import Normal

from utils.gaussianize import Gaussianize

MA_ORDER       = 20        # p in MA(p): number of squared lags. Paper uses 20.
MA_MEAN_MODEL  = 'Zero'    # We want noise with zero mean → use the Zero mean model in arch
MA_Q           = 0         # GARCH-style "q" parameter, 0 means no autoregressive variance
                           


def gaussianize_returns(log_returns: np.ndarray,
                        dt_years:    np.ndarray) -> tuple:
    
    annualised = log_returns / dt_years

    # standardise (z-score) to mean 0 and std 1.
    mean = float(np.mean(annualised))
    std  = float(np.std(annualised))
    normalised = (annualised - mean) / std

    lambert = Gaussianize()
    lambert.fit(normalised)

    # .transform() returns a 2-D array (since it's an sklearn-style transformer);
    # we flatten back to 1-D for downstream use.
    gaussianized = lambert.transform(normalised).flatten()

    fit_stats = {
        'mean':    mean,
        'std':     std,
        'lambert': lambert,    # the fitted Gaussianize object, for reuse
    }
    return gaussianized.astype(np.float64), fit_stats



def fit_ma_model(gaussianized: np.ndarray, p: int = MA_ORDER):
    series = pd.Series(gaussianized)

    model = arch.arch_model(series,
                            mean=MA_MEAN_MODEL,
                            p=p,
                            q=MA_Q,
                            rescale=True)

    res = model.fit(update_freq=0, disp='off')

    return res



class MANoiseSampler:

    def __init__(self,
                 history_residuals: np.ndarray,
                 fitted_res,
                 noise_dim: int = 4,
                 p: int = MA_ORDER,
                 seed: int = 42):
        self.history   = np.asarray(history_residuals, dtype=np.float64)
        self.res       = fitted_res
        self.noise_dim = noise_dim
        self.p         = p

    
        self.rs = np.random.RandomState(seed)

    def sample(self, sample_len: int, batch_size: int) -> np.ndarray:
        n_simulations = self.noise_dim * batch_size

        anchor = pd.Series(self.history[-self.p:])

        model = arch.arch_model(anchor,
                                mean=MA_MEAN_MODEL,
                                p=self.p,
                                q=MA_Q,
                                rescale=False)

        model.distribution = Normal(seed=self.rs)

        forecasts = model.forecast(params=self.res.params,
                                    horizon=sample_len - 1,
                                    method='simulation',
                                    simulations=n_simulations)

       
        sims = forecasts.simulations.residuals[0]  

        sims = sims.reshape(sample_len - 1, self.noise_dim, batch_size)
        sims = sims.transpose(2, 0, 1)

        return sims.astype(np.float32)




def build_ma_noise_sampler(log_returns: np.ndarray,
                            dt_years:    np.ndarray,
                            noise_dim:   int = 4,
                            p:           int = MA_ORDER,
                            seed:        int = 42) -> tuple:
    gaussianized, fit_stats = gaussianize_returns(log_returns, dt_years)

    ma_res = fit_ma_model(gaussianized, p=p)
    fit_stats['ma_res'] = ma_res

    sampler = MANoiseSampler(history_residuals=gaussianized,
                             fitted_res=ma_res,
                             noise_dim=noise_dim,
                             p=p,
                             seed=seed)

    return sampler, fit_stats
