# Generation-Of-Financial-Time-Series-Data

MSc thesis project comparing two generative models for synthetic daily S&P 500 returns under an identical evaluation framework.
Two recent generative paradigms are reimplemented in Python and compared on the same data with the same metrics:

1. **MMD-Signature model** with structured MA(20) noise — Chung & Sester (2025).
2. **Wavelet-image DDPM** — Takahashi & Mizuno (2025).

Both are trained on daily S&P 500 log returns (1995–2018) and evaluated on the standard set of stylised facts (heavy tails, near-zero autocorrelation, volatility clustering, leverage effect, gain/loss asymmetry) plus an aggregate signature-kernel MMD² distance, in-sample (TRAIN) and out-of-sample (OOS, 2018–2023).

## File overview

| File | Contents |
|---|---|
| `modeltraining.ipynb` | Kaggle notebook orchestrating training and evaluation on a Tesla P100 GPU.|
| `data_pipeline.py` | Loads the S&P 500 series, splits it into the train (1995–2018) and out-of-sample (2018–2023) periods, computes log returns, and produces the sliding windows consumed by each model. |
| `mmd_model.py` | The MMD-Signature generator: LSTM architecture, truncated signature kernel, unbiased MMD² loss, training loop with early stopping. |
| `mmd_noise.py` | Structured-noise components for the MMD-Signature model: Lambert-W Gaussianisation of the real returns and the fitted MA(20) noise sampler. |
| `gaussianize.py` | Goerg's (2015) Lambert-W transform code, imported by `mmd_noise.py`. |
| `diffusion_model.py` | The wavelet-image DDPM: Haar wavelet imaging of return windows, the UNet noise predictor, the DDPM training loop, and the inverse pipeline that reconstructs log returns from generated images. The thesis results are based on this version. |
| `diffusion_model_attempt.py` | Two iterations of targeted modifications to the diffusion implementation (standardised power transform, per-channel image layout, EMA, longer training). Neither improved out-of-sample performance; kept here for transparency about what was attempted. |
| `evaluate.py` | Loads the trained weights for both models, generates 100 synthetic paths each, and produces the stylised-fact diagnostics, summary statistics, gain/loss-asymmetry curves, and signature-kernel MMD² comparisons for the TRAIN and OOS periods. |
| `smoke_test.py` | Quick sanity checks of the data pipeline, the signature kernel, the wavelet round-trip, and the LSTM conditioning. Run this first before training. |
| `spx_20231229.csv` | Daily S&P 500 closing prices, 1995-01-01 to 2023-12-29, used as the empirical target distribution. |

## References

- Chung, L. & Sester, J. (2025). *Generative modelling of financial time series with structured noise and MMD-based signature learning.* Statistics and Risk Modeling 42(3-4), 91–122.
- Takahashi, T. & Mizuno, T. (2025). *Generation of synthetic financial time series by diffusion models.* Quantitative Finance 25(10), 1507–1516.
- Goerg, G. M. (2015). *The Lambert way to Gaussianize heavy-tailed data with the inverse of Tukey's h transformation as a special case.* The Scientific World Journal, 909231.
