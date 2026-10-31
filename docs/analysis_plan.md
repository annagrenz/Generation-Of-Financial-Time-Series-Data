# Analysis plan (pre-specification)

**Status:** DRAFT for review — to be frozen (git tag `plan-v1`) before the final experiment runs.
Anything not fixed here and decided later is reported as exploratory.

---

## 1. Motivation and model choice

Deep generative models for financial time series can be grouped by **how a sample is produced** and
**what the training objective compares**:

| | Family A: sequential, time-domain, path-distribution matching | Family B: whole-window, denoising / score-based, often in a transformed domain |
|---|---|---|
| Generation | one return at a time, recurrent or convolutional state | entire window at once, by iterative denoising from noise |
| Objective | distance between distributions of real and generated paths (adversarial, Wasserstein, MMD) | per-sample noise-prediction (denoising score-matching) loss |
| Representation | returns / log prices in time order | often transformed (wavelet, Fourier, image-like arrays) |
| Examples (to be verified in the literature review) | QuantGAN (Wiese et al., 2020), TimeGAN, SigCWGAN, COT-GAN, MMD-Signature (Chung & Sester, 2025) | DDPM-based generators incl. Takahashi & Mizuno (2025) and other time-series diffusion models |
| **Representative used here** | **MMD-Signature** | **Wavelet-image DDPM** |

Selection criteria for the representatives: (i) published recently in a peer-reviewed finance/statistics
venue, (ii) designed for or applicable to a univariate return series, (iii) method described in enough
detail to be reproduced, (iv) each model is a clear instance of its family on all three rows above.
Comparing one representative per family on identical data isolates the family-level design differences
only to the extent that the representatives are typical; this is stated as a limitation.

*Deliverable before freezing:* literature table (≈15–25 papers) classifying methods on the three rows.

## 2. Research question and hypotheses

**RQ.** When trained and evaluated on identical daily S&P 500 data, do a sequential path-matching
generator (MMD-Signature) and a wavelet-domain diffusion generator differ in how accurately they reproduce
economically relevant stylised facts, and how do both compare with a classical GJR-GARCH(1,1)-t benchmark?

All hypotheses are **two-sided**. Directional hypotheses are not used because results from an earlier
version of this analysis are already known.

For each stylised-fact discrepancy D_f (Section 5):

* **H1_f** (model comparison): E[D_f(MMD-Sig)] = E[D_f(Diffusion)]
* **H2_f** (benchmark): E[D_f(MMD-Sig)] = E[D_f(GARCH-t)] and E[D_f(Diffusion)] = E[D_f(GARCH-t)]
* **H3_f** (generalisation): for each model, E[D_f] is the same when the reference is the training period
  and when it is the subsequent test period.

Additionally, per model and fold, an omnibus question: are generated and real 250-day windows
distinguishable? (Section 6.1.)

## 3. Data and folds

* `data/spx_20231229.csv`, S&P 500 close, 1995-01-03 → 2023-12-28, from the Chung & Sester repository.
  SHA-256 `93d9412d…dde6aa`, checked on every load.
* Log returns r_t = log P_t − log P_{t−1}, computed within each period.
* **Rolling-origin folds** (expanding training window, one-year test block after it). A single
  chronological split gives one test period and therefore no replication; random splits would leak future
  information into training. Rolling-origin folds keep time order and give ten test years with different
  market regimes (calm 2017, sell-off late 2018, COVID 2020, bear market 2022).
* **Why ten folds:** the folds are the replication unit of the between-model tests (Section 6.2). An exact
  paired sign-flip test over K folds cannot give a p-value below 2/2^K: with K = 5 this is 0.0625, so no
  difference could ever be significant at 5 %. With K = 10 the minimum is 0.002, which still survives a
  Holm correction over the 12 discrepancies.

| Fold | Training (from 1995-01-01) | n train returns | Test year | n test returns |
|---|---|---|---|---|
| 1 | → 2013-12-31 | 4,783 | 2014 | 251 |
| 2 | → 2014-12-31 | 5,035 | 2015 | 251 |
| 3 | → 2015-12-31 | 5,287 | 2016 | 251 |
| 4 | → 2016-12-31 | 5,539 | 2017 | 250 |
| 5 | → 2017-12-31 | 5,790 | 2018 | 250 |
| 6 | → 2018-12-31 | 6,041 | 2019 | 251 |
| 7 | → 2019-12-31 | 6,293 | 2020 | 252 |
| 8 | → 2020-12-31 | 6,546 | 2021 | 251 |
| 9 | → 2021-12-31 | 6,798 | 2022 | 250 |
| 10 | → 2022-12-31 | 7,049 | 2023 | 248 |
| Anchor (0) | → 2018-09-18 | 5,970 | 2018-09-19 → 2023-12-28 | 1,327 |

* **Anchor split (fold 0)** = the split of Chung & Sester (2025, Sec. 5.1). It is used for the pilot, the
  capacity check, the seed-variance estimate and a replication of the original setup. It is not part of the
  between-model tests, because its test period overlaps folds 5–10.
* **Validation slice:** last 10 % of each training period. Used only in the anchor-split pilot (Section 7) for
  configuration decisions. Final models are trained on the full training period. Test blocks are not
  looked at until all runs are complete.
* **Training windows:** model-specific, as in the source papers: MMD-Sig 300 points, stride 50
  (Chung & Sester, Sec. 5.1); diffusion 256 returns, stride 20. Comparability comes from identical
  underlying returns, identical folds and identical evaluation, not from identical windows.

## 4. Models and fixed settings

Hyperparameters are fixed in `config/*.yaml`, each with its source. Main points:

* **MMD-Signature:** LSTM(64), noise dim 4, MA(20) noise on Lambert-W-Gaussianised annualised returns,
  signature kernel level 10 with RQ static kernel (σ = 0.1), lead-lag, burn-in 50 / generated 250,
  batch 38, Adam 1e-3 with plateau halving, early stopping 300 epochs on the smoothed training loss.
* **Wavelet-image DDPM:** Haar DWT of 256 returns → 16×256 image with block-repeated coefficients;
  preprocessing (r − μ) winsorised at 10σ, signed power 1/1.5, divided by σ; global [−1, 1] scaling;
  UNet 128-128-256-256-512 with attention (22.2 M parameters); T = 1000 linear β schedule; Adam 1e-4,
  batch 32, 100 epochs.
* **Baselines:** GJR-GARCH(1,1) with Student-t innovations (fitted per fold on training returns);
  stationary block bootstrap of training returns (non-parametric reference that preserves short-range
  dependence); i.i.d. resampling of training returns (keeps the marginal distribution, removes all
  temporal dependence — used to validate that temporal metrics detect missing structure).
* **Information parity:** MMD-Sig burn-in windows are drawn from the training period only, so neither
  deep model sees test-period data. (Secondary analysis: MMD-Sig conditioned on test-period history.)
* **Capacity check (anchor split only):** a reduced UNet (32-32-64-64-128) and a memorisation check
  (Section 6.4), addressing the ratio of 22 M parameters to ≈ 6,000 training returns.

Each trained model generates **1,000 paths of 250 returns**.

## 5. Stylised facts and discrepancies

A fact is included if it is (i) robust across markets (Cont, 2001; Ratliff-Crain et al., 2025),
(ii) relevant to a named use of synthetic return data, and (iii) estimable from 250-day windows.

| Fact | Statistic | Discrepancy D_f | Why it matters for synthetic data |
|---|---|---|---|
| Heavy tails | VaR and ES at 1 % and 5 %; Hill tail index (top 5 % of \|r\|) | absolute difference to real | tail risk, stress testing, capital requirements |
| Absence of linear autocorrelation | ACF(r), lags 1–20 | RMSE over lags | no artificial predictability that a backtested strategy could exploit |
| Volatility clustering | ACF(\|r\|) and ACF(r²), lags 1–20 | RMSE over lags | volatility forecasting, option pricing, risk scaling |
| Leverage effect | corr(r_t, \|r_{t+k}\|), k = 1–10 | RMSE over lags | asymmetric downside risk |
| Gain/loss asymmetry | skewness; P(R > θ \| \|R\| > θ), θ ∈ [0, 3 %] | absolute difference; RMSE over θ | asymmetric distribution of large moves |

Mean and SD are reported descriptively (log SD ratio). Excess kurtosis is reported but not tested, as it
is dominated by a few observations.

Temporal statistics are computed per 250-return path and averaged; distributional statistics on pooled
returns. The real reference for a fold is the test-year return series (≈ 250 returns, one path); for the
in-sample comparison it is all 250-return windows of the training period.

### Signature-kernel MMD² (clarification)

For samples X = {x_i}_{i=1..n} (real) and Y = {y_j}_{j=1..m} (generated) and a kernel k,

MMD²_u = 1/(n(n−1)) Σ_{i≠i'} k(x_i, x_i') + 1/(m(m−1)) Σ_{j≠j'} k(y_j, y_j') − 2/(nm) Σ_{i,j} k(x_i, y_j).

It estimates the squared distance between the two path distributions in the kernel's feature space; it is
zero in expectation when the distributions are equal. Stylised facts are specific, interpretable
functionals of the distribution; MMD² is an omnibus summary of all differences the kernel can detect.
The MMD-Signature model is *trained* to minimise MMD²_u with the signature kernel (level 10, RQ σ = 0.1,
lead-lag), so evaluating with that same kernel would grade one model on its own objective. The study
therefore does **not** use the training kernel for model comparison (Section 6.1).

## 6. Statistical analysis

α = 0.05 throughout; effect sizes with 95 % CIs are always reported alongside p-values.

### 6.1 Generated vs real (per model × fold × seed)
* **Stylised facts:** 95 % CI for each Δ_f = S_f(generated) − S_f(real). Real-data uncertainty from a
  stationary block bootstrap of the reference period (block length by Politis & White, 2004); generated
  uncertainty from resampling paths. Δ_f is called inconsistent with the real data if the CI excludes 0.
* **Omnibus tests (kernels not used in training):**
  (a) MMD² with a Gaussian RBF kernel on standardised 250-return windows (bandwidth: median heuristic on
  the real windows), p-value by permutation;
  (b) classifier two-sample test: a simple classifier trained to separate real from generated windows,
  cross-validated accuracy tested against 0.5.
  Because real windows overlap, permutation units are non-overlapping real windows; the resulting loss of
  power is reported.

### 6.2 Model vs model and vs benchmark (H1, H2)
Unit of replication = fold. For each fold, D_f per model (averaged over seeds where more than one seed is
run). Paired comparison over the 10 folds: exact sign-flip permutation test of the mean paired difference
(all 2^10 sign patterns); effect size = mean paired difference with a bootstrap CI over folds. Seed-to-seed
variation is estimated on the anchor split (3 seeds per model) and reported separately, so that a
between-model difference can be compared with the variation caused by the random seed alone.

### 6.3 Generalisation (H3)
For each model and fact: D_f against the test period minus D_f against the training period, tested across
folds with the same paired permutation test.

### 6.4 Memorisation (diffusion and MMD-Sig, anchor split)
Distance of each generated window to its nearest training window, compared with the distribution of
nearest-neighbour distances between held-out real windows and training windows. Generated samples
systematically closer than real held-out data indicate copying.

### 6.5 Multiple testing
Holm correction across the stylised-fact discrepancies within each hypothesis family (H1, H2, H3).

### 6.6 Metric validation (before any deep-model results are inspected)
On each fold: block bootstrap must be consistent with real data on all facts; i.i.d. resampling must be
flagged on volatility clustering and leverage; GJR-GARCH-t must show positive ACF(|r|). If a metric fails
these checks it is revised before the deep-model runs are evaluated, and the revision is documented.

## 7. Pilot (anchor split, validation slice) — allowed decisions

One seed per model, training with the validation slice held out. Allowed decisions, each documented with
its validation evidence: (a) MMD batch size (38, else the largest that fits the GPU); (b) diffusion
optimisation budget in gradient steps (100 epochs vs. paper-equivalent ≈ 6,300 steps); (c) number of seeds
per fold (1 or more) given the measured Kaggle time per run. No other hyperparameter is tuned.

## 8. Experiment matrix and compute

| | Splits | Seeds | Runs | Hardware |
|---|---|---|---|---|
| MMD-Signature | 10 folds | 1 (more if time allows) | 10 | Kaggle GPU |
| Wavelet DDPM | 10 folds | 1 (more if time allows) | 10 | Kaggle GPU |
| MMD-Signature, DDPM | anchor | 3 each | 6 | Kaggle GPU |
| Reduced UNet | anchor | 3 | 3 | Kaggle GPU |
| GJR-GARCH-t, block bootstrap, i.i.d. | 10 folds + anchor | 3 | 99 | CPU |

All runs from one tagged commit; every output carries `manifest.json` (commit, config, data hash,
package versions, GPU, wall time).

## 9. Known limitations (stated in advance)
One asset and one frequency; one representative per family; the diffusion model is a re-implementation
(no public code) adapted from minute-level multivariate data to daily univariate data; one-year test
blocks give noisy real reference statistics and ten folds give limited power for between-model tests,
which is why effect sizes and CIs are reported throughout.
