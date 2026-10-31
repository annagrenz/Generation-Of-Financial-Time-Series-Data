# Analysis plan — version 2

**Status:** draft for review. Revised after two external reviews (2026-10-31). To be frozen as git tag
`plan-v1` after the development-period runs (Section 8) and before any evaluation-fold result is computed.

**Nature of the study.** A transparent, documented, *retrospective* comparison of two concrete
implementations. Most of the 1995–2023 data were already inspected in an earlier version of this work
(Section 10), so the study is not a confirmatory pre-registration in the strict sense. Pre-specifying the
analysis still limits the analytic choices that can be made after seeing the new results.

---

## 1. Research question

> When trained and evaluated on identical daily S&P 500 data, how do an implementation of the
> MMD-Signature generator (Chung & Sester, 2025) and an implementation of the wavelet-image DDPM
> (Takahashi & Mizuno, 2025) differ in how well their simulated years reproduce key statistical properties
> of real S&P 500 years — and how do both compare with a classical GJR-GARCH(1,1)-t benchmark?

The answer is given as yearly paired differences between models with their average and spread
(Section 7), not as a single significance decision.

## 2. Why these two models

The two implementations differ on four separable design attributes. They are not inseparable pairs, so the
study cannot attribute a difference to any single attribute (stated limitation).

| Attribute | MMD-Signature | Wavelet-image DDPM |
|---|---|---|
| Generation architecture | sequential: an LSTM produces one return at a time | whole window at once: iterative denoising of a 16×256 array |
| Training objective | distribution-level: MMD² between sets of real and generated paths | sample-level: mean squared error of noise prediction (denoising score matching) |
| Representation | log-price paths in time order (+ time channel, lead-lag) | Haar wavelet coefficients arranged by scale (rows) and time (columns) |
| Conditioning | 50-day real burn-in; structured ARCH(20)-type input noise | none; isotropic Gaussian noise |

They are chosen as published, recent representatives of two broad groups in the literature on synthetic
financial time series: sequential generators trained by matching path distributions (e.g. QuantGAN,
SigCWGAN, COT-GAN, MMD-Signature) and denoising-diffusion generators that model a whole window, often in a
transformed domain. A literature table (≈15–25 papers, classified on the four attributes) documents this
grouping; references are verified before inclusion.

**What the MMD² loss captures** (supervisor comment). The signature of a path summarises its ordered
increments; with the lead-lag transform it also encodes quadratic variation and, through it,
autocorrelation-type information. MMD² with the signature kernel compares the *whole* distribution of
paths. The evaluation endpoints (Section 5) are a few interpretable functionals of that distribution.
The MMD model is trained on a related discrepancy, so the training-kernel MMD is not used to compare
models; it is reported only descriptively for the replication split.

**Wavelet representation** (supervisor comment). With block repetition each column refers to the same
time position on every row: the horizontal axis is time, the vertical axis is scale (row 0 the
approximation, rows 1–8 detail levels from coarse to fine; rows 9–15 zero padding). A 3×3 convolution
therefore connects neighbouring times at one scale and neighbouring scales at one time — a meaningful
time–scale neighbourhood. The earlier implementation tiled the coefficient vectors, which broke the
vertical alignment (`docs/corrections.md`, C3).

**Capacity** (supervisor comment). The UNet has 22.2 M parameters; a training fold has 4,800–7,000
returns in 227–340 strongly overlapping 256-return windows. This is addressed by (i) a reduced UNet
(32-32-64-64-128) evaluated on the development split, (ii) a memorisation check (Section 6.4), and
(iii) reporting the parameter-to-data ratio as a limitation.

## 3. Data and splits

* `data/spx_20231229.csv`: S&P 500 close, 1995-01-03 → 2023-12-28, from the Chung & Sester repository.
  SHA-256 `93d9412d…dde6aa`, verified on every load.
* Log returns are computed **once** on the full series in chronological order and assigned to the period
  containing their end date. The first return of a test year therefore uses the previous year's last close.
* All preprocessing parameters (scaling, winsorisation bounds, Lambert-W and ARCH parameters, bootstrap
  block lengths) are estimated from the training period of the respective split only; "global scaling"
  means global within that training period. No window crosses a period boundary.

| Split | Role | Training returns | Test / validation period | n returns |
|---|---|---|---|---|
| 0 | **development** | 1995-01-04 → 2011-12-30 (4,281) | validation 2012–2013 | 502 |
| 1 | evaluation | → 2013-12-31 (4,783) | 2014 | 252 |
| 2 | evaluation | → 2014-12-31 (5,035) | 2015 | 252 |
| 3 | evaluation | → 2015-12-31 (5,287) | 2016 | 252 |
| 4 | evaluation | → 2016-12-30 (5,539) | 2017 | 251 |
| 5 | evaluation | → 2017-12-29 (5,790) | 2018 | 251 |
| 6 | evaluation | → 2018-12-31 (6,041) | 2019 | 252 |
| 7 | evaluation | → 2019-12-31 (6,293) | 2020 | 253 |
| 8 | evaluation | → 2020-12-31 (6,546) | 2021 | 252 |
| 9 | evaluation | → 2021-12-31 (6,798) | 2022 | 251 |
| 10 | evaluation | → 2022-12-30 (7,049) | 2023 | 249 |
| 99 | replication | → 2018-09-18 (5,970) | 2018-09-19 → 2023-12-28 | 1,328 |

* **Development split (0).** Every data-dependent decision — preprocessing choices, training budget,
  checkpoint rule, metric definitions and their calibration, number of seeds — is made using data up to
  2013-12-31 only. After that, models for fold 1 are trained on all data up to 2013-12-31 with the frozen
  settings.
* **Evaluation folds (1–10).** Expanding training window, one test year each. Earlier test years later
  become training data, which is legitimate in a chronological design. The ten yearly results are *not*
  guaranteed to be independent (shared training data, dependence between adjacent market years).
* **Replication split (99)** reproduces the split of Chung & Sester (2025, Sec. 5.1). It is reported
  separately, never used for decisions, and not part of the between-model analysis.

## 4. Models, baselines and generation target

Hyperparameters are fixed in `config/*.yaml`, each with its source; deviations from the original methods
are listed in Section 9.

**Generation target (all models).** An *unconditional* synthetic year: a draw from the return-path
distribution of the training period, not a forecast of the coming year. All generators get the same
information set, the training period only:

| Generator | How a 250-return path is produced |
|---|---|
| MMD-Signature | burn-in = a randomly drawn 50-point window from the training period; then 250 generated returns; ARCH(20)-type noise started from the end of the training period |
| Wavelet DDPM | unconditional sample; first 250 of the 256 decoded returns |
| GJR-GARCH(1,1)-t | parameters fitted on the training returns; simulation started at the unconditional variance with a 500-day burn-in that is discarded |
| Stationary block bootstrap | blocks resampled from the training returns (mean block length by Politis & White, 2004, on training data) |
| i.i.d. resampling | days resampled independently from the training returns (keeps the marginal distribution, removes all dependence) |

GARCH: fitted by maximum likelihood with `arch`; stationarity constraints as in the package defaults; a
fit that fails to converge is refitted from the package's alternative starting values once and otherwise
recorded as failed (no silent replacement).

Each trained model (or fitted baseline) generates **N = 1,000 paths**.

## 5. Endpoints

All statistics are computed **within a path** of fixed length **L = 249 returns** (the shortest test
year): for a real test year the first 249 returns; for a generated path its first 249 returns. Equal
lengths make real and simulated statistics comparable across years.

### 5.1 Primary endpoints (four)

| | Fact | Statistic S of one path r_1..r_L | Unit |
|---|---|---|---|
| E1 | heavy lower tail | ES₅ = − mean of the k = ⌈0.05·L⌉ = 13 smallest returns | return (positive = loss) |
| E2 | volatility level | log sample SD of r | log scale |
| E3 | no linear autocorrelation | Q_r = Σ_{h=1..10} ρ̂_h(r)² | squared correlation |
| E4 | volatility clustering | C_abs = (1/10) Σ_{h=1..10} ρ̂_h(\|r\|) | correlation |

ρ̂_h is the standard sample autocorrelation (denominator: sum of squares about the path mean). A yearly ES₅
rests on about 13 tail observations and yearly autocorrelations are noisy; this uncertainty is part of what
the endpoint measures and is reported (Section 6.2).

### 5.2 Score for a generator in one year

For generator m, fold k and seed s, the 1,000 generated paths give an empirical distribution F̂ of S. The
real year gives one value y_k. The score is the **continuous ranked probability score**

CRPS(F̂, y) = (1/N) Σ_i |x_i − y| − (1/(2N²)) Σ_i Σ_j |x_i − x_j|,

which rewards a generator whose simulated years are both centred on and *as dispersed as* real years (a
generator producing near-identical paths is penalised). Lower is better; the unit is that of S.
Secondary, for interpretation: the absolute error of the median, |median(F̂) − y_k|, and whether y_k lies
in the central 90 % interval of F̂ (coverage over the ten years; nominal 90 %).

Aggregation: per fold, the CRPS is averaged over seeds; seeds are all evaluated by the same rule (no
selection of best runs).

### 5.3 Exploratory endpoints

ES₁ and VaR at 1 % and 5 %; Hill tail index (top 5 % of |r|, with threshold sensitivity 2.5–10 %);
ACF of r² ; leverage corr(r_t, |r_{t+h}|), h = 1..10; skewness; gain/loss probability P(R > θ | |R| > θ)
for θ ∈ {1 %, 2 %} with at least 5 exceedances required, otherwise reported as undefined; excess kurtosis.
Pooled statistics over all 249,000 simulated returns are reported only as descriptions of the model and are
never mixed with per-path statistics.

## 6. Analysis

### 6.1 Primary contrasts

Three contrasts × four endpoints = 12 primary comparisons:

* C1: MMD-Signature vs Wavelet DDPM
* C2: MMD-Signature vs GJR-GARCH-t
* C3: Wavelet DDPM vs GJR-GARCH-t

For contrast (a, b) and endpoint f: yearly difference d_{f,k} = CRPS_{f,k}(a) − CRPS_{f,k}(b) (negative
favours a), and the effect size δ_f = (1/10) Σ_k d_{f,k}.

### 6.2 Reporting (main result)

For each primary comparison: all ten d_{f,k} (plotted by year), δ_f, the number of years favouring each
model, and the range. Three sources of variation are reported separately:
* across years (spread of d_{f,k});
* across training seeds (spread of CRPS over seeds within a fold);
* Monte-Carlo error of the 1,000 generated paths (bootstrap over paths with the trained model fixed;
  this captures neither training-data nor market-regime uncertainty, which is stated with every interval).

### 6.3 Formal test (assumption-dependent)

A sign-flip permutation test of δ_f over the ten years (all 2¹⁰ sign patterns; smallest attainable
two-sided p = 0.002) with Holm correction over the 12 primary comparisons (first threshold 0.05/12 =
0.0042). The test is exact only if the yearly differences are sign-symmetric under the null, which is
*not* guaranteed here. Its calibration under realistic dependence is checked by simulation (Section 8.2);
the p-values are interpreted together with that calibration result, never on their own. A confidence
interval containing zero is not taken as evidence of equivalence.

### 6.4 Exploratory analyses (no error control)

Exploratory endpoints (5.3); training-period vs test-year discrepancy per model (descriptive: can reflect
overfitting, regime change or the noisier one-year reference, which cannot be separated); reduced UNet;
memorisation: nearest-neighbour distance of generated windows to training windows, compared with the same
distance for (i) the block bootstrap, which copies real segments by construction, and (ii) the real
validation/test windows; replication split 99; omnibus two-sample comparisons (RBF-kernel MMD) only on
periods long enough to provide several non-overlapping real windows, interpreted descriptively.

## 7. Seeds and compute

Three training seeds per deep model per evaluation fold (60 main runs), as a budget-dependent minimum.
The number is fixed after the development-split runtime measurement (training **and** generation) and
before any evaluation-fold result is computed. Baselines: three seeds per fold on CPU. Failed runs (non-
finite loss or output) are rerun once with the next seed; both runs are logged.

| | Splits | Seeds | Runs |
|---|---|---|---|
| MMD-Signature | dev + 10 folds | 3 | 33 |
| Wavelet DDPM | dev + 10 folds | 3 | 33 |
| Reduced UNet | dev | 3 | 3 |
| Replication (both deep models) | 99 | 1 | 2 |

## 8. Development phase (split 0 only)

### 8.1 Allowed decisions

Each decided on the 2012–2013 validation period and documented with its evidence: MMD batch size (38, or
the largest that fits the GPU); diffusion optimisation budget (100 epochs ≈ 700 updates on this split, 800–1,100 on the evaluation folds, vs.
a paper-equivalent ≈ 6,300 updates); whether the reduced UNet replaces the full UNet; seed count.
Checkpoint rules are fixed in advance: MMD = best smoothed training loss (as in the original code);
DDPM = final epoch.

### 8.2 Calibration of the evaluation by simulation

Before any evaluation-fold result: a GJR-GARCH-t fitted on 1995–2011 serves as a known "true" process.
Repeated synthetic datasets (10 "years" each, generated with dependence between years via an overlapping
training history) are evaluated with the complete pipeline, including resampling:

* **False-alarm rate:** two generators identical to the truth → share of 12 tests rejecting should be
  close to 5 % (per test) and the Holm family-wise rate close to or below 5 %.
* **Detection:** generators with known defects (Gaussian innovations; no leverage; volatility persistence
  reduced; scale × 1.2) → rejection rate per endpoint for each defect size.
* **Coverage:** do the 90 % intervals of Section 5.2 contain the true value about 90 % of the time?

Results are reported as they come out; endpoints are not altered on the basis of real-data results.

## 9. Original method → implementation

| Item | Original | This study | Reason |
|---|---|---|---|
| MMD data | daily SPX 1995–2018-09 | same data; expanding folds to 2023 | replication with years as units |
| MMD generated part | 250 log-price points after 50 burn-in (Sec. 5.1) | 250 returns after the last burn-in point (= the same 250 increments) | evaluated on returns |
| MMD noise | "MA(20)" variance model on Lambert-W-Gaussianised annualised returns | identical; called ARCH(20)-type here, as σ²_t depends on past squared values | terminology |
| MMD batch | 38 | 38 (or GPU maximum, decided on split 0) | memory |
| MMD evaluation kernel | untruncated signature kernel for distances | not used for model comparison | avoid favouring one model |
| DDPM data | AAPL minute data, 3 variables, 2,481 days, random 2,000/481 split | daily SPX log returns only, chronological folds | available data; univariate setting |
| DDPM window | 390 minutes mirror-extended to 512 | 256 daily returns, sliding windows with stride 20 | power of 2; daily data |
| DDPM image | 16×256×3, block repetition | 16×256×1, block repetition, block-mean read-out, rows 9–15 zero | univariate |
| DDPM preprocessing | (X − μ)^{1/p}/σ, p = 1.5, winsorise at 10σ | centre, winsorise at ±10σ of training returns, signed power 1/p, divide by SD of the powered training series | order made explicit |
| DDPM scaling | not reported | one affine map to [−1, 1] fitted on the training images of the split | needed for DDPM |
| DDPM schedule | Hugging Face tutorial defaults | T = 1000, linear β 1e-4 → 0.02, Adam 1e-4, batch 32 | tutorial defaults |
| Evaluation length | — | first 250 of 256 generated returns; statistics on 249 | equal length across models and years |

Winsorisation is not invertible; its effect on the tail endpoints is quantified on split 0 (share of
clipped training returns and change in ES₅ of the decoded training windows).

## 10. Already inspected data

Thesis draft v8 (July 2026) reported results of earlier implementations of both models for 1995-01 →
2018-09 (in-sample) and 2018-09 → 2023-12 (out-of-sample): summary statistics, ACFs of r, |r|, r²,
leverage, tail CCDF, gain/loss asymmetry and signature-kernel MMD². Those implementations contained the
defects listed in `docs/corrections.md`. All evaluation years 2014–2023 were therefore seen in some form.

## 11. Limitations stated in advance

One asset at one frequency; one implementation per model group, differing on four attributes at once; the
DDPM is a re-implementation adapted from minute-level multivariate data; one-year test blocks give noisy
real reference values; ten years give limited precision and are not guaranteed independent; the study is
retrospective (Section 10).
