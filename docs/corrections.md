# Corrections relative to the earlier code and thesis draft (version 8)

Each entry: what was wrong, how it was found, what changed, and how the fix is tested.
Numbers in thesis draft v8 were produced before these corrections and must not be reused.

## C1 — MA(20) noise lost its time structure (MMD-Signature)

* **Was:** `mmd_noise.MANoiseSampler.sample` reshaped `arch`'s simulated residuals, shape
  `(simulations, horizon)`, directly to `(horizon, noise_dim, batch)`. This interleaves values from
  different simulated paths and time points.
* **Evidence:** mean ACF of z² at lags 1–5 of the sampled noise: **0.003** with the old layout (same as
  i.i.d. Gaussian noise, −0.004), **0.096** with the correct layout. The "structured noise" described in the
  thesis Method section was therefore effectively unstructured in the earlier runs.
* **Fix:** keep each simulation intact: `(n_sim, horizon) → (batch, noise_dim, horizon) → (batch, horizon, noise_dim)`.
* **Test:** `tests/test_mmd.py::test_ma_noise_keeps_volatility_clustering`.

## C2 — Real burn-in data counted as generated output (evaluation)

* **Was:** `evaluate.py` differenced the full 300-point MMD path, so the first 49 of the 256 evaluated
  "generated" returns were real S&P 500 returns used as burn-in.
* **Fix:** `mmd_sig.generate_returns` returns only the 250 generated returns; both models are evaluated
  on 250 returns.
* **Test:** `tests/test_mmd.py::test_generator_keeps_history_and_generate_returns_excludes_it`.

## C3 — Wavelet image not time-aligned; preprocessing differed from the paper (diffusion)

* **Was:** each coefficient row was filled with `np.tile` (the whole coefficient vector repeated side by
  side), so column j referred to different time positions on different rows; decoding read the first n
  entries. Preprocessing lacked centring and division by σ.
* **Paper (Takahashi & Mizuno, arXiv:2410.18897):** each coefficient fills its block (block repetition);
  normalisation (X − μ)^(1/p) / σ.
* **Fix:** `np.repeat` (block repetition), block-mean read-out, paper preprocessing with exact inverse.
* **Tests:** `tests/test_diffusion_codec.py` (impulse test for time alignment, block constancy, exact
  encode→decode round-trip, oracle test that decoded real windows give identical stylised facts).
* **Open:** a noise-injection check shows that neither the old nor the new decoder strongly amplifies
  small i.i.d. image errors (SD ratio 1.0–1.3 for ε ≤ 0.1). C3 alone therefore does **not** explain the
  10× too large SD of the earlier diffusion samples. A further candidate is the optimisation budget:
  286 training images × 100 epochs / batch 32 ≈ 900 gradient steps, versus ≈ 6,300 in the paper
  (2,000 images). To be examined in the anchor-split pilot (WP3) on the validation slice.

## C4 — Inconsistent time step (MMD-Signature)

* **Was:** training used calendar days / 365 from the data; generation defaulted to a fixed 1/252; the
  thesis text states a fixed Δt = 1/252.
* **Fix:** calendar days / 365 everywhere; generation uses the real calendar grid of each burn-in window.

## C5 — Burn-in information in the out-of-sample comparison

* **Was:** in the OOS evaluation the MMD model received 50 real days from the OOS period; the diffusion
  model received none.
* **Fix (primary analysis):** burn-in windows are drawn from the training period only
  (`burnin_source: train`). OOS-conditioned generation can be reported as a secondary analysis.

## C6 — Single run, single split, no replication

* **Fix:** ten rolling-origin folds with one-year test blocks (2014–2023) as replication units for the
  significance tests, plus the Chung & Sester split as an anchor (pilot, seed variance, replication);
  see `docs/analysis_plan.md`.

## C7 — Evaluation kernel equals training kernel; real-vs-real baseline overlaps

* **Fix (WP5):** omnibus tests use kernels not used in training (Gaussian-RBF MMD with permutation test,
  classifier two-sample test); real-vs-real reference built from disjoint blocks.

## Housekeeping

* MMD training step compiled with `tf.function` (numerically identical, faster).
* `arch` fit uses `rescale=False`; the earlier `rescale=True` fit returned scale 1.0 on these data, so
  parameters are unchanged.
* Diffusion UNet parameter count reproduced exactly (22,228,865), see
  `tests/test_diffusion_codec.py::test_unet_parameter_count_matches_thesis`.
