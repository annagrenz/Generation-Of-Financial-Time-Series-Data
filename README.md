# Generation of Financial Time Series Data — MSc thesis code

Anna Grenz, MSc Behavioural Data Science, University of Amsterdam. Supervisor: Raoul Grasman.

Two generative models for daily S&P 500 log returns, trained on identical data and evaluated with
identical, pre-specified criteria:

| Model | Source | Code |
|---|---|---|
| MMD-Signature generator (sequential LSTM, structured MA(20) noise, signature-kernel MMD loss) | Chung & Sester (2025) | `src/models/mmd_sig.py`, `src/models/mmd_noise.py` |
| Wavelet-image DDPM (Haar DWT image, UNet noise predictor) | Takahashi & Mizuno (2025) | `src/models/diffusion_wavelet.py` |

The study design (research question, folds, metrics, statistical tests) is fixed in
[`docs/analysis_plan.md`](docs/analysis_plan.md). Differences from the earlier code and thesis draft are
logged in [`docs/corrections.md`](docs/corrections.md).

## Layout

```
config/       model hyperparameters (one YAML per model / variant), each value with its source
data/         spx_20231229.csv (from the Chung & Sester repository; SHA-256 checked on every load)
src/data.py   loading, checksum, rolling-origin folds, window builders
src/models/   the two generators (+ baselines, added in WP4)
src/metrics/  stylised-fact statistics and discrepancies (+ omnibus tests, WP5)
scripts/      train.py (train + generate one model/fold/seed); evaluation scripts follow in WP5
notebooks/    kaggle_runner.ipynb — clones a tagged commit and runs scripts/train.py on a GPU
tests/        pytest suite (data leakage, codec round-trips, noise structure, metrics)
```

## Running

```bash
pip install -r requirements.txt          # plus tensorflow (preinstalled on Kaggle)
pytest -q                                # must pass before any experiment
python scripts/train.py --config config/mmd_sig.yaml   --fold 0 --seed 0   # development split
python scripts/train.py --config config/diffusion.yaml --fold 0 --seed 0
```

Splits: `0` = development (train 1995–2011, validate 2012–2013; all configuration decisions are made
here), `1`–`10` = evaluation folds with test years 2014–2023, `99` = replication of the Chung & Sester
split.

Every run writes `manifest.json` with the git commit, config, data hash, package versions, GPU and timings,
so every number in the thesis can be traced to the code and data that produced it.

## Reference code

The original PyTorch implementation of the MMD-Signature model (including the RL experiments) is at
<https://github.com/luchungi/Generative-Model-Signature-MMD>. It is not duplicated here.
