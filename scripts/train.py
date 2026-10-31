"""
Train one model on one fold with one seed, then generate synthetic returns.

    python scripts/train.py --config config/mmd_sig.yaml   --fold 0 --seed 0     # development split
    python scripts/train.py --config config/diffusion.yaml --fold 3 --seed 1     # evaluation fold 3 (test 2016)

Splits: 0 = development (train 1995-2011, validation 2012-2013), 1-10 = evaluation folds with test
years 2014-2023, 99 = replication of the Chung & Sester split. See docs/analysis_plan.md.

Outputs in runs/<model>[_<variant>]/fold<k>/seed<s>/:
    generated_returns.npy   (n_generate, 250) log returns — the only input to evaluation
    history.json            training-loss history
    *.weights.h5            model weights
    preprocessing.json      fitted preprocessing state (diffusion codec / MA-noise fit)
    manifest.json           commit, config, data hash, package versions, GPU, timings
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src import data
from src.runlog import set_seed, write_manifest


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--override", nargs="*", default=[],
                    help="key=value config overrides, recorded in the manifest (smoke tests only)")
    return ap.parse_args()


def apply_overrides(cfg, overrides):
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def run_mmd(cfg, fold, seed, out_dir):
    import tensorflow as tf
    from src.models import mmd_sig
    from src.models.mmd_noise import build_ma_noise_sampler

    prices = fold.train_prices()
    windows = data.log_price_windows(prices, cfg["sample_len"], cfg["stride"])
    tr = fold.train_returns()
    r, dt = tr["r"].values, tr["dt"].values
    sampler, fit = build_ma_noise_sampler(r, dt, noise_dim=cfg["noise_dim"], p=cfg["ma_order"], seed=seed)

    gen = mmd_sig.build_generator(cfg)
    kernel = mmd_sig.build_kernel(cfg)
    t0 = time.time()
    result = mmd_sig.train(gen, kernel, windows, sampler, cfg, seed, str(out_dir))
    train_s = time.time() - t0

    gen.load_weights(str(out_dir / "best.weights.h5"))
    burnin = data.log_price_windows(prices, cfg["sample_len"], 1)      # training period only
    t0 = time.time()
    gen_r = mmd_sig.generate_returns(gen, burnin, sampler, cfg, cfg["n_generate"], seed=seed + 10_000)
    gen_s = time.time() - t0

    prep = {"annualised_mean": fit["mean"], "annualised_std": fit["std"],
            "lambert_tau": [float(v) for v in fit["lambert"].coefs_[0]],
            "ma_params": {k: float(v) for k, v in fit["ma_res"].params.items()}}
    info = {"n_train_windows": int(len(windows)), "n_params": int(gen.count_params()),
            "n_train_returns": int(len(r)), "train_period": [str(prices.index[0].date()), str(prices.index[-1].date())]}
    return gen_r, result, prep, info, train_s, gen_s


def run_diffusion(cfg, fold, seed, out_dir):
    from src.models import diffusion_wavelet as dw

    prices = fold.train_prices()
    r = fold.train_returns()["r"].values
    windows = data.return_windows(r, cfg["window"], cfg["stride"])
    codec = dw.WaveletImageCodec(cfg["power_p"], cfg["winsor_z"], cfg["image_scaling"]).fit(r, windows)
    images = codec.encode(windows)

    model = dw.build_unet(cfg)
    diff = dw.GaussianDiffusion(cfg["T"], cfg["beta_start"], cfg["beta_end"])
    t0 = time.time()
    result = dw.train(model, diff, images, cfg, seed, str(out_dir))
    train_s = time.time() - t0

    t0 = time.time()
    gen_r = dw.generate_returns(model, diff, codec, cfg["n_generate"], cfg["out_len"])
    gen_s = time.time() - t0
    info = {"n_train_windows": int(len(windows)), "n_params": int(model.count_params()),
            "n_train_returns": int(len(r)), "train_period": [str(prices.index[0].date()), str(prices.index[-1].date())],
            "image_range_train": [float(images.min()), float(images.max())]}
    return gen_r, result, codec.state(), info, train_s, gen_s


def main():
    a = parse()
    cfg = apply_overrides(yaml.safe_load(open(a.config)), a.override)
    fold = data.get_fold(a.fold)
    name = cfg["model"] + (f"_{cfg['variant']}" if cfg.get("variant") else "")
    out_dir = Path(a.out_root) / name / f"fold{a.fold}" / f"seed{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed, a.deterministic)

    runner = {"mmd_sig": run_mmd, "diffusion": run_diffusion}[cfg["model"]]
    gen_r, result, prep, info, train_s, gen_s = runner(cfg, fold, a.seed, out_dir)

    if not np.all(np.isfinite(gen_r)):
        print("WARNING: non-finite generated returns", file=sys.stderr)
    np.save(out_dir / "generated_returns.npy", gen_r)
    (out_dir / "history.json").write_text(json.dumps(result, indent=1))
    (out_dir / "preprocessing.json").write_text(json.dumps(prep, indent=1, default=float))
    write_manifest(out_dir, config=cfg, config_file=a.config, overrides=a.override, fold=vars(fold),
                   seed=a.seed, deterministic=a.deterministic, info=info,
                   timing_seconds={"train": round(train_s, 1), "generate": round(gen_s, 1)},
                   generated={"shape": list(gen_r.shape), "mean": float(np.mean(gen_r)),
                              "std": float(np.std(gen_r))})
    print(f"done -> {out_dir}  train {train_s/60:.1f} min, generate {gen_s/60:.1f} min, "
          f"generated mean {gen_r.mean():+.5f} sd {gen_r.std():.5f}")


if __name__ == "__main__":
    main()
