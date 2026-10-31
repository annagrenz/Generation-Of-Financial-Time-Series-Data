"""Reproducibility helpers: seeding and the per-run manifest."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data import DATA_PATH, file_sha256, REPO_ROOT


def set_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    import keras
    keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args],
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
            "tag": run("describe", "--tags", "--always")}


def environment() -> dict:
    import tensorflow as tf
    import keras
    import arch
    import scipy
    import pandas as pd
    gpus = tf.config.list_physical_devices("GPU")
    details = []
    for g in gpus:
        try:
            details.append(tf.config.experimental.get_device_details(g).get("device_name"))
        except Exception:
            details.append(g.name)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "arch": arch.__version__,
            "gpus": details, "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))}


def write_manifest(out_dir: str | Path, **fields) -> Path:
    out = Path(out_dir) / "manifest.json"
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git": git_state(), "environment": environment(),
                "data": {"path": str(DATA_PATH.relative_to(REPO_ROOT)), "sha256": file_sha256()},
                **fields}
    out.write_text(json.dumps(manifest, indent=2, default=_jsonable))
    return out


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
