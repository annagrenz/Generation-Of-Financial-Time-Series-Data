"""
MMD-Signature generator (Chung & Sester, 2025) — Keras/TensorFlow port.

The architecture, signature kernel and training loop follow the authors'
PyTorch repository (github.com/luchungi/Generative-Model-Signature-MMD) and the
paper (Sec. 3-5). Hyperparameters live in config/mmd_sig.yaml.

Generator
  single-layer LSTM; input at step t = [previous return, 4-d MA noise, dt];
  a linear head maps the hidden state to the next log return. The first
  `hist_len` points of each window are real data used only as burn-in.

Loss
  unbiased MMD^2 between real and generated *generated-part* paths
  (time, log price), lead-lag augmented, under a truncated signature kernel
  with a rational-quadratic static kernel.

Only the generated part of a path is ever returned for evaluation
(`generate_returns`), so real burn-in data cannot leak into the metrics.
"""

from __future__ import annotations

import os
from collections import deque

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf
import keras
from keras import layers


# ---------------------------------------------------------------------------
# Static kernels and signature kernel
# ---------------------------------------------------------------------------

def _sq_dist(X, Y):
    xx = tf.reduce_sum(X ** 2, -1)
    yy = tf.reduce_sum(Y ** 2, -1)
    return xx[:, None] + yy[None, :] - 2.0 * tf.matmul(X, Y, transpose_b=True)


class LinearKernel:
    def gram_matrix(self, X, Y):
        return tf.matmul(X, Y, transpose_b=True)


class RBFKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def gram_matrix(self, X, Y):
        return tf.exp(-_sq_dist(X, Y) / self.sigma ** 2)


class RationalQuadraticKernel:
    def __init__(self, sigma, alpha=1.0):
        self.sigma, self.alpha = sigma, alpha

    def gram_matrix(self, X, Y):
        return tf.pow(1.0 + _sq_dist(X, Y) / (2.0 * self.alpha * self.sigma ** 2), -self.alpha)


def get_static_kernel(kind: str, sigma: float, alpha: float = 1.0):
    if kind == "linear":
        return LinearKernel()
    if kind == "rbf":
        return RBFKernel(sigma)
    if kind == "rq":
        return RationalQuadraticKernel(sigma, alpha)
    raise ValueError(f"Unknown static kernel {kind}")


def _multi_cumsum(M, axes=(1, 3)):
    nd = len(M.shape)
    sl = tuple(slice(None, -1) if i in axes else slice(None) for i in range(nd))
    M = M[sl]
    for ax in axes:
        M = tf.cumsum(M, axis=ax)
    return tf.pad(M, [[1, 0] if i in axes else [0, 0] for i in range(nd)])


class SignatureKernel:
    """Truncated signature kernel (Kiraly & Oberhauser, 2019), dynamic-programming form."""

    def __init__(self, n_levels: int, static_kernel):
        self.n_levels, self.static_kernel = n_levels, static_kernel

    def __call__(self, X, Y):
        nX, TX = tf.shape(X)[0], tf.shape(X)[1]
        nY, TY = tf.shape(Y)[0], tf.shape(Y)[1]
        d = X.shape[-1]
        G = self.static_kernel.gram_matrix(tf.reshape(X, (-1, d)), tf.reshape(Y, (-1, d)))
        M = tf.reshape(G, (nX, TX, nY, TY))
        M = M[:, 1:] - M[:, :-1]
        M = M[:, :, :, 1:] - M[:, :, :, :-1]
        K = tf.ones((nX, nY), M.dtype) + tf.reduce_sum(M, axis=(1, 3))
        R = M
        for _ in range(1, self.n_levels):
            R = M * _multi_cumsum(R, axes=(1, 3))
            K = K + tf.reduce_sum(R, axis=(1, 3))
        return K


def mmd2_unbiased(X, Y, kernel):
    Kxx, Kyy, Kxy = kernel(X, X), kernel(Y, Y), kernel(X, Y)
    n = tf.cast(tf.shape(Kxx)[0], Kxx.dtype)
    m = tf.cast(tf.shape(Kyy)[0], Kyy.dtype)
    sxx = tf.reduce_sum(Kxx) - tf.linalg.trace(Kxx)
    syy = tf.reduce_sum(Kyy) - tf.linalg.trace(Kyy)
    return sxx / (n * (n - 1)) + syy / (m * (m - 1)) - 2.0 * tf.reduce_sum(Kxy) / (n * m)


def lead_lag(x, t, lags=(1,)):
    """Lead-lag augmentation: [t, x, x lagged by each lag]; time and data padded at the end."""
    lags = list(lags)
    max_lag = max(lags)
    B, T, D = tf.shape(x)[0], x.shape[1], x.shape[2]
    t_ext = tf.concat([t, tf.repeat(t[:, -1:], max_lag, axis=1)], axis=1)
    cols = [tf.concat([x, tf.repeat(x[:, -1:], max_lag, axis=1)], axis=1)]
    for lag in lags:
        parts = [tf.zeros((B, lag, D), x.dtype), x]
        if max_lag - lag > 0:
            parts.append(tf.repeat(x[:, -1:], max_lag - lag, axis=1))
        cols.append(tf.concat(parts, axis=1)[:, :T + max_lag])
    return tf.concat([t_ext] + cols, axis=-1)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class GenLSTM(keras.Model):
    def __init__(self, noise_dim=4, seq_dim=1, seq_len=300, hidden_size=64, n_lstm_layers=1, **kw):
        super().__init__(**kw)
        self.noise_dim, self.seq_dim, self.seq_len = noise_dim, seq_dim, seq_len
        self.hidden_size, self.n_lstm_layers = hidden_size, n_lstm_layers
        self.cells = [layers.LSTMCell(hidden_size, name=f"lstm_cell_{i}") for i in range(n_lstm_layers)]
        self.head = layers.Dense(seq_dim)

    def _step(self, inp, h, c):
        x, nh, nc = inp, [], []
        for i, cell in enumerate(self.cells):
            x, (hi, ci) = cell(x, [h[i], c[i]])
            nh.append(hi)
            nc.append(ci)
        return x, nh, nc

    def call(self, inputs, training=False):
        """
        inputs = (noise (B, T-1, noise_dim), t (B, T, 1), hist_x (B, H, seq_dim))
        returns full log-price path (B, T, seq_dim): real history followed by generated part.
        """
        noise, t, hist_x = inputs
        B = tf.shape(noise)[0]
        T = self.seq_len
        dts = t[:, 1:] - t[:, :-1]
        h = [tf.zeros((B, self.hidden_size)) for _ in self.cells]
        c = [tf.zeros((B, self.hidden_size)) for _ in self.cells]

        H = hist_x.shape[1]
        dh = hist_x[:, 1:] - hist_x[:, :-1]
        out = None
        for i in range(H - 1):                      # burn-in on real returns
            out, h, c = self._step(tf.concat([dh[:, i], noise[:, i], dts[:, i]], -1), h, c)

        gen = []
        for i in range(T - H):                      # generate T - H returns
            x = self.head(out)
            gen.append(x[:, None])
            ni = H - 1 + i
            if ni < T - 1:
                out, h, c = self._step(tf.concat([x, noise[:, ni], dts[:, ni]], -1), h, c)
        gen = tf.concat(gen, axis=1)
        return tf.concat([hist_x, hist_x[:, -1:] + tf.cumsum(gen, axis=1)], axis=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_generator(cfg: dict) -> GenLSTM:
    g = GenLSTM(noise_dim=cfg["noise_dim"], seq_dim=1, seq_len=cfg["sample_len"],
                hidden_size=cfg["hidden_size"], n_lstm_layers=cfg["n_lstm_layers"])
    T, H = cfg["sample_len"], cfg["hist_len"]
    g((tf.zeros((1, T - 1, cfg["noise_dim"])), tf.zeros((1, T, 1)), tf.zeros((1, H, 1))))
    return g


def build_kernel(cfg: dict) -> SignatureKernel:
    return SignatureKernel(cfg["n_levels"],
                           get_static_kernel(cfg["static_kernel"], cfg["kernel_sigma"], cfg["kernel_alpha"]))


def train(generator, kernel, windows: np.ndarray, noise_sampler, cfg: dict, seed: int,
          save_dir: str, log=print) -> dict:
    """Paper training loop: Adam, LR halved on plateau of the smoothed loss, early stopping."""
    os.makedirs(save_dir, exist_ok=True)
    T, H, B = cfg["sample_len"], cfg["hist_len"], cfg["batch_size"]
    lags = cfg["lags"]

    ds = (tf.data.Dataset.from_tensor_slices(windows)
          .shuffle(len(windows), seed=seed, reshuffle_each_iteration=True)
          .batch(B, drop_remainder=True))
    opt = keras.optimizers.Adam(learning_rate=cfg["start_lr"])

    @tf.function(reduce_retracing=True)
    def step(X, noise):
        t, lp = X[:, :, :1], X[:, :, 1:]
        with tf.GradientTape() as tape:
            path = generator((noise, t, lp[:, :H]), training=True)
            real, fake = X[:, H:], tf.concat([t, path], -1)[:, H:]
            if cfg["lead_lag"]:
                real = lead_lag(real[:, :, 1:], real[:, :, :1], lags)
                fake = lead_lag(fake[:, :, 1:], fake[:, :, :1], lags)
            loss = mmd2_unbiased(real, fake, kernel)
        grads = tape.gradient(loss, generator.trainable_variables)
        opt.apply_gradients(zip(grads, generator.trainable_variables))
        return loss

    rng = np.random.default_rng(seed)
    last = deque(maxlen=cfg["num_losses"])
    best, best_epoch, wait_lr, no_improve = np.inf, 0, 0, 0
    history = []
    for epoch in range(1, cfg["max_epochs"] + 1):
        losses = []
        for X in ds:
            if cfg["use_ma_noise"]:
                noise = noise_sampler.sample(T, B)
            else:
                noise = rng.standard_normal((B, T - 1, cfg["noise_dim"])).astype(np.float32)
            losses.append(float(step(X, tf.constant(noise))))
        avg = float(np.mean(losses))
        last.append(avg)
        smooth = float(np.mean(last))
        history.append({"epoch": epoch, "loss": avg, "smoothed": smooth,
                        "lr": float(opt.learning_rate.numpy())})
        if epoch % cfg["log_every"] == 0:
            log(f"epoch {epoch:5d}  mmd2={avg:+.6f}  smooth={smooth:+.6f}  lr={history[-1]['lr']:.2e}")

        if smooth < best:
            best, best_epoch, wait_lr, no_improve = smooth, epoch, 0, 0
            generator.save_weights(os.path.join(save_dir, "best.weights.h5"))
        else:
            wait_lr += 1
            no_improve += 1
            if wait_lr >= cfg["patience"]:
                opt.learning_rate.assign(max(float(opt.learning_rate.numpy()) * cfg["lr_factor"], cfg["min_lr"]))
                wait_lr = 0
            if no_improve >= cfg["early_stopping"]:
                log(f"early stop at epoch {epoch} (best smoothed {best:+.6f} @ {best_epoch})")
                break
    generator.save_weights(os.path.join(save_dir, "final.weights.h5"))
    return {"history": history, "best_epoch": best_epoch, "best_smoothed_loss": best,
            "epochs_run": len(history)}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_returns(generator, burnin_windows: np.ndarray, noise_sampler, cfg: dict,
                     n_paths: int, seed: int, batch: int = 250) -> np.ndarray:
    """
    Generate `n_paths` paths and return ONLY the generated log returns, shape
    (n_paths, sample_len - hist_len). Burn-in histories and their calendar
    time grids are drawn (with replacement) from `burnin_windows`, which the
    caller takes from the training period in the primary analysis.
    """
    T, H = cfg["sample_len"], cfg["hist_len"]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(burnin_windows), n_paths)
    out = []
    for s in range(0, n_paths, batch):
        w = burnin_windows[idx[s:s + batch]]
        b = len(w)
        t = tf.constant(w[:, :, :1])
        hist = tf.constant(w[:, :H, 1:])
        if cfg["use_ma_noise"]:
            noise = noise_sampler.sample(T, b)
        else:
            noise = rng.standard_normal((b, T - 1, cfg["noise_dim"])).astype(np.float32)
        path = generator((tf.constant(noise), t, hist), training=False).numpy()[:, :, 0]
        out.append(np.diff(path[:, H - 1:], axis=1))     # first generated return onwards
    r = np.concatenate(out, axis=0)
    assert r.shape == (n_paths, T - H), r.shape
    return r.astype(np.float64)
