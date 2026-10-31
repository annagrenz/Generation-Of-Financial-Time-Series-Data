"""
Wavelet-image DDPM (Takahashi & Mizuno, 2025) for univariate daily log returns.

Takahashi & Mizuno publish no code; this re-implementation follows their
method section (arXiv:2410.18897). Hyperparameters live in config/diffusion.yaml.

Pipeline for one window of W = 256 log returns
  1. preprocessing (paper):  x = r - mu, winsorised at +-z*sd(r),
                             y = sign(x)|x|^(1/p) / s     (p = 1.5, s = sd of the powered training series)
  2. Haar DWT -> 9 coefficient sets of sizes 1 (approximation), 1, 2, 4, ..., 128 (details)
  3. image: coefficient set k becomes row k; each coefficient is REPEATED over its
     block of 256 / len_k columns (paper: "each coefficient fills its block"),
     so every column refers to the same time location on every row;
     rows 9..15 are zero padding -> 16 x 256 single-channel image
  4. one global affine map of all training images to [-1, 1]
  5. DDPM (Ho et al., 2020) with a UNet noise predictor (128-128-256-256-512)

Decoding reverses every step; within a block the generated values are averaged.

Corrections relative to the earlier thesis code
  * step 3 used np.tile (whole coefficient vector repeated side by side), so
    columns were not time-aligned across rows and the vertical neighbourhood
    of the 2-D convolutions had no consistent meaning;
  * step 1 lacked centring and the division by s used in the paper.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf
import keras
from keras import layers

WINDOW = 256
N_LEVELS = 8          # log2(256)
N_ROWS_USED = N_LEVELS + 1
IMAGE_H, IMAGE_W = 16, WINDOW
COEF_LENGTHS = [1] + [2 ** i for i in range(N_LEVELS)]   # 1, 1, 2, 4, ..., 128


# ---------------------------------------------------------------------------
# Preprocessing (paper) and Haar transform
# ---------------------------------------------------------------------------

class Preprocessor:
    def __init__(self, p: float = 1.5, winsor_z: float = 10.0):
        self.p, self.winsor_z = p, winsor_z
        self.mu = self.sd_r = self.s = None

    def fit(self, r: np.ndarray) -> "Preprocessor":
        r = np.asarray(r, np.float64)
        self.mu, self.sd_r = float(r.mean()), float(r.std())
        self.s = 1.0
        self.s = float(self._power(r).std())
        return self

    def _power(self, r):
        x = np.clip(r - self.mu, -self.winsor_z * self.sd_r, self.winsor_z * self.sd_r)
        return np.sign(x) * np.abs(x) ** (1.0 / self.p) / self.s

    def transform(self, r):
        return self._power(np.asarray(r, np.float64))

    def inverse(self, y):
        x = np.asarray(y, np.float64) * self.s
        return np.sign(x) * np.abs(x) ** self.p + self.mu

    def state(self) -> dict:
        return {"p": self.p, "winsor_z": self.winsor_z, "mu": self.mu, "sd_r": self.sd_r, "s": self.s}


def haar_forward(x: np.ndarray) -> list[np.ndarray]:
    """Orthonormal Haar DWT; returns [approx(1), detail(1), detail(2), ..., detail(N/2)]."""
    x = np.asarray(x, np.float64)
    assert len(x) & (len(x) - 1) == 0, "length must be a power of 2"
    details, cur = [], x
    while len(cur) > 1:
        details.append((cur[0::2] - cur[1::2]) / np.sqrt(2))
        cur = (cur[0::2] + cur[1::2]) / np.sqrt(2)
    return [cur] + details[::-1]


def haar_inverse(coeffs: list[np.ndarray]) -> np.ndarray:
    cur = np.asarray(coeffs[0], np.float64)
    for d in coeffs[1:]:
        rec = np.empty(2 * len(d))
        rec[0::2] = (cur + d) / np.sqrt(2)
        rec[1::2] = (cur - d) / np.sqrt(2)
        cur = rec
    return cur


def coeffs_to_image(coeffs: list[np.ndarray]) -> np.ndarray:
    rows = [np.repeat(c, IMAGE_W // len(c)) for c in coeffs]
    rows += [np.zeros(IMAGE_W)] * (IMAGE_H - len(rows))
    return np.stack(rows)


def image_to_coeffs(img: np.ndarray) -> list[np.ndarray]:
    return [img[k].reshape(n, IMAGE_W // n).mean(axis=1) for k, n in enumerate(COEF_LENGTHS)]


class WaveletImageCodec:
    """Maps return windows <-> normalised 16x256 images (fit on training windows only)."""

    def __init__(self, p=1.5, winsor_z=10.0, scaling="minmax"):
        self.pre = Preprocessor(p, winsor_z)
        self.scaling = scaling
        self.lo = self.hi = None

    def fit(self, train_returns: np.ndarray, train_windows: np.ndarray) -> "WaveletImageCodec":
        self.pre.fit(train_returns)
        raw = np.stack([self._raw_image(w) for w in train_windows])
        if self.scaling == "minmax":
            self.lo, self.hi = float(raw.min()), float(raw.max())
        else:
            raise ValueError(f"unknown scaling {self.scaling}")
        return self

    def _raw_image(self, window):
        return coeffs_to_image(haar_forward(self.pre.transform(window)))

    def encode(self, windows: np.ndarray) -> np.ndarray:
        raw = np.stack([self._raw_image(w) for w in windows])
        img = 2.0 * (raw - self.lo) / (self.hi - self.lo) - 1.0
        return img[..., None].astype(np.float32)

    def decode(self, images: np.ndarray) -> np.ndarray:
        imgs = np.asarray(images, np.float64)
        if imgs.ndim == 4:
            imgs = imgs[..., 0]
        raw = (imgs + 1.0) / 2.0 * (self.hi - self.lo) + self.lo
        return np.stack([self.pre.inverse(haar_inverse(image_to_coeffs(im))) for im in raw])

    def state(self) -> dict:
        return {**self.pre.state(), "scaling": self.scaling, "lo": self.lo, "hi": self.hi}


# ---------------------------------------------------------------------------
# UNet noise predictor
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(layers.Layer):
    def __init__(self, dim, **kw):
        super().__init__(**kw)
        self.dim = dim

    def call(self, t):
        half = self.dim // 2
        f = tf.exp(-tf.math.log(10000.0) / (half - 1) * tf.cast(tf.range(half), tf.float32))
        a = tf.cast(t, tf.float32)[:, None] * f[None]
        return tf.concat([tf.sin(a), tf.cos(a)], -1)


class ResBlock(layers.Layer):
    def __init__(self, cin, cout, groups=32, **kw):
        super().__init__(**kw)
        self.n1 = layers.GroupNormalization(groups=min(groups, cin))
        self.c1 = layers.Conv2D(cout, 3, padding="same")
        self.tp = layers.Dense(cout)
        self.n2 = layers.GroupNormalization(groups=min(groups, cout))
        self.c2 = layers.Conv2D(cout, 3, padding="same")
        self.skip = layers.Conv2D(cout, 1) if cin != cout else None

    def call(self, x, temb):
        h = self.c1(tf.nn.silu(self.n1(x)))
        h = h + self.tp(tf.nn.silu(temb))[:, None, None]
        h = self.c2(tf.nn.silu(self.n2(h)))
        return (self.skip(x) if self.skip is not None else x) + h


class Attention(layers.Layer):
    def __init__(self, ch, heads=8, **kw):
        super().__init__(**kw)
        self.ch = ch
        self.norm = layers.GroupNormalization(groups=min(32, ch))
        self.mha = layers.MultiHeadAttention(num_heads=heads, key_dim=ch // heads)

    def call(self, x):
        b, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        y = tf.reshape(self.norm(x), (b, h * w, self.ch))
        return x + tf.reshape(self.mha(y, y), (b, h, w, self.ch))


class UNet(keras.Model):
    def __init__(self, base=128, mults=(1, 1, 2, 2, 4), temb_dim=512, heads=8, **kw):
        super().__init__(**kw)
        ch = [base * m for m in mults]
        self.temb = keras.Sequential([SinusoidalEmbedding(base),
                                      layers.Dense(temb_dim, activation="silu"), layers.Dense(temb_dim)])
        self.inc = layers.Conv2D(ch[0], 3, padding="same")
        self.enc = [ResBlock(ch[max(i - 1, 0)], ch[i]) for i in range(4)]
        self.down = [layers.Conv2D(ch[i], 3, strides=2, padding="same") for i in range(4)]
        self.mid1, self.attn, self.mid2 = ResBlock(ch[3], ch[4]), Attention(ch[4], heads), ResBlock(ch[4], ch[4])
        self.up = [keras.Sequential([layers.UpSampling2D(2, interpolation="bilinear"),
                                     layers.Conv2D(ch[i], 3, padding="same")]) for i in range(4)]
        self.dec = [ResBlock(2 * ch[i], ch[i]) for i in range(4)]
        self.out_norm = layers.GroupNormalization(groups=min(32, ch[0]))
        self.out_conv = layers.Conv2D(1, 3, padding="same")

    def call(self, inputs, training=False):
        x, t = inputs
        e = self.temb(t)
        x = self.inc(x)
        skips = []
        for i in range(4):
            x = self.enc[i](x, e)
            skips.append(x)
            x = self.down[i](x)
        x = self.mid2(self.attn(self.mid1(x, e)), e)
        for i in reversed(range(4)):
            x = self.dec[i](tf.concat([self.up[i](x), skips[i]], -1), e)
        return self.out_conv(tf.nn.silu(self.out_norm(x)))


def build_unet(cfg: dict) -> UNet:
    m = UNet(base=cfg["base_channels"], mults=tuple(cfg["channel_mults"]),
             temb_dim=cfg["time_emb_dim"], heads=cfg["num_heads"])
    m([tf.zeros((1, IMAGE_H, IMAGE_W, 1)), tf.zeros((1,), tf.int32)])
    return m


# ---------------------------------------------------------------------------
# DDPM
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        b = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        a = 1.0 - b
        ab = np.cumprod(a)
        f = lambda v: tf.constant(v, tf.float32)
        self.betas, self.alphas, self.ab = f(b), f(a), f(ab)
        self.sqrt_ab, self.sqrt_1mab = f(np.sqrt(ab)), f(np.sqrt(1 - ab))

    def q_sample(self, x0, t, eps):
        return tf.gather(self.sqrt_ab, t)[:, None, None, None] * x0 + \
            tf.gather(self.sqrt_1mab, t)[:, None, None, None] * eps


def train(model, diffusion, images: np.ndarray, cfg: dict, seed: int, save_dir: str, log=print) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    ds = (tf.data.Dataset.from_tensor_slices(images)
          .shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
          .batch(cfg["batch_size"]))
    opt = keras.optimizers.Adam(learning_rate=cfg["lr"])

    @tf.function(reduce_retracing=True)
    def step(x0):
        b = tf.shape(x0)[0]
        t = tf.random.uniform([b], 0, diffusion.T, tf.int32)
        eps = tf.random.normal(tf.shape(x0))
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(eps - model([diffusion.q_sample(x0, t, eps), t], training=True)))
        g = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        loss = float(np.mean([float(step(b)) for b in ds]))
        history.append({"epoch": epoch, "loss": loss})
        if epoch % cfg["log_every"] == 0:
            log(f"epoch {epoch:4d}  mse={loss:.6f}")
    model.save_weights(os.path.join(save_dir, "final.weights.h5"))
    return {"history": history, "epochs_run": len(history)}


def sample_images(model, diffusion, n: int, batch: int = 100) -> np.ndarray:
    @tf.function(reduce_retracing=True)
    def p_step(x, t):
        tb = tf.fill([tf.shape(x)[0]], t)
        eps = model([x, tb], training=False)
        bt, at, s1 = tf.gather(diffusion.betas, t), tf.gather(diffusion.alphas, t), tf.gather(diffusion.sqrt_1mab, t)
        mean = (x - bt / s1 * eps) / tf.sqrt(at)
        noise = tf.random.normal(tf.shape(x))
        return tf.where(t > 0, mean + tf.sqrt(bt) * noise, mean)

    out = []
    for s in range(0, n, batch):
        x = tf.random.normal((min(batch, n - s), IMAGE_H, IMAGE_W, 1))
        for t in reversed(range(diffusion.T)):
            x = p_step(x, tf.constant(t, tf.int32))
        out.append(x.numpy())
    return np.concatenate(out)


def generate_returns(model, diffusion, codec: WaveletImageCodec, n_paths: int, out_len: int,
                     batch: int = 100) -> np.ndarray:
    """Generate n_paths windows and return the first `out_len` returns of each."""
    r = codec.decode(sample_images(model, diffusion, n_paths, batch))
    return r[:, :out_len]
