"""
diffusion_model.py — Wavelet-based DDPM generator for daily SPX log-returns.

v3 ARCHITECTURE CHANGE 
-----------------------
After v2 still showed a strong oscillating ACF at lags 4, 8, 12, 16 and a
blown-up std, the diagnosis pointed to a STRUCTURAL problem with the original
paper's image layout when applied to univariate data:

    The paper stacks 9 DWT coefficient levels as 9 rows of a 16×256 image.
    UNet 3×3 spatial convolutions then cross row boundaries, mixing
    level-3 coefficients with level-4 etc. For the paper's RGB minute-data
    setup (3 channels of richly-varying intraday spectrograms) this cross-row
    leakage is masked by the channel structure. For our univariate daily
    data it shows up as periodic artefacts at multiples of the wavelet
    block sizes (4, 8, 16, 32, …).

v3 fix: SEPARATE THE DWT LEVELS INTO CHANNELS, not rows.

    image shape:  (16, 256, 1)   →   (1, 256, 9)
                  9 levels stacked      9 levels each in own channel,
                  + 7 padding rows      no padding, no cross-level spatial
                                        convolutions, all signal meaningful.

The UNet keeps its overall design but spatial convolutions become 1×3 (the
H=1 dimension is degenerate), height downsampling is dropped, and only the
width is downsampled through the encoder. Channel mixing still happens through
1×1 projections inside each ResBlock, but in a controlled, learnable way
rather than being forced by 3×3 kernels crossing rows.

What is RETAINED from v2 that did work:
  - np.repeat for block-style coefficient tiling (forward).
  - Block-averaging on the inverse.
  - EMA, cosine LR schedule, 200 epochs.
  - Full-pipeline series→image→series roundtrip test.

What is REVERTED from v2 that didn't work:
  - Standardisation (x-μ)/σ in PowerTransform. Empirically this BLEW UP the
    generated std from ~0.05 to ~0.15. Reverted to the no-standardisation
    behaviour of v1: signed power-transform on the winsorised return directly.

Per-channel image normalisation:
  - v1/v2 used a global min/max over the whole image. The coarsest DWT
    coefficient has magnitude ~sqrt(256) larger than the finest, so global
    scaling forced the finest-level coefficients into [-0.01, 0.01] of the
    [-1, 1] image range — i.e. only 1% of the dynamic range available to
    the network. v3 normalises each channel independently so every level
    uses the full [-1, 1] range.

ADAPTATION FROM THE PAPER  (unchanged across versions — keep for the thesis)
----------------------------------------------------------------------------
The original paper applies its method to MINUTE-LEVEL AAPL.O data with THREE
channels (log-returns, bid-ask spreads, trading volumes) packed into a single
RGB image, and one of its headline contributions is reproducing INTRADAY
SEASONALITY. None of those properties are available from daily SPX prices.

For a fair head-to-head comparison against the MMD-Signature model (which is
univariate daily index data by design), we deliberately reduce the diffusion
model to its SINGLE-CHANNEL core. What remains — and is what we evaluate — is:
  - The wavelet-imaging trick (Haar DWT → tiled coefficient image)
  - The DDPM training objective (MSE noise-prediction)
  - The UNet noise predictor with attention
  - The fat-tail + volatility-clustering replication
We do NOT claim to replicate intraday seasonality or cross-channel correlations
because the daily univariate data cannot support those claims.

ARCHITECTURE OVERVIEW (v3)
--------------------------
    SPX prices ─→ log returns
                      ↓
                 winsorise at ±10σ
                      ↓
                 power transform sign(x)·|x|^(1/p) with p=1.5     [no standardisation]
                      ↓
                 Haar DWT (one image per sliding window)
                      ↓
                 tile each level across IMAGE_W (np.repeat, block-style)
                      ↓
                 stack 9 levels as 9 channels — image shape (1, 256, 9)
                      ↓
                 per-channel normalise to [-1, 1]
                      ↓
                 DDPM training: UNet ε_θ predicts the noise   (200 ep, cosine LR, EMA)
                      ↓
                 reverse process (EMA weights) from N(0,I) → synthetic image
                      ↓
                 per-channel denormalise → average within tile blocks
                      ↓
                 inverse Haar DWT → inverse power → synthetic log-returns

UNet channel progression: 128 - 128 - 256 - 256 - 512  (5 stages, exact match
to the paper Sec. 4.1 and the HuggingFace DDPM tutorial reference).
Spatial: height stays 1 throughout; width 256→128→64→32→16 then back up.

Run from MMD-Model_VSCode/:
    python -m diffusion_model
"""

import os
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers

import data_pipeline as dp


# ================================================================
# HYPERPARAMETERS
# ================================================================
# Wavelet-image setup
WINDOW_SIZE   = 256        # log-returns per sample. Must be a power of 2 (DWT requirement).
STRIDE        = 20         # sliding-window stride across the return series.
P_POWER       = 1.5        # power-transform exponent for log returns (paper, Sec. 4.1).
WINSOR_Z      = 10.0       # winsorisation clip level in σ.
N_DWT_LEVELS  = 8          # log2(256) = 8 → 9 coefficient sets.
N_CHANNELS    = N_DWT_LEVELS + 1   # = 9. One channel per DWT level (v3 layout).
IMAGE_H       = 1          # degenerate height: levels live in CHANNELS, not rows.
IMAGE_W       = 256        # image width (= WINDOW_SIZE).

# Diffusion schedule
T_STEPS       = 1000       # diffusion-step count T.

# UNet architecture (paper exact: 128-128-256-256-512 → 5 stages, channel multipliers below)
BASE_CHANNELS = 128
CHANNEL_MULTS = (1, 1, 2, 2, 4)
TIME_EMB_DIM  = 512
NUM_HEADS     = 8

# Training
EPOCHS        = 200        # bumped from 100 — loss was still decreasing at epoch 100.
BATCH_SIZE    = 32
LR            = 1e-4
LR_MIN        = 1e-6       # cosine decay floor.
EMA_DECAY     = 0.999      # standard DDPM EMA decay. Used for sampling, not training.
SEED          = 42

# Output
SAVE_DIR      = 'runs/diffusion_paper'


# ================================================================
# PREPROCESSING  (winsorisation + power transform + image scaling)
# ================================================================

class PowerTransform:
    """
    Paper preprocessing (Sec. 4.1) for the return series — v3 NO STANDARDISATION.

      forward:
        x_clean  = clip(x, -z·σ, +z·σ)                      (winsorise)
        x_power  = sign(x_clean) · |x_clean|^(1/p)          (signed power, no standardise)

      inverse:
        x_inv    = sign(y) · |y|^p

    Why no standardisation in v3
    ----------------------------
    v2 added a (x - μ) / σ standardisation step before the power transform, in
    line with what the paper literally writes. Empirically this BLEW UP the
    generated std from ~0.05 to ~0.15 — the standardisation pushed the inputs
    to magnitude ~1, then the inverse |.|^p applied to noisy network outputs
    near 1 created very large values that the missing σ multiplication on the
    way back couldn't tame.

    v3 reverts to the no-standardisation behaviour of v1 for this step. The
    structural fix in v3 (DWT levels as channels) is what addresses the
    underlying problem; the standardisation was a red herring that turned out
    to make the scale worse, not better.

    Per-channel image normalisation
    -------------------------------
    Different DWT levels have hugely different magnitudes — the coarsest
    coefficient is ~sqrt(256) = 16× the size of the finest. With a single
    global min/max scaling (v1, v2), the finest-level coefficients got squeezed
    into ~1% of the [-1, 1] image range. v3 fits min/max PER CHANNEL so every
    level uses the full dynamic range, giving the UNet a chance to model
    detail at every scale.
    """

    def __init__(self, p: float = P_POWER, winsor_z: float = WINSOR_Z):
        self.p        = p
        self.winsor_z = winsor_z
        # Fit-time statistics; set by .fit(). μ, σ used only for the winsor
        # bounds — NOT for standardisation in v3.
        self.mean:    float = None
        self.std:     float = None
        # Per-channel image min/max for [-1, 1] scaling; set by .fit_image_scale().
        # Shape: (N_CHANNELS,) for both.
        self.img_min: np.ndarray = None
        self.img_max: np.ndarray = None

    def fit(self, x: np.ndarray) -> 'PowerTransform':
        """Record μ and σ of the training returns (used for winsor bounds only)."""
        self.mean = float(np.mean(x))
        self.std  = float(np.std(x))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Forward: winsorise → signed power-transform (NO standardisation)."""
        # Step 1: winsorise to ±z·σ.
        lo = self.mean - self.winsor_z * self.std
        hi = self.mean + self.winsor_z * self.std
        x  = np.clip(x, lo, hi)
        # Step 2: signed power transform DIRECTLY on the winsorised return.
        return np.sign(x) * np.abs(x) ** (1.0 / self.p)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Backward: inverse signed power-transform."""
        return np.sign(y) * np.abs(y) ** self.p

    def fit_image_scale(self, images: np.ndarray) -> 'PowerTransform':
        """
        Find per-channel min/max across all training wavelet images.

        Expected shape: (N, IMAGE_H, IMAGE_W, N_CHANNELS).
        Result: img_min, img_max are each shape (N_CHANNELS,).
        """
        # Reduce over batch, H, W axes; keep channels.
        self.img_min = images.min(axis=(0, 1, 2)).astype(np.float64)
        self.img_max = images.max(axis=(0, 1, 2)).astype(np.float64)
        # Guard against a degenerate channel where min==max (would happen on
        # a tiny test dataset of one window; never on real training data).
        eps = 1e-6
        flat = (self.img_max - self.img_min) < eps
        if np.any(flat):
            self.img_max[flat] = self.img_min[flat] + 1.0
        return self

    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Map image values to [-1, 1] using per-channel img_min / img_max."""
        # img shape: (..., N_CHANNELS). Broadcast min/max over leading axes.
        r = self.img_max - self.img_min + 1e-8
        return 2.0 * (img - self.img_min) / r - 1.0

    def denormalize_image(self, img: np.ndarray) -> np.ndarray:
        """Map [-1, 1] back to original wavelet-coefficient scale (per-channel)."""
        r = self.img_max - self.img_min + 1e-8
        return (img + 1.0) / 2.0 * r + self.img_min


# ================================================================
# HAAR WAVELET TRANSFORM  (1-D, orthonormal)
# ================================================================

def haar_forward(x: np.ndarray) -> list:
    """
    Orthonormal Haar DWT on a 1-D array of length 2^n.

    Returns a list of (n+1) arrays ordered from COARSEST to FINEST:
      [approx(1), detail_coarsest(1), detail(2), detail(4), ..., detail(2^(n-1))]

    Total elements sum to 2^n — that's the "perfect reconstruction" property
    of the Haar wavelet, meaning we can losslessly invert this transform.
    """
    # Defensive check: input length must be a power of 2.
    assert len(x) >= 2 and (len(x) & (len(x) - 1)) == 0, \
        'haar_forward requires input length to be a power of 2'

    details = []
    cur     = x.astype(np.float64)
    while len(cur) > 1:
        # Average-and-difference of adjacent pairs. The √2 keeps the basis
        # orthonormal so we can use the same factors during inverse.
        approx = (cur[0::2] + cur[1::2]) / np.sqrt(2)
        detail = (cur[0::2] - cur[1::2]) / np.sqrt(2)
        details.append(detail)
        cur = approx

    # Reverse so index 0 = coarsest detail, last = finest detail.
    return [cur] + list(reversed(details))


def haar_inverse(coeffs: list) -> np.ndarray:
    """Inverse Haar DWT. Exact inverse of haar_forward (up to float precision)."""
    cur = coeffs[0].astype(np.float64)
    for detail in coeffs[1:]:
        n         = len(detail)
        rec       = np.empty(2 * n)
        rec[0::2] = (cur + detail) / np.sqrt(2)
        rec[1::2] = (cur - detail) / np.sqrt(2)
        cur = rec
    return cur


def series_to_image(returns: np.ndarray, pt: PowerTransform) -> np.ndarray:
    """
    Convert WINDOW_SIZE log returns into an (IMAGE_H × IMAGE_W × N_CHANNELS)
    wavelet image — v3 channel layout.

    Pipeline:
      1. Winsorise + power-transform.
      2. Haar DWT → 9 coefficient arrays (1, 1, 2, 4, ..., 128).
      3. Tile each array across IMAGE_W using np.repeat (block-style).
      4. Stack the 9 tiled rows along the CHANNEL axis (not the row axis).
         The image is now (H=1, W=256, C=9): height degenerate, width is
         time, channels are DWT levels. No padding needed.

    Returns: (IMAGE_H=1, IMAGE_W=256, N_CHANNELS=9) float64 array,
             NOT yet rescaled to [-1, 1].
    """
    transformed = pt.transform(returns)
    coeffs      = haar_forward(transformed)         # 9 arrays for 256-length input

    channels = []
    for c in coeffs:
        # Block-style tiling: each coefficient fills its own block of
        # IMAGE_W / len(c) consecutive pixels. np.repeat gives
        # [c0,c0,…,c0, c1,c1,…,c1, …] — the layout that lets the inverse
        # cleanly recover by block-averaging.
        block = IMAGE_W // len(c)
        row   = np.repeat(c, block)[:IMAGE_W]
        channels.append(row)

    # Stack into (W, C) and add the degenerate H=1 axis up front.
    img = np.stack(channels, axis=-1)   # (IMAGE_W, N_CHANNELS)
    img = img[np.newaxis, :, :]         # (IMAGE_H=1, IMAGE_W, N_CHANNELS)
    return img


def image_to_series(image: np.ndarray, pt: PowerTransform) -> np.ndarray:
    """
    Inverse of series_to_image: a generated (1, 256, 9) image → WINDOW_SIZE returns.

    Pipeline:
      1. Drop the leading H=1 axis if present (or handle (W, C) input too).
      2. Denormalise per-channel from [-1, 1] back to wavelet-coefficient scale.
      3. For each DWT level (channel), reshape the 256-length signal into
         (n_coeffs, block_size) and AVERAGE along the block dimension to
         recover one coefficient per block.
      4. Inverse Haar DWT.
      5. Inverse power transform.
    """
    # Accept (H, W, C), (W, C), or even with a Keras batch axis stripped already.
    if image.ndim == 3 and image.shape[0] == IMAGE_H:
        image = image[0]                  # (W, C)
    elif image.ndim == 2:
        pass                              # already (W, C)
    else:
        raise ValueError(f'Unexpected image shape {image.shape}; '
                         f'expected (H, W, C) or (W, C).')

    # Denormalise per-channel back to wavelet-coefficient scale.
    image = pt.denormalize_image(image)   # still (W, C)

    n_per_level = [1] + [2 ** i for i in range(N_DWT_LEVELS)]
    coeffs = []
    for ch, n in enumerate(n_per_level):
        block  = IMAGE_W // n
        signal = image[:, ch]             # length-W coefficient signal for this level
        # Reshape into (n_coeffs, block) and average over block: each
        # coefficient is the network's average estimate over its `block` pixels.
        # This averaging is what suppresses the network's noise — particularly
        # important for the finest levels (which have only 1-2 pixels per block).
        coeffs.append(signal[:n * block].reshape(n, block).mean(axis=1))

    power_returns = haar_inverse(coeffs)
    return pt.inverse_transform(power_returns)


# ================================================================
# DATASET BUILDER
# ================================================================

def build_image_dataset(log_returns: np.ndarray,
                         pt: PowerTransform,
                         window_size: int = WINDOW_SIZE,
                         stride:      int = STRIDE) -> np.ndarray:
    """
    Slide a window across log_returns, convert each window to a wavelet image
    of shape (IMAGE_H, IMAGE_W, N_CHANNELS), fit the per-channel [-1, 1]
    normalisation, and return a (N, IMAGE_H, IMAGE_W, N_CHANNELS) array
    ready for the UNet.
    """
    n   = len(log_returns)
    raw = []
    for start in range(0, n - window_size + 1, stride):
        window = log_returns[start:start + window_size]
        # Each image is (IMAGE_H=1, IMAGE_W, N_CHANNELS).
        raw.append(series_to_image(window, pt))
    raw = np.stack(raw)                  # (N, IMAGE_H, IMAGE_W, N_CHANNELS)

    # FIT image scale (only on training data — must NOT include OOS at this step).
    # Per-channel min/max for v3.
    pt.fit_image_scale(raw)
    normalised = pt.normalize_image(raw)

    return normalised.astype(np.float32)


def build_tf_dataset(images: np.ndarray, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    return (tf.data.Dataset
            .from_tensor_slices(images)
            .shuffle(buffer_size=len(images), seed=SEED)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))


# ================================================================
# UNET BUILDING BLOCKS  (matches the HuggingFace DDPM tutorial defaults)
# ================================================================

class SinusoidalTimeEmbedding(layers.Layer):
    """Standard sinusoidal embedding of the integer diffusion step t."""

    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, t):
        # Build geometric frequencies, same recipe as Transformer position
        # embeddings. Half the dimensions are sin, half are cos.
        half  = self.dim // 2
        scale = tf.math.log(10000.0) / tf.cast(half - 1, tf.float32)
        freqs = tf.exp(-scale * tf.cast(tf.range(half), tf.float32))
        args  = tf.cast(t, tf.float32)[:, None] * freqs[None, :]
        return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)


class ResBlock(layers.Layer):
    """
    Residual 2-D conv block conditioned on the diffusion-step embedding t.

    Structure:  GroupNorm → Swish → Conv → (+ time projection) → GroupNorm → Swish → Conv
    Skip connection uses a 1×1 conv when the channel count changes.

    This is the standard "ResNet block with time conditioning" used by DDPM
    implementations.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, groups=32, **kwargs):
        super().__init__(**kwargs)
        self.norm1     = layers.GroupNormalization(groups=groups, axis=-1)
        self.conv1     = layers.Conv2D(out_channels, 3, padding='same')
        self.time_proj = layers.Dense(out_channels)
        self.norm2     = layers.GroupNormalization(groups=groups, axis=-1)
        self.conv2     = layers.Conv2D(out_channels, 3, padding='same')
        self.skip_conv = (layers.Conv2D(out_channels, 1, padding='same')
                          if in_channels != out_channels else None)

    def call(self, inputs, training=False):
        x, t_emb = inputs
        h = tf.nn.swish(self.norm1(x))
        h = self.conv1(h)
        # Inject the time embedding by broadcasting over spatial dims.
        h = h + self.time_proj(tf.nn.swish(t_emb))[:, None, None, :]
        h = tf.nn.swish(self.norm2(h))
        h = self.conv2(h)
        skip = self.skip_conv(x) if self.skip_conv is not None else x
        return skip + h


class AttentionBlock(layers.Layer):
    """
    Multi-head self-attention over spatial positions (used at the bottleneck).
    Spatial dims H×W are flattened into a sequence of H·W tokens; each token
    attends to every other token — useful for capturing long-range structure.
    """

    def __init__(self, channels, num_heads=8, groups=32, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.norm = layers.GroupNormalization(groups=groups, axis=-1)
        self.mha  = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=channels // num_heads)

    def call(self, x, training=False):
        b, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        c       = self.channels
        residual = x
        x = self.norm(x)
        x = tf.reshape(x, (b, h * w, c))
        x = self.mha(x, x, training=training)
        x = tf.reshape(x, (b, h, w, c))
        return x + residual


class Downsample(layers.Layer):
    """
    Strided Conv2D: halves the WIDTH only, keeps the HEIGHT.

    v3: in the channel-layout image (H=1, W=256, C=9) the height dimension is
    degenerate. Downsampling it with stride=2 would collapse it from 1 to 1
    (no-op) but also waste a convolutional axis. We use strides=(1, 2) so
    only the time axis (width) is downsampled — the natural choice for a
    multi-channel 1-D signal in a 2-D wrapper.

    Kernel stays 3×3 (cheap, lets the channel-mixing happen in the 1×1
    projections inside ResBlock). Strides 1×2: H stays, W halves.
    """

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(out_channels, 3, strides=(1, 2), padding='same')

    def call(self, x):
        return self.conv(x)


class Upsample(layers.Layer):
    """
    2× upsample on WIDTH only, followed by a Conv2D channel projection.

    Symmetric to Downsample for v3 — height stays at 1 throughout the
    encoder/decoder; only the time axis is up/down-sampled.
    """

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.up   = layers.UpSampling2D(size=(1, 2), interpolation='bilinear')
        self.conv = layers.Conv2D(out_channels, 3, padding='same')

    def call(self, x):
        return self.conv(self.up(x))


# ================================================================
# UNET  (5 stages, exact paper progression)
# ================================================================

class UNet(keras.Model):
    """
    UNet noise predictor ε_θ(x_t, t).

    Channel progression (paper exact, Sec. 4.1):
        128 → 128 → 256 → 256 → 512        (encoder)
        → 512 (bottleneck with self-attention) →
        512 → 256 → 256 → 128 → 128        (decoder, with channel-concat skips)

    Input:  [x_noisy (B, 16, 256, 1), t (B,)]
    Output: predicted noise (B, 16, 256, 1)
    """

    def __init__(self,
                 base_channels: int   = BASE_CHANNELS,
                 channel_mults: tuple = CHANNEL_MULTS,
                 time_emb_dim:  int   = TIME_EMB_DIM,
                 num_heads:     int   = NUM_HEADS,
                 **kwargs):
        super().__init__(**kwargs)
        # Channel counts at each stage. (1,1,2,2,4) × 128 = [128, 128, 256, 256, 512].
        ch = [base_channels * m for m in channel_mults]

        # Time embedding: sinusoidal → Dense → Dense (the standard recipe).
        self.time_embed = keras.Sequential([
            SinusoidalTimeEmbedding(base_channels),
            layers.Dense(time_emb_dim, activation='swish'),
            layers.Dense(time_emb_dim),
        ])

        # ── Encoder ──────────────────────────────────────────────
        # Initial channel lift: 1 → ch[0].
        self.in_conv  = layers.Conv2D(ch[0], 3, padding='same')

        # Stage 1: ch[0] → ch[0]
        self.enc_res1 = ResBlock(ch[0], ch[0], time_emb_dim)
        self.down1    = Downsample(ch[0])
        # Stage 2: ch[0] → ch[1]
        self.enc_res2 = ResBlock(ch[0], ch[1], time_emb_dim)
        self.down2    = Downsample(ch[1])
        # Stage 3: ch[1] → ch[2]
        self.enc_res3 = ResBlock(ch[1], ch[2], time_emb_dim)
        self.down3    = Downsample(ch[2])
        # Stage 4: ch[2] → ch[3]
        self.enc_res4 = ResBlock(ch[2], ch[3], time_emb_dim)
        self.down4    = Downsample(ch[3])

        # ── Bottleneck ───────────────────────────────────────────
        # Stage 5: ch[3] → ch[4]  (channels = 512)
        self.mid_res1 = ResBlock(ch[3], ch[4], time_emb_dim)
        self.mid_attn = AttentionBlock(ch[4], num_heads=num_heads)
        self.mid_res2 = ResBlock(ch[4], ch[4], time_emb_dim)

        # ── Decoder ──────────────────────────────────────────────
        # Each decoder stage: upsample → concat skip (doubles channels) → ResBlock.
        self.up4      = Upsample(ch[3])
        self.dec_res4 = ResBlock(ch[3] + ch[3], ch[3], time_emb_dim)

        self.up3      = Upsample(ch[2])
        self.dec_res3 = ResBlock(ch[2] + ch[2], ch[2], time_emb_dim)

        self.up2      = Upsample(ch[1])
        self.dec_res2 = ResBlock(ch[1] + ch[1], ch[1], time_emb_dim)

        self.up1      = Upsample(ch[0])
        self.dec_res1 = ResBlock(ch[0] + ch[0], ch[0], time_emb_dim)

        # ── Output ───────────────────────────────────────────────
        self.out_norm = layers.GroupNormalization(groups=32, axis=-1)
        self.out_conv = layers.Conv2D(N_CHANNELS, 3, padding='same')

    def call(self, inputs, training=False):
        x, t  = inputs
        t_emb = self.time_embed(t, training=training)

        # Encoder. Keep the skip activations (s1..s4) to concatenate later.
        x  = self.in_conv(x)
        s1 = self.enc_res1([x, t_emb], training=training);  x = self.down1(s1)
        s2 = self.enc_res2([x, t_emb], training=training);  x = self.down2(s2)
        s3 = self.enc_res3([x, t_emb], training=training);  x = self.down3(s3)
        s4 = self.enc_res4([x, t_emb], training=training);  x = self.down4(s4)

        # Bottleneck (with attention).
        x = self.mid_res1([x, t_emb], training=training)
        x = self.mid_attn(x, training=training)
        x = self.mid_res2([x, t_emb], training=training)

        # Decoder — concatenate skips before each ResBlock.
        x = self.up4(x);   x = tf.concat([x, s4], axis=-1)
        x = self.dec_res4([x, t_emb], training=training)
        x = self.up3(x);   x = tf.concat([x, s3], axis=-1)
        x = self.dec_res3([x, t_emb], training=training)
        x = self.up2(x);   x = tf.concat([x, s2], axis=-1)
        x = self.dec_res2([x, t_emb], training=training)
        x = self.up1(x);   x = tf.concat([x, s1], axis=-1)
        x = self.dec_res1([x, t_emb], training=training)

        x = tf.nn.swish(self.out_norm(x))
        return self.out_conv(x)


# ================================================================
# DDPM  (Ho et al. 2020 forward + reverse process)
# ================================================================

class GaussianDiffusion:
    """
    Forward (training):
      q(x_t | x_0) = N(x_t; √ᾱ_t · x_0,  (1-ᾱ_t) I)
      closed form: x_t = √ᾱ_t · x_0  +  √(1-ᾱ_t) · ε,   ε ~ N(0, I)

    Loss:
      L(θ) = E_{x_0, ε, t}[ ||ε - ε_θ(x_t, t)||² ]

    Reverse (sampling):
      p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t),  β_t I)
      μ_θ = (x_t  -  β_t/√(1-ᾱ_t) · ε_θ)  /  √α_t
    """

    def __init__(self, T: int = T_STEPS,
                 beta_start: float = 1e-4,
                 beta_end:   float = 0.02):
        self.T = T
        # Linear β schedule between beta_start and beta_end (Ho et al. default).
        betas      = np.linspace(beta_start, beta_end, T, dtype=np.float32)
        alphas     = 1.0 - betas
        alpha_bars = np.cumprod(alphas)

        # Pre-compute the things we need at training and sampling time.
        self.betas             = tf.constant(betas)
        self.alphas            = tf.constant(alphas)
        self.alpha_bars        = tf.constant(alpha_bars)
        self.sqrt_ab           = tf.constant(np.sqrt(alpha_bars))
        self.sqrt_one_minus_ab = tf.constant(np.sqrt(1.0 - alpha_bars))

    def q_sample(self, x0: tf.Tensor, t: tf.Tensor, noise: tf.Tensor = None):
        """Forward: corrupt x_0 to x_t via the closed-form formula."""
        if noise is None:
            noise = tf.random.normal(tf.shape(x0))
        s_ab   = tf.gather(self.sqrt_ab,           t)[:, None, None, None]
        s_1mab = tf.gather(self.sqrt_one_minus_ab, t)[:, None, None, None]
        return s_ab * x0 + s_1mab * noise, noise

    def p_sample(self, model, x_t, t_scalar):
        """Single reverse step: x_t → x_{t-1}."""
        batch    = tf.shape(x_t)[0]
        t_batch  = tf.fill([batch], t_scalar)
        eps_hat  = model([x_t, t_batch], training=False)

        beta_t   = self.betas[t_scalar]
        alpha_t  = self.alphas[t_scalar]
        s_1mab_t = self.sqrt_one_minus_ab[t_scalar]

        coeff = beta_t / s_1mab_t
        mean  = (x_t - coeff * eps_hat) / tf.sqrt(alpha_t)

        if t_scalar > 0:
            return mean + tf.sqrt(beta_t) * tf.random.normal(tf.shape(x_t))
        return mean

    def p_sample_loop(self, model, shape):
        """Full reverse chain from pure Gaussian noise to a data sample."""
        x = tf.random.normal(shape)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
            if t % 200 == 0:
                print(f'  reverse step {self.T - t:4d}/{self.T}', end='\r')
        print()
        return x


# ================================================================
# TRAINING
# ================================================================

def make_train_step(model, diffusion, optimizer):
    """Returns a @tf.function-compiled training step."""

    @tf.function
    def train_step(x0):
        batch = tf.shape(x0)[0]
        # Sample a random diffusion step t for each example in the batch.
        t     = tf.random.uniform([batch], 0, diffusion.T, dtype=tf.int32)
        # Sample noise and corrupt x0 to x_t.
        noise = tf.random.normal(tf.shape(x0))
        x_t, _ = diffusion.q_sample(x0, t, noise)

        with tf.GradientTape() as tape:
            pred = model([x_t, t], training=True)
            # The DDPM loss is just MSE between predicted and actual noise.
            loss = tf.reduce_mean(tf.square(noise - pred))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    return train_step


def train_diffusion(model, diffusion, dataset,
                     epochs:   int   = EPOCHS,
                     lr:       float = LR,
                     lr_min:   float = LR_MIN,
                     save_dir: str   = SAVE_DIR) -> tuple:
    """
    Train the UNet noise predictor and save checkpoints every 10 epochs.

    v2 changes:
      - Cosine LR schedule from `lr` down to `lr_min` over `epochs`.
      - EMA of weights, decay=EMA_DECAY. Generation always uses EMA weights.

    Returns: (epoch_losses, ema)  — pass `ema` to generate_samples().
    """
    os.makedirs(save_dir, exist_ok=True)

    # Cosine decay schedule. Counted in OPTIMISER STEPS, so we need to know
    # how many batches per epoch. The dataset is cached/finite so len(...) works.
    steps_per_epoch = sum(1 for _ in dataset)
    total_steps     = max(1, epochs * steps_per_epoch)
    lr_schedule     = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=total_steps,
        alpha=lr_min / lr,  # CosineDecay's `alpha` is the floor as a fraction of initial.
    )
    optimizer  = keras.optimizers.Adam(learning_rate=lr_schedule)
    train_step = make_train_step(model, diffusion, optimizer)
    losses     = []

    # EMA initialised AFTER the first forward pass (which has already happened
    # in __main__) so model.trainable_variables is populated.
    ema = EMA(model, decay=EMA_DECAY)

    for epoch in range(1, epochs + 1):
        batch_losses = []
        for batch in dataset:
            loss = float(train_step(batch))
            batch_losses.append(loss)
            ema.update(model)

        avg = float(np.mean(batch_losses))
        losses.append(avg)
        print(f'Epoch {epoch:4d}/{epochs}  loss={avg:.6f}')

        if epoch % 10 == 0:
            model.save_weights(
                os.path.join(save_dir, f'weights_ep{epoch:04d}.weights.h5'))

    model.save_weights(os.path.join(save_dir, 'weights_final.weights.h5'))
    return losses, ema


# ================================================================
# GENERATION
# ================================================================

def generate_samples(model, diffusion, n_samples, pt, ema=None):
    """
    Sample n_samples synthetic log-return sequences of length WINDOW_SIZE.

    v2: if `ema` is provided, sampling uses the EMA copy of the weights
    (standard DDPM practice — usually visibly better than the instantaneous
    training weights).
    """
    print(f'Generating {n_samples} samples ({diffusion.T} reverse steps)...')
    if ema is not None:
        ema.apply_to(model)
    try:
        images = diffusion.p_sample_loop(model, (n_samples, IMAGE_H, IMAGE_W, N_CHANNELS))
        images = images.numpy()
        return np.array([image_to_series(img, pt) for img in images])
    finally:
        if ema is not None:
            ema.restore(model)


# ================================================================
# SANITY TESTS
# ================================================================

def verify_wavelet_roundtrip():
    """Assert that Haar DWT → inverse Haar DWT is exact (tolerance < 1e-10)."""
    rng   = np.random.default_rng(42)
    x     = rng.standard_normal(WINDOW_SIZE)
    x_rec = haar_inverse(haar_forward(x))
    err   = float(np.max(np.abs(x - x_rec)))
    assert err < 1e-10, f'Wavelet roundtrip error too large: {err:.2e}'
    print(f'Wavelet roundtrip OK — max reconstruction error: {err:.2e}')


def verify_full_pipeline_roundtrip(log_returns: np.ndarray):
    """
    v2 sanity check — series → image → series must be lossless when no
    network is in the middle. If this errors out, training is pointless.

    Tolerance is generous (5e-3 on a single return) because the winsorisation
    step is intentionally not invertible: returns beyond ±10σ get clipped, and
    we don't unclip them on the way back. But anything inside the winsor band
    should round-trip exactly.
    """
    # Take a clean training window away from any winsor cap.
    window = log_returns[:WINDOW_SIZE].copy()
    pt     = PowerTransform().fit(log_returns)

    # Verify the windowed sample is well inside the winsor band so no clipping happens.
    lo = pt.mean - pt.winsor_z * pt.std
    hi = pt.mean + pt.winsor_z * pt.std
    if np.any((window < lo) | (window > hi)):
        # Replace any out-of-band returns with random in-band returns for the test.
        bad = (window < lo) | (window > hi)
        window[bad] = pt.mean

    img    = series_to_image(window, pt)
    # Build a single-image dataset to fit the image scale, then normalise.
    # image_to_series itself does the denormalise step internally, so we pass
    # the [-1, 1]-normalised image to mirror what comes out of the UNet.
    pt.fit_image_scale(img[np.newaxis, ...])
    img_norm   = pt.normalize_image(img)
    recovered  = image_to_series(img_norm, pt)

    err_max  = float(np.max(np.abs(window - recovered)))
    err_mean = float(np.mean(np.abs(window - recovered)))
    print(f'Full pipeline roundtrip — max err: {err_max:.2e}, mean err: {err_mean:.2e}')
    assert err_max < 5e-3, (
        f'Pipeline roundtrip error too large: {err_max:.2e}. '
        'Power transform or wavelet tile/untile is broken.')


# ================================================================
# EMA (Exponential Moving Average) of model weights
# ================================================================

class EMA:
    """
    Keep a slow-moving copy of the model weights for sample generation.

    Why: DDPM training is noisy and the instantaneous weights at any given
    step can produce visibly worse samples than a running average. The EMA
    weights typically generate noticeably better samples. This is standard
    in DDPM implementations (Ho et al. 2020, all HuggingFace tutorials).

    Usage:
      ema = EMA(model, decay=0.999)
      ... training loop, after each optimiser step: ema.update(model)
      ... at sample time:                            ema.apply_to(model)
                                                     ... generate ...
                                                     ema.restore(model)
    """

    def __init__(self, model, decay: float = EMA_DECAY):
        self.decay   = decay
        # Make a deep copy of the model variables at construction time.
        self.shadow  = [tf.Variable(v, trainable=False) for v in model.trainable_variables]
        self._backup = None

    def update(self, model):
        """Pull the model's weights one EMA step closer to its current value."""
        for s, v in zip(self.shadow, model.trainable_variables):
            s.assign(self.decay * s + (1.0 - self.decay) * v)

    def apply_to(self, model):
        """Swap the model's trainable weights for the EMA copy (saves originals)."""
        self._backup = [v.numpy() for v in model.trainable_variables]
        for v, s in zip(model.trainable_variables, self.shadow):
            v.assign(s)

    def restore(self, model):
        """Undo apply_to — put the live training weights back."""
        assert self._backup is not None, 'restore called without a prior apply_to'
        for v, b in zip(model.trainable_variables, self._backup):
            v.assign(b)
        self._backup = None


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print('=' * 64)
    print('Diffusion Model — paper-faithful training run')
    print(f'  Train: {dp.TRAIN_START} → {dp.TRAIN_END}')
    print(f'  Window {WINDOW_SIZE},  stride {STRIDE},  image {IMAGE_H}x{IMAGE_W}')
    print(f'  UNet channels: {[BASE_CHANNELS * m for m in CHANNEL_MULTS]}')
    print(f'  T={T_STEPS} diffusion steps, {EPOCHS} epochs, batch {BATCH_SIZE}')
    print('=' * 64)

    # 1. Wavelet sanity check.
    print('\n[1] Sanity-checking wavelet roundtrip ...')
    verify_wavelet_roundtrip()

    # 2. Load training log returns from the SHARED pipeline (matches MMD model).
    print(f'\n[2] Loading SPX training returns ...')
    train_log_returns, _ = dp.get_train_and_oos_log_returns()
    print(f'    {len(train_log_returns)} daily log returns, '
          f'mean={train_log_returns.mean():+.5f}, std={train_log_returns.std():.5f}')

    # 2b. v2: full pipeline round-trip on REAL data. If this fails, the
    # preprocessing/inverse chain is broken and training would be wasted.
    print('\n[2b] Sanity-checking full series→image→series roundtrip ...')
    verify_full_pipeline_roundtrip(train_log_returns)

    # 3. Fit PowerTransform on training returns and build the wavelet image dataset.
    print('\n[3] Building wavelet image dataset ...')
    pt     = PowerTransform().fit(train_log_returns)
    images = build_image_dataset(train_log_returns, pt)
    print(f'    {images.shape[0]} samples of shape {images.shape[1:]}  '
          f'[image range: {images.min():.2f}, {images.max():.2f}]')

    dataset = build_tf_dataset(images, batch_size=BATCH_SIZE)

    # 4. UNet + DDPM.
    print('\n[4] Building UNet + DDPM ...')
    model = UNet(base_channels=BASE_CHANNELS,
                  channel_mults=CHANNEL_MULTS,
                  time_emb_dim=TIME_EMB_DIM,
                  num_heads=NUM_HEADS)
    # Warm up the model with a single forward call so .count_params() works.
    _x = tf.zeros((1, IMAGE_H, IMAGE_W, N_CHANNELS))
    _t = tf.zeros((1,), dtype=tf.int32)
    _  = model([_x, _t], training=False)
    print(f'    UNet trainable parameters: {model.count_params():,}')

    diffusion = GaussianDiffusion(T=T_STEPS)

    # 5. Train.
    print(f'\n[5] Training for {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}→{LR_MIN} (cosine), EMA={EMA_DECAY} ...')
    epoch_losses, ema = train_diffusion(model, diffusion, dataset,
                                         epochs=EPOCHS, lr=LR, lr_min=LR_MIN,
                                         save_dir=SAVE_DIR)

    # 6. Generate and plot — using EMA weights.
    print('\n[6] Generating samples (EMA weights) ...')
    n_gen = 20
    gen_returns = generate_samples(model, diffusion, n_gen, pt, ema=ema)
    print(f'    Generated array shape: {gen_returns.shape}')
    print(f'    Generated mean={gen_returns.mean():+.5f}, std={gen_returns.std():.5f} '
          f'(real: mean={train_log_returns.mean():+.5f}, std={train_log_returns.std():.5f})')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(epoch_losses, lw=1.0)
    axes[0].set_yscale('log')
    axes[0].set_title('Training Loss (MSE)')
    axes[0].grid(True, alpha=0.4)

    axes[1].hist(train_log_returns, bins=100, alpha=0.55, density=True,
                  color='steelblue', label='Real SPX')
    axes[1].hist(gen_returns.flatten(), bins=100, alpha=0.55, density=True,
                  color='tomato',   label='Generated')
    axes[1].set_title('Log Return Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for i in range(min(5, n_gen)):
        axes[2].plot(np.cumsum(gen_returns[i]), alpha=0.75, lw=0.9)
    axes[2].set_title('Generated Log-Price Paths')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, 'results.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f'\nResults saved → {out_path}')
    plt.close(fig)
