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


WINDOW_SIZE   = 256        # log-returns per sample. Must be a power of 2 (DWT requirement).
STRIDE        = 20         # sliding-window stride across the return series.
P_POWER       = 1.5        # power-transform exponent for log returns (paper, Sec. 4.1).
WINSOR_Z      = 10.0       # winsorisation clip level in σ.
N_DWT_LEVELS  = 8          # log2(256) = 8 → 9 coefficient sets.
IMAGE_H       = 16         # image height (9 DWT rows + 7 rows of zero padding).
IMAGE_W       = 256        # image width (= WINDOW_SIZE).

# Diffusion schedule
T_STEPS       = 1000       # diffusion-step count T.

BASE_CHANNELS = 128
CHANNEL_MULTS = (1, 1, 2, 2, 4)
TIME_EMB_DIM  = 512
NUM_HEADS     = 8

# Training
EPOCHS        = 100
BATCH_SIZE    = 32
LR            = 1e-4
SEED          = 42

# Output
SAVE_DIR      = 'runs/diffusion_paper'



class PowerTransform:

    def __init__(self, p: float = P_POWER, winsor_z: float = WINSOR_Z):
        self.p        = p
        self.winsor_z = winsor_z
        
        self.mean:    float = None
        self.std:     float = None
        
        self.img_min: float = None
        self.img_max: float = None

    def fit(self, x: np.ndarray) -> 'PowerTransform':
        
        self.mean = float(np.mean(x))
        self.std  = float(np.std(x))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        
        lo = self.mean - self.winsor_z * self.std
        hi = self.mean + self.winsor_z * self.std
        x  = np.clip(x, lo, hi)

        
        return np.sign(x) * np.abs(x) ** (1.0 / self.p)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
       
        return np.sign(y) * np.abs(y) ** self.p

    def fit_image_scale(self, images: np.ndarray) -> 'PowerTransform':
        
        self.img_min = float(images.min())
        self.img_max = float(images.max())
        return self

    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        
        r = self.img_max - self.img_min + 1e-8
        return 2.0 * (img - self.img_min) / r - 1.0

    def denormalize_image(self, img: np.ndarray) -> np.ndarray:
       
        r = self.img_max - self.img_min + 1e-8
        return (img + 1.0) / 2.0 * r + self.img_min




def haar_forward(x: np.ndarray) -> list:
    
    assert len(x) >= 2 and (len(x) & (len(x) - 1)) == 0, \
        'haar_forward requires input length to be a power of 2'

    details = []
    cur     = x.astype(np.float64)
    while len(cur) > 1:
        
        approx = (cur[0::2] + cur[1::2]) / np.sqrt(2)
        detail = (cur[0::2] - cur[1::2]) / np.sqrt(2)
        details.append(detail)
        cur = approx

    
    return [cur] + list(reversed(details))


def haar_inverse(coeffs: list) -> np.ndarray:
    
    cur = coeffs[0].astype(np.float64)
    for detail in coeffs[1:]:
        n         = len(detail)
        rec       = np.empty(2 * n)
        rec[0::2] = (cur + detail) / np.sqrt(2)
        rec[1::2] = (cur - detail) / np.sqrt(2)
        cur = rec
    return cur


def series_to_image(returns: np.ndarray, pt: PowerTransform) -> np.ndarray:
   
    transformed = pt.transform(returns)
    coeffs      = haar_forward(transformed)         

    rows = []
    for c in coeffs:
        repeats = IMAGE_W // len(c)
        row     = np.tile(c, repeats)[:IMAGE_W]
        rows.append(row)

    
    while len(rows) < IMAGE_H:
        rows.append(np.zeros(IMAGE_W, dtype=np.float64))

    return np.stack(rows, axis=0)


def image_to_series(image: np.ndarray, pt: PowerTransform) -> np.ndarray:
    
    if image.ndim == 3:
        image = image[:, :, 0]
    image = pt.denormalize_image(image)

    
    n_per_level = [1] + [2 ** i for i in range(N_DWT_LEVELS)]
    
    coeffs = [image[i, :n] for i, n in enumerate(n_per_level)]

    power_returns = haar_inverse(coeffs)
    return pt.inverse_transform(power_returns)




def build_image_dataset(log_returns: np.ndarray,
                         pt: PowerTransform,
                         window_size: int = WINDOW_SIZE,
                         stride:      int = STRIDE) -> np.ndarray:
   
    n   = len(log_returns)
    raw = []
    for start in range(0, n - window_size + 1, stride):
        window = log_returns[start:start + window_size]
        raw.append(series_to_image(window, pt))
    raw = np.stack(raw)                  

  
    pt.fit_image_scale(raw)
    normalised = pt.normalize_image(raw)

    # Add a channel dim (single channel) so the UNet sees a 4-D tensor.
    return normalised[..., np.newaxis].astype(np.float32)


def build_tf_dataset(images: np.ndarray, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    return (tf.data.Dataset
            .from_tensor_slices(images)
            .shuffle(buffer_size=len(images), seed=SEED)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))



class SinusoidalTimeEmbedding(layers.Layer):
   

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
    

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(out_channels, 3, strides=2, padding='same')

    def call(self, x):
        return self.conv(x)


class Upsample(layers.Layer):
    

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.up   = layers.UpSampling2D(size=2, interpolation='bilinear')
        self.conv = layers.Conv2D(out_channels, 3, padding='same')

    def call(self, x):
        return self.conv(self.up(x))




class UNet(keras.Model):
   

    def __init__(self,
                 base_channels: int   = BASE_CHANNELS,
                 channel_mults: tuple = CHANNEL_MULTS,
                 time_emb_dim:  int   = TIME_EMB_DIM,
                 num_heads:     int   = NUM_HEADS,
                 **kwargs):
        super().__init__(**kwargs)
        
        ch = [base_channels * m for m in channel_mults]

        
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
        self.out_conv = layers.Conv2D(1, 3, padding='same')

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


class GaussianDiffusion:
  

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
        
        if noise is None:
            noise = tf.random.normal(tf.shape(x0))
        s_ab   = tf.gather(self.sqrt_ab,           t)[:, None, None, None]
        s_1mab = tf.gather(self.sqrt_one_minus_ab, t)[:, None, None, None]
        return s_ab * x0 + s_1mab * noise, noise

    def p_sample(self, model, x_t, t_scalar):
        
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
        
        x = tf.random.normal(shape)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
            if t % 200 == 0:
                print(f'  reverse step {self.T - t:4d}/{self.T}', end='\r')
        print()
        return x




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
                     save_dir: str   = SAVE_DIR) -> list:
    
    os.makedirs(save_dir, exist_ok=True)
    optimizer  = keras.optimizers.Adam(learning_rate=lr)
    train_step = make_train_step(model, diffusion, optimizer)
    losses     = []

    for epoch in range(1, epochs + 1):
        batch_losses = [float(train_step(batch)) for batch in dataset]
        avg = float(np.mean(batch_losses))
        losses.append(avg)
        print(f'Epoch {epoch:4d}/{epochs}  loss={avg:.6f}')

        if epoch % 10 == 0:
            model.save_weights(
                os.path.join(save_dir, f'weights_ep{epoch:04d}.weights.h5'))

    model.save_weights(os.path.join(save_dir, 'weights_final.weights.h5'))
    return losses


# GENERATION


def generate_samples(model, diffusion, n_samples, pt):
    
    print(f'Generating {n_samples} samples ({diffusion.T} reverse steps)...')
    images  = diffusion.p_sample_loop(model, (n_samples, IMAGE_H, IMAGE_W, 1))
    images  = images.numpy()
    return np.array([image_to_series(img, pt) for img in images])



# SANITY TESTS


def verify_wavelet_roundtrip():
    
    rng   = np.random.default_rng(42)
    x     = rng.standard_normal(WINDOW_SIZE)
    x_rec = haar_inverse(haar_forward(x))
    err   = float(np.max(np.abs(x - x_rec)))
    assert err < 1e-10, f'Wavelet roundtrip error too large: {err:.2e}'
    print(f'Wavelet roundtrip OK — max reconstruction error: {err:.2e}')



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

    print('\n[1] Sanity-checking wavelet roundtrip ...')
    verify_wavelet_roundtrip()

    print(f'\n[2] Loading SPX training returns ...')
    train_log_returns, _ = dp.get_train_and_oos_log_returns()
    print(f'    {len(train_log_returns)} daily log returns, '
          f'mean={train_log_returns.mean():+.5f}, std={train_log_returns.std():.5f}')

    print('\n[3] Building wavelet image dataset ...')
    pt     = PowerTransform().fit(train_log_returns)
    images = build_image_dataset(train_log_returns, pt)
    print(f'    {images.shape[0]} samples of shape {images.shape[1:]}  '
          f'[image range: {images.min():.2f}, {images.max():.2f}]')

    dataset = build_tf_dataset(images, batch_size=BATCH_SIZE)

    print('\n[4] Building UNet + DDPM ...')
    model = UNet(base_channels=BASE_CHANNELS,
                  channel_mults=CHANNEL_MULTS,
                  time_emb_dim=TIME_EMB_DIM,
                  num_heads=NUM_HEADS)
    _x = tf.zeros((1, IMAGE_H, IMAGE_W, 1))
    _t = tf.zeros((1,), dtype=tf.int32)
    _  = model([_x, _t], training=False)
    print(f'    UNet trainable parameters: {model.count_params():,}')

    diffusion = GaussianDiffusion(T=T_STEPS)

    print(f'\n[5] Training for {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR} ...')
    epoch_losses = train_diffusion(model, diffusion, dataset,
                                    epochs=EPOCHS, lr=LR, save_dir=SAVE_DIR)

    print('\n[6] Generating samples ...')
    n_gen = 20
    gen_returns = generate_samples(model, diffusion, n_gen, pt)
    print(f'    Generated array shape: {gen_returns.shape}')

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
