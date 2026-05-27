
import os
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers

import data_pipeline as dp
from mmd_noise import build_ma_noise_sampler, MANoiseSampler


SAMPLE_LEN          = 300
HIST_LEN            = 50          # k=50 conditioning steps
GEN_LEN             = SAMPLE_LEN - HIST_LEN   # 250 generated returns

# Sliding-window stride for building the training dataset (Sec. 5.1).
STRIDE              = 50

# Batch size (Sec. 5.1).
BATCH_SIZE          = 38

# Generator architecture (Sec. 4.1).
SEQ_DIM             = 1           # 1 asset (SPX index)
NOISE_DIM           = 4           # noise dimension d_z (Sec. 5.3)
HIDDEN_SIZE         = 64          # LSTM hidden units (Sec. 4.1)
N_LSTM_LAYERS       = 1           # paper uses a single LSTM layer

# Signature kernel + static kernel (Sec. 5.3).
STATIC_KERNEL_TYPE  = 'rq'        # rational quadratic
N_LEVELS            = 10          # truncation order m of signature kernel
KERNEL_SIGMA        = 0.1         # length scale l of RQ kernel
KERNEL_ALPHA        = 1.0         # shape parameter α of RQ kernel

# Lead-lag augmentation (Sec. 3.5).
LEAD_LAG            = True
LAGS                = [1]

# MA noise (Sec. 4.3, 5.3).
USE_MA_NOISE        = True        # Set False for i.i.d. Gaussian baseline.
MA_ORDER            = 20          # MA(p) order

# Training schedule (Sec. 5; defaults from run_training.py in reference repo).
EPOCHS              = 10000       # caps training; early stopping kicks in much sooner
START_LR            = 1e-3
LR_FACTOR           = 0.5
PATIENCE            = 100         # epochs without improvement → LR decay
EARLY_STOPPING      = PATIENCE * 3
NUM_LOSSES          = 20          # rolling-window length for "smoothed" loss

SEED                = 42

# Output directory for weights and plots.
SAVE_DIR            = 'runs/mmd_paper'


# These are the "base" kernels used inside the signature kernel. They each
# define how similar two single (time, log_price) points are.

def _squared_euclid_dist(X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
    """Squared Euclidean distance matrix: D[i,j] = ||X[i] - Y[j]||²."""
    X_sq = tf.reduce_sum(X ** 2, axis=-1)
    Y_sq = tf.reduce_sum(Y ** 2, axis=-1)
    return (X_sq[:, None] + Y_sq[None, :]
            - 2.0 * tf.matmul(X, Y, transpose_b=True))


class LinearKernel:
    """K(x, y) = <x, y>  — simplest possible kernel, the baseline."""
    def gram_matrix(self, X, Y):
        return tf.matmul(X, Y, transpose_b=True)


class RBFKernel:
    """K(x, y) = exp(-||x-y||² / σ²)  — Gaussian / squared-exp kernel."""
    def __init__(self, sigma=KERNEL_SIGMA):
        self.sigma = sigma

    def gram_matrix(self, X, Y):
        return tf.exp(-_squared_euclid_dist(X, Y) / self.sigma ** 2)


class RationalQuadraticKernel:
    def __init__(self, sigma=KERNEL_SIGMA, alpha=KERNEL_ALPHA):
        self.sigma = sigma
        self.alpha = alpha

    def gram_matrix(self, X, Y):
        scaled = _squared_euclid_dist(X, Y) / (2.0 * self.alpha * self.sigma ** 2)
        return tf.pow(1.0 + scaled, -self.alpha)


def get_static_kernel(kernel_type=STATIC_KERNEL_TYPE,
                      sigma=KERNEL_SIGMA,
                      alpha=KERNEL_ALPHA):
    if kernel_type == 'linear':
        return LinearKernel()
    if kernel_type == 'rbf':
        return RBFKernel(sigma)
    if kernel_type == 'rq':
        return RationalQuadraticKernel(sigma, alpha)
    raise ValueError(f'Unknown kernel type: {kernel_type}')



def multi_cumsum(M: tf.Tensor, axes=(1, 3)) -> tf.Tensor:

    ndim = len(M.shape)
    axes = [ndim + a if a < 0 else a for a in axes]

    # Slice off the LAST element along every relevant axis simultaneously.
    slices = tuple(slice(None, -1) if i in axes else slice(None)
                   for i in range(ndim))
    M = M[slices]

    # Standard cumulative sum along each axis.
    for ax in axes:
        M = tf.cumsum(M, axis=ax)

    # Pad with a single zero at the start of each relevant axis.
    paddings = [[1, 0] if i in axes else [0, 0] for i in range(ndim)]
    return tf.pad(M, paddings)


class SignatureKernel:

    def __init__(self, n_levels=N_LEVELS, static_kernel=None):
        self.n_levels      = n_levels
        self.static_kernel = static_kernel or LinearKernel()

    def __call__(self, X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:

        n_X = tf.shape(X)[0]
        T_X = tf.shape(X)[1]
        n_Y = tf.shape(Y)[0]
        T_Y = tf.shape(Y)[1]
        d   = X.shape[-1]

        # Static-kernel Gram matrix over all (path, time-step) pairs.
        # Flatten so the kernel sees only "vectors of features".
        X_flat = tf.reshape(X, (-1, d))
        Y_flat = tf.reshape(Y, (-1, d))
        G      = self.static_kernel.gram_matrix(X_flat, Y_flat)
        M      = tf.reshape(G, (n_X, T_X, n_Y, T_Y))

        M = M[:, 1:, :, :] - M[:, :-1, :, :]     # diff along X's time
        M = M[:, :, :, 1:] - M[:, :, :, :-1]     # diff along Y's time

        K = tf.ones((n_X, n_Y), dtype=M.dtype) + tf.reduce_sum(M, axis=(1, 3))
        R = tf.identity(M)
        for _ in range(1, self.n_levels):
            R = M * multi_cumsum(R, axes=(1, 3))   # Király-Oberhauser update
            K = K + tf.reduce_sum(R, axis=(1, 3))

        return K


def mmd_loss(X: tf.Tensor, Y: tf.Tensor, kernel: SignatureKernel) -> tf.Tensor:
    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)

    n = tf.cast(tf.shape(K_XX)[0], tf.float32)
    m = tf.cast(tf.shape(K_YY)[0], tf.float32)

    # Diagonal masks: zero out the i==j terms so we only average over i≠j.
    mask_XX = 1.0 - tf.eye(tf.shape(K_XX)[0])
    mask_YY = 1.0 - tf.eye(tf.shape(K_YY)[0])

    return (tf.reduce_sum(K_XX * mask_XX) / (n * (n - 1))
          + tf.reduce_sum(K_YY * mask_YY) / (m * (m - 1))
          - 2.0 * tf.reduce_sum(K_XY)     / (n * m))




def batch_lead_lag_transform(data: tf.Tensor,
                             t:    tf.Tensor,
                             lags: list = None) -> tf.Tensor:
    if lags is None:
        lags = LAGS
    if isinstance(lags, int):
        lags = [lags]

    B       = tf.shape(data)[0]
    T       = data.shape[1]
    D       = data.shape[2]
    max_lag = max(lags)

    # Pad the time axis with the last timestamp repeated max_lag times.
    t_ext = tf.concat(
        [t, tf.repeat(t[:, -1:, :], max_lag, axis=1)], axis=1)

    # Pad the original data the same way.
    data_tail   = tf.repeat(data[:, -1:, :], max_lag, axis=1)
    data_padded = tf.concat([data, data_tail], axis=1)

    cols = [data_padded]
    for lag in lags:
        # Pad with `lag` zeros at the front, then the data, then padding at the back.
        zeros = tf.zeros((B, lag, D), dtype=data.dtype)
        rem   = max_lag - lag
        if rem > 0:
            extra  = tf.repeat(data[:, -1:, :], rem, axis=1)
            lagged = tf.concat([zeros, data, extra], axis=1)
        else:
            lagged = tf.concat([zeros, data], axis=1)
        cols.append(lagged[:, :T + max_lag, :])

    all_data = tf.concat(cols, axis=-1)
    return tf.concat([t_ext, all_data], axis=-1)



class GenLSTM(keras.Model):

    def __init__(self,
                 noise_dim:     int = NOISE_DIM,
                 seq_dim:       int = SEQ_DIM,
                 seq_len:       int = SAMPLE_LEN,
                 hidden_size:   int = HIDDEN_SIZE,
                 n_lstm_layers: int = N_LSTM_LAYERS,
                 **kwargs):
        super().__init__(**kwargs)
        self.noise_dim     = noise_dim
        self.seq_dim       = seq_dim
        self.seq_len       = seq_len
        self.hidden_size   = hidden_size
        self.n_lstm_layers = n_lstm_layers
        input_size = seq_dim + noise_dim + 1

        self.lstm_cells = [
            layers.LSTMCell(hidden_size, name=f'lstm_cell_{i}')
            for i in range(n_lstm_layers)
        ]


        self.output_net = layers.Dense(seq_dim)

   
    def _init_states(self, batch_size):
        h = [tf.zeros((batch_size, self.hidden_size))
             for _ in range(self.n_lstm_layers)]
        c = [tf.zeros((batch_size, self.hidden_size))
             for _ in range(self.n_lstm_layers)]
        return h, c

    def _lstm_step(self, inp, h_states, c_states):
        x          = inp
        new_h, new_c = [], []
        for i, cell in enumerate(self.lstm_cells):
            out, [h_new, c_new] = cell(x, [h_states[i], c_states[i]])
            new_h.append(h_new)
            new_c.append(c_new)
            x = out      # output of layer i → input of layer i+1
        return x, new_h, new_c


    def call(self, inputs, training=False):
        if len(inputs) == 3:
            noise, t, hist_x = inputs
        else:
            noise, t = inputs
            hist_x = None

        batch_size = tf.shape(noise)[0]
        T          = self.seq_len

        dts = t[:, 1:, :] - t[:, :-1, :]

        h_states, c_states = self._init_states(batch_size)

        if hist_x is not None:
            hist_len = hist_x.shape[1]

            diff_hist = hist_x[:, 1:, :] - hist_x[:, :-1, :]

            last_out = None
            for i in range(hist_len - 1):
                step_in = tf.concat([diff_hist[:, i, :],
                                      noise[:, i, :],
                                      dts[:, i, :]], axis=-1)
                last_out, h_states, c_states = self._lstm_step(
                    step_in, h_states, c_states)

            n_consumed    = hist_len - 1       # noise indices consumed so far
            n_to_generate = T - hist_len       # remaining predictions we need

        else:
            zero_return = tf.zeros((batch_size, self.seq_dim))
            init_in = tf.concat([zero_return,
                                  noise[:, 0, :],
                                  dts[:, 0, :]], axis=-1)
            last_out, h_states, c_states = self._lstm_step(
                init_in, h_states, c_states)

            n_consumed    = 1
            n_to_generate = T - 1

        
        gen_returns = []
        for i in range(n_to_generate):
            x = self.output_net(last_out)            # (B, seq_dim)
            gen_returns.append(x[:, tf.newaxis, :])  # collect as (B, 1, seq_dim)

            ni = n_consumed + i                      # next noise index to use
            if ni < T - 1:                           # still noise available?
                step_in = tf.concat([x,
                                      noise[:, ni, :],
                                      dts[:, ni, :]], axis=-1)
                last_out, h_states, c_states = self._lstm_step(
                    step_in, h_states, c_states)

        # Stack into (B, n_to_generate, seq_dim).
        gen_seq = tf.concat(gen_returns, axis=1)

        
        if hist_x is not None:
            # The historical log-price prefix is real data, so we keep it
            # verbatim and append the cumulatively-summed generated returns.
            # First, the synthetic returns become log-price increments starting
            # from the last historical log price.
            last_hist_price = hist_x[:, -1:, :]                 # (B, 1, seq_dim)
            gen_cum         = tf.cumsum(gen_seq, axis=1)        # (B, n_gen, seq_dim)
            gen_prices      = last_hist_price + gen_cum         # offset to continue from history
            full_path       = tf.concat([hist_x, gen_prices], axis=1)
        else:
            # No conditioning: prepend a zero return so the cumsum starts at 0.
            init_zero = tf.zeros((batch_size, 1, self.seq_dim))
            full_ret  = tf.concat([init_zero, gen_seq], axis=1)
            full_path = tf.cumsum(full_ret, axis=1)

        return full_path



def build_tf_dataset(windows: np.ndarray,
                     batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    return (tf.data.Dataset
            .from_tensor_slices(windows)
            .shuffle(buffer_size=len(windows), seed=SEED)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))


class ReduceLROnPlateau:

    def __init__(self, optimizer, patience=PATIENCE, factor=LR_FACTOR, min_lr=1e-6):
        self.optimizer = optimizer
        self.patience  = patience
        self.factor    = factor
        self.min_lr    = min_lr
        self.best      = float('inf')
        self.wait      = 0

    def step(self, metric: float) -> bool:
        if metric < self.best:
            self.best = metric
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            cur    = float(self.optimizer.learning_rate)
            new_lr = max(cur * self.factor, self.min_lr)
            self.optimizer.learning_rate.assign(new_lr)
            self.wait = 0
            print(f'  LR → {new_lr:.2e}')
            return True
        return False



def _compute_mmd_for_batch(generator, kernel, X, noise,
                            hist_len:  int,
                            lead_lag:  bool,
                            lags:      list) -> tf.Tensor:
    # X is (B, T, 2) with [time, log_price] columns.
    t          = X[:, :, :1]          # time column   (B, T, 1)
    log_prices = X[:, :, 1:]          # log-price col (B, T, 1)

    # Slice out the history prefix used for LSTM conditioning.
    hist_x     = log_prices[:, :hist_len, :]    # (B, hist_len, 1)

    # Forward pass through the generator — returns the FULL log-price path
    # (history + generated) of shape (B, T, 1).
    generated_path = generator((noise, t, hist_x), training=True)

    # Re-attach the time column so the path is (B, T, 2) again.
    full_gen = tf.concat([t, generated_path], axis=-1)

    # For a FAIR MMD comparison, we drop the conditioning prefix from BOTH
    # the real and generated paths — otherwise the kernel would see the
    # same hist_x on both sides and trivially match.
    real_for_mmd = X[:, hist_len:, :]
    gen_for_mmd  = full_gen[:, hist_len:, :]

    if lead_lag:
        # Augment with lagged copies so the signature kernel can read
        # autocorrelation information.
        real_aug = batch_lead_lag_transform(real_for_mmd[:, :, 1:],
                                              real_for_mmd[:, :, :1], lags)
        gen_aug  = batch_lead_lag_transform(gen_for_mmd[:, :, 1:],
                                              gen_for_mmd[:, :, :1],  lags)
        return mmd_loss(real_aug, gen_aug, kernel)

    return mmd_loss(real_for_mmd, gen_for_mmd, kernel)


def train_mmd(generator: GenLSTM,
              kernel:    SignatureKernel,
              dataset:   tf.data.Dataset,
              noise_sampler:  MANoiseSampler,
              epochs:         int   = EPOCHS,
              start_lr:       float = START_LR,
              patience:       int   = PATIENCE,
              early_stopping: int   = EARLY_STOPPING,
              num_losses:     int   = NUM_LOSSES,
              sample_len:     int   = SAMPLE_LEN,
              hist_len:       int   = HIST_LEN,
              noise_dim:      int   = NOISE_DIM,
              batch_size:     int   = BATCH_SIZE,
              lead_lag:       bool  = LEAD_LAG,
              lags:           list  = None,
              use_ma_noise:   bool  = USE_MA_NOISE,
              save_dir:       str   = SAVE_DIR) -> list:
    if lags is None:
        lags = LAGS
    os.makedirs(save_dir, exist_ok=True)

    optimizer = keras.optimizers.Adam(learning_rate=start_lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=LR_FACTOR)

    def train_step(X, noise):
        with tf.GradientTape() as tape:
            loss = _compute_mmd_for_batch(generator, kernel, X, noise,
                                            hist_len, lead_lag, lags)
        grads = tape.gradient(loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        return loss

    last_k        = deque(maxlen=num_losses)
    best          = [float('inf'), 0]
    epoch_losses  = []
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        batch_losses = []
        for X in dataset:
            if use_ma_noise:
                noise_np = noise_sampler.sample(sample_len=sample_len,
                                                  batch_size=batch_size)
                noise = tf.constant(noise_np, dtype=tf.float32)
            else:
                noise = tf.random.normal((batch_size, sample_len - 1, noise_dim),
                                          seed=SEED)

            loss = train_step(X, noise)
            batch_losses.append(float(loss))

        avg   = float(np.mean(batch_losses))
        last_k.append(avg)
        avg_k = float(np.mean(last_k))
        epoch_losses.append(avg)

        lr_cur = float(optimizer.learning_rate)
        print(f'Epoch {epoch:5d}/{epochs}  mmd={avg:+.6f}  '
              f'avg{num_losses}={avg_k:+.6f}  lr={lr_cur:.2e}')

        # LR scheduler reads the smoothed (avg-over-last-k) metric.
        scheduler.step(avg_k)

        if avg_k < best[0]:
            best = [avg_k, epoch]
            no_improve = 0
            generator.save_weights(os.path.join(save_dir, 'best_model.weights.h5'))
        else:
            no_improve += 1
            if no_improve >= early_stopping:
                print(f'Early stopping at epoch {epoch}  '
                      f'(best avg_k={best[0]:+.6f} @ epoch {best[1]})')
                break

        if epoch % 50 == 0:
            generator.save_weights(
                os.path.join(save_dir, f'ckpt_ep{epoch:05d}.weights.h5'))

    generator.save_weights(os.path.join(save_dir, 'final_model.weights.h5'))
    return epoch_losses


def generate_paths(generator: GenLSTM,
                   n_samples:     int,
                   sample_len:    int = SAMPLE_LEN,
                   hist_len:      int = HIST_LEN,
                   noise_sampler: MANoiseSampler = None,
                   hist_x:        np.ndarray     = None,
                   t_vec:         np.ndarray     = None) -> np.ndarray:
   
    if t_vec is None:
        dt_per_step = 1.0 / 252.0
        t_vals = np.arange(sample_len, dtype=np.float32) * dt_per_step
    else:
        
        t_vals = np.asarray(t_vec, dtype=np.float32).reshape(sample_len)
    
    t = tf.constant(np.broadcast_to(t_vals[None, :, None],
                                     (n_samples, sample_len, 1)).copy())

    # Noise.
    if noise_sampler is not None:
        noise_np = noise_sampler.sample(sample_len=sample_len, batch_size=n_samples)
        noise = tf.constant(noise_np, dtype=tf.float32)
    else:
        noise = tf.random.normal((n_samples, sample_len - 1, generator.noise_dim),
                                  seed=SEED)

    # History.
    if hist_x is None:
        hist_x_tf = None
    else:
       
        assert hist_x.shape == (n_samples, hist_len, generator.seq_dim), \
            f'hist_x must be (n_samples, hist_len, seq_dim); got {hist_x.shape}'
        hist_x_tf = tf.constant(hist_x, dtype=tf.float32)

    paths = generator((noise, t, hist_x_tf), training=False).numpy()
    return paths[:, :, 0]   



if __name__ == '__main__':
    import time

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print('=' * 64)
    print('MMD-Signature Model — paper-faithful training run')
    print(f'  Train: {dp.TRAIN_START} → {dp.TRAIN_END}')
    print(f'  Sample length {SAMPLE_LEN}  '
          f'(hist={HIST_LEN}, generated={GEN_LEN})  stride={STRIDE}')
    print(f'  Batch {BATCH_SIZE}, hidden {HIDDEN_SIZE}, '
          f'noise dim {NOISE_DIM}, MA noise: {USE_MA_NOISE}')
    print('=' * 64)

    print('\n[1] Building training windows ...')
    train_windows = dp.make_log_price_windows(sample_len=SAMPLE_LEN, stride=STRIDE)
    print(f'    {train_windows.shape[0]} windows of shape {train_windows.shape[1:]}')

    dataset = build_tf_dataset(train_windows, batch_size=BATCH_SIZE)


    print('\n[2] Fitting Lambert W transform and MA(20) noise model ...')
    train_log_returns, _ = dp.get_train_and_oos_log_returns()
    # Build calendar-time deltas (dt in years) for the same training rows.
    df_train = dp.load_spx_prices(start_date=dp.TRAIN_START, end_date=dp.TRAIN_END)
    t_full   = dp.calendar_time_vector(df_train.index)
    dt_years = np.diff(t_full)              # one Δt per log return

    if USE_MA_NOISE:
        sampler, fit_stats = build_ma_noise_sampler(train_log_returns,
                                                     dt_years,
                                                     noise_dim=NOISE_DIM,
                                                     p=MA_ORDER,
                                                     seed=SEED)
        print(f'    MA({MA_ORDER}) model fitted on {len(train_log_returns)} '
              f'Gaussianised returns.')
    else:
        sampler, fit_stats = None, {}
        print(f'    Skipping MA noise — using i.i.d. Gaussian baseline.')

    # Generator + kernel 
    print('\n[3] Building generator and signature kernel ...')
    generator = GenLSTM(noise_dim=NOISE_DIM, seq_dim=SEQ_DIM,
                        seq_len=SAMPLE_LEN, hidden_size=HIDDEN_SIZE,
                        n_lstm_layers=N_LSTM_LAYERS)

    # Warm up the model weights with a dummy forward pass so .count_params() works.
    _noise = tf.zeros((1, SAMPLE_LEN - 1, NOISE_DIM))
    _t     = tf.zeros((1, SAMPLE_LEN, 1))
    _hist  = tf.zeros((1, HIST_LEN, SEQ_DIM))
    _ = generator((_noise, _t, _hist), training=False)
    print(f'    Generator parameters: {generator.count_params():,}')

    kernel = SignatureKernel(
        n_levels=N_LEVELS,
        static_kernel=get_static_kernel(STATIC_KERNEL_TYPE, KERNEL_SIGMA))

    # Train 
    print(f'\n[4] Training (max {EPOCHS} epochs, early stop after '
          f'{EARLY_STOPPING} stagnant epochs) ...')
    t0 = time.time()
    losses = train_mmd(generator, kernel, dataset, sampler,
                        sample_len=SAMPLE_LEN, hist_len=HIST_LEN,
                        batch_size=BATCH_SIZE)
    print(f'\nTotal training time: {(time.time() - t0)/60:.1f} min')

    # Save preprocessing parameters with the weights 
    # We need fit_stats at generation time to reproduce the MA noise.
    # `lambert` and `ma_res` are non-picklable inside fit_stats — for the
    # thesis run, just keeping the trained weights + the SEED is enough
    # to fully reproduce, since fit_stats is built deterministically from
    # the same train data and the same seed.
    print(f'\n[5] Weights saved under {SAVE_DIR}/')

    # Generate synthetic paths and plot 
    n_gen = 20
    print(f'\n[6] Generating {n_gen} synthetic paths ...')

    # Use the last `hist_len` real log prices as conditioning.
    last_window = train_windows[-1:, :HIST_LEN, 1:]      # (1, hist_len, 1)
    hist_x_gen  = np.repeat(last_window, n_gen, axis=0)  # (n_gen, hist_len, 1)
    gen_paths   = generate_paths(generator, n_gen,
                                  sample_len=SAMPLE_LEN,
                                  hist_len=HIST_LEN,
                                  noise_sampler=sampler,
                                  hist_x=hist_x_gen)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'MMD-Signature Model  | T={SAMPLE_LEN}, m={N_LEVELS}', y=1.02)

    axes[0].plot(losses, linewidth=1.0)
    axes[0].set_yscale('log')
    axes[0].set_title('MMD² training loss')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(True, alpha=0.4)

    real_log_returns = np.diff(train_windows[:, :, 1], axis=1).flatten()
    gen_log_returns  = np.diff(gen_paths, axis=1).flatten()
    axes[1].hist(real_log_returns, bins=80, alpha=0.55, density=True,
                  color='steelblue', label='Real SPX')
    axes[1].hist(gen_log_returns, bins=80, alpha=0.55, density=True,
                  color='tomato',   label='Generated')
    axes[1].set_title('Log return distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for i in range(min(10, n_gen)):
        axes[2].plot(gen_paths[i], alpha=0.7, lw=0.9)
    axes[2].set_title('Generated log-price paths')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, 'results.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f'    Plot saved → {out_path}')
    plt.close(fig)
