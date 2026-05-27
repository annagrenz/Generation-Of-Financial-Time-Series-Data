"""
1. Loads the SAME daily SPX data both models were trained on.
2. Rebuilds each model's architecture and loads the trained weights.
3. Generates a batch of synthetic paths from each model.
4. Computes stylised-fact statistics on:
     - REAL SPX windows (the "ground truth")
     - Diffusion-generated windows
     - MMD-Signature generated windows
5. Plots a 3×3 stylised-facts figure + a moments table.
6. Computes signature-kernel MMD² distances between real and each model,
   with a bootstrap confidence interval.
7. Re-runs everything on the OUT-OF-SAMPLE period (2018-09 to 2023-12) so we
   can see how well each model generalises to data it never saw.

- Everything reads from `data_pipeline.py` so the two models are evaluated
  against the SAME real windows — no data-window mismatches.
- The MMD model is given REAL HISTORICAL PRICES as a conditioning prefix
  (mirroring how it was trained). The diffusion model has no such concept
  and just samples from noise.
- Time vectors for MMD evaluation come from the actual SPX dates (calendar
  days / 365), so the signature kernel sees the same time scale that was
  used during MMD training.

"""

import os
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

import tensorflow as tf

import data_pipeline as dp
from mmd_model     import (GenLSTM, generate_paths,
                            SignatureKernel, get_static_kernel, mmd_loss,
                            NOISE_DIM, SEQ_DIM, HIDDEN_SIZE, N_LSTM_LAYERS,
                            SAMPLE_LEN, HIST_LEN, N_LEVELS, KERNEL_SIGMA,
                            STATIC_KERNEL_TYPE, MA_ORDER, SEED)
from diffusion_model import (PowerTransform, build_image_dataset,
                              UNet, GaussianDiffusion, generate_samples,
                              IMAGE_H, IMAGE_W,
                              BASE_CHANNELS, CHANNEL_MULTS, TIME_EMB_DIM, NUM_HEADS,
                              T_STEPS, WINDOW_SIZE, STRIDE as DIFF_STRIDE)
from mmd_noise     import build_ma_noise_sampler



# Model weights — paths to the .weights.h5 files saved during training.
DIFF_WEIGHTS = 'runs/diffusion_paper/weights_final.weights.h5'
MMD_WEIGHTS  = 'runs/mmd_paper/best_model.weights.h5'

# Evaluation settings
N_GEN        = 100      # synthetic samples per model
ACF_LAGS     = 30       # max lag for ACF plots
LEV_LAGS     = 20       # max lag for leverage-effect plot
MMD_EVAL_LEN = 100      # path length used in MMD² comparison
MMD_BATCH    = 64       # paths per bootstrap draw
MMD_REPS     = 20       # number of bootstrap repeats → mean ± std
USE_MA_NOISE = True     # use the MA(20) sampler at generation time too

OUT_DIR = 'runs/evaluation'


def _acf(x: np.ndarray, n_lags: int) -> np.ndarray:
    """
    Compute the autocorrelation of `x` at lags 0..n_lags.

    The autocorrelation at lag k tells us "how similar is x_t to x_{t+k}?":
      - acf[0] is always 1 (correlation with itself).
      - For log RETURNS, we expect acf ≈ 0 at all positive lags (no linear
        predictability — that's the efficient-market hypothesis baseline).
      - For ABSOLUTE / SQUARED returns, we expect slow positive decay
        (volatility clustering — big moves cluster in time).
    """
    x   = x - x.mean()
    var = float(np.dot(x, x))
    if var < 1e-15:
        return np.zeros(n_lags + 1)
    n = len(x)
    return np.array([float(np.dot(x[:n - k], x[k:])) / var
                     for k in range(n_lags + 1)])


def _leverage(r: np.ndarray, n_lags: int) -> np.ndarray:
    """
    Leverage effect: corr(r_t, |r_{t+k}|) for k = 1..n_lags.

    For real equity returns this is negative for small lags, a down day
    today tends to be followed by larger absolute moves tomorrow. Captures
    the empirical asymmetry that volatility spikes after losses, not gains.
    """
    rc    = r - r.mean()
    arc   = np.abs(r) - np.abs(r).mean()
    n     = len(rc)
    denom = np.std(r) * np.std(np.abs(r)) * n + 1e-15
    return np.array([float(np.dot(rc[:n - k], arc[k:])) / denom
                     for k in range(1, n_lags + 1)])


def _gain_loss_asymmetry(r: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    out = []
    for t in thresholds:
        extreme = np.abs(r) > t
        if extreme.sum() == 0:
            out.append(np.nan)
        else:
            # Fraction of extreme moves that are positive.
            out.append(float((r[extreme] > 0).mean()))
    return np.array(out)


def stylised_stats(ret_matrix: np.ndarray,
                   n_acf:      int = ACF_LAGS,
                   n_lev:      int = LEV_LAGS) -> dict:
    """
    Compute all stylised-fact statistics for a (N, T) log-return matrix.

    ACF/leverage are computed per-path and averaged — gives smoother
    estimates and works for any dataset size N.
    Returns are pooled across all paths for distributional statistics.
    """
    acf_r   = np.mean([_acf(r,         n_acf) for r in ret_matrix], axis=0)
    acf_abs = np.mean([_acf(np.abs(r), n_acf) for r in ret_matrix], axis=0)
    acf_sq  = np.mean([_acf(r ** 2,    n_acf) for r in ret_matrix], axis=0)
    lev     = np.mean([_leverage(r,    n_lev) for r in ret_matrix], axis=0)
    pooled  = ret_matrix.flatten()
    return dict(
        acf_r=acf_r, acf_abs=acf_abs, acf_sq=acf_sq, lev=lev,
        pooled=pooled,
        mean=float(pooled.mean()),
        std=float(pooled.std()),
        kurt=float(stats.kurtosis(pooled)),    # excess kurtosis (Gaussian = 0)
        skew=float(stats.skew(pooled)),
    )


def load_diffusion_model(weights_path: str):
    model = UNet(base_channels=BASE_CHANNELS,
                  channel_mults=CHANNEL_MULTS,
                  time_emb_dim=TIME_EMB_DIM,
                  num_heads=NUM_HEADS)
    
    _ = model([tf.zeros((1, IMAGE_H, IMAGE_W, 1)),
                tf.zeros((1,), dtype=tf.int32)], training=False)
    diffusion = GaussianDiffusion(T=T_STEPS)
    model.load_weights(weights_path)
    print(f'  Diffusion model loaded  ({model.count_params():,} params)  '
          f'← {weights_path}')
    return model, diffusion


def load_mmd_model(weights_path: str, sample_len: int = SAMPLE_LEN):
    
    gen = GenLSTM(noise_dim=NOISE_DIM, seq_dim=SEQ_DIM,
                   seq_len=sample_len, hidden_size=HIDDEN_SIZE,
                   n_lstm_layers=N_LSTM_LAYERS)
  
    _ = gen((tf.zeros((1, sample_len - 1, NOISE_DIM)),
              tf.zeros((1, sample_len, 1)),
              tf.zeros((1, HIST_LEN, SEQ_DIM))), training=False)
    gen.load_weights(weights_path)
    print(f'  MMD generator loaded    ({gen.count_params():,} params)  '
          f'← {weights_path}')
    return gen




def generate_diffusion_paths(model, diffusion, pt: PowerTransform,
                              n_samples: int) -> np.ndarray:
    
    print(f'\n  Generating {n_samples} diffusion samples '
          f'({diffusion.T} reverse steps each) ...')
    return generate_samples(model, diffusion, n_samples, pt)
    # Shape: (n_samples, WINDOW_SIZE=256) log returns.


def generate_mmd_paths(generator, real_windows: np.ndarray,
                       noise_sampler, n_samples: int) -> np.ndarray:
    
 
    rs   = np.random.RandomState(SEED)
    idx  = rs.choice(len(real_windows), n_samples, replace=True)
    wnd  = real_windows[idx]                            # (n_samples, SAMPLE_LEN, 2)

    
    hist_x = wnd[:, :HIST_LEN, 1:]                       # (n, HIST_LEN, 1)

   
    t_vec = wnd[:, :, 0].mean(axis=0).astype(np.float32) # (SAMPLE_LEN,)

    print(f'\n  Generating {n_samples} MMD-Signature samples ...')
    paths = generate_paths(generator, n_samples=n_samples,
                            sample_len=SAMPLE_LEN, hist_len=HIST_LEN,
                            noise_sampler=noise_sampler,
                            hist_x=hist_x.astype(np.float32),
                            t_vec=t_vec)
    return paths   # (n_samples, SAMPLE_LEN)




def plot_stylised_facts(stats_dict: dict, T_common: int, out_path: str,
                         title_suffix: str = ''):
   
    names  = list(stats_dict.keys())
    colors = ['steelblue', 'tomato', 'seagreen']
    data   = [stats_dict[n] for n in names]
    ci     = 1.96 / np.sqrt(T_common)              

    lags_a = np.arange(1, ACF_LAGS + 1)
    lags_l = np.arange(1, LEV_LAGS + 1)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f'Stylised facts — Real SPX vs. Generated{title_suffix}',
                  fontsize=13, y=1.01)

    
    ax = axes[0, 0]
    x_lim = 0.06
    for d, n, c in zip(data, names, colors):
        ax.hist(d['pooled'], bins=120, density=True, alpha=0.4, color=c, label=n)
   
    xs = np.linspace(-x_lim, x_lim, 400)
    r0 = data[0]['pooled']
    ax.plot(xs, stats.norm.pdf(xs, r0.mean(), r0.std()),
             'k--', lw=1.3, label='Normal fit')
    ax.set_xlim(-x_lim, x_lim)
    ax.set_title('Return distribution')
    ax.set_xlabel('Log return')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

   
    ax = axes[0, 1]
    ref_slope, ref_int = None, None
    for d, n, c in zip(data, names, colors):
        (osm, osr), (slope, intercept, _) = stats.probplot(d['pooled'])
        ax.scatter(osm, osr, s=1, alpha=0.15, color=c, label=n)
        if ref_slope is None:
            ref_slope, ref_int = slope, intercept
    xs_q = np.array([-5.0, 5.0])
    ax.plot(xs_q, ref_slope * xs_q + ref_int, 'k--', lw=1.2)
    ax.set_title('QQ-plot vs. Normal')
    ax.set_xlabel('Theoretical quantile')
    ax.set_ylabel('Sample quantile')
    ax.legend(fontsize=8, markerscale=6)
    ax.grid(True, alpha=0.3)

    
    ax = axes[0, 2]
    x_ = np.arange(len(names))
    w  = 0.35
    ax.bar(x_ - w/2, [d['kurt'] for d in data], w,
            color=colors, alpha=0.8, label='Excess kurtosis')
    ax.bar(x_ + w/2, [d['skew'] for d in data], w,
            color=colors, alpha=0.45, hatch='//', label='Skewness')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x_); ax.set_xticklabels(names, fontsize=9)
    ax.set_title('Excess kurtosis & skewness')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    
    ax = axes[1, 0]
    for d, n, c in zip(data, names, colors):
        ax.plot(lags_a, d['acf_r'][1:], color=c, label=n, lw=1.3, marker='o', ms=2)
    ax.axhline( ci, color='gray', lw=0.8, ls='--', label='95% CI')
    ax.axhline(-ci, color='gray', lw=0.8, ls='--')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('ACF of returns  (should ≈ 0)')
    ax.set_xlabel('Lag'); ax.set_ylabel('Autocorrelation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    
    ax = axes[1, 1]
    for d, n, c in zip(data, names, colors):
        ax.plot(lags_a, d['acf_abs'][1:], color=c, label=n, lw=1.3, marker='o', ms=2)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('ACF of |returns|  (volatility clustering)')
    ax.set_xlabel('Lag'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    for d, n, c in zip(data, names, colors):
        ax.plot(lags_a, d['acf_sq'][1:], color=c, label=n, lw=1.3, marker='o', ms=2)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('ACF of returns²  (squared clustering)')
    ax.set_xlabel('Lag'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    
    ax = axes[2, 0]
    for d, n, c in zip(data, names, colors):
        ax.plot(lags_l, d['lev'], color=c, label=n, lw=1.3, marker='o', ms=2)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('Leverage effect  corr(r_t, |r_{t+k}|)')
    ax.set_xlabel('Lag k'); ax.set_ylabel('Cross-correlation')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    
    ax = axes[2, 1]
    for d, n, c in zip(data, names, colors):
        r_s  = np.sort(np.abs(d['pooled']))
        ccdf = 1.0 - np.arange(1, len(r_s) + 1) / len(r_s)
        mask = (r_s > 1e-6) & (ccdf > 0)
        ax.loglog(r_s[mask], ccdf[mask], color=c, alpha=0.85, lw=1, label=n)
    ax.set_title('Tail log-log CCDF of |returns|')
    ax.set_xlabel('|Log return|'); ax.set_ylabel('P(|R| > x)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

    
    ax = axes[2, 2]
    ax.axis('off')
    col_labels = ['Model', 'Mean', 'Std', 'Skew', 'Excess Kurt']
    rows = []
    for d, n in zip(data, names):
        rows.append([n,
                      f'{d["mean"]:+.5f}',
                      f'{d["std"]:.5f}',
                      f'{d["skew"]:+.2f}',
                      f'{d["kurt"]:+.2f}'])
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                    cellLoc='center', loc='center',
                    bbox=[0.0, 0.25, 1.0, 0.65])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    ax.set_title('Summary statistics', pad=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'    → {out_path}')


def plot_gain_loss_and_endpoint(real_rets, diff_rets, mmd_rets,
                                  real_paths, diff_paths, mmd_paths,
                                  out_path: str, title_suffix: str = ''):
   
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'Gain/loss asymmetry & end-point distribution{title_suffix}',
                  fontsize=12, y=1.03)

    
    thresholds = np.linspace(0, np.percentile(np.abs(real_rets), 95), 30)

    
    real_glo = _gain_loss_asymmetry(real_rets, thresholds)
    diff_glo = _gain_loss_asymmetry(diff_rets, thresholds)
    mmd_glo  = _gain_loss_asymmetry(mmd_rets,  thresholds)

    axes[0].plot(thresholds, real_glo, 'steelblue', lw=1.5, label='Real SPX')
    axes[0].plot(thresholds, diff_glo, 'tomato',    lw=1.5, label='Diffusion')
    axes[0].plot(thresholds, mmd_glo,  'seagreen',  lw=1.5, label='MMD-Sig')
    axes[0].axhline(0.5, color='k', lw=0.8, ls='--', label='Symmetric (0.5)')
    axes[0].set_xlabel('Threshold on |return|')
    axes[0].set_ylabel('P(R > 0 | |R| > threshold)')
    axes[0].set_title('Gain/loss asymmetry')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    
    real_end = np.exp(real_paths[:, -1] - real_paths[:, 0])
    diff_end = np.exp(np.cumsum(diff_rets.reshape(-1, real_paths.shape[1]-1),
                                 axis=1)[:, -1])
    mmd_end  = np.exp(mmd_paths[:, -1] - mmd_paths[:, 0])

    axes[1].hist(real_end, bins=40, alpha=0.55, color='steelblue',
                  density=True, label='Real SPX')
    axes[1].hist(diff_end, bins=40, alpha=0.55, color='tomato',
                  density=True, label='Diffusion')
    axes[1].hist(mmd_end,  bins=40, alpha=0.55, color='seagreen',
                  density=True, label='MMD-Sig')
    axes[1].axvline(1.0, color='k', lw=0.8, ls='--', label='Flat (1.0)')
    axes[1].set_xlabel('End-point wealth multiple  exp(P_T - P_0)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('End-point distribution')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'    → {out_path}')




def cum_log_prices(returns: np.ndarray) -> np.ndarray:
    
    n, T = returns.shape
    cum  = np.cumsum(np.concatenate([np.zeros((n, 1)), returns], axis=1), axis=1)
    return cum                                          # (N, T+1)


def to_signature_paths(log_prices: np.ndarray,
                        eval_len:   int,
                        time_vec:   np.ndarray = None) -> np.ndarray:
    
    N   = log_prices.shape[0]
    T_u = min(log_prices.shape[1], eval_len)
    lp  = np.zeros((N, eval_len), dtype=np.float32)
    lp[:, :T_u] = log_prices[:, :T_u].astype(np.float32)
    if T_u < eval_len:
        lp[:, T_u:] = lp[:, T_u - 1:T_u]                # pad with last value

    if time_vec is None:
        
        time_vec = np.arange(eval_len, dtype=np.float32) / 252.0
    t_ax = np.broadcast_to(time_vec[None, :], (N, eval_len)).astype(np.float32)

    return np.stack([t_ax, lp], axis=-1)                # (N, eval_len, 2)


def mmd_bootstrap(X: np.ndarray, Y: np.ndarray, kernel: SignatureKernel,
                   batch: int, n_rep: int) -> tuple:
    
    rs   = np.random.RandomState(SEED)
    vals = []
    for _ in range(n_rep):
        ix = rs.choice(len(X), min(batch, len(X)), replace=False)
        iy = rs.choice(len(Y), min(batch, len(Y)), replace=True)
        v  = float(mmd_loss(tf.constant(X[ix], dtype=tf.float32),
                              tf.constant(Y[iy], dtype=tf.float32),
                              kernel))
        vals.append(v)
    return float(np.mean(vals)), float(np.std(vals))


def plot_mmd_bars(results: dict, out_path: str, title_suffix: str = ''):
    
    labels = list(results.keys())
    means  = [results[l][0] for l in labels]
    stds   = [results[l][1] for l in labels]
    colors = ['steelblue', 'tomato', 'seagreen', 'dimgray'][:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, means, color=colors, alpha=0.8,
                   yerr=stds, capsize=6,
                   error_kw=dict(elinewidth=1.5, ecolor='black'))
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_title(f'Signature-kernel MMD²{title_suffix}', fontsize=11)
    ax.set_ylabel('MMD²    ↓ lower = closer to real SPX')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate each bar with its value.
    offset = max(abs(m) for m in means) * 0.04 + max(stds) * 0.5
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                 (bar.get_height() if m >= 0 else 0) + s + offset,
                 f'{m:+.5f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'    → {out_path}')




def run_evaluation(label:    str,           # "TRAIN" or "OOS" — for filenames
                    real_windows: np.ndarray, # (N, SAMPLE_LEN, 2) [time, log_price]
                    diff_model, diff_diffusion, pt: PowerTransform,
                    mmd_gen, mmd_noise_sampler):
    
    out_subdir = os.path.join(OUT_DIR, label.lower())
    os.makedirs(out_subdir, exist_ok=True)

    title_suffix = f'  [{label}]'
    print(f'\n{"="*60}')
    print(f'  Evaluation pass: {label}')
    print(f'  {real_windows.shape[0]} real windows of length {real_windows.shape[1]}')
    print('=' * 60)

    
    diff_returns = generate_diffusion_paths(diff_model, diff_diffusion, pt, N_GEN)
    

    mmd_paths    = generate_mmd_paths(mmd_gen, real_windows, mmd_noise_sampler, N_GEN)
    
    mmd_returns  = np.diff(mmd_paths, axis=1)            

    real_returns = np.diff(real_windows[:, :, 1], axis=1)  

    T_sf = min(real_returns.shape[1], diff_returns.shape[1], mmd_returns.shape[1])
    print(f'\n[{label}] Stylised facts over T={T_sf} returns / window')

    stats_dict = {
        'Real SPX':  stylised_stats(real_returns[:, :T_sf]),
        'Diffusion': stylised_stats(diff_returns[:, :T_sf]),
        'MMD-Sig':   stylised_stats(mmd_returns[:, :T_sf]),
    }

    print(f'\n    {"Model":12s}  {"Mean":>10s}  {"Std":>10s}  '
          f'{"Skew":>10s}  {"Kurt":>10s}')
    print('    ' + '-' * 56)
    for name, s in stats_dict.items():
        print(f'    {name:12s}  {s["mean"]:>+10.5f}  {s["std"]:>10.5f}  '
              f'{s["skew"]:>+10.2f}  {s["kurt"]:>+10.2f}')

    plot_stylised_facts(stats_dict, T_common=T_sf,
                         out_path=os.path.join(out_subdir, 'stylised_facts.png'),
                         title_suffix=title_suffix)

    real_paths_eval = cum_log_prices(real_returns[:, :T_sf])
    diff_paths_eval = cum_log_prices(diff_returns[:, :T_sf])
    mmd_paths_eval  = cum_log_prices(mmd_returns[:, :T_sf])

    plot_gain_loss_and_endpoint(
        real_rets=real_returns[:, :T_sf].flatten(),
        diff_rets=diff_returns[:, :T_sf].flatten(),
        mmd_rets =mmd_returns[:, :T_sf].flatten(),
        real_paths=real_paths_eval, diff_paths=diff_paths_eval, mmd_paths=mmd_paths_eval,
        out_path=os.path.join(out_subdir, 'gain_loss_endpoint.png'),
        title_suffix=title_suffix)

    print(f'\n[{label}] Signature-kernel MMD²  '
          f'(m={N_LEVELS}, T={MMD_EVAL_LEN}, batch={MMD_BATCH}, '
          f'{MMD_REPS}× bootstrap)')

    kernel = SignatureKernel(
        n_levels=N_LEVELS,
        static_kernel=get_static_kernel(STATIC_KERNEL_TYPE, KERNEL_SIGMA))

    X_real = to_signature_paths(real_paths_eval, MMD_EVAL_LEN)
    X_diff = to_signature_paths(diff_paths_eval, MMD_EVAL_LEN)
    X_mmd  = to_signature_paths(mmd_paths_eval,  MMD_EVAL_LEN)

    comparisons = [
        ('Real vs Real',      X_real, X_real),        # bias baseline; should ≈ 0
        ('Real vs Diffusion', X_real, X_diff),
        ('Real vs MMD-Sig',   X_real, X_mmd),
    ]

    mmd_results = {}
    for name, A, B in comparisons:
        m, s = mmd_bootstrap(A, B, kernel, MMD_BATCH, MMD_REPS)
        mmd_results[name] = (m, s)
        print(f'    {name:25s}  {m:+.5f} ± {s:.5f}')

    plot_mmd_bars(mmd_results,
                   out_path=os.path.join(out_subdir, 'mmd_comparison.png'),
                   title_suffix=title_suffix)

    return stats_dict, mmd_results



if __name__ == '__main__':
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print('=' * 62)
    print('EVALUATION — Diffusion vs MMD-Signature on daily SPX')
    print(f'  Train range: {dp.TRAIN_START} → {dp.TRAIN_END}')
    print(f'  OOS range:   {dp.OOS_START} → {dp.OOS_END}')
    print('=' * 62)

    print('\n[1] Loading SPX windows from data_pipeline ...')
    train_windows, oos_windows = dp.get_train_and_oos_log_price_windows(
        sample_len=SAMPLE_LEN, stride=50)
    print(f'    train: {train_windows.shape}    oos: {oos_windows.shape}')

    print('\n[2] Refitting diffusion PowerTransform on training returns ...')
    train_log_returns, _ = dp.get_train_and_oos_log_returns()
    pt = PowerTransform().fit(train_log_returns)
    
    _ = build_image_dataset(train_log_returns, pt)
    print(f'    PowerTransform: μ={pt.mean:+.5f}, σ={pt.std:.5f}, '
          f'img range=[{pt.img_min:.2f}, {pt.img_max:.2f}]')

    print('\n[3] Loading trained models ...')
    diff_model, diff_diffusion = load_diffusion_model(DIFF_WEIGHTS)
    mmd_gen                    = load_mmd_model(MMD_WEIGHTS)

    if USE_MA_NOISE:
        print('\n[4] Rebuilding MA(20) noise sampler ...')
        df_train = dp.load_spx_prices(start_date=dp.TRAIN_START, end_date=dp.TRAIN_END)
        t_full   = dp.calendar_time_vector(df_train.index)
        dt_years = np.diff(t_full)
        mmd_sampler, _ = build_ma_noise_sampler(
            train_log_returns, dt_years,
            noise_dim=NOISE_DIM, p=MA_ORDER, seed=SEED)
        print(f'    MA({MA_ORDER}) sampler ready.')
    else:
        mmd_sampler = None

    train_stats, train_mmd = run_evaluation(
        'TRAIN', train_windows, diff_model, diff_diffusion, pt,
        mmd_gen, mmd_sampler)

    oos_stats, oos_mmd = run_evaluation(
        'OOS', oos_windows, diff_model, diff_diffusion, pt,
        mmd_gen, mmd_sampler)

    print('\n' + '=' * 62)
    print('FINAL SUMMARY')
    print(f'  Outputs in {OUT_DIR}/')
    print('\n  Signature kernel MMD²  (lower = better, in-sample):')
    for label, (m, s) in train_mmd.items():
        print(f'    {label:25s}  {m:+.5f} ± {s:.5f}')
    print('\n  Signature kernel MMD²  (lower = better, OUT-OF-SAMPLE):')
    for label, (m, s) in oos_mmd.items():
        print(f'    {label:25s}  {m:+.5f} ± {s:.5f}')
    print('=' * 62)
