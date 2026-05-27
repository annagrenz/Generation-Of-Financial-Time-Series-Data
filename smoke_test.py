"""
Runs a short, cheap test of every component WITHOUT doing real training.
If all checks pass, you can confidently start a real training run.
If something fails, the error message tells you exactly where to look.

Each test is wrapped in try/except so one failure doesn't stop the others.

Run from MMD-Model_VSCode/:
    python smoke_test.py
"""

import os
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

import sys
import traceback
import numpy as np



def _run_test(name: str, fn):
    print(f'\n[TEST] {name}')
    try:
        fn()
        print(f'  ✓ PASSED')
        return True
    except Exception as e:
        print(f'  ✗ FAILED:  {type(e).__name__}: {e}')
        traceback.print_exc(limit=2)
        return False




def test_data_pipeline():
    import data_pipeline as dp

    # Total CSV
    df = dp.load_spx_prices()
    assert len(df) > 5000, f'Too few rows: {len(df)}'

    # Train / OOS split
    train_rets, oos_rets = dp.get_train_and_oos_log_returns()
    assert len(train_rets) > 5000, f'Too few train returns: {len(train_rets)}'
    assert len(oos_rets)   > 1000, f'Too few oos returns:   {len(oos_rets)}'

    # MMD-style windows
    train_w = dp.make_log_price_windows(sample_len=300, stride=50)
    assert train_w.shape[1:] == (300, 2), \
        f'Wrong window shape: {train_w.shape}'

    # Diffusion-style windows
    diff_w = dp.make_log_return_windows(sample_len=256, stride=20)
    assert diff_w.shape[1] == 256, f'Wrong return-window length: {diff_w.shape}'

    print(f'  → SPX rows: {len(df)}, train returns: {len(train_rets)}, '
          f'oos returns: {len(oos_rets)}')
    print(f'  → MMD windows: {train_w.shape},  diff windows: {diff_w.shape}')




def test_gaussianize():
    import data_pipeline as dp
    from mmd_noise import gaussianize_returns

    train_rets, _ = dp.get_train_and_oos_log_returns()
    df = dp.load_spx_prices(start_date=dp.TRAIN_START, end_date=dp.TRAIN_END)
    dt = np.diff(dp.calendar_time_vector(df.index))

    g, stats = gaussianize_returns(train_rets, dt)
    # After Gaussianisation, the result should be approximately mean 0, std 1.
    assert abs(g.mean()) < 0.1, f'Gaussianised mean too far from 0: {g.mean()}'
    assert 0.5 < g.std() < 2.0, f'Gaussianised std unreasonable: {g.std()}'
    print(f'  → input  kurtosis (excess) ≈ {((train_rets - train_rets.mean())**4).mean()/train_rets.var()**2 - 3:.2f}')
    print(f'  → output kurtosis (excess) ≈ {((g - g.mean())**4).mean()/g.var()**2 - 3:.2f}')
    print(f'  → fitted μ={stats["mean"]:+.5f}, σ={stats["std"]:.5f}')



def test_ma_noise():
    import data_pipeline as dp
    from mmd_noise import build_ma_noise_sampler

    train_rets, _ = dp.get_train_and_oos_log_returns()
    df = dp.load_spx_prices(start_date=dp.TRAIN_START, end_date=dp.TRAIN_END)
    dt = np.diff(dp.calendar_time_vector(df.index))

    sampler, _ = build_ma_noise_sampler(train_rets, dt,
                                         noise_dim=4, p=20, seed=42)

    # Sample a small batch.
    noise = sampler.sample(sample_len=300, batch_size=8)
    assert noise.shape == (8, 299, 4), f'Bad noise shape: {noise.shape}'

    # Sanity: noise should have approximately zero mean and finite variance.
    assert abs(noise.mean()) < 0.5, f'Noise mean too large: {noise.mean()}'
    assert 0.01 < noise.std() < 5.0, f'Noise std unreasonable: {noise.std()}'
    print(f'  → noise shape {noise.shape}, mean={noise.mean():+.4f}, std={noise.std():.4f}')




def test_mmd_forward():
    import tensorflow as tf
    from mmd_model import GenLSTM, NOISE_DIM, SEQ_DIM, HIDDEN_SIZE, SAMPLE_LEN, HIST_LEN

    gen = GenLSTM(noise_dim=NOISE_DIM, seq_dim=SEQ_DIM,
                   seq_len=SAMPLE_LEN, hidden_size=HIDDEN_SIZE)

    batch = 4
    noise  = tf.random.normal((batch, SAMPLE_LEN - 1, NOISE_DIM))
    t      = tf.constant(np.broadcast_to(
        (np.arange(SAMPLE_LEN, dtype=np.float32)/252)[None, :, None],
        (batch, SAMPLE_LEN, 1)).copy())
    hist_x = tf.zeros((batch, HIST_LEN, SEQ_DIM))

    out = gen((noise, t, hist_x), training=False)
    assert out.shape == (batch, SAMPLE_LEN, SEQ_DIM), \
        f'Bad generator output shape: {out.shape}'
    print(f'  → generator params: {gen.count_params():,}')
    print(f'  → output shape: {tuple(out.shape)}')




def test_signature_kernel():
    import tensorflow as tf
    from mmd_model import SignatureKernel, get_static_kernel, mmd_loss

    kernel = SignatureKernel(n_levels=3,
                              static_kernel=get_static_kernel('rq', 0.1))


    X = tf.random.normal((8, 20, 2))
    Y = tf.random.normal((8, 20, 2))

    K_XX = kernel(X, X)
    assert K_XX.shape == (8, 8), f'Wrong Gram shape: {K_XX.shape}'

    
    loss = float(mmd_loss(X, Y, kernel))
    assert np.isfinite(loss), f'MMD² is not finite: {loss}'
    print(f'  → MMD²(X, Y) for random paths = {loss:+.5f}')




def test_diffusion_forward():
    import tensorflow as tf
    from diffusion_model import (UNet, BASE_CHANNELS, CHANNEL_MULTS,
                                  TIME_EMB_DIM, NUM_HEADS,
                                  IMAGE_H, IMAGE_W, verify_wavelet_roundtrip)

 
    verify_wavelet_roundtrip()

   
    model = UNet(base_channels=BASE_CHANNELS,
                  channel_mults=CHANNEL_MULTS,
                  time_emb_dim=TIME_EMB_DIM,
                  num_heads=NUM_HEADS)
    x = tf.zeros((2, IMAGE_H, IMAGE_W, 1))
    t = tf.zeros((2,), dtype=tf.int32)
    out = model([x, t], training=False)
    assert out.shape == (2, IMAGE_H, IMAGE_W, 1), \
        f'Bad UNet output shape: {out.shape}'
    print(f'  → UNet params: {model.count_params():,}')
    print(f'  → output shape: {tuple(out.shape)}')



def test_conditioning_has_effect():
   
    import tensorflow as tf
    from mmd_model import GenLSTM, NOISE_DIM, SEQ_DIM, HIDDEN_SIZE, SAMPLE_LEN, HIST_LEN

    gen = GenLSTM(noise_dim=NOISE_DIM, seq_dim=SEQ_DIM,
                   seq_len=SAMPLE_LEN, hidden_size=HIDDEN_SIZE)

    batch  = 2
    noise  = tf.random.normal((batch, SAMPLE_LEN - 1, NOISE_DIM), seed=42)
    t      = tf.constant(np.broadcast_to(
        (np.arange(SAMPLE_LEN, dtype=np.float32)/252)[None, :, None],
        (batch, SAMPLE_LEN, 1)).copy())

    hist_rising  = tf.constant(np.linspace(0, 0.3, HIST_LEN)
                                .reshape(1, HIST_LEN, 1)
                                .repeat(batch, axis=0).astype(np.float32))
    hist_falling = tf.constant(np.linspace(0, -0.3, HIST_LEN)
                                .reshape(1, HIST_LEN, 1)
                                .repeat(batch, axis=0).astype(np.float32))

    out_rising  = gen((noise, t, hist_rising),  training=False).numpy()
    out_falling = gen((noise, t, hist_falling), training=False).numpy()

    
    diff = np.abs(out_rising[:, HIST_LEN:, :] - out_falling[:, HIST_LEN:, :]).mean()
    assert diff > 1e-5, \
        f'Conditioning had no effect on generated section (mean diff = {diff})'
    print(f'  → mean abs diff in generated section: {diff:.6f} (must be > 0)')




if __name__ == '__main__':
    print('=' * 60)
    print('Smoke tests — verify everything is wired correctly')
    print('=' * 60)

    tests = [
        ('Data pipeline loads SPX',           test_data_pipeline),
        ('Lambert W Gaussianisation works',   test_gaussianize),
        ('MA(20) noise sampler produces noise', test_ma_noise),
        ('MMD generator forward pass',        test_mmd_forward),
        ('Signature kernel + MMD² compute',   test_signature_kernel),
        ('Diffusion UNet forward + wavelet',  test_diffusion_forward),
        ('LSTM conditioning has an effect',   test_conditioning_has_effect),
    ]

    results = [(_run_test(name, fn), name) for name, fn in tests]
    passed  = sum(1 for ok, _ in results if ok)

    print('\n' + '=' * 60)
    print(f'  {passed}/{len(tests)} tests passed')
    if passed == len(tests):
        print('  ✓ All systems go — safe to start a real training run.')
    else:
        print('  ✗ Some checks failed. Fix the errors above before training.')
        for ok, name in results:
            mark = '✓' if ok else '✗'
            print(f'    {mark} {name}')
    print('=' * 60)

    sys.exit(0 if passed == len(tests) else 1)
