import numpy as np
import pytest

from src import data
from src.models import diffusion_wavelet as dw
from src.metrics.stylised import stylised_stats


@pytest.fixture(scope="module")
def fold4():
    r = data.log_returns(data.get_fold(0).train_prices()).values
    w = data.return_windows(r, 256, 20)
    codec = dw.WaveletImageCodec(1.5, 10.0, "minmax").fit(r, w)
    return r, w, codec


def test_haar_roundtrip():
    x = np.random.default_rng(0).standard_normal(256)
    np.testing.assert_allclose(dw.haar_inverse(dw.haar_forward(x)), x, atol=1e-12)


def test_coefficient_lengths():
    assert [len(c) for c in dw.haar_forward(np.zeros(256))] == dw.COEF_LENGTHS


def test_preprocessor_roundtrip_without_clipping():
    r = np.random.default_rng(1).standard_t(4, 5000) * 0.01
    pre = dw.Preprocessor(1.5, 1e9).fit(r)
    np.testing.assert_allclose(pre.inverse(pre.transform(r)), r, atol=1e-14)
    assert abs(pre.transform(r).std() - 1.0) < 1e-12          # paper: divide by sd after power


def test_image_columns_are_time_aligned():
    """An impulse at time j may only affect, on every row, the block that contains column j."""
    j = 77
    x = np.zeros(256)
    x[j] = 1.0
    img = dw.coeffs_to_image(dw.haar_forward(x))
    for k, n in enumerate(dw.COEF_LENGTHS):
        block = 256 // n
        nz = np.nonzero(np.abs(img[k]) > 1e-12)[0]
        if len(nz):
            assert nz.min() >= (j // block) * block and nz.max() < (j // block + 1) * block


def test_rows_are_blockwise_constant(fold4):
    _, w, codec = fold4
    img = codec.encode(w[:5])[..., 0]
    for k, n in enumerate(dw.COEF_LENGTHS):
        blocks = img[:, k].reshape(5, n, 256 // n)
        assert np.allclose(blocks, blocks[..., :1])


def test_full_encode_decode_roundtrip(fold4):
    r, w, codec = fold4
    img = codec.encode(w)
    assert img.shape == (len(w), 16, 256, 1)
    assert img.min() >= -1 - 1e-6 and img.max() <= 1 + 1e-6
    rec = codec.decode(img)
    assert np.max(np.abs(rec - w)) < 1e-6                       # float32 image storage


def test_oracle_stylised_facts_survive_codec(fold4):
    """Decoded real windows must give the same stylised facts as raw real windows.
    Non-overlapping windows avoid duplicated returns, whose copies would carry different
    float32 rounding and could switch sides of the VaR quantile."""
    r, _, codec = fold4
    w = data.return_windows(r, 256, 256)
    a, b = stylised_stats(w[:, :250]), stylised_stats(codec.decode(codec.encode(w))[:, :250])
    for key in ["std", "skew", "excess_kurtosis", "var_1", "es_1"]:
        assert abs(a[key] - b[key]) < 1e-4 * max(1.0, abs(a[key])), key
    for key in ["acf_r", "acf_abs", "acf_sq", "leverage"]:
        np.testing.assert_allclose(a[key], b[key], atol=1e-5)


def test_unet_parameter_count_matches_thesis():
    m = dw.build_unet({"base_channels": 128, "channel_mults": [1, 1, 2, 2, 4],
                       "time_emb_dim": 512, "num_heads": 8})
    assert m.count_params() == 22_228_865
