"""1-D DSP (smoothing + temporal filtering) — synthetic regression tests."""
import numpy as np
import pytest

from neitz.analysis import dsp


# ---------------------------------------------------------------------------------------------
# smooth_1d
# ---------------------------------------------------------------------------------------------
def test_moving_matches_matlab_smooth_span5():
    # MATLAB smooth(1:5, 5) == [1 2 3 4 5] (linear ramp is unchanged by a symmetric mean,
    # and the shrinking end-windows preserve the endpoints exactly).
    y = np.arange(1, 6, dtype=float)
    out = dsp.smooth_1d(y, window=5, method="moving")
    assert np.allclose(out, y)


def test_moving_matlab_reference_values():
    # Hand-computed MATLAB smooth(y, 3) with the shrinking-end rule:
    #   yy[0]=y[0]; yy[i]=mean(y[i-1:i+2]); yy[-1]=y[-1]
    y = np.array([2.0, 4.0, 6.0, 8.0, 2.0])
    out = dsp.smooth_1d(y, window=3, method="moving")
    expect = np.array([2.0, (2 + 4 + 6) / 3, (4 + 6 + 8) / 3, (6 + 8 + 2) / 3, 2.0])
    assert np.allclose(out, expect)


def test_moving_even_span_reduced_to_odd():
    # span 4 -> 3 (MATLAB forces odd), so smooth(y,4) == smooth(y,3)
    y = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0])
    assert np.allclose(dsp.smooth_1d(y, 4, "moving"), dsp.smooth_1d(y, 3, "moving"))


def test_smooth_preserves_length_and_mean_interior():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200)
    for method in dsp.SMOOTH_METHODS:
        out = dsp.smooth_1d(y, window=7, method=method)
        assert out.shape == y.shape
        assert np.isfinite(out).all()


def test_smooth_reduces_variance_of_noise():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(500)
    out = dsp.smooth_1d(y, window=9, method="moving")
    assert out.var() < y.var()          # smoothing attenuates white noise


def test_smooth_noop_for_tiny_window():
    y = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    assert np.allclose(dsp.smooth_1d(y, window=1), y)
    assert np.allclose(dsp.smooth_1d(y, window=0), y)


def test_smooth_rejects_2d():
    with pytest.raises(ValueError):
        dsp.smooth_1d(np.ones((3, 3)), 3)


# ---------------------------------------------------------------------------------------------
# temporal_filter_1d
# ---------------------------------------------------------------------------------------------
def test_lowpass_passes_dc_kills_nyquist():
    fs = 1000.0
    t = np.arange(2000) / fs
    dc = np.ones_like(t) * 2.0
    hi = np.sin(2 * np.pi * 400.0 * t)          # 400 Hz, well above a 50 Hz cutoff
    y = dc + hi
    out = dsp.apply_temporal_filter(y, ftype="butterworth", mode="lowpass",
                                    cutoff_hz=50.0, fs=fs, order=4, taps=65)
    # DC preserved (unity gain), high frequency strongly attenuated
    assert abs(out.mean() - 2.0) < 0.05
    interior = slice(100, -100)
    assert np.std(out[interior] - 2.0) < 0.2 * np.std(hi[interior])


def test_highpass_removes_dc():
    fs = 1000.0
    t = np.arange(2000) / fs
    y = 5.0 + np.sin(2 * np.pi * 200.0 * t)     # DC offset + a 200 Hz tone
    out = dsp.apply_temporal_filter(y, ftype="butterworth", mode="highpass",
                                    cutoff_hz=50.0, fs=fs, order=4, taps=65)
    assert abs(out[100:-100].mean()) < 0.1      # DC (below cutoff) removed


def test_lowpass_unity_dc_gain_all_types():
    fs = 500.0
    for ft in dsp.FILTER_TYPES:
        k = dsp.temporal_kernel(ft, "lowpass", fc_cps=0.1, order=2, taps=33)
        assert abs(k.sum() - 1.0) < 1e-6, ft    # DC gain exactly 1


def test_highpass_zero_dc_gain_all_types():
    for ft in dsp.FILTER_TYPES:
        k = dsp.temporal_kernel(ft, "highpass", fc_cps=0.1, order=2, taps=33)
        assert abs(k.sum()) < 1e-6, ft          # DC blocked


def test_filter_preserves_length():
    y = np.random.default_rng(2).standard_normal(300)
    out = dsp.apply_temporal_filter(y, ftype="gaussian", mode="lowpass",
                                    cutoff_hz=30.0, fs=1000.0, taps=33)
    assert out.shape == y.shape


def test_design_bundle_shape_and_cutoff():
    d = dsp.temporal_filter_design("butterworth", "lowpass", cutoff_hz=20.0,
                                   fs=200.0, order=2, taps=33)
    assert len(d["kernel"]) == d["taps"] == 33
    assert len(d["freq_hz"]) == len(d["response"])
    assert d["nyquist_hz"] == 100.0
    assert abs(d["cutoff_hz"] - 20.0) < 1e-6
    # response is monotone-ish: near 1 at low f, small near Nyquist for a lowpass
    assert d["response"][0] > 0.9
    assert d["response"][-1] < 0.2


def test_filter_short_signal_is_noop():
    y = np.array([1.0, 2.0, 3.0])
    out = dsp.apply_temporal_filter(y, ftype="ideal", mode="lowpass",
                                    cutoff_hz=10.0, fs=100.0, taps=33)
    assert np.allclose(out, y)                   # signal shorter than the kernel -> unchanged


def test_filter_zero_fs_is_noop():
    y = np.arange(50.0)
    assert np.allclose(dsp.apply_temporal_filter(
        y, ftype="gaussian", mode="lowpass", cutoff_hz=10.0, fs=0.0), y)
