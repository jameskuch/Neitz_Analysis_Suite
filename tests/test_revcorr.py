"""Reverse correlation / linear filter — synthetic regression tests."""
import numpy as np
from neitz.analysis import revcorr


def test_reverse_correlation_recovers_delay():
    rng = np.random.default_rng(1)
    stim = rng.standard_normal(3600)
    resp = np.zeros(3600)
    resp[100:] = stim[:-100]                 # response = stimulus delayed by 100 bins
    f = revcorr.reverse_correlation(stim, resp, filter_len=360, zero_pad=60)
    assert f.shape == (360,)
    assert int(np.argmax(np.abs(f))) == 100


def test_normalize_max_std_none():
    f = np.array([0.0, -4.0, 2.0, 1.0])
    assert np.isclose(np.max(np.abs(revcorr.normalize_filter(f, "max"))), 1.0)
    assert np.isclose(np.std(revcorr.normalize_filter(f, "std"), ddof=1), 1.0)
    assert np.allclose(revcorr.normalize_filter(f, None), f)


def test_average_filter():
    rng = np.random.default_rng(2)
    stimuli = [rng.standard_normal(3600) for _ in range(5)]
    responses = [np.roll(s, 50) for s in stimuli]      # each delayed by 50
    out = revcorr.average_filter(stimuli, responses, filter_len=360,
                                 zero_pad=60, normalize="max")
    assert out["per_epoch"].shape == (5, 360)
    assert out["average"].shape == (360,)
    assert abs(int(np.argmax(np.abs(out["average"]))) - 50) <= 1
    assert np.isclose(np.max(np.abs(out["average"])), 1.0)


def test_temporal_tuning_axes():
    f = np.random.default_rng(3).standard_normal(360)
    freqs, amp = revcorr.temporal_tuning(f, bin_rate=360)
    assert len(freqs) == len(amp) == 181        # rfft of length-360 -> 181 bins
    assert np.isclose(freqs[-1], 180.0)         # Nyquist = bin_rate/2
