"""Spike detection — synthetic, deterministic regression tests."""
import numpy as np
from neitz.spikes import detect_spikes, SpikeTrain


def make_signal(fs=10000, dur=2.0, spike_times=(0.5, 1.0, 1.5), amp=-100.0,
                noise=2.0, seed=0):
    """Gaussian noise with sharp spikes added at the given times."""
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    x = rng.normal(0.0, noise, n)
    for t in spike_times:
        x[int(t * fs)] += amp
    return x, fs


def test_detects_count_and_times():
    x, fs = make_signal()
    st = detect_spikes(x, fs, polarity="neg", method="mad", k=6)
    assert len(st) == 3
    assert np.allclose(sorted(st.times), [0.5, 1.0, 1.5], atol=0.001)


def test_matlab_method_recovers_spikes_and_gates_noise():
    """Sara's spikeDetectorOnline: polarity-driven + max/3 threshold + 4σ noise gate."""
    x, fs = make_signal(amp=-120.0, noise=2.0)          # 3 clear (negative) spikes
    for pol in ("neg", "abs"):                          # both orient the downward spikes up
        times = sorted(detect_spikes(x, fs, method="matlab", polarity=pol).times)
        for true_t in (0.5, 1.0, 1.5):
            assert any(abs(t - true_t) < 0.002 for t in times), f"{pol}: missed spike at {true_t}"
    # an explicit threshold (well above the spikes) suppresses them — the abs boxes are wired in
    assert len(detect_spikes(x, fs, method="matlab", polarity="neg", abs_threshold=500)) == 0
    # pure noise → the 4σ noise gate must yield zero spikes
    rng = np.random.default_rng(1)
    noise_only = rng.normal(0.0, 2.0, int(fs * 2.0))
    assert len(detect_spikes(noise_only, fs, method="matlab")) == 0


def test_polarity():
    x, fs = make_signal(amp=+100.0)            # upward spikes
    assert len(detect_spikes(x, fs, polarity="pos", k=6)) == 3
    assert len(detect_spikes(x, fs, polarity="neg", k=6)) == 0
    assert len(detect_spikes(x, fs, polarity="abs", k=6)) == 3


def test_mad_floor_takes_max_of_threshold():
    # spikes ~30 pA, noise sigma ~2 -> k*MAD(k=6) ~12
    x, fs = make_signal(amp=-30.0)
    lo = detect_spikes(x, fs, polarity="neg", method="mad_floor", k=6, abs_threshold=20)
    assert np.isclose(lo.threshold, max(6 * lo.sigma, 20))   # 20 wins (>12)
    assert len(lo) == 3                                       # 30 > 20 -> kept
    hi = detect_spikes(x, fs, polarity="neg", method="mad_floor", k=6, abs_threshold=50)
    assert np.isclose(hi.threshold, 50)                       # floor wins
    assert len(hi) == 0                                       # 30 < 50 -> rejected


def test_refractory_merges_close_events():
    x, fs = make_signal(spike_times=(0.5, 0.5005))           # 0.5 ms apart
    assert len(detect_spikes(x, fs, polarity="neg", k=6, refractory_s=0.002)) == 1


def test_binary_train():
    x, fs = make_signal()
    st = detect_spikes(x, fs, polarity="neg", k=6)
    b = st.binary(out_rate=10000, duration=2.0)
    assert isinstance(st, SpikeTrain)
    assert b.dtype == np.int8
    assert int(b.sum()) == 3
