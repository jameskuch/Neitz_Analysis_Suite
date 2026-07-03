"""Flicker analysis — synthetic regression tests."""
import numpy as np
from neitz.analysis import flicker


def make_ttl(fs=20000, dur=80.0, flick_start=15.0, flick_end=75.0,
             flick_hz=2.0, carrier_hz=120.0, lo=0.27, hi=0.33):
    """Square-wave flicker on a carrier, flat baseline outside the flicker region."""
    t = np.arange(int(fs * dur)) / fs
    carrier = (np.mod(t * carrier_hz, 1.0) < 0.5).astype(float)     # 120 Hz, 50% duty
    phase = np.mod((t - flick_start) * flick_hz, 1.0)              # 0..1 each cycle
    white = (t >= flick_start) & (t < flick_end) & (phase < 0.5)   # carrier on
    ttl = np.full(t.shape, lo)
    ttl[white] = lo + (hi - lo) * carrier[white]
    return ttl, fs


def test_detect_flicker_freq_and_region():
    ttl, fs = make_ttl()
    fl = flicker.detect_flicker(ttl, fs)
    assert fl is not None
    assert abs(fl.freq - 2.0) < 0.05
    assert 14.0 < fl.t0 < 16.0
    assert 74.0 < fl.t1 < 76.0
    assert len(fl.on_edges) >= 100
    # the region must extend a full period past the last ON transition (the final cycle is
    # included, not clipped). Before the fix t1 sat AT the last onset, so this gap was ~0.
    assert fl.t1 - fl.on_edges[-1] > 0.4


def test_detect_flicker_none_when_flat():
    flat = np.full(int(20000 * 5), 0.27)
    assert flicker.detect_flicker(flat, 20000) is None


def test_vector_strength_locked_vs_uniform():
    locked = np.arange(0.0, 10.0, 0.5)          # one spike per 2 Hz cycle
    vs, p = flicker.vector_strength(locked, t0=0.0, freq=2.0)
    assert vs > 0.99 and p < 1e-3
    rng = np.random.default_rng(0)
    rand = np.sort(rng.uniform(0, 10, 300))
    vs2, _ = flicker.vector_strength(rand, t0=0.0, freq=2.0)
    assert vs2 < 0.25


def test_triggered_psth_shape():
    spikes = np.arange(0.5, 30.0, 0.5)
    edges = np.arange(0.5, 30.0, 0.5)
    centers, rate = flicker.triggered_psth(spikes, edges, pre_s=0.1, post_s=0.4, bin_s=0.01)
    assert centers.shape == rate.shape
    assert np.isclose(centers[0], -0.1 + 0.005, atol=1e-6)


def test_shift_test_jitter_null_detects_locked_response():
    # The default 'jitter' null disperses time-locking, so it CAN detect a real
    # response on a periodic stimulus (unlike the old whole-train 'shift' rotation).
    on = np.arange(0.5, 30.0, 0.5)
    locked = dict(spikes=on + 0.05, on=on, off=on + 0.25, dur=30.0)   # 50 ms after each ON
    rng = np.random.default_rng(0)
    rnd = dict(spikes=np.sort(rng.uniform(0, 30, 200)), on=on, off=on + 0.25, dur=30.0)

    r_lock = flicker.shift_test([locked], "on", pre_s=0.1, post_s=0.4, bin_s=0.01,
                                n_shuffle=200, rng=np.random.default_rng(1))
    r_rand = flicker.shift_test([rnd], "on", pre_s=0.1, post_s=0.4, bin_s=0.01,
                                n_shuffle=200, rng=np.random.default_rng(2))

    assert r_lock["null"] == "jitter"
    assert r_lock["p"] < 0.05                     # locked response IS detected now
    assert 30 <= r_lock["peak_ms"] <= 70          # latency ~50 ms
    assert r_rand["p"] > 0.05                     # random firing is not significant


def test_shift_test_rotation_null_has_low_power():
    # Documents the old behavior: whole-train rotation cannot reach significance
    # for a perfectly periodic, fully locked train (p floored ~window/period).
    on = np.arange(0.5, 30.0, 0.5)
    locked = dict(spikes=on + 0.05, on=on, off=on + 0.25, dur=30.0)
    r = flicker.shift_test([locked], "on", pre_s=0.1, post_s=0.4, bin_s=0.01,
                           n_shuffle=200, rng=np.random.default_rng(1), null="shift")
    assert r["p"] > 0.05
