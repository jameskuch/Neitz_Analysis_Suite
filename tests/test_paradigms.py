"""Paradigm classes — synthetic regression tests."""
import numpy as np
from neitz.stimulus import NoiseParadigm, FlickerParadigm, FlickerResult


def make_ttl(fs=20000, dur=40.0, start=10.0, end=35.0, fhz=2.0, chz=120.0, lo=0.27, hi=0.33):
    t = np.arange(int(fs * dur)) / fs
    carrier = (np.mod(t * chz, 1.0) < 0.5).astype(float)
    phase = np.mod((t - start) * fhz, 1.0)
    white = (t >= start) & (t < end) & (phase < 0.5)
    ttl = np.full(t.shape, lo)
    ttl[white] = lo + (hi - lo) * carrier[white]
    return ttl


def make_current(fs=20000, dur=40.0, spike_times=(), amp=-100.0, noise=2.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, noise, int(fs * dur))
    for t in spike_times:
        x[int(t * fs)] += amp
    return x


class FakeRec:
    """Minimal Recording stand-in for paradigm tests (no .abf needed)."""
    def __init__(self, im, ttl, fs, dur):
        self._c = {"Im_prime": im, "TTL": ttl}
        self.fs = fs
        self.duration = dur
        self.channel_names = ["Im_prime", "TTL"]

    def channel(self, name):
        return self._c[name]


def test_noise_paradigm_recovers_delay():
    rng = np.random.default_rng(1)
    stimuli = [rng.standard_normal(3600) for _ in range(4)]
    responses = [np.roll(s, 80) for s in stimuli]
    out = NoiseParadigm().analyze(stimuli, responses)
    assert abs(int(np.argmax(np.abs(out["average"]))) - 80) <= 1
    assert out["tuning"].shape == out["freqs"].shape
    assert out["time_ms"].shape == (360,)


def test_flicker_paradigm_recording():
    fs, dur = 20000, 40.0
    on_times = np.arange(10.0, 35.0, 0.5)
    rec = FakeRec(make_current(fs, dur, spike_times=on_times + 0.05),
                  make_ttl(fs, dur), fs, dur)
    res = FlickerParadigm(k=6).analyze_recording(rec, name="synthetic")
    assert isinstance(res, FlickerResult)
    assert res.flicker is not None
    assert abs(res.freq - 2.0) < 0.1
    assert res.n_in_region > 0
    assert res.vector_strength > 0.5          # cycle-locked spikes


def test_flicker_paradigm_group_detects_locking():
    fs, dur = 20000, 40.0
    on_times = np.arange(10.0, 35.0, 0.5)
    recs = [FakeRec(make_current(fs, dur, spike_times=on_times + 0.05, seed=i),
                    make_ttl(fs, dur), fs, dur) for i in range(2)]
    grp = FlickerParadigm(k=6).analyze_group(recs, n_shuffle=100,
                                             rng=np.random.default_rng(0))
    assert grp["n_trials"] == 2
    assert grp["on"]["p"] < 0.05              # locked at ON onset
