"""Headless run layer + Result — synthetic tests."""
import numpy as np
from neitz.run import Result, run_flicker


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
    def __init__(self, im, ttl, fs, dur):
        self._c = {"Im_prime": im, "TTL": ttl}
        self.fs = fs
        self.duration = dur
        self.channel_names = ["Im_prime", "TTL"]

    def channel(self, name):
        return self._c[name]


def test_result_save_load_and_csv(tmp_path):
    r = Result("flicker",
               summary=[{"file": "a", "vs": 0.5}, {"file": "b", "vs": 0.1}],
               tables={"pooled": [{"cell": "c", "p": 0.03}]},
               arrays={"psth": np.arange(5.0)},
               meta={"n_files": 2})
    r.save(tmp_path / "res")
    loaded = Result.load(tmp_path / "res")
    assert loaded.kind == "flicker"
    assert loaded.summary[1]["file"] == "b"
    assert loaded.tables["pooled"][0]["p"] == 0.03
    assert np.allclose(loaded.arrays["psth"], np.arange(5.0))

    csv = r.save_csv(tmp_path / "summary.csv")
    text = open(csv).read()
    assert text.splitlines()[0] == "file,vs"
    assert len(text.splitlines()) == 3


def test_run_flicker_synthetic():
    fs, dur = 20000, 40.0
    on = np.arange(10.0, 35.0, 0.5)
    recs = [(f"c{i}.abf", "cell", FakeRec(make_current(fs, dur, on + 0.05, seed=i),
                                          make_ttl(fs, dur), fs, dur)) for i in range(2)]
    res = run_flicker(recs, n_shuffle=100, seed=0)
    assert res.kind == "flicker"
    assert len(res.summary) == 2
    assert abs(res.summary[0]["flicker_hz"] - 2.0) < 0.1
    pooled = res.tables["pooled_onoff"]
    assert pooled[0]["n_trials"] == 2
    assert pooled[0]["on_p"] < 0.05            # locked at ON -> significant
