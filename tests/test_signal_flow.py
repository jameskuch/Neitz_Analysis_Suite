"""Signal-flow engine — discrete DSP blocks, sinks, and the graph evaluator."""
import numpy as np
import pytest

from neitz import signal_flow as sf


# ---------------------------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------------------------
def test_events_and_series_signal():
    ev = sf.events_signal([[0.1, 0.2], [0.3]], t0=0.0, t1=0.5)
    assert ev.kind == "events" and ev.n_epochs == 2 and ev.fs is None and ev.t1 == 0.5
    s = sf.series_signal([[1, 2, 3], [4, 5, 6]], fs=100.0)
    assert s.kind == "series" and s.fs == 100.0 and s.n_epochs == 2


# ---------------------------------------------------------------------------------------------
# Bin
# ---------------------------------------------------------------------------------------------
def test_bin_spikes_counts_and_rate():
    ev = sf.events_signal([[0.05, 0.15, 0.25]], t0=0.0, t1=0.3)
    # 100 ms bins over [0,0.3] -> 3 bins, one spike each
    counts = sf.bin_spikes(ev, bin_ms=100, unit="count")
    assert counts.kind == "series"
    assert np.allclose(counts.epochs[0], [1, 1, 1])
    assert counts.fs == pytest.approx(10.0)          # 1 / 0.1 s
    rate = sf.bin_spikes(ev, bin_ms=100, unit="hz")
    assert np.allclose(rate.epochs[0], [10, 10, 10])  # count / 0.1 s


def test_bin_requires_events():
    s = sf.series_signal([[1, 2, 3]], fs=10)
    with pytest.raises(ValueError):
        sf.bin_spikes(s)


# ---------------------------------------------------------------------------------------------
# Resample / smooth / filter / detrend / window
# ---------------------------------------------------------------------------------------------
def test_resample_up_and_down_changes_fs():
    x = np.sin(2 * np.pi * 5 * np.arange(200) / 100.0)
    s = sf.series_signal([x], fs=100.0)
    up = sf.resample(s, mode="factor", factor=2.0)
    assert up.fs == pytest.approx(200.0)
    assert up.epochs[0].size == pytest.approx(400, abs=2)
    down = sf.resample(s, mode="rate", new_fs=50.0)
    assert down.fs == pytest.approx(50.0)
    assert down.epochs[0].size == pytest.approx(100, abs=2)


def test_smooth_and_filter_delegate_to_dsp():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    s = sf.series_signal([x], fs=200.0)
    sm = sf.smooth(s, method="moving", window=9)
    assert sm.epochs[0].var() < x.var()
    lp = sf.freq_filter(s, fmode="lowpass", cutoff_hz=20.0, taps=65)
    assert lp.epochs[0].shape == x.shape


def test_detrend_removes_mean_and_line():
    s = sf.series_signal([np.arange(100.0) + 5.0], fs=10)
    assert abs(sf.detrend(s, mode="mean").epochs[0].mean()) < 1e-9
    lin = sf.detrend(s, mode="linear").epochs[0]
    assert np.allclose(lin, 0.0, atol=1e-6)          # a pure ramp detrends to ~0


def test_window_tapers_edges():
    s = sf.series_signal([np.ones(64)], fs=64)
    w = sf.window(s, wtype="hann").epochs[0]
    assert w[0] < 0.05 and w[-1] < 0.05 and w[32] > 0.9


# ---------------------------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------------------------
def test_fft_power_peaks_at_tone():
    fs = 1000.0
    t = np.arange(1000) / fs
    s = sf.series_signal([np.sin(2 * np.pi * 10.0 * t)], fs=fs)
    spec = sf.fft_power(s, epoch_mode="overlay")
    assert spec["kind"] == "fft" and spec["traces"]
    f = np.array(spec["traces"][0]["x"]); p = np.array(spec["traces"][0]["y"])
    assert abs(f[np.argmax(p)] - 10.0) < 1.0         # peak at the 10 Hz tone


def test_fft_power_average_across_epochs():
    fs = 500.0
    t = np.arange(500) / fs
    s = sf.series_signal([np.sin(2 * np.pi * 8 * t), np.sin(2 * np.pi * 8 * t)], fs=fs)
    spec = sf.fft_power(s, epoch_mode="average")
    assert len(spec["traces"]) == 1 and spec["traces"][0]["name"] == "epoch mean"


def test_isi_hist_pools_intervals():
    ev = sf.events_signal([np.arange(0, 1.0, 0.1)], t0=0, t1=1.0)   # 10 Hz -> ISI ~100 ms
    h = sf.isi_hist(ev, bin_ms=5)
    assert h["kind"] == "hist" and h["traces"]
    x = np.array(h["traces"][0]["x"]); y = np.array(h["traces"][0]["y"])
    assert abs(x[np.argmax(y)] - 100.0) < 10.0       # mode near 100 ms


# ---------------------------------------------------------------------------------------------
# Evaluator — full pathway
# ---------------------------------------------------------------------------------------------
def _graph(nodes, conns):
    return {"nodes": [dict(id=i, type=t, params=p) for i, t, p in nodes],
            "connections": [dict(from_node=a, from_port="x", to_node=b, to_port="y")
                            for a, b in conns]}


def test_evaluate_spikes_bin_fft_pathway():
    # region(spikes) -> Bin -> FFT ; spikes a regular 10 Hz train
    g = _graph([("region0", "region", {}),
                ("bin0", "sf_bin", {"bin_ms": 1, "unit": "hz"}),
                ("fft0", "sf_fft", {"epoch_mode": "overlay"})],
               [("region0", "bin0"), ("bin0", "fft0")])
    spikes = [np.arange(0.1, 1.0001, 0.1)]           # 10 spikes, 10 Hz
    res = sf.evaluate(g, {"spikes": spikes, "t0": 0.0, "t1": 1.0})
    assert "fft0" in res and res["fft0"]["kind"] == "fft" and res["fft0"]["traces"]
    f = np.array(res["fft0"]["traces"][0]["x"]); p = np.array(res["fft0"]["traces"][0]["y"])
    pk = f[np.argmax(p)]
    assert abs(pk % 10.0) < 1.0 or abs(pk % 10.0 - 10.0) < 1.0   # peak on a 10 Hz harmonic


def test_evaluate_chain_smooth_filter_between():
    g = _graph([("region0", "region", {}),
                ("bin0", "sf_bin", {"bin_ms": 5}),
                ("sm0", "sf_smooth", {"window": 5}),
                ("f0", "sf_filter", {"fmode": "lowpass", "cutoff_hz": 20}),
                ("t0n", "sf_time", {})],
               [("region0", "bin0"), ("bin0", "sm0"), ("sm0", "f0"), ("f0", "t0n")])
    spikes = [np.sort(np.random.default_rng(1).uniform(0, 2, 40))]
    res = sf.evaluate(g, {"spikes": spikes, "t0": 0.0, "t1": 2.0})
    assert res["t0n"]["kind"] == "time" and res["t0n"]["traces"]


def test_evaluate_sink_without_source_is_error_spec_not_crash():
    g = _graph([("fft0", "sf_fft", {})], [])
    res = sf.evaluate(g, {"spikes": None})
    assert res["fft0"]["kind"] == "error" and res["fft0"]["traces"] == []


def test_evaluate_isi_needs_events_not_binned():
    # ISI wired AFTER a bin (series) → error spec, not a crash
    g = _graph([("region0", "region", {}), ("bin0", "sf_bin", {}), ("isi0", "sf_isi", {})],
               [("region0", "bin0"), ("bin0", "isi0")])
    res = sf.evaluate(g, {"spikes": [np.array([0.1, 0.2, 0.3])], "t0": 0, "t1": 0.4})
    assert res["isi0"]["kind"] == "error"


def test_evaluate_cycle_is_reported():
    g = _graph([("region0", "region", {}), ("a", "sf_smooth", {}), ("b", "sf_smooth", {})],
               [("region0", "a"), ("a", "b"), ("b", "a")])
    res = sf.evaluate(g, {"spikes": [np.array([0.1])], "t0": 0, "t1": 1})
    # both smooth nodes are non-sinks; add a sink downstream to observe the cycle error
    g["nodes"].append({"id": "t", "type": "sf_time", "params": {}})
    g["connections"].append({"from_node": "b", "from_port": "x", "to_node": "t", "to_port": "y"})
    res = sf.evaluate(g, {"spikes": [np.array([0.1, 0.5])], "t0": 0, "t1": 1})
    assert res["t"]["kind"] == "error"
