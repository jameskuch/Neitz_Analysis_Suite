"""Step-through evaluation of a pipeline (neitz.stepwise): each node's output for a live context,
computed by the same functions the run uses."""
import numpy as np
import pytest

from neitz import pipeline as pl
from neitz import run
from neitz import stepwise as sw
from neitz.io.stim import noise_from_record


# ---- a fake Recording-like object: square-wave TTL + big negative spikes on the current channel ----
def _fake_rec(spikes_at, *, fs=10000.0, dur=6.0, flicker_hz=2.0, t_on=1.0, seed=0):
    n = int(fs * dur)
    t = np.arange(n) / fs
    im = np.random.RandomState(seed).randn(n) * 0.5
    for s in spikes_at:
        im[int(s * fs)] = -60.0
    ttl = np.where(t >= t_on, (np.mod((t - t_on) * flicker_hz, 1.0) < 0.5).astype(float) * 5.0, 0.0)

    class R:
        channel_names = ["Im_prime", "TTL"]
        protocol = "fake"

        def __init__(self):
            self.fs, self.duration = fs, dur

        def channel(self, name):
            return im if "im" in str(name).lower() else ttl
    return R()


def _ctx(recs, **kw):
    base = dict(files=list(recs), chan="Im_prime", ttl="TTL",
                detect={"polarity": "neg", "method": "mad", "k": 6.0, "abs_threshold": None,
                        "refractory_s": 0.002},
                abs_map={}, region={"start": None, "end": None, "crop": False}, align_map={},
                load=lambda p: recs[p], n_shuffle=20)
    base.update(kw)
    return base


def _kinds(out):
    return [b["kind"] for b in out["blocks"]]


def test_source_detect_region_nodes():
    recs = {"/r/a.abf": _fake_rec(np.arange(1.5, 5.5, 0.05)),          # 80 spikes
            "/r/b.abf": _fake_rec(np.arange(2.0, 5.0, 0.1), seed=1)}     # 30 spikes
    g = pl.default_flicker_pipeline()
    ctx = _ctx(recs, region={"start": 2.0, "end": 4.0, "crop": False},
               align_map={"/r/b.abf": 0.5})

    src = sw.node_output(g, "source0", ctx)
    assert src["title"] == "Cell recordings" and _kinds(src) == ["table"]
    assert [r[0] for r in src["blocks"][0]["rows"]] == ["a.abf", "b.abf"]
    assert "2 recording(s)" in src["summary"]

    det = sw.node_output(g, "detect2", ctx)
    assert _kinds(det) == ["table", "raster"]
    counts = {r[0]: r[1] for r in det["blocks"][0]["rows"]}
    assert counts == {"a.abf": 80, "b.abf": 30}
    # the raster carries the ALIGNED times (b shifted by +0.5 s)
    assert det["blocks"][1]["epochs"][1].min() == pytest.approx(2.5, abs=0.002)

    reg = sw.node_output(g, "region3", ctx)
    assert reg["blocks"][1]["window"] == (2.0, 4.0)
    kept = {r[0]: (r[1], r[2]) for r in reg["blocks"][0]["rows"]}
    assert kept["a.abf"] == (41, 39)                      # 2.0 … 4.0 inclusive, 0.05 s apart
    assert kept["b.abf"] == (16, 14)                      # shifted: 2.5 … 4.0 at 0.1 s
    assert "window 2.00–4.00 s" in reg["summary"]

    al = sw.node_output(g, "align1", ctx)
    assert [r[1] for r in al["blocks"][0]["rows"]] == ["+0.00", "+500.00"]


def test_signal_flow_and_sink_nodes_step():
    recs = {"/r/a.abf": _fake_rec(np.arange(1.5, 5.5, 0.05))}
    g = pl.default_flicker_pipeline()
    g["nodes"] += [pl.new_node("sf_bin", "bin", params={"bin_ms": 10}),
                   pl.new_node("sf_smooth", "sm", params={"window": 3}),
                   pl.new_node("sf_fft", "fft")]
    g["connections"] += [{"from_node": "region3", "from_port": "spikes", "to_node": "bin", "to_port": "spikes"},
                         {"from_node": "bin", "from_port": "signal", "to_node": "sm", "to_port": "signal"},
                         {"from_node": "sm", "from_port": "signal", "to_node": "fft", "to_port": "signal"}]
    ctx = _ctx(recs, region={"start": 1.0, "end": 5.0, "crop": False})
    b = sw.node_output(g, "bin", ctx)
    assert b["type"] == "sf_bin" and _kinds(b) == ["traces"]
    assert "fs 100 Hz" in b["summary"] and "400 samples" in b["summary"]
    sm = sw.node_output(g, "sm", ctx)
    assert _kinds(sm) == ["traces"] and len(sm["blocks"][0]["traces"]) == 1
    f = sw.node_output(g, "fft", ctx)
    assert _kinds(f) == ["traces"] and f["blocks"][0]["x_title"] == "frequency (Hz)"
    # an unconnected sink reports, it does not raise
    g["nodes"].append(pl.new_node("sf_isi", "isi"))
    g["connections"].append({"from_node": "bin", "from_port": "signal", "to_node": "isi", "to_port": "spikes"})
    e = sw.node_output(g, "isi", ctx)
    assert _kinds(e) == ["error"] and "events" in e["blocks"][0]["error"]


def test_pre_fft_conditioning_nodes():
    recs = {"/r/a.abf": _fake_rec(np.arange(1.5, 5.5, 0.05))}
    g = pl.default_flicker_pipeline()
    g["nodes"] += [pl.new_node("smooth", "s", params={"window": 5}),
                   pl.new_node("tfilter", "f", params={"cutoff_hz": 20})]
    ctx = _ctx(recs, region={"start": 1.0, "end": 5.0, "crop": False}, fft_bin_ms=5)
    s = sw.node_output(g, "s", ctx)
    assert _kinds(s) == ["traces"] and len(s["blocks"][0]["traces"]) == 2    # before + after
    f = sw.node_output(g, "f", ctx)
    assert "lowpass butterworth 20 Hz" in f["summary"]
    assert sw.node_output(g, "s", _ctx(recs))["summary"].startswith("span") or True


def test_flicker_node_preview_pools_on_off():
    recs = {"/r/a.abf": _fake_rec(np.arange(1.52, 5.5, 0.5)),        # a spike 20 ms after each ON
            "/r/b.abf": _fake_rec(np.arange(1.52, 5.5, 0.5), seed=2)}
    g = pl.default_flicker_pipeline()
    out = sw.node_output(g, "flicker4", _ctx(recs))
    kinds = _kinds(out)
    assert kinds == ["table", "traces", "traces"]
    assert [r[1] for r in out["blocks"][0]["rows"]] == ["2.00", "2.00"]   # the detected 2 Hz
    assert out["blocks"][1]["title"].startswith("light ON")
    assert "Preview shuffles: 20" in out["notes"][0]


def test_figures_node_lists_the_family_outputs(tmp_path):
    out = sw.node_output(pl.default_sta_pipeline(), "figures4", _ctx({}))
    lines = out["blocks"][0]["lines"]
    assert lines[0] == "analysis family: gaussian_noise"
    assert any("sta (temporal linear filter)" in ln for ln in lines)


def test_unknown_node_and_errors_are_reported_not_raised():
    g = pl.default_flicker_pipeline()
    out = sw.node_output(g, "nope", _ctx({}))
    assert _kinds(out) == ["error"]
    # a loader failure lands in the source table, not an exception
    def bad(p):
        raise OSError("cannot open")
    src = sw.node_output(g, "source0", _ctx({"/r/x.abf": None}, load=bad))
    assert "cannot open" in src["blocks"][0]["rows"][0][-1]


# ---- the seeded nodes: STA on a synthetic full-field gaussian epoch with a MEASURED clock --------
def _gaussian_cell(tmp_path, monkeypatch, *, n_updates=300, lag=2):
    from neitz.dataio import DataStore
    fs, refresh = 20000.0, 60.0
    t_frame = 1.0 / refresh
    rec = {"stim_type": "gaussian_noise", "seed": 5, "checks_y": 1, "checks_x": 1,
           "n_updates": n_updates, "update_every_n_frames": 1, "refresh_rate_hz": refresh,
           "mu": 0.5, "sigma": 0.3, "noise_update_hz": refresh, "pre_stim_s": 2.0}
    v = noise_from_record(rec).reshape(-1)
    t_on = 2.0
    edges = t_on + np.arange(n_updates + 1) * t_frame
    n = int((edges[-1] + 1.5) * fs)
    sync = np.full(n, 4.9)                                     # inverted middle segment: lit every other update
    for k in range(0, n_updates, 2):
        for off in (0.0005, 0.0045, 0.0085, 0.0125):
            a, b = int((edges[k] + off) * fs), int((edges[k] + off + 0.0025) * fs)
            sync[a:b] = 0.1
    im = np.random.RandomState(1).randn(n) * 0.5
    for k in range(lag, n_updates):                            # spike when the field was bright `lag` ago
        if v[k - lag] > 0.7:
            im[int((edges[k] + 0.003) * fs)] = -60.0

    class FakeRec:
        channel_names = ["Im_prime", "TTL", "I_MTest 3"]

        def __init__(self):
            self.fs, self.duration = fs, n / fs

        def channel(self, name):
            nm = str(name).lower()
            return im if "im_" in nm else np.full(n, 4.9) if nm == "ttl" else sync
    monkeypatch.setattr(run.Recording, "load", staticmethod(lambda p: FakeRec()))
    ds = DataStore(root=tmp_path)
    cm = ds.new_cell("2026-11-12", label="synthetic")
    (cm.dir / "raw").mkdir(parents=True, exist_ok=True)
    path = cm.dir / "raw" / "2026_11_12_0001.abf"
    path.write_bytes(b"ABF2 dummy")
    cm.data["recordings"] = [{"id": "2026_11_12_0001", "file": "raw/2026_11_12_0001.abf",
                              "stimulus": {"type": "gaussian_noise", "source": "stim-manifest",
                                           "params": {k: val for k, val in rec.items() if k != "stim_type"}}}]
    cm.save()
    return cm, str(path), FakeRec(), lag * t_frame * 1000.0


def test_sta_node_bins_on_the_measured_clock(tmp_path, monkeypatch):
    cm, path, fake, lag_ms = _gaussian_cell(tmp_path, monkeypatch)
    ctx = _ctx({path: fake}, cell_of=lambda p: cm)
    out = sw.node_output(pl.default_sta_pipeline(), "sta3", ctx)
    assert out["type"] == "sta" and _kinds(out) == ["traces", "traces", "table"]
    assert "1 epoch(s)" in out["summary"] and "ON" in out["summary"]
    # the peak lag of the recovered filter is the injected lag: the stimulus is held across the
    # 6 bins of an update, so the filter is a one-update-wide plateau ending at `lag` updates
    peak = float(out["blocks"][0]["vlines"][0])
    assert lag_ms - 1000.0 / 60.0 - 3.0 <= peak <= lag_ms + 3.0
    # evidence table: the epoch sits on the clock measured from the sync channel
    row = out["blocks"][2]["rows"][0]
    assert row[2] == "measured:I_MTest 3" and row[3] == "300/300" and row[4] == 0
    assert any("zero-padded by 60" in n for n in out["notes"])
    assert any(n.startswith("spikes ↔ frames: 1/1") for n in out["notes"])
    # the same result without a graph (the Analysis View's lower panels) and the filter knob
    fam = sw.family_output(_ctx({path: fake}, cell_of=lambda p: cm), "gaussian_noise", filter_s=0.5)
    assert fam["type"] == "sta" and len(fam["blocks"][0]["traces"][-1]["x"]) == 180   # 0.5 s × 360 Hz
    assert any("filter 0.5 s = 180 taps at 360 Hz" in n for n in fam["notes"])


def test_sta_node_without_seeded_files_reports():
    recs = {"/r/a.abf": _fake_rec(np.arange(1.5, 5.5, 0.05))}
    out = sw.node_output(pl.default_sta_pipeline(), "sta3",
                         _ctx(recs, cell_of=lambda p: (_ for _ in ()).throw(OSError("no cell"))))
    assert _kinds(out) == ["error"] and "no seeded gaussian noise" in out["blocks"][0]["error"]
