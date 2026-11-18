"""Analysis View: the lower panels follow the pipeline (`lower_family`) and a file's default region
comes from its stimulus (`stim_window`)."""
from pathlib import Path

import numpy as np
import pytest

import viewer
from neitz import pipeline as pl


def test_lower_family_prefers_the_chosen_pipeline_then_the_files(monkeypatch):
    monkeypatch.setattr(viewer.pipe, "load_pipeline",
                        lambda name: {"STA": pl.default_sta_pipeline(),
                                      "SQ": pl.default_flicker_pipeline()}.get(name))
    stims = {"/r/a.abf": {"type": "sq_wave"}, "/r/b.abf": {"type": "checkerboard"}}
    monkeypatch.setattr(viewer, "_recording_for_path",
                        lambda p: {"stimulus": stims[p]} if p in stims else None)
    assert viewer.lower_family("STA", ["/r/a.abf"]) == "gaussian_noise"
    assert viewer.lower_family("SQ", ["/r/b.abf"]) == "sq_wave"
    assert viewer.lower_family(None, ["/r/a.abf", "/r/b.abf"]) == "checkerboard"   # first seeded file
    assert viewer.lower_family(None, ["/r/a.abf"]) == "sq_wave"
    assert viewer.lower_family("missing", []) == "sq_wave"


class _Rec:
    protocol = "abf"
    channel_names = ["Im_prime", "TTL"]

    def __init__(self, ttl, fs=20000.0):
        self.fs, self._ttl = fs, ttl
        self.duration = len(ttl) / fs

    def channel(self, name):
        return self._ttl if "ttl" in name.lower() else np.zeros_like(self._ttl)


def _inverted_envelope(t_on, t_off, dur=10.0, fs=20000.0):
    """The rig's envelope segment as Clampex records it: dark ≈ +4.9 V, lit ≈ 0 with bit-plane pulses."""
    t = np.arange(int(dur * fs)) / fs
    sig = np.full(t.size, 4.9)
    lit = (t >= t_on) & (t < t_off) & (np.mod(t * 240.0, 1.0) < 0.3)
    sig[lit] = 0.05
    return sig


def test_stim_window_measures_the_envelope_for_seeded_noise(monkeypatch):
    path = "/store/2026-11-12/c04/raw/n.abf"
    rec = _Rec(_inverted_envelope(2.69, 8.0))
    viewer._CACHE[path] = dict(rec=rec, chans={}, flicker={})
    monkeypatch.setattr(viewer, "_recording_for_path",
                        lambda p: {"stimulus": {"type": "gaussian_noise",
                                                "params": {"pre_stim_s": 2.0, "post_stim_s": 1.5}}})
    t0, t1, how = viewer.stim_window(path, "TTL")
    assert (t0, t1, how) == (2.69, 8.0, "sync-bar envelope")
    assert viewer.stim_window(path, "TTL") is viewer._CACHE[path]["window"]["TTL"]   # cached
    viewer._CACHE.pop(path, None)


def test_stim_window_keeps_the_flicker_region_for_square_waves(monkeypatch):
    path = "/store/2026-11-12/c02/raw/s.abf"
    fs = 20000.0
    t = np.arange(int(10.0 * fs)) / fs
    ttl = np.where(t >= 1.0, (np.mod((t - 1.0) * 2.0, 1.0) < 0.5).astype(float) * 5.0, 0.0)
    viewer._CACHE[path] = dict(rec=_Rec(ttl), chans={}, flicker={})
    monkeypatch.setattr(viewer, "_recording_for_path", lambda p: {"stimulus": {"type": "sq_wave"}})
    t0, t1, how = viewer.stim_window(path, "TTL")
    assert how == "flicker steady region" and abs(t0 - 1.0) < 0.02 and t1 > 9.0
    # no stimulus record at all (a file outside the store) → the flicker path too, else the whole file
    viewer._CACHE[path]["window"].clear()
    monkeypatch.setattr(viewer, "_recording_for_path", lambda p: None)
    assert viewer.stim_window(path, "TTL")[2] == "flicker steady region"
    flat = "/store/x/flat.abf"
    viewer._CACHE[flat] = dict(rec=_Rec(np.zeros(int(3 * fs))), chans={}, flicker={})
    assert viewer.stim_window(flat, "TTL") == (0.0, 3.0, "whole recording")
    viewer._CACHE.pop(path, None); viewer._CACHE.pop(flat, None)


def test_block_figure_renders_every_kind():
    tr = {"kind": "traces", "title": "t", "x_title": "x", "y_title": "y",
          "traces": [{"x": [0, 1], "y": [0, 1], "name": "a"}], "vlines": [0.5], "xrange": [0, 1]}
    hm = {"kind": "heatmap", "title": "h", "z": np.arange(6.0).reshape(2, 3) - 2, "mark": (1, 2)}
    ra = {"kind": "raster", "epochs": [np.array([0.1, 0.2]), np.array([0.3])], "names": ["a", "b"],
          "window": (0.0, 0.25)}
    ba = {"kind": "bars", "title": "b", "traces": [{"x": [1, 2], "y": [3, 4], "name": "n"}]}
    for blk in (tr, hm, ra, ba):
        fig = viewer._block_figure(blk, dark=True, height=200)
        assert fig.layout.height == 200 and len(fig.data) >= 1
    assert viewer._block_figure(hm).layout.yaxis.autorange == "reversed"
    assert viewer._block_figure(tr, title="override").layout.title.text == "override"


def test_reveal_in_file_manager_per_platform(monkeypatch):
    calls = []
    monkeypatch.setattr(viewer.subprocess, "run", lambda cmd, *a, **k: calls.append(list(cmd)))
    monkeypatch.setattr(viewer.platform, "system", lambda: "Darwin")
    assert viewer.reveal_in_file_manager(["/s/p/a.py", "/s/p/b.py"]) == "✓ revealed 2 item(s) in Finder"
    assert calls[-1] == ["open", "-R", "/s/p/a.py", "/s/p/b.py"]
    monkeypatch.setattr(viewer.platform, "system", lambda: "Windows")
    assert viewer.reveal_in_file_manager("/s/p/a.py") == "✓ revealed in Explorer"
    assert calls[-1] == ["explorer", "/select,", "/s/p/a.py"]
    assert viewer.reveal_in_file_manager(["/s/p/a.py", "/s/q/b.py"]) == "✓ opened 2 folder(s) in Explorer"
    assert calls[-2:] == [["explorer", "/s/p"], ["explorer", "/s/q"]]
    monkeypatch.setattr(viewer.platform, "system", lambda: "Linux")
    assert viewer.reveal_in_file_manager(["/s/p/a.py"]) == "✓ opened 1 folder(s)"
    assert calls[-1] == ["xdg-open", "/s/p"]
    assert viewer.reveal_in_file_manager([]) == "nothing to reveal"
    def boom(cmd, *a, **k):
        raise OSError("no gui")
    monkeypatch.setattr(viewer.subprocess, "run", boom)
    assert viewer.reveal_in_file_manager("/s/p/a.py").startswith("open failed")
    # the saved-listing path follows the pipeline name
    assert viewer._pipeline_code_path({"name": "Gaussian-noise STA"}).name == "Gaussian-noise_STA_code.py"


def _strf_out():
    cube = np.zeros((3, 4, 5))
    cube[1, 2] = [0, 1.0, 0.5, 0, 0]              # the peak check, ON at lag 1
    cube[0, 0] = [0, 0, -0.8, 0, 0]               # an OFF check at lag 2
    cube[2, 3] = [0, 0.2, 0, 0, 0]
    return {"cube": cube, "time_ms": np.array([0, 16.7, 33.4, 50.1, 66.8]), "peak_yx": (1, 2),
            "summary": "1 epoch(s) · peak check (y=1, x=2) @ 16.7 ms ON", "clock": "1/1 epochs on the clock",
            "blocks": [{"kind": "heatmap", "title": "spatial RF", "z": cube[:, :, 1], "mark": (1, 2)}]}


def test_strf_checks_from_events():
    f = viewer.strf_checks_from_events
    # the sink (assets/strfpick.js): a held click → that check; a double-click → [] = the peak
    assert f("strf-pick-sink.value", "check|1|2|7", None, 3, 4) == [(1, 2)]
    assert f("strf-pick-sink.value", "check|9|2|8", None, 3, 4) is None                     # off-grid
    assert f("strf-pick-sink.value", "check|a|b|9", None, 3, 4) is None
    assert f("strf-pick-sink.value", "peak|3", None, 3, 4) == []
    assert f("strf-pick-sink.value", "", None, 3, 4) is None
    # a box selection → every check inside; an EMPTY one (Plotly re-emits those) → no change
    sel = {"points": [{"x": 0, "y": 0}, {"x": 3, "y": 2}, {"x": 3, "y": 2}, {"curveNumber": 1}]}
    assert f("fft.selectedData", None, sel, 3, 4) == [(0, 0), (2, 3)]
    assert f("fft.selectedData", None, {"points": [{"x": 2.4, "y": 0.6}]}, 3, 4) == [(1, 2)]  # rounds
    assert f("fft.selectedData", None, {"points": []}, 3, 4) is None
    assert f("fft.selectedData", None, None, 3, 4) is None
    assert f("fft.clickData", "check|1|2|0", None, 3, 4) is None                              # not wired
    js = (Path(viewer.__file__).parent / "assets" / "strfpick.js").read_text()
    assert "plotly_deselect" in js and "plotly_click" in js and "strf-pick-sink" in js
    # both fired in one tick (a double-click that also clears a box selection): the sink wins
    g = viewer.strf_pick_checks
    assert g(["fft.selectedData", "strf-pick-sink.value"], "peak|4", None, 3, 4) == []
    assert g(["strf-pick-sink.value", "fft.selectedData"], "check|1|2|5", {"points": []}, 3, 4) == [(1, 2)]
    assert g(["fft.selectedData"], "peak|4", sel, 3, 4) == [(0, 0), (2, 3)]                   # a box alone
    assert g(["fft.selectedData"], "peak|4", {"points": []}, 3, 4) is None
    assert g(["strf-pick-sink.value"], "", None, 3, 4) is None
    assert g([], "peak|1", None, 3, 4) is None


def test_strf_panels_single_check_mean_and_peak_fallback():
    out = _strf_out()
    left, right = viewer._strf_panels(out)                          # no choice → the peak check
    assert "(y=1, x=2) = peak check" in right.layout.title.text and "[ON]" in right.layout.title.text
    assert "peak 16.7 ms" in right.layout.title.text
    assert left.layout.dragmode == "select" and "double-click" in left.layout.xaxis.title.text
    kinds = [tr.type for tr in left.data]
    assert kinds == ["heatmap", "scatter", "scatter"]                # map · pickable grid · the X mark(s)
    assert len(left.data[1].x) == 12 and left.data[1].marker.opacity == 0
    assert list(left.data[2].x) == [2] and list(left.data[2].y) == [1]
    # one chosen check → its own kernel, its own peak and sign
    _, r1 = viewer._strf_panels(out, [(0, 0)])
    assert "(y=0, x=0)" in r1.layout.title.text and "[OFF]" in r1.layout.title.text
    assert "peak 33.4 ms" in r1.layout.title.text and len(r1.data) >= 1
    # several → the mean, each one faint behind it, every mark on the map
    l2, r2 = viewer._strf_panels(out, [(0, 0), (1, 2), (2, 3), (9, 9)])
    assert "mean of 3 checks (y 0–2, x 0–3)" in r2.layout.title.text
    named = [tr.name for tr in r2.data if tr.name]
    assert named[-1] == "mean" and len(named) == 4
    mean_y = np.asarray(r2.data[-1].y)
    assert np.allclose(mean_y, (out["cube"][0, 0] + out["cube"][1, 2] + out["cube"][2, 3]) / 3)
    assert sorted(zip(l2.data[2].y, l2.data[2].x)) == [(0, 0), (1, 2), (2, 3)]
