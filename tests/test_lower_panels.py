"""Analysis View: the lower panels follow the pipeline (`lower_family`) and a file's default region
comes from its stimulus (`stim_window`)."""
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
