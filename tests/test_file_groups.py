"""Analysis-View files list — the "group" toggle (like-stimulus blocks) and a block's own all / none.

`viewer.build_file_options(paths, grouped=True)` opens each block with a DISABLED header option
(value `GROUP_HDR + key`) carrying all / none buttons; `file.value` keeps holding real paths only,
and the callbacks that read `file.options` drop the headers via `_paths_from_options`."""
import json
import types
import numpy as np
import pytest
import viewer
from neitz.dataio.manifest import CellManifest


def _stim(t, **params):
    return {"type": t, "params": params}


def _rec(file, stim=None, protocol=None):
    return {"id": file.split(".")[0], "file": f"raw/{file}", "stimulus": stim, "protocol": protocol}


# ---- labels ------------------------------------------------------------------------------------
def test_stim_group_label_drops_the_per_epoch_bits():
    assert viewer._stim_group_label(_stim("sq_wave", flicker_hz=1, cone_isolation="achromatic",
                                          seed=12)) == "1 Hz sq wave · achromatic"
    assert viewer._stim_group_label(_stim("flicker", flicker_hz=4.0, cone_isolation="S")) == "4 Hz sq wave · S"
    assert viewer._stim_group_label(_stim("checkerboard", checks_x=16, checks_y=10, flicker_hz=59.94,
                                          cone_isolation="achromatic", seed=5)) == "checkerboard 16×10 · achromatic"
    assert viewer._stim_group_label(_stim("gaussian_noise", checks_x=1, checks_y=1,
                                          cone_isolation="achromatic")) == "gaussian noise full field · achromatic"
    assert viewer._stim_group_label(_stim("grating")) == "grating"
    assert viewer._stim_group_label(None) is None
    assert viewer._stim_group_label({"type": None, "params": {"flicker_hz": 1}}) is None


# ---- keys: manifest stimulus first ---------------------------------------------------------------
def test_file_group_keys_by_stimulus_signature_across_runs_and_cells(tmp_path):
    """Two protocol runs of the same stimulus (different seeds / epochs, even different cells) land
    in ONE group; a different signature is a different group."""
    store = tmp_path / "store"
    sq = lambda seed: _stim("sq_wave", flicker_hz=1, cone_isolation="achromatic", seed=seed,
                            stim_signature="692bc4ce68020ddd")
    paths = {}
    for cell, files in (("c02", ["a.abf", "b.abf", "c.abf"]), ("c04", ["d.abf"])):
        cm = CellManifest.open(store / "2026-11-12" / cell, date="2026-11-12", cell=cell, label=cell)
        cm.data["recordings"] = [
            _rec(files[0], sq(1), {"index": 1, "label": "block 1", "signature": "692bc4ce68020ddd"}),
            *([_rec(files[1], _stim("sq_wave", flicker_hz=1, cone_isolation="S", seed=2,
                                    stim_signature="7c3dc3c25a61242d"))] if len(files) > 1 else []),
            *([_rec(files[2], sq(3), {"index": 3, "label": "block 3", "signature": "692bc4ce68020ddd"})]
              if len(files) > 2 else []),
        ]
        cm.save()
        for f in files:
            paths[f] = str(cm.dir / "raw" / f)
    ka, la = viewer.file_group(paths["a.abf"])
    kb, lb = viewer.file_group(paths["b.abf"])
    kc, _ = viewer.file_group(paths["c.abf"])
    kd, _ = viewer.file_group(paths["d.abf"])
    assert ka == kc == kd == "s:692bc4ce68020ddd" and la == "1 Hz sq wave · achromatic"
    assert kb == "s:7c3dc3c25a61242d" and lb == "1 Hz sq wave · S"
    # a typed stimulus with no signature still groups by its label
    cm = CellManifest.open(store / "2026-11-12" / "c02")
    cm.data["recordings"].append(_rec("e.abf", _stim("sq_wave", flicker_hz=2)))
    cm.save()
    assert viewer.file_group(str(cm.dir / "raw" / "e.abf")) == ("l:2_hz_sq_wave", "2 Hz sq wave")


def test_recording_for_path_outside_the_store_is_none(tmp_path):
    assert viewer._recording_for_path(str(tmp_path / "x.abf")) is None
    assert viewer._recording_for_path("/no/such/raw/file.abf") is None


# ---- keys: measured frequency, then duration, for un-manifested files -------------------------------
def test_file_group_falls_back_to_measured_flicker_then_duration(monkeypatch):
    monkeypatch.setattr(viewer, "_recording_for_path", lambda p: None)
    rec = types.SimpleNamespace(channel_names=["Im_prime", "Vm_sec", "TTL"], duration=24.4)
    monkeypatch.setattr(viewer, "get_recording", lambda p: rec)
    seen = {}
    monkeypatch.setattr(viewer, "get_flicker",
                        lambda p, chan: seen.setdefault("chan", chan) and
                        (types.SimpleNamespace(freq=1.004) if "sq" in p else None))
    assert viewer.file_group("/x/raw/sq_0005.abf") == ("m:1.0", "~1 Hz sq wave (measured)")
    assert seen["chan"] == "TTL"                                   # the sync channel, not the last one
    assert viewer.file_group("/x/raw/other.abf") == ("d:24", "unlabelled · 24 s")
    # a file that won't open at all
    monkeypatch.setattr(viewer, "get_recording", lambda p: (_ for _ in ()).throw(OSError("bad")))
    assert viewer.file_group("/x/raw/broken.abf") == ("d:?", "unlabelled")


# ---- the options list -----------------------------------------------------------------------------
def _fake_groups(monkeypatch):
    table = {"/r/a.abf": ("s:1", "1 Hz sq wave · achromatic"), "/r/b.abf": ("s:2", "checkerboard 16×10"),
             "/r/c.abf": ("s:1", "1 Hz sq wave · achromatic"), "/r/d.abf": ("s:2", "checkerboard 16×10")}
    monkeypatch.setattr(viewer, "file_group", lambda p: table[p])
    return list(table)


def test_build_file_options_flat_vs_grouped(monkeypatch):
    paths = _fake_groups(monkeypatch)
    flat = viewer.build_file_options(paths + ["/r/a.abf", None])     # dupes / blanks dropped
    assert [o["value"] for o in flat] == paths and not any(viewer._is_header(o) for o in flat)

    opts = viewer.build_file_options(paths, grouped=True)
    kinds = [o["value"] if viewer._is_header(o) else "f:" + o["value"] for o in opts]
    assert kinds == [viewer.GROUP_HDR + "s:1", "f:/r/a.abf", "f:/r/c.abf",
                     viewer.GROUP_HDR + "s:2", "f:/r/b.abf", "f:/r/d.abf"]
    hdr = opts[0]
    assert hdr["disabled"] is True
    # the header label carries this group's own all / none buttons
    btn_ids = [c.id for c in hdr["label"].children[1].children if getattr(c, "id", None)]
    assert btn_ids == [{"type": "file-grp-all", "grp": "s:1"}, {"type": "file-grp-none", "grp": "s:1"}]
    assert hdr["label"].children[0].children[1] == "1 Hz sq wave · achromatic"   # [arrow, title]
    assert "2 files" in hdr["label"].children[1].children[0].children
    # downstream helpers never see the headers as files
    assert viewer._paths_from_options(opts) == ["/r/a.abf", "/r/c.abf", "/r/b.abf", "/r/d.abf"]
    assert viewer._group_members(opts, "s:2") == ["/r/b.abf", "/r/d.abf"]
    assert viewer._group_members(opts, "s:9") == []
    assert viewer.file_count(["/r/a.abf"], opts) == "(1/4)"
    assert viewer.file_count([], []) == ""


def test_apply_group_select_only_touches_that_group(monkeypatch):
    paths = _fake_groups(monkeypatch)
    opts = viewer.build_file_options(paths, grouped=True)
    # "all" on group 2 with a.abf already checked → a + both of group 2, in checklist order
    assert viewer._apply_group_select(opts, ["/r/a.abf"], "s:2", True) == ["/r/a.abf", "/r/b.abf", "/r/d.abf"]
    # "none" on group 1 leaves group 2's selection alone
    assert viewer._apply_group_select(opts, ["/r/a.abf", "/r/c.abf", "/r/d.abf"], "s:1", False) == ["/r/d.abf"]
    assert viewer._apply_group_select(opts, [], "s:1", False) == []


def test_regroup_keeps_the_files_and_flattens_back(monkeypatch):
    paths = _fake_groups(monkeypatch)
    grouped = viewer.regroup_files(["stim"], viewer.build_file_options(paths))
    assert sum(viewer._is_header(o) for o in grouped) == 2
    flat = viewer.regroup_files([], grouped)
    assert [o["value"] for o in flat] == ["/r/a.abf", "/r/c.abf", "/r/b.abf", "/r/d.abf"]
    assert viewer.regroup_files([], []) is viewer.no_update


def test_unsorted_file_borrows_the_stimulus_of_its_same_day_twin(tmp_path):
    """An '(unsorted)' re-import of the day's Clampex folder has no stimulus records, but the same
    file names were already filed WITH stimuli in other cells of that date → same group."""
    day = tmp_path / "store" / "2026-11-12"
    cb = _stim("checkerboard", checks_x=16, checks_y=10, cone_isolation="achromatic", seed=1,
               stim_signature="6228c01c58417c5b")
    c04 = CellManifest.open(day / "c04", date="2026-11-12", cell="c04", label="cell3")
    c04.data["recordings"] = [_rec("2026_11_12_0042.abf", cb, {"index": 1, "signature": "6228c01c58417c5b"})]
    c04.save()
    c06 = CellManifest.open(day / "c06", date="2026-11-12", cell="c06", label="(unsorted)")
    c06.data["recordings"] = [_rec("2026_11_12_0042.abf"), _rec("2026_11_12_0001.abf")]
    c06.save()
    twin = viewer.file_group(str(c06.dir / "raw" / "2026_11_12_0042.abf"))
    assert twin == viewer.file_group(str(c04.dir / "raw" / "2026_11_12_0042.abf"))
    assert twin == ("s:6228c01c58417c5b", "checkerboard 16×10 · achromatic")
    # no twin anywhere (and the file itself is absent) → the last-resort bucket
    assert viewer.file_group(str(c06.dir / "raw" / "2026_11_12_0001.abf")) == ("d:?", "unlabelled")
