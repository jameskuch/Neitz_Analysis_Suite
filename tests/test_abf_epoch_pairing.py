"""ABF <-> stim-epoch pairing: align_abfs_to_epochs + build_import_plan.

Replaces the old jsonl count/timestamp guard (test_manifest_countcheck): the nested importer
no longer refuses on a count mismatch — it ALIGNS abfs to epochs by timestamp, skipping
aborted/discarded epochs and surfacing stray abfs, so a real session (false starts, a manual
sweep) imports correctly instead of aborting.
"""
import json
from neitz.io import stim


def _fs(wall=10.0, drift=-9.0):
    return {"client_wall_s": wall, "drift_frames": drift}


# ---------------- align_abfs_to_epochs ----------------
def test_align_one_to_one():
    ep = ["2026-07-16T10:00:00", "2026-07-16T10:00:20"]
    ab = ["2026-07-16T10:00:03", "2026-07-16T10:00:23"]     # ~3 s after each epoch
    ae, ea = stim.align_abfs_to_epochs(ab, ep)
    assert ae == [0, 1] and ea == [0, 1]


def test_align_aborted_epoch_skipped():
    # epoch 0 is a false start with no nearby abf; recording begins at epoch 1
    ep = ["2026-07-16T10:00:00", "2026-07-16T10:05:00", "2026-07-16T10:05:20"]
    ab = ["2026-07-16T10:05:03", "2026-07-16T10:05:23"]
    ae, ea = stim.align_abfs_to_epochs(ab, ep, epoch_skippable=[True, False, False])
    assert ea == [None, 0, 1]                               # epoch 0 unmatched, then aligned
    assert ae == [1, 2]


def test_align_stray_abf_unmatched():
    ep = ["2026-07-16T10:00:00", "2026-07-16T10:00:20"]
    ab = ["2026-07-16T10:00:03", "2026-07-16T10:00:23", "2026-07-16T11:00:00"]  # 3rd = stray sweep
    ae, ea = stim.align_abfs_to_epochs(ab, ep)
    assert ae == [0, 1, None]                               # the far stray abf is left unmatched
    assert ea == [0, 1]


def test_align_prefers_correct_shift_over_blind_zip():
    # a blind zip-by-order would pair abf0 with the (aborted) epoch0; timestamps force the
    # correct shift so no abf is mislabeled with the wrong seed.
    ep = ["2026-07-16T10:00:00", "2026-07-16T10:10:00"]     # epoch0 aborted (no abf)
    ab = ["2026-07-16T10:10:04"]                            # single abf belongs to epoch1
    ae, ea = stim.align_abfs_to_epochs(ab, ep, epoch_skippable=[True, False])
    assert ae == [1] and ea == [None, 0]


# ---------------- build_import_plan ----------------
def _today_shape_tree(date="2026-07-16"):
    """A real-session shape: cell 'mac' = a false start (2 aborted epochs, no abf), then cell
    'c1' with 2 recorded epochs. Mirrors 2026-07-16's mac-C1 → c1."""
    gparams = {"mu": 0.5, "sigma": 0.3, "checks_x": 1, "checks_y": 1,
               "n_updates": 600, "stim_frames": 600, "refresh_rate_hz": 60}
    return {"format": "neitz-stim-manifest/2", "date": date, "cells": [
        {"cell_name": "mac", "blocks": [
            {"block_index": 1, "label": "grey", "stim_type": "gaussian_noise",
             "cone_isolation": "achromatic", "stim_signature": "A", "params": gparams,
             "epochs": [
                 {"epoch": 1, "seed": 2, "timestamp": f"{date}T10:00:00", "frame_sync": _fs(1.0, -540)},
                 {"epoch": 2, "seed": 3, "timestamp": f"{date}T10:00:10", "frame_sync": _fs(1.0, -540)}]}]},
        {"cell_name": "c1", "blocks": [
            {"block_index": 1, "label": "grey", "stim_type": "gaussian_noise",
             "cone_isolation": "achromatic", "stim_signature": "A", "params": gparams,
             "epochs": [
                 {"epoch": 1, "seed": 2, "timestamp": f"{date}T10:05:00", "frame_sync": _fs()},
                 {"epoch": 2, "seed": 3, "timestamp": f"{date}T10:05:20", "frame_sync": _fs()}]}]},
    ]}


def _write(tmp, tree, date="2026-07-16"):
    p = tmp / f"{date.replace('-', '_')}_stim_manifest.json"
    p.write_text(json.dumps(tree))
    return p


def test_build_import_plan_false_start_and_stray(tmp_path):
    _write(tmp_path, _today_shape_tree())
    abfs = ["a0.abf", "a1.abf", "a2.abf"]                   # a0,a1 -> c1; a2 -> a stray sweep
    dts = ["2026-07-16T10:05:03", "2026-07-16T10:05:23", "2026-07-16T11:30:00"]
    plan = stim.build_import_plan(tmp_path, "2026-07-16", abfs, abf_datetimes=dts)
    names = [c["cell_name"] for c in plan["cells"]]
    assert names == ["mac", "c1"]
    mac = plan["cells"][0]
    assert all(e["abf"] is None for e in mac["protocols"][0]["epochs"])   # false start: no abf
    c1 = plan["cells"][1]
    got = [e["abf"] for e in c1["protocols"][0]["epochs"]]
    assert got == ["a0.abf", "a1.abf"]                     # correctly paired by time
    assert plan["unmatched_abfs"] == ["a2.abf"]            # the stray sweep -> unsorted
    assert plan["discarded_abfs"] == []


def test_build_import_plan_discarded_epoch_skips_abf(tmp_path):
    tree = _today_shape_tree()
    tree["discarded"] = [{"timestamp": "2026-07-16T09:00:00", "stim_signature": "A"}]
    _write(tmp_path, tree)
    abfs = ["d.abf", "a0.abf", "a1.abf"]                    # d.abf = a discarded-block sweep
    dts = ["2026-07-16T09:00:02", "2026-07-16T10:05:03", "2026-07-16T10:05:23"]
    plan = stim.build_import_plan(tmp_path, "2026-07-16", abfs, abf_datetimes=dts)
    assert plan["discarded_abfs"] == ["d.abf"]             # matched a discarded timestamp -> skip
    assert plan["unmatched_abfs"] == []
    assert [e["abf"] for e in plan["cells"][1]["protocols"][0]["epochs"]] == ["a0.abf", "a1.abf"]


def test_build_import_plan_no_manifest_is_unsorted(tmp_path):
    abfs = ["x0.abf", "x1.abf"]
    plan = stim.build_import_plan(tmp_path, "2026-07-16", abfs,
                                  abf_datetimes=[None, None])
    assert plan["manifest"] is None
    assert plan["cells"] == []
    assert plan["unmatched_abfs"] == abfs
    assert any("no nested stim manifest" in w for w in plan["warnings"])
