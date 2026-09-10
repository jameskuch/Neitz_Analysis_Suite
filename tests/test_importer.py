"""neitz.io.importer.apply_import_plan + DataStore.move_recording — the day → cell →
protocol → epoch store-writing path, and integrity-safe cross-cell moves."""
import json
from neitz.io import stim
from neitz.io.importer import apply_import_plan, renumber_protocols
from neitz.dataio import DataStore


def _fs(wall=10.0, drift=-9.0):
    return {"client_wall_s": wall, "drift_frames": drift}


def _session(tmp, date="2026-07-16"):
    """Write a manifest + dummy .abf files; return (source_dir, [abf_paths], [abf_dts])."""
    src = tmp / "src"; src.mkdir()
    gparams = {"mu": 0.5, "sigma": 0.3, "checks_x": 1, "checks_y": 1,
               "n_updates": 600, "stim_frames": 600, "refresh_rate_hz": 60}
    tree = {"format": "neitz-stim-manifest/2", "date": date, "cells": [
        {"cell_name": "mac", "blocks": [                    # false start: aborted, no abf
            {"block_index": 1, "label": "grey", "stim_type": "gaussian_noise",
             "cone_isolation": "achromatic", "stim_signature": "A", "params": gparams,
             "epochs": [{"epoch": 1, "seed": 2, "timestamp": f"{date}T10:00:00",
                         "frame_sync": _fs(1.0, -540)}]}]},
        {"cell_name": "c1", "blocks": [
            {"block_index": 1, "label": "grey", "stim_type": "gaussian_noise",
             "cone_isolation": "achromatic", "stim_signature": "A", "params": gparams,
             "epochs": [{"epoch": 1, "seed": 2, "timestamp": f"{date}T10:05:00", "frame_sync": _fs()},
                        {"epoch": 2, "seed": 3, "timestamp": f"{date}T10:05:20", "frame_sync": _fs()}]},
            {"block_index": 2, "label": "4hz", "stim_type": "sq_wave",
             "cone_isolation": "achromatic", "stim_signature": "B",
             "params": {"flicker_hz": 4, "stim_frames": 600, "refresh_rate_hz": 60},
             "epochs": [{"epoch": 1, "timestamp": f"{date}T10:06:00", "frame_sync": _fs()}]}]},
    ]}
    (src / f"{date.replace('-', '_')}_stim_manifest.json").write_text(json.dumps(tree))
    abfs, dts = [], []
    for i, t in enumerate(["10:05:03", "10:05:23", "10:06:04", "11:30:00"]):  # last = stray
        p = src / f"{date.replace('-', '_')}_{i:04d}.abf"
        p.write_bytes(b"ABF\x00 dummy")
        abfs.append(str(p))
        dts.append(f"{date}T{t}")
    return src, abfs, dts


def test_apply_import_plan_builds_hierarchy(tmp_path):
    src, abfs, dts = _session(tmp_path)
    ds = DataStore(root=tmp_path / "store")
    plan = stim.build_import_plan(src, "2026-07-16", abfs, abf_datetimes=dts)
    summ = apply_import_plan(ds, "2026-07-16", plan)

    # mac (only-aborted) skipped; c1 (3 recorded) + (unsorted) (the stray sweep) created
    by_name = {c["name"]: c for c in summ["cells"]}
    assert set(by_name) == {"c1", "(unsorted)"}
    assert by_name["c1"]["n"] == 3 and by_name["(unsorted)"]["n"] == 1
    assert summ["n_imported"] == 4
    assert any("mac" in n and "skipped" in n for n in summ["notes"])

    cm = ds.cell_by_name("2026-07-16", "c1")
    # two protocols under the cell, stamped on each recording
    groups = stim.epoch_groups(cm.data["recordings"])
    assert [(g["protocol_label"], g["n_epochs"]) for g in groups] == [("grey", 2), ("4hz", 1)]
    r0 = cm.data["recordings"][0]
    assert r0["protocol"] == {"index": 1, "label": "grey", "signature": "A",
                              "epoch": 1, "n_epochs": 2}
    assert r0["stimulus"]["type"] == "gaussian_noise" and r0["stimulus"]["params"]["seed"] == 2
    assert (cm.dir / r0["file"]).exists()                  # raw file copied in


def test_move_recording_is_integrity_safe(tmp_path):
    src, abfs, dts = _session(tmp_path)
    ds = DataStore(root=tmp_path / "store")
    plan = stim.build_import_plan(src, "2026-07-16", abfs, abf_datetimes=dts)
    apply_import_plan(ds, "2026-07-16", plan)

    c1 = ds.cell_by_name("2026-07-16", "c1")
    src_cell = c1.data["cell"]
    rec = c1.data["recordings"][0]
    rec_id, old_path = rec["id"], c1.dir / rec["file"]
    dst = ds.new_cell("2026-07-16", label="c1b")

    moved = ds.move_recording("2026-07-16", src_cell, dst.data["cell"], rec_id)

    src_after = ds.cell("2026-07-16", src_cell)
    dst_after = ds.cell("2026-07-16", dst.data["cell"])
    assert rec_id not in [r["id"] for r in src_after.data["recordings"]]   # gone from src
    assert rec_id in [r["id"] for r in dst_after.data["recordings"]]       # in dst
    assert not old_path.exists()                                          # file physically moved
    assert (dst_after.dir / moved["file"]).exists()                       # ...to dst/raw
    assert moved["protocol"]["label"] == "grey"                           # metadata preserved
    assert moved["stimulus"]["type"] == "gaussian_noise"


class _FakeCM:
    def __init__(self, recs):
        self.data = {"recordings": recs}


def test_renumber_protocols_reassigns_subset():
    # a cell of 4 recordings; re-categorize (assign a protocol label to a subset), then renumber
    recs = [{"id": f"r{i}", "protocol": None} for i in range(4)]
    cm = _FakeCM(recs)
    # user checks r0, r1 → "grey"; r2, r3 → "flash"
    for r in recs[:2]:
        r["protocol"] = {"label": "grey"}
    for r in recs[2:]:
        r["protocol"] = {"label": "flash"}
    renumber_protocols(cm)
    assert [r["protocol"]["index"] for r in recs] == [1, 1, 2, 2]      # grey=1, flash=2
    assert [r["protocol"]["epoch"] for r in recs] == [1, 2, 1, 2]      # numbered within each
    assert [r["protocol"]["n_epochs"] for r in recs] == [2, 2, 2, 2]
    groups = stim.epoch_groups(recs)
    assert [(g["protocol_label"], g["n_epochs"]) for g in groups] == [("grey", 2), ("flash", 2)]


def test_renumber_protocols_leaves_unlabeled_alone():
    recs = [{"id": "a", "protocol": {"label": "grey"}}, {"id": "b", "protocol": None},
            {"id": "c", "protocol": {"label": "grey"}}]
    renumber_protocols(_FakeCM(recs))
    assert recs[0]["protocol"]["epoch"] == 1 and recs[2]["protocol"]["epoch"] == 2  # grey run
    assert recs[1]["protocol"] is None                                # ungrouped stays ungrouped


def test_move_recording_same_cell_is_noop(tmp_path):
    src, abfs, dts = _session(tmp_path)
    ds = DataStore(root=tmp_path / "store")
    apply_import_plan(ds, "2026-07-16",
                      stim.build_import_plan(src, "2026-07-16", abfs, abf_datetimes=dts))
    c1 = ds.cell_by_name("2026-07-16", "c1")
    rid = c1.data["recordings"][0]["id"]
    n_before = len(c1.data["recordings"])
    ds.move_recording("2026-07-16", c1.data["cell"], c1.data["cell"], rid)
    assert len(ds.cell("2026-07-16", c1.data["cell"]).data["recordings"]) == n_before
