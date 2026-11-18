"""neitz.io.stim — nested session-manifest reading + seed-based stimulus reproduction."""
import json
import numpy as np
import pytest
from neitz.io import stim
from neitz.stimulus import reproduce_noise

# flat records (the shape stimulus_metadata / noise_from_record consume)
CHECKER = {"stimulus": "AASeededGaussianCheckerboardSConeIsoStimFinal",
           "stim_type": "checkerboard", "cone_isolation": "S",
           "seed": 2, "mu": 0.5, "sigma": 0.3,
           "checks_x": 40, "checks_y": 32, "n_updates": 5,
           "update_every_n_frames": 8, "refresh_rate_hz": 60, "stim_frames": 40,
           "gamma": 2.2056, "noise_method": "mt19937ar+invCDF", "fill_order": "F",
           "timestamp": "2026-07-03T09:14:02"}


def _fs(wall=10.0, drift=-9.0):
    return {"client_wall_s": wall, "drift_frames": drift, "n_flips": 595}


def _tree(date="2026-07-16"):
    """A small nested manifest: cell 'c1' with a gaussian block (2 epochs) + a sq_wave block
    (1 epoch); cell '(standalone)' with one gaussian epoch."""
    return {
        "format": "neitz-stim-manifest/2", "date": date,
        "rig": {"projector": "TI LightCrafter 4500"},
        "cells": [
            {"cell_name": "c1", "blocks": [
                {"block_index": 1, "label": "grey", "stim_type": "gaussian_noise",
                 "cone_isolation": "achromatic", "stim_signature": "AAA",
                 "params": {"mu": 0.5, "sigma": 0.3, "checks_x": 1, "checks_y": 1,
                            "n_updates": 600, "stim_frames": 600, "refresh_rate_hz": 60},
                 "epochs": [
                     {"epoch": 1, "seed": 2, "timestamp": "2026-07-16T10:00:00", "frame_sync": _fs()},
                     {"epoch": 2, "seed": 3, "timestamp": "2026-07-16T10:00:20", "frame_sync": _fs()}]},
                {"block_index": 1, "label": "4hz", "stim_type": "sq_wave",
                 "cone_isolation": "achromatic", "stim_signature": "BBB",
                 "params": {"flicker_hz": 4, "stim_frames": 600, "refresh_rate_hz": 60},
                 "epochs": [
                     {"epoch": 1, "timestamp": "2026-07-16T10:01:00", "frame_sync": _fs()}]}]},
            {"cell_name": "(standalone)", "blocks": [
                {"block_index": 1, "label": "grey", "stim_type": "gaussian_noise",
                 "cone_isolation": "achromatic", "stim_signature": "AAA",
                 "params": {"mu": 0.5, "sigma": 0.3, "checks_x": 1, "checks_y": 1,
                            "n_updates": 600, "stim_frames": 600, "refresh_rate_hz": 60},
                 "epochs": [
                     {"epoch": 1, "seed": 2, "timestamp": "2026-07-16T10:02:00", "frame_sync": _fs()}]}]},
        ]}


def _write(tmp_path, tree, date="2026-07-16"):
    p = tmp_path / f"{date.replace('-', '_')}_stim_manifest.json"
    p.write_text(json.dumps(tree))
    return p


# ---- find / load / flatten ----
def test_find_and_load(tmp_path):
    p = _write(tmp_path, _tree())
    assert stim.find_manifest(tmp_path) == p
    assert stim.find_manifest(tmp_path, date="2026-07-16") == p
    assert stim.find_manifest(tmp_path, date="1999-01-01") is None   # strict: no wrong-date fallback
    tree = stim.load_manifest(p)
    assert tree["format"] == "neitz-stim-manifest/2"


def test_iter_epochs_flattens_in_order_with_merged_params(tmp_path):
    eps = stim.iter_epochs(_tree())
    assert len(eps) == 4                                             # 2 + 1 + 1
    assert [e["cell_name"] for e in eps] == ["c1", "c1", "c1", "(standalone)"]
    assert [e["block_label"] for e in eps] == ["grey", "grey", "4hz", "grey"]
    e0 = eps[0]
    assert e0["stim_type"] == "gaussian_noise" and e0["seed"] == 2
    # block params merged onto the epoch, with seed + block identity folded into `params`
    assert e0["params"]["mu"] == 0.5 and e0["params"]["seed"] == 2
    assert e0["params"]["stim_signature"] == "AAA"
    assert e0["aborted"] is False


def test_iter_epochs_flags_aborted():
    tree = _tree()
    tree["cells"][0]["blocks"][0]["epochs"][0]["frame_sync"] = _fs(wall=1.0, drift=-540)  # false start
    eps = stim.iter_epochs(tree)
    assert eps[0]["aborted"] is True and eps[1]["aborted"] is False


def test_stimulus_metadata_split():
    stim_type, params = stim.stimulus_metadata(CHECKER)
    assert stim_type == "checkerboard"
    assert "stim_type" not in params
    assert params["seed"] == 2 and params["cone_isolation"] == "S"


# ---- seed-based reproduction (unchanged core) ----
def test_noise_from_record_matches_reproduce():
    v = stim.noise_from_record(CHECKER)
    assert v.shape == (32, 40, 5)
    assert np.array_equal(v, reproduce_noise(2, 32, 40, 5, 0.5, 0.3))


def test_noise_per_frame_expands():
    v = stim.noise_from_record(CHECKER, per_frame=True)
    assert v.shape == (32, 40, 40)
    assert np.array_equal(v[..., 0], v[..., 7])
    assert not np.array_equal(v[..., 7], v[..., 8])


def test_noise_from_sq_wave_raises():
    with pytest.raises(ValueError):
        stim.noise_from_record({"stim_type": "sq_wave", "seed": 2})


def test_sent_codes_siso():
    codes = stim.sent_codes_from_record(CHECKER)
    assert codes.shape == (32, 40, 5, 3)
    assert codes[..., 2].max() == 0                                 # B = 0 for S-iso
    assert codes.min() >= 0 and codes.max() <= 255


# ---------------- noise seed files (per-check seeds, gaussCheckSeeds/1) ----------------
import os
import shutil
from neitz.stimulus import reproduce_noise_per_check

RIG_SEED_FILE = os.path.join(os.path.dirname(__file__), "data",
                             "2026_11_12_140021_cell3_b1_e1_check16x10_grey.json")


def _per_check_record(check_seeds, n_updates=5, **over):
    rec = {"stim_type": "checkerboard", "seed": 2, "mu": 0.5, "sigma": 0.3,
           "checks_y": len(check_seeds), "checks_x": len(check_seeds[0]), "n_updates": n_updates,
           "update_every_n_frames": 1, "refresh_rate_hz": 60, "stim_frames": n_updates,
           "seed_scheme": "gaussCheckSeeds/1", "check_seeds": check_seeds}
    rec.update(over)
    return rec


def test_noise_from_record_uses_check_seeds():
    seeds = [[11, 12, 13], [21, 22, 23]]
    v = stim.noise_from_record(_per_check_record(seeds))
    assert v.shape == (2, 3, 5)
    assert np.array_equal(v, reproduce_noise_per_check(seeds, 5, 0.5, 0.3))
    assert not np.array_equal(v, reproduce_noise(2, 2, 3, 5))        # NOT the legacy epoch-seed fill
    with pytest.raises(ValueError):                                   # grid must match checks_y/x
        stim.noise_from_record(_per_check_record(seeds, checks_x=4))


def test_noise_from_record_refuses_per_check_checkerboard_without_seeds():
    rec = _per_check_record([[1, 2], [3, 4]]); rec.pop("check_seeds")
    with pytest.raises(ValueError, match="per-check seeds"):
        stim.noise_from_record(rec)
    ff = _per_check_record([[7]]); ff.pop("check_seeds"); ff["seed"] = 7   # 1x1: epoch seed suffices
    assert np.array_equal(stim.noise_from_record(ff), reproduce_noise(7, 1, 1, 5))


def test_seed_file_resolve_params_and_checksum(tmp_path):
    (tmp_path / "noise_seeds").mkdir()
    shutil.copy(RIG_SEED_FILE, tmp_path / "noise_seeds" / "x.json")
    rec = {"stim_type": "checkerboard", "seed": 228464, "mu": 0.5, "sigma": 0.3, "checks_y": 10,
           "checks_x": 16, "n_updates": 1199, "seed_scheme": "gaussCheckSeeds/1",
           "noise_seed_file": "noise_seeds/x.json"}
    p = stim.resolve_noise_seed_file(rec, tmp_path)
    assert p == tmp_path / "noise_seeds" / "x.json"
    assert stim.resolve_noise_seed_file({"noise_seed_file": "x.json"}, tmp_path) == p   # bare name ok
    assert stim.resolve_noise_seed_file({"noise_seed_file": "nope.json"}, tmp_path) is None
    assert stim.resolve_noise_seed_file({}, tmp_path) is None
    sf = stim.load_noise_seed_file(p)
    params = stim.seed_params_from_file(sf)
    assert np.asarray(params["check_seeds"]).shape == (10, 16) and params["seed_scheme"] == "gaussCheckSeeds/1"
    rec.update(params)
    assert stim.verify_noise_checksum(rec) is True                    # the rig's own checksum
    bad = dict(rec, noise_checksum={"sum": rec["noise_checksum"]["sum"] + 1.0})
    assert stim.verify_noise_checksum(bad) is False
    assert stim.verify_noise_checksum({k: v for k, v in rec.items() if k != "noise_checksum"}) is None
    with pytest.raises(ValueError):
        (tmp_path / "bad.json").write_text(json.dumps({"format": "other"}))
        stim.load_noise_seed_file(tmp_path / "bad.json")


def test_build_import_plan_embeds_check_seeds_and_warns_on_missing(tmp_path):
    """Import-time resolution: an epoch's noise_seed_file (relative to the manifest) is loaded, its
    per-check seeds embedded in the epoch params + the checksum verified; a missing file for a
    checkerboard is a warning (that epoch can't be regenerated)."""
    date = "2026-11-12"
    src = tmp_path / "docs"; (src / "noise_seeds").mkdir(parents=True)
    shutil.copy(RIG_SEED_FILE, src / "noise_seeds" / "e1.json")
    params = {"mu": 0.5, "sigma": 0.3, "checks_x": 16, "checks_y": 10, "n_updates": 1199,
              "update_every_n_frames": 1, "refresh_rate_hz": 59.9435, "stim_frames": 1199,
              "seed_scheme": "gaussCheckSeeds/1"}
    tree = {"format": "neitz-stim-manifest/2", "date": date, "cells": [
        {"cell_name": "cell3", "blocks": [
            {"block_index": 1, "label": "block 1", "stim_type": "checkerboard", "cone_isolation": "achromatic",
             "stim_signature": "S", "params": params, "epochs": [
                 {"epoch": 1, "seed": 228464, "timestamp": f"{date}T14:00:21", "frame_sync": _fs(24.1, -0.5),
                  "noise_seed_file": "noise_seeds/e1.json", "noise_checksum": {"sum": 95938.29989430605}},
                 {"epoch": 2, "seed": 996841, "timestamp": f"{date}T14:00:51", "frame_sync": _fs(24.1, -0.5),
                  "noise_seed_file": "noise_seeds/missing.json"}]}]}]}
    (src / f"{date.replace('-', '_')}_stim_manifest.json").write_text(json.dumps(tree))
    abfs = [str(src / f"2026_11_12_00{i}.abf") for i in (42, 43)]
    plan = stim.build_import_plan(src, date, abfs, abf_datetimes=[f"{date}T14:00:22", f"{date}T14:00:52"])
    eps = plan["cells"][0]["protocols"][0]["epochs"]
    assert np.asarray(eps[0]["params"]["check_seeds"]).shape == (10, 16)
    assert eps[0]["seed_file_path"] == str(src / "noise_seeds" / "e1.json")
    assert eps[0]["params"]["noise_checksum"]["sum"] == 95938.29989430605
    assert "check_seeds" not in eps[1]["params"] and "seed_file_path" not in eps[1]
    assert any("missing.json" in w and "CANNOT" in w for w in plan["warnings"])
    assert not any("does NOT match" in w for w in plan["warnings"])
    # the resolved record regenerates the rig's exact noise
    rec = dict(eps[0]["params"], stim_type="checkerboard")
    assert stim.verify_noise_checksum(rec) is True
