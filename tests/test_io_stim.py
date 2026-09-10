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
