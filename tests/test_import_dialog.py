"""Data Explorer import — the optional "stimulus JSON folder": the day's manifest + noise_seeds/
can live in a different folder than the Clampex .abf files (2026-11-12: pCLAMP/ vs
docs.ephys_data/). `_import_recordings(..., stim_dir=)` searches that folder FIRST and resolves the
per-check seed files relative to it."""
import json
import os
import shutil
import numpy as np
import viewer
from neitz.dataio import DataStore
from neitz.io import stim

FIXTURE = os.path.join(os.path.dirname(__file__), "data",
                       "2026_11_12_140021_cell3_b1_e1_check16x10_grey.json")


def _two_folder_drop(tmp_path, date="2026-11-12"):
    """pCLAMP/<abf> in one folder, docs/<manifest> + docs/noise_seeds/<seed file> in another."""
    abf_dir = tmp_path / "pCLAMP"; abf_dir.mkdir()
    docs = tmp_path / "docs"; (docs / "noise_seeds").mkdir(parents=True)
    shutil.copy(FIXTURE, docs / "noise_seeds" / "e1.json")
    params = {"mu": 0.5, "sigma": 0.3, "checks_x": 16, "checks_y": 10, "n_updates": 1199,
              "update_every_n_frames": 1, "refresh_rate_hz": 59.9435, "stim_frames": 1199,
              "seed_scheme": "gaussCheckSeeds/1"}
    tree = {"format": "neitz-stim-manifest/2", "date": date, "cells": [
        {"cell_name": "cell3", "blocks": [
            {"block_index": 1, "label": "block 1", "stim_type": "checkerboard", "cone_isolation": "achromatic",
             "stim_signature": "S", "params": params, "epochs": [
                 {"epoch": 1, "seed": 228464, "timestamp": f"{date}T14:00:21",
                  "frame_sync": {"client_wall_s": 24.1, "drift_frames": -0.5},
                  "noise_seed_file": "noise_seeds/e1.json"}]}]}]}
    (docs / f"{date.replace('-', '_')}_stim_manifest.json").write_text(json.dumps(tree))
    abf = abf_dir / "2026_11_12_0042.abf"
    abf.write_bytes(b"ABF2 dummy")
    return abf, docs


def test_import_with_stimulus_json_folder(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("EPHYSDATAIO_ROOT", str(store))
    abf, docs = _two_folder_drop(tmp_path)
    opts, rev, msg, msg2 = viewer._import_recordings([str(abf)], None, None, None, 0, stim_dir=str(docs))
    assert rev == 1 and msg == msg2 and "imported 1 recording" in msg
    assert "manifest:" in msg and "stim_manifest.json" in msg
    cm = DataStore(str(store)).cell_by_name("2026-11-12", "cell3")
    rec = cm.data["recordings"][0]
    assert rec["stimulus"]["type"] == "checkerboard"
    assert np.asarray(rec["stimulus"]["params"]["check_seeds"]).shape == (10, 16)
    assert (cm.dir / "noise_seeds" / "e1.json").exists()           # seed file travelled with the cell
    assert stim.verify_noise_checksum(dict(rec["stimulus"]["params"], stim_type="checkerboard")) is True


def test_import_without_stimulus_folder_falls_back(tmp_path, monkeypatch):
    """Blank stim folder = the old behaviour: look next to the abfs → here nothing → '(unsorted)'."""
    store = tmp_path / "store"
    monkeypatch.setenv("EPHYSDATAIO_ROOT", str(store))
    abf, _docs = _two_folder_drop(tmp_path)
    _o, _r, msg, _m = viewer._import_recordings([str(abf)], None, None, None, 0, stim_dir="   ")
    assert "[no manifest]" in msg
    ds = DataStore(str(store))
    assert ds.cell_by_name("2026-11-12", "(unsorted)") is not None
    assert ds.cell_by_name("2026-11-12", "cell3") is None


def test_import_rejects_missing_stimulus_folder(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("EPHYSDATAIO_ROOT", str(store))
    abf, _docs = _two_folder_drop(tmp_path)
    o, r, msg, _m = viewer._import_recordings([str(abf)], None, None, None, 0, stim_dir=str(tmp_path / "nope"))
    assert "not found" in msg and not DataStore(str(store)).cells()     # nothing imported
