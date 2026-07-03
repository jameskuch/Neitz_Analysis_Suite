"""neitz.io.stim — session-manifest reading + seed-based stimulus reproduction."""
import json
import numpy as np
import pytest
from neitz.io import stim
from neitz.stimulus import reproduce_noise

CHECKER = {"stimulus": "AASeededGaussianCheckerboardSConeIsoStimFinal",
           "stim_type": "checkerboard", "cone_isolation": "S",
           "seed": 2, "mu": 0.5, "sigma": 0.3,
           "checks_x": 40, "checks_y": 32, "n_updates": 5,
           "update_every_n_frames": 8, "refresh_rate_hz": 60, "stim_frames": 40,
           "gamma": 2.2056, "noise_method": "mt19937ar+invCDF", "fill_order": "F",
           "timestamp": "2026-07-03T09:14:02"}
FLICKER = {"stimulus": "AAGreyScaleFullFieldNoiseFinal2026", "stim_type": "sq_wave",
           "cone_isolation": "achromatic", "flicker_hz": 4, "refresh_rate_hz": 60,
           "stim_frames": 40, "timestamp": "2026-07-03T09:15:00"}


def _write_manifest(tmp_path, rows, date="2026_07_03"):
    p = tmp_path / f"{date}_stim_manifest.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_load_and_find(tmp_path):
    p = _write_manifest(tmp_path, [CHECKER, FLICKER])
    rows = stim.load_session_manifest(p)
    assert len(rows) == 2 and rows[0]["stim_type"] == "checkerboard"
    assert stim.find_session_manifest(tmp_path) == p
    assert stim.find_session_manifest(tmp_path, date="2026-07-03") == p
    assert stim.find_session_manifest(tmp_path, date="1999-01-01") is None  # strict: no wrong-date fallback


def test_stimulus_metadata_split():
    stim_type, params = stim.stimulus_metadata(CHECKER)
    assert stim_type == "checkerboard"
    assert "stim_type" not in params
    assert params["seed"] == 2 and params["cone_isolation"] == "S"


def test_noise_from_record_matches_reproduce():
    v = stim.noise_from_record(CHECKER)
    assert v.shape == (32, 40, 5)
    assert np.array_equal(v, reproduce_noise(2, 32, 40, 5, 0.5, 0.3))


def test_noise_per_frame_expands():
    v = stim.noise_from_record(CHECKER, per_frame=True)
    assert v.shape == (32, 40, 40)                     # stim_frames = 40
    assert np.array_equal(v[..., 0], v[..., 7])        # held within an update...
    assert not np.array_equal(v[..., 7], v[..., 8])    # ...changes at the next update


def test_noise_from_flicker_raises():
    with pytest.raises(ValueError):
        stim.noise_from_record(FLICKER)


def test_pair_by_order():
    pairs = stim.pair_by_order([CHECKER, FLICKER], ["2026_07_03_0001", "2026_07_03_0002"])
    assert pairs[0][0] == "2026_07_03_0001"
    assert pairs[0][1]["stim_type"] == "checkerboard"


def test_sent_codes_siso():
    codes = stim.sent_codes_from_record(CHECKER)
    assert codes.shape == (32, 40, 5, 3)
    assert codes[..., 2].max() == 0                    # B = 0 for S-iso
    assert codes.min() >= 0 and codes.max() <= 255


def test_apply_session_manifest_tags_recordings(tmp_path):
    """Import wiring: manifest rows pair to recordings BY ORDER, set stimulus metadata with
    source='stim-manifest', and the manifest is copied into the cell dir (self-contained store)."""
    from neitz.dataio import DataStore
    src = tmp_path / "src"; src.mkdir()
    _write_manifest(src, [CHECKER, FLICKER], date="2026_07_03")

    cm = DataStore(root=tmp_path / "store").cell("2026-07-03", "c01")
    cm.dir.mkdir(parents=True, exist_ok=True)
    cm.data["recordings"] = [{"id": "2026_07_03_0001"}, {"id": "2026_07_03_0002"}]

    n = stim.apply_session_manifest(cm, src, date="2026-07-03")
    assert n == 2
    s0 = cm.get_stimulus("2026_07_03_0001")
    assert s0["type"] == "checkerboard" and s0["source"] == "stim-manifest"
    assert s0["params"]["seed"] == 2 and s0["params"]["cone_isolation"] == "S"
    assert "stim_type" not in s0["params"]                       # split out into .type
    assert cm.get_stimulus("2026_07_03_0002")["type"] == "sq_wave"
    assert (cm.dir / "2026_07_03_stim_manifest.jsonl").exists()  # copied into the store

    # no manifest for a different date → no-op (0), so the caller can fall back to hand-entry
    assert stim.apply_session_manifest(cm, src, date="1999-01-01") == 0
