"""apply_session_manifest: refuse to silently mispair when manifest rows != recordings."""
import json
import pytest
from neitz.io import stim


class MockCM:
    """Minimal duck-typed CellManifest (needs .data['recordings'], .set_stimulus, .dir)."""
    def __init__(self, tmp, rec_ids):
        self.dir = tmp / "cell"; self.dir.mkdir(parents=True, exist_ok=True)
        self.data = {"recordings": [{"id": r} for r in rec_ids]}
        self.set = {}

    def set_stimulus(self, rec_id, stim_type, params, source="user"):
        self.set[rec_id] = {"type": stim_type, "params": params, "source": source}


def _manifest(dirp, n_rows, date="2026_07_04"):
    p = dirp / f"{date}_stim_manifest.jsonl"
    p.write_text("".join(
        json.dumps({"stimulus": "g", "stim_type": "gaussian_noise",
                    "seed": 2 + i, "stim_signature": "A"}) + "\n"
        for i in range(n_rows)))
    return p


def test_matched_counts_tag_all(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _manifest(src, 3)
    cm = MockCM(tmp_path, ["r1", "r2", "r3"])
    assert stim.apply_session_manifest(cm, src, date="2026-07-04") == 3
    assert len(cm.set) == 3


def test_mismatch_raises_by_default(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _manifest(src, 3)                       # 3 manifest rows
    cm = MockCM(tmp_path, ["r1", "r2"])     # but 2 recordings
    with pytest.raises(ValueError, match="count mismatch"):
        stim.apply_session_manifest(cm, src, date="2026-07-04")
    assert cm.set == {}                     # nothing tagged (no partial mispairing)


def test_mismatch_non_strict_warns_and_noops(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _manifest(src, 3)
    cm = MockCM(tmp_path, ["r1", "r2"])
    with pytest.warns(UserWarning, match="count mismatch"):
        n = stim.apply_session_manifest(cm, src, date="2026-07-04", strict=False)
    assert n == 0 and cm.set == {}


def test_no_manifest_is_noop(tmp_path):
    src = tmp_path / "src"; src.mkdir()     # no manifest file present
    cm = MockCM(tmp_path, ["r1"])
    assert stim.apply_session_manifest(cm, src, date="2026-07-04") == 0
