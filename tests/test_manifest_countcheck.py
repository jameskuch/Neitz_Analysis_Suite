"""apply_session_manifest: refuse to silently mispair when manifest rows != recordings."""
import json
import pytest
from neitz.io import stim


class MockCM:
    """Minimal duck-typed CellManifest (needs .data['recordings'], .set_stimulus, .dir).

    `rec_times` (parallel to `rec_ids`) seeds each recording's `recorded_at`, which
    `apply_session_manifest`'s timestamp cross-check reads without touching a real .abf.
    """
    def __init__(self, tmp, rec_ids, ref_ids=(), rec_times=None):
        self.dir = tmp / "cell"; self.dir.mkdir(parents=True, exist_ok=True)
        rec_times = rec_times if rec_times is not None else [None] * len(rec_ids)
        recs = []
        for r, t in zip(rec_ids, rec_times):
            d = {"id": r, "kind": "recording"}
            if t is not None:
                d["recorded_at"] = t
            recs.append(d)
        recs += [{"id": r, "kind": "reference"} for r in ref_ids]
        self.data = {"recordings": recs}
        self.set = {}

    def set_stimulus(self, rec_id, stim_type, params, source="user"):
        self.set[rec_id] = {"type": stim_type, "params": params, "source": source}


def _manifest(dirp, n_rows, date="2026_07_04", timestamps=None):
    p = dirp / f"{date}_stim_manifest.jsonl"
    lines = []
    for i in range(n_rows):
        row = {"stimulus": "g", "stim_type": "gaussian_noise", "seed": 2 + i, "stim_signature": "A"}
        if timestamps is not None:
            row["timestamp"] = timestamps[i]
        lines.append(json.dumps(row))
    p.write_text("\n".join(lines) + "\n")
    return p


# manifest rows written ~4 s after each acquisition start; trials spaced 17 s apart.
_MTS = ["2026-07-04T09:00:04", "2026-07-04T09:00:21", "2026-07-04T09:00:38"]
_ATS_OK      = ["2026-07-04T09:00:00", "2026-07-04T09:00:17", "2026-07-04T09:00:34"]  # aligned
_ATS_SHIFTED = ["2026-07-04T09:00:00", "2026-07-04T08:00:00", "2026-07-04T09:00:17"]  # r2 = stray


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


def test_reference_recordings_are_excluded(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _manifest(src, 3)                                            # 3 stimulus rows
    cm = MockCM(tmp_path, ["r1", "r2", "r3"], ref_ids=["ref1"])  # + 1 reference recording
    assert stim.apply_session_manifest(cm, src, date="2026-07-04") == 3   # no false mismatch
    assert set(cm.set) == {"r1", "r2", "r3"}                    # the reference is NOT tagged


def test_mismatch_still_fires_ignoring_references(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _manifest(src, 3)                                            # 3 rows
    cm = MockCM(tmp_path, ["r1", "r2"], ref_ids=["ref1", "ref2"])  # only 2 stimulus recs
    with pytest.raises(ValueError, match="count mismatch"):
        stim.apply_session_manifest(cm, src, date="2026-07-04")


# ---- timestamp order cross-check ----

def test_time_aligned_tags_all(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _manifest(src, 3, timestamps=_MTS)
    cm = MockCM(tmp_path, ["r1", "r2", "r3"], rec_times=_ATS_OK)   # constant ~4 s offset
    assert stim.apply_session_manifest(cm, src, date="2026-07-04") == 3
    assert set(cm.set) == {"r1", "r2", "r3"}


def test_time_shifted_refuses(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _manifest(src, 3, timestamps=_MTS)
    cm = MockCM(tmp_path, ["r1", "r2", "r3"], rec_times=_ATS_SHIFTED)  # r2 is a stray .abf
    with pytest.raises(ValueError, match="do not line up"):
        stim.apply_session_manifest(cm, src, date="2026-07-04")
    assert cm.set == {}                                           # nothing tagged


def test_time_mismatch_non_strict_warns(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _manifest(src, 3, timestamps=_MTS)
    cm = MockCM(tmp_path, ["r1", "r2", "r3"], rec_times=_ATS_SHIFTED)
    with pytest.warns(UserWarning, match="do not line up"):
        n = stim.apply_session_manifest(cm, src, date="2026-07-04", strict=False)
    assert n == 0 and cm.set == {}


def test_time_tol_override_allows_shift(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _manifest(src, 3, timestamps=_MTS)
    cm = MockCM(tmp_path, ["r1", "r2", "r3"], rec_times=_ATS_SHIFTED)
    # a huge absolute tolerance disables the order check
    assert stim.apply_session_manifest(cm, src, date="2026-07-04", time_tol_s=100000) == 3


def test_check_time_false_disables(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _manifest(src, 3, timestamps=_MTS)
    cm = MockCM(tmp_path, ["r1", "r2", "r3"], rec_times=_ATS_SHIFTED)
    assert stim.apply_session_manifest(cm, src, date="2026-07-04", check_time=False) == 3


def test_time_check_skips_without_timestamps(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _manifest(src, 3)                                             # rows carry no timestamp
    cm = MockCM(tmp_path, ["r1", "r2", "r3"], rec_times=_ATS_OK)  # recordings ARE timed
    assert stim.apply_session_manifest(cm, src, date="2026-07-04") == 3   # can't verify -> tags


def test_time_check_skips_when_too_close(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    # trials only 4 s apart -> below the min spacing, so the clock can't resolve order
    mts = ["2026-07-04T09:00:04", "2026-07-04T09:00:08", "2026-07-04T09:00:12"]
    ats = ["2026-07-04T09:00:00", "2026-07-04T08:00:00", "2026-07-04T09:00:08"]  # would-be shift
    _manifest(src, 3, timestamps=mts)
    cm = MockCM(tmp_path, ["r1", "r2", "r3"], rec_times=ats)
    assert stim.apply_session_manifest(cm, src, date="2026-07-04") == 3   # skipped -> tags


def test_recorded_datetime_none_on_bad_path(tmp_path):
    from neitz.io.abf import recorded_datetime
    assert recorded_datetime(tmp_path / "does_not_exist.abf") is None
