"""
neitz.io.stim — read the Stage/LightCrafter stimulus session manifest and rebuild
the stimulus from it (no per-frame values are stored anywhere).

The MATLAB stimuli write ONE JSON-Lines file per day,
``YYYY_MM_DD_stim_manifest.jsonl`` (see June2026StageMATLAB/writeStimManifest.m),
with one record per trial, e.g.::

    {"stimulus": "AASeededGaussianCheckerboardSConeIsoStimFinal",
     "stim_type": "checkerboard", "cone_isolation": "S",
     "seed": 2, "mu": 0.5, "sigma": 0.3,
     "checks_x": 40, "checks_y": 32, "n_updates": 75,
     "update_every_n_frames": 8, "refresh_rate_hz": 60, "stim_frames": 600,
     "gamma": 2.2056, "noise_method": "mt19937ar+invCDF", "fill_order": "F",
     "timestamp": "2026-07-03T09:14:02"}

Trials pair to Clampex ``.abf`` recordings BY ORDER (manifest row order == acquisition
order); :func:`apply_session_manifest` refuses to pair when the row and (non-reference)
recording counts disagree, AND cross-checks each row's ``timestamp`` against the paired
``.abf``'s recorded time so an equal-count-but-shifted pairing is caught too. The seed is
the source of truth — :func:`noise_from_record` regenerates the exact linear stimulus via
:mod:`neitz.stimulus.reproduce`; :func:`sent_codes_from_record` recovers the 8-bit
codes that were actually displayed.
"""
from __future__ import annotations
import json
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np

from ..stimulus.reproduce import reproduce_noise, gamma_adjust, expand_to_frames

MANIFEST_GLOB = "*_stim_manifest.jsonl"
NOISE_TYPES = ("gaussian_noise", "checkerboard")   # records that carry reproducible noise
_TIME_MIN_SPACING_S = 8.0    # below this inter-trial spacing, the clock can't resolve order
_TIME_TOL_FRACTION = 0.4     # a real slip moves a pair ~one full spacing; flag at 0.4 of it


def load_session_manifest(path) -> list[dict]:
    """Read a ``*_stim_manifest.jsonl`` file -> list of per-trial dicts, in order."""
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def find_session_manifest(directory, date=None):
    """Locate the stimulus manifest inside `directory`.

    If `date` (``YYYY-MM-DD`` or ``YYYY_MM_DD``) is given, require the exact
    ``<date>_stim_manifest.jsonl`` (returns ``None`` if that date's manifest is absent —
    it will not fall back to a different date). With no `date`, returns the sole
    ``*_stim_manifest.jsonl`` in the directory (the earliest if several). Returns a
    :class:`~pathlib.Path` or ``None``.
    """
    directory = Path(directory)
    if date:
        cand = directory / f"{str(date).replace('-', '_')}_stim_manifest.jsonl"
        return cand if cand.exists() else None
    hits = sorted(directory.glob(MANIFEST_GLOB))
    return hits[0] if hits else None


def stimulus_metadata(record) -> tuple[str, dict]:
    """Split a manifest record into ``(stim_type, params)`` for
    ``CellManifest.set_stimulus(rec_id, stim_type, params, source="stim-manifest")``.

    ``stim_type`` is the record's ``stim_type`` (``gaussian_noise`` | ``checkerboard`` |
    ``sq_wave`` | ``jitter``); ``params`` is everything else — the fields the analysis
    and the Data Explorer may display or sort on (seed, cone_isolation, checks_x/y, ...).
    """
    params = dict(record)
    stim_type = params.pop("stim_type", None)
    return stim_type, params


def pair_by_order(manifest_rows, recording_ids) -> list[tuple]:
    """Zip manifest trial rows to recording ids IN ORDER (the session-manifest
    association). Returns ``[(recording_id, record), ...]``; the excess of whichever
    list is longer is dropped, so check ``len`` beforehand if a mismatch matters.
    """
    return list(zip(recording_ids, manifest_rows))


def _parse_dt(value):
    """Parse an ISO-ish datetime string (or pass a ``datetime`` through); ``None`` on failure."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    txt = str(value).strip().replace(" ", "T")
    for candidate in (txt, txt[:19]):        # full, then trimmed to seconds (drops frac/tz)
        try:
            return datetime.fromisoformat(candidate).replace(tzinfo=None)
        except ValueError:
            continue
    return None


def _recording_datetime(cm, rec):
    """Best-effort acquisition ``datetime`` for a recording, or ``None``. Never raises.

    Prefers a datetime already stored on the recording dict (a seam for future callers /
    tests); otherwise reads the ``.abf`` header cheaply (the raw copy under ``cm.dir``, then
    the original ``source`` path).
    """
    for key in ("recorded_at", "abf_datetime"):
        dt = _parse_dt(rec.get(key))
        if dt is not None:
            return dt
    try:
        from .abf import recorded_datetime
    except Exception:
        return None
    candidates = []
    if rec.get("file"):
        try:
            candidates.append(Path(cm.dir) / rec["file"])
        except Exception:
            pass
    if rec.get("source"):
        candidates.append(Path(rec["source"]))
    for path in candidates:
        try:
            if path.exists():
                dt = recorded_datetime(path)
                if dt is not None:
                    return dt.replace(tzinfo=None)
        except Exception:
            pass
    return None


def _timecheck(rows, recs, cm, time_tol_s):
    """Cross-check that the manifest rows line up, IN ORDER, with the ``.abf`` recorded times.

    Returns ``None`` when the timeline is consistent OR cannot be verified (missing
    timestamps on either side, or recordings spaced too close to resolve order from the
    clock); returns a message string when the timeline is inconsistent (the by-order
    pairing is probably shifted).

    Both clocks are the same acquisition PC, and each row is written a few seconds AFTER
    Clampex starts that recording, so a correct pairing has a small, ~constant offset
    ``manifest_time - abf_time``. A one-trial slip pushes the mispaired offset off by ~one
    inter-trial interval, which stands out from the median offset. Note: a *perfectly
    uniform* slide (e.g. a missing first recording plus a stray last one whose times happen
    to align) keeps the offset constant and can still slip through — only deterministic
    filename read-back closes that fully.
    """
    m = [_parse_dt(r.get("timestamp")) for r in rows]
    a = [_recording_datetime(cm, rec) for rec in recs]
    idx = [i for i in range(len(m)) if m[i] is not None and a[i] is not None]
    if len(idx) < 2:
        return None                                          # not enough timed pairs to verify
    deltas = [(m[i] - a[i]).total_seconds() for i in idx]
    offset = statistics.median(deltas)
    worst = max(abs(d - offset) for d in deltas)
    if time_tol_s is not None:
        tol = float(time_tol_s)
    else:
        mt = sorted(m[i] for i in idx)                       # manifest timeline is the clean one
        gaps = [(mt[j + 1] - mt[j]).total_seconds() for j in range(len(mt) - 1)]
        gaps = [g for g in gaps if g > 0]
        if not gaps:
            return None
        spacing = statistics.median(gaps)
        if spacing < _TIME_MIN_SPACING_S:
            return None                                      # too close to tell order from clock
        tol = _TIME_TOL_FRACTION * spacing
    if worst > tol:
        return (f"stim-manifest timestamps do not line up with the .abf recorded times: the "
                f"worst-matched trial is {worst:.0f}s off the ~{offset:.0f}s median offset "
                f"(tolerance {tol:.0f}s). The row<->recording order is probably shifted (an "
                f"aborted trial plus a stray .abf?). Refusing to auto-pair by order alone.")
    return None


def apply_session_manifest(cm, source_dir, *, date=None, copy_into_store=True, strict=True,
                           check_time=True, time_tol_s=None) -> int:
    """Auto-fill a cell's recordings' stimulus metadata from the day's stim manifest.

    Finds ``<date>_stim_manifest.jsonl`` in `source_dir`, pairs its rows to the cell's
    recordings BY ORDER (the session-manifest association), and calls
    ``cm.set_stimulus(rec_id, stim_type, params, source="stim-manifest")`` for each.
    When `copy_into_store` (default) the manifest is copied into ``cm.dir`` beside ``raw/``
    so the stored cell is self-contained. Returns the number of recordings tagged — ``0``
    (a no-op) when no manifest is found, so callers can fall back to hand-entered stimulus.

    Only ``kind == "recording"`` recordings are paired; ``reference``/baseline recordings
    have no manifest row and are skipped (matching ``run.py``'s trial collection).

    By-order pairing is only valid with exactly one manifest row per stimulus recording, in
    the same order. Two guards enforce that:
      * cardinality — if the counts differ (a session that crossed midnight into a second
        manifest file, an aborted trial, or an orphan row) this raises ``ValueError``
        (``strict``, default) or warns + returns 0 (``strict=False``).
      * order (when ``check_time``) — each row's ``timestamp`` is compared to its paired
        ``.abf``'s recorded time; since both clocks are the same machine and the row is
        written seconds after Clampex starts the recording, a correct pairing has a small
        ~constant offset, so an equal-count-but-shifted pairing (aborted trial + stray
        file) stands out and is likewise refused. The check self-skips when it cannot verify
        (missing timestamps, or recordings spaced too close to tell order from the clock),
        so it never false-alarms on older data. ``time_tol_s`` pins an absolute residual
        tolerance in seconds; by default it adapts to a fraction of the inter-trial spacing.

    ``strict=False`` warns and returns 0 on either failure so the caller can fall back to
    hand-entry.

    `cm` is duck-typed (needs ``.data["recordings"]``, ``.set_stimulus``, ``.dir``) so this
    stays free of a `dataio` import.
    """
    mf = find_session_manifest(source_dir, date=date)
    if not mf:
        return 0
    rows = load_session_manifest(mf)
    recs = [r for r in cm.data.get("recordings", [])
            if r.get("kind", "recording") == "recording"]
    rec_ids = [r["id"] for r in recs]
    if len(rows) != len(rec_ids):
        msg = (f"stim-manifest / recording count mismatch: {len(rows)} rows in {mf.name} vs "
               f"{len(rec_ids)} stimulus recordings. By-order pairing would silently mislabel "
               f"trials (session crossed midnight? aborted/orphan trial?). Refusing to auto-pair.")
        if strict:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg)
        return 0
    if check_time:
        problem = _timecheck(rows, recs, cm, time_tol_s)
        if problem is not None:
            if strict:
                raise ValueError(problem)
            import warnings
            warnings.warn(problem)
            return 0
    n = 0
    for rec_id, rec in pair_by_order(rows, rec_ids):
        stim_type, params = stimulus_metadata(rec)
        cm.set_stimulus(rec_id, stim_type, params, source="stim-manifest")
        n += 1
    if copy_into_store:
        import shutil
        try:
            shutil.copy2(mf, Path(cm.dir) / mf.name)
        except Exception:
            pass
    return n


def epoch_groups(recordings) -> list[dict]:
    """Group a cell's recordings (in acquisition order) into runs of the SAME stimulus.

    Consecutive recordings that share ``stimulus.params.stim_signature`` form one group —
    N epochs of one stimulus (e.g. a 2 Hz square wave presented 3x). A recording with no
    signature (imported before the manifest existed, or hand-entered) is its own group of
    one and never merges. The signature is stamped on the rig (writeStimManifest.m) and
    lands in ``stimulus.params.stim_signature`` via :func:`apply_session_manifest`.

    `recordings` is the manifest's ``recordings`` list (dicts with a ``stimulus`` field,
    in acquisition order). Returns a list of groups, each::

        {"stim_signature", "n_epochs", "stim_type", "cone_isolation",
         "recording_ids", "recordings"}
    """
    def _sig(rec):
        stim = rec.get("stimulus") or {}
        return (stim.get("params") or {}).get("stim_signature")

    groups = []
    for rec in recordings:
        sig = _sig(rec)
        if sig is not None and groups and groups[-1]["stim_signature"] == sig:
            groups[-1]["recordings"].append(rec)
        else:
            groups.append({"stim_signature": sig, "recordings": [rec]})

    for g in groups:
        recs = g["recordings"]
        stim = recs[0].get("stimulus") or {}
        g["n_epochs"] = len(recs)
        g["recording_ids"] = [r.get("id") for r in recs]
        g["stim_type"] = stim.get("type")
        g["cone_isolation"] = (stim.get("params") or {}).get("cone_isolation")
    return groups


def noise_from_record(record, *, per_frame=False) -> np.ndarray:
    """Regenerate the exact LINEAR noise tensor `v` in [0, 1] for a manifest record.

    Shape ``(checks_y, checks_x, n_updates)`` — this is the stimulus to reverse-
    correlate against (the projector is gamma-corrected, so emitted light is linear in
    `v`). With ``per_frame=True`` it is expanded to ``(checks_y, checks_x, stim_frames)``
    by holding each update. Raises ``ValueError`` for ``sq_wave`` / ``jitter`` records,
    which carry no reproducible noise field.
    """
    if record.get("stim_type") not in NOISE_TYPES:
        raise ValueError(
            f"stim_type={record.get('stim_type')!r} has no noise to reproduce "
            f"(only {NOISE_TYPES} do)")
    v = reproduce_noise(record["seed"], record["checks_y"], record["checks_x"],
                        record["n_updates"], record.get("mu", 0.5), record.get("sigma", 0.3))
    if per_frame:
        v = expand_to_frames(v, record["update_every_n_frames"], record.get("stim_frames"))
    return v


def sent_codes_from_record(record, *, per_frame=False) -> np.ndarray:
    """The 8-bit RGB codes actually sent to the projector (for display verification).

    Greyscale -> ``R=G=B=round(255*gamma_adjust(v))``; S-cone-iso -> ``R`` from ``v``,
    ``G`` from ``1-v``, ``B=0``. Returns an int array of shape ``(..., 3)``.
    """
    v = noise_from_record(record, per_frame=per_frame)
    if str(record.get("cone_isolation")).upper() == "S":
        r = np.round(255 * gamma_adjust(v))
        g = np.round(255 * gamma_adjust(1.0 - v))
        b = np.zeros_like(r)
    else:
        r = g = b = np.round(255 * gamma_adjust(v))
    return np.stack([r, g, b], axis=-1).astype(int)
