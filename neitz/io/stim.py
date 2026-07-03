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
order); ``timestamp`` is available as a cross-check. The seed is the source of truth —
:func:`noise_from_record` regenerates the exact linear stimulus via
:mod:`neitz.stimulus.reproduce`; :func:`sent_codes_from_record` recovers the 8-bit
codes that were actually displayed.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from ..stimulus.reproduce import reproduce_noise, gamma_adjust, expand_to_frames

MANIFEST_GLOB = "*_stim_manifest.jsonl"
NOISE_TYPES = ("gaussian_noise", "checkerboard")   # records that carry reproducible noise


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
