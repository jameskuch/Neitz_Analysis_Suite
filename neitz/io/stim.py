"""
neitz.io.stim — read the Stage/LightCrafter *nested* stimulus session manifest
(day -> cells -> blocks -> epochs) and rebuild the stimulus from seeds.

The MATLAB rig writes ONE JSON document per day, ``YYYY_MM_DD_stim_manifest.json``
(``format: neitz-stim-manifest/2``; see June2026StageMATLAB/writeStimManifest.m). It
mirrors how the session was actually run::

    { "format": "neitz-stim-manifest/2", "date": "2026-07-16", "rig": {...},
      "cells": [
        { "cell_name": "c1",
          "blocks": [                                      # a "block" = one PROTOCOL
            { "block_index", "label", "stim_type", "cone_isolation", "stim_signature",
              "leds": {...}, "params": {...constant across the block's epochs...},
              "epochs": [ {"epoch", "seed", "timestamp", "frame_sync": {...}} ] } ] } ],
      "discarded": [ {"timestamp", ...} ] }               # epochs the operator Discarded at the rig

Epoch document order == acquisition order == the Clampex ``.abf`` order. But a run can
ABORT an epoch before Clampex records it (a false start), the operator can DISCARD a
finished block, and a stray manual sweep can leave an ``.abf`` with no epoch — so there is
NOT always one ``.abf`` per epoch. We therefore pair ABFs to epochs by a monotonic
**timestamp alignment** (:func:`align_abfs_to_epochs`), not a blind zip-by-order, and
surface whatever is left over instead of mislabeling it. The seed is the source of truth —
:func:`noise_from_record` regenerates the exact linear stimulus via
:mod:`neitz.stimulus.reproduce`.

The legacy flat ``*_stim_manifest.jsonl`` per-trial log is no longer written by the rig
(``writeStimManifest.m`` now writes only the nested ``.json``) nor read here.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ..stimulus.reproduce import reproduce_noise, gamma_adjust, expand_to_frames

MANIFEST_GLOB = "*_stim_manifest.json"                 # nested (neitz-stim-manifest/2)
NOISE_TYPES = ("gaussian_noise", "checkerboard")       # records that carry reproducible noise


# ======================= find / load the nested manifest =======================
def find_manifest(directory, date=None):
    """Locate the nested stim manifest (``*_stim_manifest.json``) in `directory`.

    With `date` (``YYYY-MM-DD`` or ``YYYY_MM_DD``) require exactly
    ``<date>_stim_manifest.json`` (returns ``None`` if that date's manifest is absent — it
    will not fall back to a different date). With no `date`, returns the sole
    ``*_stim_manifest.json`` (the earliest if several). Returns a :class:`~pathlib.Path` or
    ``None``.
    """
    directory = Path(directory)
    if date:
        cand = directory / f"{str(date).replace('-', '_')}_stim_manifest.json"
        return cand if cand.exists() else None
    hits = sorted(directory.glob(MANIFEST_GLOB))
    return hits[0] if hits else None


def load_manifest(path) -> dict:
    """Read a nested ``*_stim_manifest.json`` -> the tree dict (``neitz-stim-manifest/2``)."""
    return json.loads(Path(path).read_text())


def _as_list(x):
    """Normalize a manifest collection to a list (a lone dict -> [dict], None -> [])."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def stimulus_metadata(record) -> tuple[str, dict]:
    """Split a flat record into ``(stim_type, params)`` — the shape
    ``CellManifest.set_stimulus(rec_id, stim_type, params, ...)`` wants. ``params`` is the
    record minus ``stim_type`` (seed, cone_isolation, checks_x/y, stim_signature, …)."""
    params = dict(record)
    stim_type = params.pop("stim_type", None)
    return stim_type, params


# ============================ flatten epochs ============================
def _epoch_aborted(epoch, block_params) -> bool:
    """Heuristic: did this presentation run far short of its expected duration (a false
    start that Clampex probably never recorded)? A normal epoch runs ``stim_frames /
    refresh`` seconds with drift ~ -10 frames; an aborted one is ~1 s with drift ~ -540.
    Used only as a *hint* to the timestamp alignment (it makes skipping the epoch free);
    the actual pairing is decided by the timestamps, so a wrong flag can't mislabel data."""
    fs = epoch.get("frame_sync") or {}
    wall = fs.get("client_wall_s")
    drift = fs.get("drift_frames")
    frames = block_params.get("stim_frames")
    refresh = block_params.get("refresh_rate_hz") or 60
    expected = (frames / refresh) if frames else None
    if wall is not None and expected and wall < 0.5 * expected:
        return True
    if drift is not None and abs(drift) > 100:              # normal |drift| ~ 10 frames
        return True
    return False


def iter_epochs(tree) -> list[dict]:
    """Flatten a nested manifest tree into per-epoch context dicts, in acquisition order.

    Each dict merges its block context onto the epoch so it is self-describing::

        {cell_name, block_index, block_label, stim_type, cone_isolation, stim_signature,
         seed, epoch, timestamp, frame_sync, aborted, params}

    where ``params`` is the block's constant params + this epoch's ``seed`` (the exact dict
    to hand to :func:`noise_from_record` / ``set_stimulus``). ``aborted`` is a derived hint
    (see :func:`_epoch_aborted`).
    """
    out = []
    for cell in _as_list(tree.get("cells")):
        cname = cell.get("cell_name")
        for bi, block in enumerate(_as_list(cell.get("blocks"))):
            bparams = dict(block.get("params") or {})
            for ep in _as_list(block.get("epochs")):
                params = dict(bparams)
                if ep.get("seed") is not None:
                    params["seed"] = ep["seed"]
                if block.get("cone_isolation") is not None:
                    params.setdefault("cone_isolation", block["cone_isolation"])
                if block.get("stim_signature") is not None:
                    params.setdefault("stim_signature", block["stim_signature"])
                if block.get("stimulus") is not None:
                    params.setdefault("stimulus", block["stimulus"])
                out.append({
                    "cell_name": cname,
                    "block_index": block.get("block_index", bi + 1),
                    "block_label": block.get("label") or "",
                    "stim_type": block.get("stim_type"),
                    "cone_isolation": block.get("cone_isolation"),
                    "stim_signature": block.get("stim_signature"),
                    "seed": ep.get("seed"),
                    "epoch": ep.get("epoch"),
                    "timestamp": ep.get("timestamp"),
                    "frame_sync": ep.get("frame_sync") or {},
                    "aborted": _epoch_aborted(ep, bparams),
                    "params": params,
                })
    return out


# ============================ ABF <-> epoch alignment ============================
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


def _secs(t):
    """A datetime-or-string -> float POSIX seconds, or ``None``."""
    dt = t if isinstance(t, datetime) else _parse_dt(t)
    return dt.timestamp() if dt is not None else None


def align_abfs_to_epochs(abf_times, epoch_times, *, epoch_skippable=None,
                         match_cap_s=90.0, skip_epoch_s=25.0, skip_abf_s=45.0):
    """Monotonic timestamp alignment of ABFs to stimulus epochs (both in acquisition order).

    Returns ``(abf_epoch, epoch_abf)``: ``abf_epoch[i]`` is the epoch index paired with ABF
    ``i`` (or ``None``); ``epoch_abf[j]`` the ABF paired with epoch ``j`` (or ``None``).

    Both sequences are ordered, but an epoch can have NO abf (a false start Clampex never
    recorded, or a Discarded block) and an abf can have NO epoch (a stray manual sweep). So
    this is a global sequence alignment (Needleman–Wunsch): a match costs ``|Δt|`` (capped
    at `match_cap_s`), leaving an epoch unmatched costs `skip_epoch_s` (or 0 when
    ``epoch_skippable[j]`` — an aborted/discarded epoch), and leaving an abf unmatched costs
    `skip_abf_s`. The costs are in seconds, so an epoch farther than ~`skip_epoch_s` from
    every abf is skipped rather than force-matched. Ties resolve toward matching.

    `abf_times` / `epoch_times` are datetimes or ISO strings (``None`` allowed).
    """
    na, ne = len(abf_times), len(epoch_times)
    if epoch_skippable is None:
        epoch_skippable = [False] * ne
    ta = [_secs(t) for t in abf_times]
    te = [_secs(t) for t in epoch_times]

    def mcost(i, j):
        if ta[i] is None or te[j] is None:
            return match_cap_s * 0.5           # unknown time: allowed but not preferred
        return min(abs(ta[i] - te[j]), match_cap_s)

    def skip_ep(j):
        return 0.0 if epoch_skippable[j] else skip_epoch_s

    dp = [[0.0] * (ne + 1) for _ in range(na + 1)]
    bk = [[""] * (ne + 1) for _ in range(na + 1)]
    for j in range(1, ne + 1):
        dp[0][j] = dp[0][j - 1] + skip_ep(j - 1); bk[0][j] = "e"
    for i in range(1, na + 1):
        dp[i][0] = dp[i - 1][0] + skip_abf_s; bk[i][0] = "a"
    for i in range(1, na + 1):
        for j in range(1, ne + 1):
            best, mv = dp[i - 1][j - 1] + mcost(i - 1, j - 1), "m"
            c = dp[i - 1][j] + skip_abf_s
            if c < best:
                best, mv = c, "a"
            c = dp[i][j - 1] + skip_ep(j - 1)
            if c < best:
                best, mv = c, "e"
            dp[i][j], bk[i][j] = best, mv

    abf_epoch = [None] * na
    epoch_abf = [None] * ne
    i, j = na, ne
    while i > 0 or j > 0:
        mv = bk[i][j]
        if mv == "m":
            abf_epoch[i - 1] = j - 1
            epoch_abf[j - 1] = i - 1
            i -= 1
            j -= 1
        elif mv == "a":
            i -= 1
        else:
            j -= 1
    return abf_epoch, epoch_abf


def _abf_dt(path):
    """Best-effort .abf acquisition datetime (naive), or ``None`` — never raises."""
    try:
        from .abf import recorded_datetime
        dt = recorded_datetime(path)
        return dt.replace(tzinfo=None) if dt is not None else None
    except Exception:
        return None


# ============================ build the import plan ============================
def build_import_plan(source_dir, date, abf_paths, *, abf_datetimes=None):
    """Plan a day's import against the nested stim manifest — the day -> cell -> protocol ->
    epoch tree, with each epoch's ABF resolved by :func:`align_abfs_to_epochs`.

    Returns a dict::

        {"manifest": Path|None, "date": str|None,
         "cells": [ {"cell_name",
                     "protocols": [ {"index","label","stim_type","cone_isolation",
                                     "stim_signature",
                                     "epochs": [ {"abf": path|None, "epoch","seed",
                                                  "timestamp","stim_type","params",
                                                  "aborted"} ]} ]} ],
         "unmatched_abfs": [path,...],     # abfs with no epoch -> import "unsorted"
         "discarded_abfs": [path,...],     # abfs whose epoch was Discarded at the rig -> skip
         "warnings": [str,...]}

    When no nested manifest is present the abfs all fall to ``unmatched_abfs`` with a
    warning, so the caller can still import them (unsorted, no stimulus metadata).
    """
    source_dir = Path(source_dir)
    abf_paths = list(abf_paths)
    mf = find_manifest(source_dir, date=date)
    plan = {"manifest": mf, "date": str(date) if date else None,
            "cells": [], "unmatched_abfs": [], "discarded_abfs": [], "warnings": []}
    if abf_datetimes is None:
        abf_datetimes = [_abf_dt(p) for p in abf_paths]
    else:                                                  # accept ISO strings too -> datetimes
        abf_datetimes = [t if isinstance(t, datetime) else _parse_dt(t) for t in abf_datetimes]

    if mf is None:
        plan["unmatched_abfs"] = list(abf_paths)
        want = f"{str(date).replace('-', '_')}_stim_manifest.json" if date else "*_stim_manifest.json"
        plan["warnings"].append(
            f"no nested stim manifest ({want}) in {source_dir}: imported "
            f"{len(abf_paths)} recording(s) unsorted, with no stimulus metadata.")
        return plan

    tree = load_manifest(mf)
    epochs = iter_epochs(tree)
    discarded_dts = [_parse_dt(d.get("timestamp")) for d in _as_list(tree.get("discarded"))]
    skippable = [e["aborted"] for e in epochs]
    abf_epoch, epoch_abf = align_abfs_to_epochs(
        abf_datetimes, [e["timestamp"] for e in epochs], epoch_skippable=skippable)

    for j, e in enumerate(epochs):
        i = epoch_abf[j]
        e["abf"] = abf_paths[i] if i is not None else None

    plan["cells"] = _epochs_to_cells(epochs)

    for i, ep_idx in enumerate(abf_epoch):                  # abfs with no epoch
        if ep_idx is not None:
            continue
        at = abf_datetimes[i]
        if at is not None and any(dd is not None and abs((at - dd).total_seconds()) < 30
                                  for dd in discarded_dts):
            plan["discarded_abfs"].append(abf_paths[i])
        else:
            plan["unmatched_abfs"].append(abf_paths[i])

    lost = [e for e in epochs if e["abf"] is None and not e["aborted"]]
    if lost:
        plan["warnings"].append(
            f"{len(lost)} recorded-looking epoch(s) had no matching .abf "
            f"(e.g. {lost[0]['cell_name']} '{lost[0]['block_label']}' @ {lost[0]['timestamp']}).")
    if plan["unmatched_abfs"]:
        plan["warnings"].append(
            f"{len(plan['unmatched_abfs'])} .abf(s) matched no stim epoch -> imported to an "
            f"'(unsorted)' cell for review.")
    return plan


def _epochs_to_cells(epochs):
    """Regroup the flat epoch list back into cells -> protocols (consecutive runs)."""
    cells = []
    for e in epochs:
        if not cells or cells[-1]["cell_name"] != e["cell_name"]:
            cells.append({"cell_name": e["cell_name"], "protocols": []})
        protos = cells[-1]["protocols"]
        pkey = (e["block_index"], e["block_label"], e["stim_signature"])
        if not protos or protos[-1]["_key"] != pkey:
            protos.append({"_key": pkey, "index": e["block_index"], "label": e["block_label"],
                           "stim_type": e["stim_type"], "cone_isolation": e["cone_isolation"],
                           "stim_signature": e["stim_signature"], "epochs": []})
        protos[-1]["epochs"].append(e)
    for c in cells:
        for p in c["protocols"]:
            p.pop("_key", None)
    return cells


# ============================ protocol / epoch grouping ============================
def epoch_groups(recordings) -> list[dict]:
    """Group a cell's recordings (acquisition order) into PROTOCOL runs = N epochs of one
    stimulus.

    Prefers an explicit ``recording['protocol']`` (``index`` + ``label``, stamped at import
    or re-categorization); falls back to consecutive ``stimulus.params.stim_signature`` runs
    for older / hand-entered cells with no protocol assignment. A recording with neither is
    its own group of one and never merges. Each group::

        {"stim_signature", "protocol_label", "protocol_index", "n_epochs", "stim_type",
         "cone_isolation", "recording_ids", "recordings"}
    """
    def _key(rec):
        pr = rec.get("protocol") or {}
        if pr.get("index") is not None or pr.get("label"):
            return ("p", pr.get("index"), pr.get("label"))
        sig = ((rec.get("stimulus") or {}).get("params") or {}).get("stim_signature")
        return ("s", sig) if sig is not None else None

    groups = []
    for rec in recordings:
        key = _key(rec)
        if key is not None and groups and groups[-1]["_key"] == key:
            groups[-1]["recordings"].append(rec)
        else:
            groups.append({"_key": key, "recordings": [rec]})

    for g in groups:
        recs = g["recordings"]
        stim = recs[0].get("stimulus") or {}
        pr = recs[0].get("protocol") or {}
        g["n_epochs"] = len(recs)
        g["recording_ids"] = [r.get("id") for r in recs]
        g["stim_type"] = stim.get("type")
        g["cone_isolation"] = (stim.get("params") or {}).get("cone_isolation")
        g["stim_signature"] = (stim.get("params") or {}).get("stim_signature")
        g["protocol_label"] = pr.get("label")
        g["protocol_index"] = pr.get("index")
        g.pop("_key", None)
    return groups


# ============================ stimulus reproduction ============================
def noise_from_record(record, *, per_frame=False) -> np.ndarray:
    """Regenerate the exact LINEAR noise tensor `v` in [0, 1] for a manifest record.

    Shape ``(checks_y, checks_x, n_updates)`` — the stimulus to reverse-correlate against
    (the projector is gamma-corrected, so emitted light is linear in `v`). With
    ``per_frame=True`` it is expanded to ``(checks_y, checks_x, stim_frames)`` by holding
    each update. Raises ``ValueError`` for ``sq_wave`` / ``jitter`` records (no noise field).

    `record` is a flat dict (an :func:`iter_epochs` entry's ``params`` with ``stim_type``,
    or any dict carrying ``stim_type`` + seed/checks/n_updates).
    """
    stype = record.get("stim_type")
    if stype is None and "type" in record:                 # tolerate {"type": ...} shape too
        stype = record.get("type")
    if stype not in NOISE_TYPES:
        raise ValueError(
            f"stim_type={stype!r} has no noise to reproduce (only {NOISE_TYPES} do)")
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
