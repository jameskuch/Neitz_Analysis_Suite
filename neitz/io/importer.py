"""
neitz.io.importer — apply a stim-manifest import plan to a DataStore.

Turns a plan from :func:`neitz.io.stim.build_import_plan` (day → cell → protocol → epochs,
ABFs already paired to epochs by timestamp) into real store cells, integrity-safe: each
manifest cell_name becomes a numbered store cell (its human name kept as the ``label``),
each block a PROTOCOL stamped on every recording, matched ABFs its epochs. Stray ABFs (no
epoch) and any non-abf files go to one '(unsorted)' cell for the day; a cell whose epochs
were all aborted false-starts is skipped; discarded-epoch ABFs are skipped.

Kept out of ``viewer.py`` so the CLI, tests, and a batch re-import can reuse the exact same
store-writing path the GUI import uses.
"""
from __future__ import annotations
import os
import shutil
from collections import OrderedDict
from pathlib import Path


def renumber_protocols(cm):
    """Recompute each recording's protocol ``{index, epoch, n_epochs}`` after a label change or
    a cross-cell move: group recordings by protocol LABEL (in manifest order), index the groups
    by first appearance, and number epochs 1..n within each. Recordings with no protocol label
    are left ungrouped. Keeps ``epoch_groups`` + the Explorer protocol headers consistent after
    re-categorization. Mutates the recordings in place; the caller saves."""
    order = OrderedDict()
    for r in cm.data.get("recordings", []):
        lab = (r.get("protocol") or {}).get("label")
        if lab:
            order.setdefault(lab, []).append(r)
    for idx, (lab, recs) in enumerate(order.items(), start=1):
        for ep, r in enumerate(recs, start=1):
            pr = dict(r.get("protocol") or {})
            pr.update({"index": idx, "label": lab, "epoch": ep, "n_epochs": len(recs)})
            r["protocol"] = pr


def _copy_manifest_into(cm, mf):
    """Copy the day's nested stim manifest into a cell dir (self-contained stored cell)."""
    try:
        shutil.copy2(mf, Path(cm.dir) / Path(mf).name)
    except Exception:
        pass


def apply_import_plan(ds, date, plan, *, other_paths=(), hand_stim=None) -> dict:
    """Create a day's cells from an import `plan`. Returns a summary dict::

        {"date", "cells": [{"cell","name","n"}], "n_imported", "notes": [str,...]}

    `other_paths` are non-abf files (e.g. spike CSVs) with no manifest epoch — they join the
    '(unsorted)' cell. `hand_stim` (optional ``{"type","params","source"}``) tags those
    non-abf leftovers only (matched abfs always take their stimulus from the manifest).
    """
    mf = plan.get("manifest")
    made_cells, notes, n_imported = [], list(plan.get("warnings", [])), 0

    for cnode in plan.get("cells", []):
        proto_eps = [(pr, [e for e in pr["epochs"] if e["abf"]]) for pr in cnode["protocols"]]
        if sum(len(eps) for _, eps in proto_eps) == 0:       # only aborted / no-abf epochs
            notes.append(f"cell '{cnode['cell_name']}': only aborted/no-abf epochs — skipped")
            continue
        cm = ds.new_cell(date, label=cnode["cell_name"])
        nrec = 0
        for proto, eps in proto_eps:
            for k, e in enumerate(eps, start=1):
                pr = {"index": proto["index"], "label": proto["label"],
                      "signature": proto["stim_signature"], "epoch": k, "n_epochs": len(eps)}
                stim = ({"type": e["stim_type"], "params": e["params"], "source": "stim-manifest"}
                        if e["stim_type"] else None)
                cm.add_recording(e["abf"], label=os.path.basename(e["abf"]),
                                 stimulus=stim, protocol=pr)
                nrec += 1
        if mf is not None:
            _copy_manifest_into(cm, mf)
        cm.save()
        n_imported += nrec
        made_cells.append({"cell": cm.data["cell"], "name": cnode["cell_name"], "n": nrec})

    leftovers = list(plan.get("unmatched_abfs", [])) + list(other_paths)
    if leftovers:
        cu = ds.new_cell(date, label="(unsorted)")
        for p in leftovers:
            st = hand_stim if not str(p).lower().endswith(".abf") else None
            cu.add_recording(p, label=os.path.basename(p), stimulus=st)
        if mf is not None:
            _copy_manifest_into(cu, mf)
        cu.save()
        n_imported += len(leftovers)
        made_cells.append({"cell": cu.data["cell"], "name": "(unsorted)", "n": len(leftovers)})

    if plan.get("discarded_abfs"):
        notes.append(f"{len(plan['discarded_abfs'])} discarded-epoch abf(s) skipped")
    ds.update_index()
    return {"date": date, "cells": made_cells, "n_imported": n_imported, "notes": notes}
