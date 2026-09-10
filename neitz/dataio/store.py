"""
DataStore — manage the ephysdataio root: date/cell hierarchy + a top-level index.

    from neitz.dataio import DataStore
    ds   = DataStore()                               # ~/Documents/ephysdataio
    cell = ds.new_cell("2026-06-02", label="ipRGC")  # -> c01
    cell.add_recording("…/2026_06_02_0040.abf", stimulus={...}); cell.save()
    ds.update_index()
"""
from __future__ import annotations
import json
import re
import shutil
from pathlib import Path

from .config import data_root
from .manifest import CellManifest

_CELL_RE = re.compile(r"^c(\d+)")


class DataStore:
    def __init__(self, root=None):
        self.root = Path(root) if root else data_root()

    def _date_dir(self, date) -> Path:
        return self.root / str(date)

    # -- cells --------------------------------------------------------------
    def next_cell_number(self, date) -> int:
        d = self._date_dir(date)
        nums = []
        if d.exists():
            for p in d.iterdir():
                m = _CELL_RE.match(p.name)
                if m and p.is_dir():
                    nums.append(int(m.group(1)))
        return (max(nums) + 1) if nums else 1

    def new_cell(self, date, *, label=None, cell_type=None) -> CellManifest:
        cell = f"c{self.next_cell_number(date):02d}"
        cm = CellManifest.open(self._date_dir(date) / cell, date=str(date), cell=cell, label=label)
        cm.data["cell_type"] = cell_type
        cm.save()
        self.update_index()
        return cm

    def cell(self, date, cell) -> CellManifest:
        return CellManifest.open(self._date_dir(date) / cell)

    def cell_by_name(self, date, name) -> CellManifest | None:
        """Find a date's cell by its human name (the manifest ``label``, e.g. 'mac-C1'), or
        by its folder id ('c01'). Returns the :class:`CellManifest` or ``None``. Used by the
        Explorer 'move to cell' so a re-file can target an existing cell by the name the user
        sees. First match wins if two cells share a label."""
        for d, c in self.cells():
            if d != str(date):
                continue
            if c == name:
                return self.cell(d, c)
            cm = self.cell(d, c)
            if (cm.data.get("label") or "") == name:
                return cm
        return None

    def move_recording(self, date, src_cell, dst_cell, rec_id) -> dict:
        """Move ONE recording from ``src_cell`` to ``dst_cell`` (same date), preserving the
        data-store invariants: the raw file physically MOVES into the destination's ``raw/``
        and BOTH manifests are updated (removed from src, added to dst) so no orphan file and
        no dangling record are left behind. The recording keeps its stimulus + protocol
        metadata; its id/filename stay the same (abf names are unique within a day). Existing
        outputs already produced under the SOURCE cell stay there (they belong to the run that
        made them); future analyses run under the destination. Returns the moved record.
        """
        src = self.cell(date, src_cell)
        dst = self.cell(date, dst_cell)
        if src.dir.resolve() == dst.dir.resolve():
            return src.recording(rec_id)                       # no-op: same cell
        rec = dict(src.recording(rec_id))                      # copy before we mutate src
        rel = rec.get("file")
        if rel:
            old = src.dir / rel
            (dst.dir / "raw").mkdir(parents=True, exist_ok=True)
            target = dst.dir / rel                             # keep raw/<name>
            if old.resolve() != target.resolve():
                if target.exists():                            # never clobber a dst file
                    raise FileExistsError(f"{target} already exists in {dst_cell}")
                if old.exists():
                    shutil.move(str(old), str(target))
        src.remove_recording(rec_id)
        dst.data["recordings"].append(rec)
        src.save()
        dst.save()
        self.update_index()
        return rec

    def cells(self) -> list:
        out = []
        if self.root.exists():
            for dd in sorted(self.root.iterdir()):
                if not dd.is_dir():
                    continue
                for cd in sorted(dd.iterdir()):
                    if (cd / CellManifest.FILENAME).exists():
                        out.append((dd.name, cd.name))
        return out

    # -- index --------------------------------------------------------------
    def update_index(self) -> list:
        cells = []
        for date, cell in self.cells():
            d = json.loads((self.root / date / cell / CellManifest.FILENAME).read_text())
            cells.append({"date": date, "cell": cell, "label": d.get("label"),
                          "cell_type": d.get("cell_type"),
                          "n_recordings": len(d.get("recordings", [])),
                          "n_outputs": len(d.get("outputs", []))})
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "index.json").write_text(json.dumps({"version": 1, "cells": cells}, indent=2))
        return cells

    def index(self) -> list:
        idx = self.root / "index.json"
        return json.loads(idx.read_text())["cells"] if idx.exists() else []
