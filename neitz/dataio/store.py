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
