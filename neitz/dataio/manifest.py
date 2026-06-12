"""
CellManifest — the per-cell JSON record (recordings, stimulus metadata, outputs).

A manifest lives at <cell_dir>/manifest.json. Raw data is copied into <cell_dir>/raw/
(renamed to a proper title if needed); outputs go under <cell_dir>/outputs/<analysis>/.
"""
from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path

from .config import proper_name


class CellManifest:
    FILENAME = "manifest.json"

    def __init__(self, cell_dir, data):
        self.dir = Path(cell_dir)
        self.data = data

    # -- open / save --------------------------------------------------------
    @classmethod
    def open(cls, cell_dir, *, date=None, cell=None, label=None):
        cell_dir = Path(cell_dir)
        mf = cell_dir / cls.FILENAME
        if mf.exists():
            return cls(cell_dir, json.loads(mf.read_text()))
        data = {"version": 1, "date": date, "cell": cell, "label": label,
                "tissue": None, "cell_type": None, "notes": "",
                "recordings": [], "outputs": []}
        return cls(cell_dir, data)

    def save(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        mf = self.dir / self.FILENAME
        mf.write_text(json.dumps(self.data, indent=2, default=str))
        return mf

    # -- recordings ---------------------------------------------------------
    def add_recording(self, source, *, rec_id=None, stimulus=None, channels=None,
                      fs=None, duration_s=None, copy=True) -> dict:
        """Copy `source` into raw/ (proper-named, de-duped) and add to the manifest."""
        source = Path(source)
        target_name = proper_name(source.name, self.data.get("date"))
        raw = self.dir / "raw"
        target = raw / target_name
        if copy:
            raw.mkdir(parents=True, exist_ok=True)
            same = target.exists() and target.stat().st_size == source.stat().st_size
            if not same:
                shutil.copy2(source, target)
        rec = {"id": rec_id or Path(target_name).stem,
               "file": f"raw/{target_name}",
               "source": str(source),
               "stimulus": stimulus, "channels": channels,
               "fs": fs, "duration_s": duration_s}
        self.data["recordings"].append(rec)
        return rec

    def recording(self, rec_id):
        for r in self.data["recordings"]:
            if r["id"] == rec_id:
                return r
        raise KeyError(rec_id)

    def get_stimulus(self, rec_id):
        return self.recording(rec_id).get("stimulus")

    def set_stimulus(self, rec_id, stim_type, params, source="user") -> dict:
        r = self.recording(rec_id)
        r["stimulus"] = {"type": stim_type, "params": params, "source": source}
        return r

    # -- outputs ------------------------------------------------------------
    def output_dir(self, analysis) -> Path:
        d = self.dir / "outputs" / analysis
        d.mkdir(parents=True, exist_ok=True)
        return d

    def record_output(self, analysis, *, files, params=None, summary=None,
                      inputs=None, created=None) -> dict:
        out = {"analysis": analysis,
               "created": created or datetime.now().isoformat(timespec="seconds"),
               "params": params or {}, "inputs": inputs or [],
               "files": files, "summary": summary or {}}
        self.data["outputs"].append(out)
        return out
