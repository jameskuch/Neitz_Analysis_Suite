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
    def add_recording(self, source, *, rec_id=None, label=None, kind="recording",
                      stimulus=None, channels=None, fs=None, duration_s=None,
                      copy=True) -> dict:
        """Copy `source` into raw/ (proper-named, de-duped) and add to the manifest.

        `label` is a friendly display name (what the GUI shows); the formatted file
        name stays canonical. Defaults to the original source filename.
        """
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
               "label": label or source.name,          # friendly display name
               "kind": kind,                            # 'recording' | 'reference'
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

    # -- saved Analysis-View states -----------------------------------------
    # A named snapshot of the GUI's analysis/display controls for this cell (region, detection,
    # alignment, per-trace thresholds, display toggles, …). Stored right in the manifest so the
    # settings you worked out on a cell can be restored later. Keyed by `name` (re-save REPLACES).
    def view_states(self) -> list:
        return self.data.get("view_states", [])

    def save_view_state(self, name, state) -> dict:
        rec = {"name": name, "created": datetime.now().isoformat(timespec="seconds"), "state": state}
        states = self.data.setdefault("view_states", [])
        for i, s in enumerate(states):
            if s.get("name") == name:
                states[i] = rec
                return rec
        states.append(rec)
        return rec

    def delete_view_state(self, name) -> bool:
        states = self.data.get("view_states", [])
        kept = [s for s in states if s.get("name") != name]
        self.data["view_states"] = kept
        return len(kept) < len(states)

    def record_output(self, analysis, *, files, params=None, summary=None,
                      inputs=None, created=None, label=None) -> dict:
        out = {"analysis": analysis,
               "created": created or datetime.now().isoformat(timespec="seconds"),
               "params": params or {}, "inputs": inputs or [],
               "files": files, "summary": summary or {}}
        if label and label != analysis:          # the user's friendly run name (folder key is sanitized)
            out["label"] = label
        # a re-run of the same analysis OVERWRITES its output folder, so REPLACE the existing
        # record in place rather than appending (otherwise the manifest accrues stale duplicates).
        outs = self.data.setdefault("outputs", [])
        for i, o in enumerate(outs):
            if o.get("analysis") == analysis:
                outs[i] = out
                return out
        outs.append(out)
        return out
