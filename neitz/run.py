"""
neitz.run — headless orchestration: folder/recordings -> structured, saveable results.

Reuses the Paradigm classes so the CLI, notebooks, and the GUI share one core.

    from neitz.run import run_flicker, from_folder, Result
    res = run_flicker(from_folder("data/ipRGC barak"))
    res.save("flicker_results")        # -> flicker_results.json (+ .npz if arrays)
    res.save_csv("flicker_summary.csv")
"""
from __future__ import annotations
import os
import glob
import json
from dataclasses import dataclass, field

import numpy as np

from .io.abf import Recording
from .io import csv as ncsv
from .stimulus import FlickerParadigm, NoiseParadigm


# ---------------------------------------------------------------- result model
def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


@dataclass
class Result:
    kind: str                                   # 'flicker' | 'noise' | 'spikes'
    summary: list = field(default_factory=list)   # per-item scalar rows (dicts)
    tables: dict = field(default_factory=dict)    # name -> list[dict]
    arrays: dict = field(default_factory=dict)    # name -> np.ndarray
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(kind=self.kind, summary=self.summary, tables=self.tables, meta=self.meta)

    def save(self, path) -> str:
        """Write <path>.json (summary/tables/meta) and <path>.npz (arrays, if any)."""
        base = str(path)[:-5] if str(path).endswith(".json") else str(path)
        with open(base + ".json", "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_jsonable)
        if self.arrays:
            np.savez(base + ".npz", **self.arrays)
        return base + ".json"

    def save_csv(self, path) -> str:
        """Write the per-item `summary` rows as a CSV."""
        rows = self.summary
        if not rows:
            open(path, "w").close()
            return path
        cols = list(rows[0].keys())
        with open(path, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
        return path

    @classmethod
    def load(cls, path) -> "Result":
        base = str(path)[:-5] if str(path).endswith(".json") else str(path)
        with open(base + ".json") as f:
            d = json.load(f)
        arrays = {}
        if os.path.exists(base + ".npz"):
            arrays = dict(np.load(base + ".npz"))
        return cls(kind=d["kind"], summary=d["summary"], tables=d.get("tables", {}),
                   arrays=arrays, meta=d.get("meta", {}))


# ---------------------------------------------------------------- discovery
def discover(folder, pattern="**/*.abf") -> list:
    return sorted(glob.glob(os.path.join(str(folder), pattern), recursive=True))


def from_folder(folder, pattern="**/*.abf") -> list:
    """-> [(basename, parent_dir_name, Recording), ...] for use with run_flicker."""
    out = []
    for p in discover(folder, pattern):
        out.append((os.path.basename(p), os.path.basename(os.path.dirname(p)), Recording.load(p)))
    return out


# ---------------------------------------------------------------- flicker
def run_flicker(records, *, paradigm=None, pooled=True, n_shuffle=1000, seed=0) -> Result:
    """records: iterable of (name, group, Recording). Returns per-file summary and,
    if pooled, a per-cell (group) transition-triggered ON/OFF table."""
    paradigm = paradigm or FlickerParadigm()
    records = list(records)

    summary, cells = [], {}
    for name, group, rec in records:
        res = paradigm.analyze_recording(rec, name=name)
        summary.append(dict(file=name, group=group, flicker_hz=res.freq,
                            n_spikes=res.n_spikes, n_in_region=res.n_in_region,
                            vector_strength=res.vector_strength, rayleigh_p=res.rayleigh_p))
        cells.setdefault(group, []).append(rec)

    tables = {}
    if pooled:
        rng = np.random.default_rng(seed)
        rows = []
        for cell, recs in cells.items():
            grp = paradigm.analyze_group(recs, n_shuffle=n_shuffle, rng=rng)
            if grp["n_trials"] == 0:
                continue
            on, off = grp["on"], grp["off"]
            cls = ("ON-OFF" if on["p"] < 0.01 and off["p"] < 0.01 else
                   "ON" if on["p"] < 0.01 else "OFF" if off["p"] < 0.01 else "no sig.")
            rows.append(dict(cell=cell, n_trials=grp["n_trials"], flicker_hz=grp["freq"],
                             on_ratio=on["ratio"], on_peak_ms=on["peak_ms"], on_p=on["p"],
                             off_ratio=off["ratio"], off_peak_ms=off["peak_ms"], off_p=off["p"],
                             verdict=cls))
        tables["pooled_onoff"] = rows

    return Result("flicker", summary=summary, tables=tables,
                  meta=dict(n_files=len(records), pooled=pooled, n_shuffle=n_shuffle,
                            detect=paradigm._det))


def run_flicker_folder(folder, **kwargs) -> Result:
    res = run_flicker(from_folder(folder), **kwargs)
    res.meta["folder"] = str(folder)
    return res


# ---------------------------------------------------------------- spike export
def run_spike_export(folder, out_csv, *, channel="Im_prime", out_rate=10000,
                     pattern="**/*.abf", truncate=True, **detect) -> Result:
    """Detect spikes per file and write a binary spike-train CSV (siso-spikes.csv layout)."""
    from .spikes import detect_spikes
    detect = {**dict(polarity="neg", method="mad", k=6.0, refractory_s=0.002), **detect}
    files = discover(folder, pattern)
    cols, summary = [], []
    for f in files:
        rec = Recording.load(f)
        st = detect_spikes(rec.channel(channel), rec.fs, **detect)
        cols.append(st.binary(out_rate, duration=rec.duration))
        summary.append(dict(file=os.path.basename(f), duration_s=rec.duration,
                            n_spikes=len(st), rate_hz=st.rate(rec.duration)))
    if not cols:
        raise SystemExit(f"No .abf files matched {folder}/{pattern}")
    n = min(len(c) for c in cols)
    if truncate:
        cols = [c[:n] for c in cols]
    mat = np.column_stack(cols)
    t = np.arange(mat.shape[0]) / out_rate
    np.savetxt(out_csv, np.column_stack([t, mat]),
               fmt=["%.6g"] + ["%d"] * mat.shape[1], delimiter=",")
    return Result("spikes", summary=summary,
                  meta=dict(out_csv=out_csv, out_rate=out_rate, n_files=len(files), detect=detect))


# ---------------------------------------------------------------- noise (revcorr)
def run_noise(spike_csv, stim_csv, *, paradigm=None, trim_s=0.5, stim_le_s=10.0,
              bins_per_frame=6, fps=60, zero_pad=None) -> Result:
    """S-iso / Gaussian-noise reverse correlation from a spike CSV + stimulus CSV."""
    paradigm = paradigm or NoiseParadigm(bin_rate=bins_per_frame * fps,
                                         filter_len=bins_per_frame * fps,
                                         zero_pad=(zero_pad if zero_pad is not None else 60))
    bin_rate = bins_per_frame * fps
    bins_total = int(stim_le_s * bin_rate)

    time_vec, spike_ch = ncsv.load_spikes_csv(spike_csv)
    mask = (time_vec >= time_vec[0] + trim_s) & (time_vec < time_vec[-1] - trim_s)
    time_s = time_vec[mask] - time_vec[mask][0]
    spike_times = ncsv.spike_times_from_matrix(spike_ch[mask, :], time_s)

    stim_ep, phases = ncsv.load_stimulus_epochs_csv(stim_csv)
    stim = ncsv.stim_phase_only(stim_ep, phases)
    stim_up = np.repeat(stim, bins_per_frame, axis=0)        # (bins_total, n_epochs)

    edges = np.linspace(0, stim_le_s, bins_total + 1)
    stimuli, responses = [], []
    for ep in range(stim.shape[1]):
        counts, _ = np.histogram(spike_times[ep], bins=edges)
        responses.append(counts.astype(float) * bin_rate)
        stimuli.append(stim_up[:, ep].astype(float))

    out = paradigm.analyze(stimuli, responses)
    pk = int(np.argmax(np.abs(out["average"])))
    summary = [dict(n_epochs=int(out["n_epochs"]), filter_len=len(out["average"]),
                    peak_ms=float(out["time_ms"][pk]),
                    peak_sign="OFF" if out["average"][pk] < 0 else "ON")]
    arrays = dict(average=out["average"], time_ms=out["time_ms"],
                  freqs=out["freqs"], tuning=out["tuning"], per_epoch=out["per_epoch"])
    return Result("noise", summary=summary, arrays=arrays,
                  meta=dict(spike_csv=str(spike_csv), stim_csv=str(stim_csv),
                            normalize=paradigm.normalize, zero_pad=paradigm.zero_pad))
