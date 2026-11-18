"""
neitz.run — headless orchestration: folder/recordings -> structured, saveable results.

Reuses the Paradigm classes so the CLI, notebooks, and the GUI share one core.

    from neitz.run import run_flicker, from_folder, Result
    res = run_flicker(from_folder("~/Documents/ephysdataio/2026-06-02"))
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
from .io.stim import noise_from_record
from .stimulus import FlickerParadigm, NoiseParadigm, CheckerboardParadigm


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


# ---------------------------------------------------------------- store-driven cell run
def run_cell_flicker(store, date, cell, *, paradigm=None, n_shuffle=1000,
                     formats=("png", "pdf", "svg"), save=True,
                     name="flicker", include=None, detect=None, abs_map=None,
                     run_label=None) -> Result:
    """
    Run the flicker analysis on a stored cell: load its .abf recordings, compute
    per-file + pooled ON/OFF, and (if save) write figures (PNG/PDF/SVG) + metrics.csv
    + result.json into <cell>/outputs/<name>/, recording them in the manifest.

    name      : output sub-folder + analysis key (default "flicker"). Use a distinct
                name to keep a variant run alongside earlier ones (instead of overwriting).
    include   : optional collection of recording ids OR file paths/basenames to analyze;
                default = every .abf recording in the cell. Lets you EXCLUDE recordings.
    detect    : optional spike-detection settings (polarity/method/k/abs_threshold/
                refractory_s) — overrides the paradigm defaults so the run USES (and
                records) the caller's choices instead of the hard-coded defaults.
    abs_map   : optional {file path or basename: abs_threshold} for per-trace absolute
                thresholds; each file detected with its own value (falls back to
                detect['abs_threshold']).
    run_label : the user's friendly run name, preserved in the manifest output record
                even though `name` (the folder key) is sanitized.
    """
    import os as _os
    from .io.figures import save_figure
    from . import plots

    cm = store.cell(date, cell)
    if paradigm is None:
        paradigm = FlickerParadigm(**{k: v for k, v in (detect or {}).items() if v is not None})
    base_det = dict(paradigm._det)          # canonical settings, before any per-trace override

    recs = []
    for r in cm.data.get("recordings", []):
        if r.get("kind", "recording") == "recording" and str(r.get("file", "")).endswith(".abf"):
            fpath = cm.dir / r["file"]
            recs.append((r["id"], r.get("label", r["id"]), Recording.load(fpath), str(fpath)))
    if not recs:
        raise SystemExit(f"no .abf recordings in {date}/{cell}")

    if include is not None:                              # restrict to a chosen subset
        inc = {str(x) for x in include} | {_os.path.basename(str(x)) for x in include}
        recs = [t for t in recs
                if t[0] in inc or t[3] in inc or _os.path.basename(t[3]) in inc]
        if not recs:
            raise SystemExit(f"none of the selected recordings are .abf files in {date}/{cell}")

    def _abs_for(fp):                       # this file's per-trace threshold, else the base value
        if abs_map:
            v = abs_map.get(fp, abs_map.get(_os.path.basename(fp)))
            if v is not None:
                return float(v)
        return base_det.get("abs_threshold")

    summary, per_file, trials = [], [], []
    for rid, label, rec, _fp in recs:
        paradigm.abs_threshold = _abs_for(_fp)         # detect THIS file with its own threshold
        res = paradigm.analyze_recording(rec, name=label)
        per_file.append(res)
        summary.append(dict(file=label, flicker_hz=res.freq, n_in_region=res.n_in_region,
                            vector_strength=res.vector_strength, rayleigh_p=res.rayleigh_p))
        st, fl = paradigm.detect(rec)                  # same per-file detection → pool below
        if fl is not None:
            trials.append(dict(spikes=st.times, on=fl.on_edges, off=fl.off_edges,
                               dur=rec.duration, freq=fl.freq))
    paradigm.abs_threshold = base_det.get("abs_threshold")    # restore for _det recording

    grp = paradigm.group_from_trials(trials, n_shuffle=n_shuffle)
    on, off = grp["on"], grp["off"]
    verdict = ("ON-OFF" if on["p"] < 0.01 and off["p"] < 0.01 else
               "ON" if on["p"] < 0.01 else "OFF" if off["p"] < 0.01 else "no sig.")
    pooled = dict(n_trials=grp["n_trials"], flicker_hz=grp["freq"],
                  on_ratio=on["ratio"], on_p=on["p"], off_ratio=off["ratio"], off_p=off["p"],
                  verdict=verdict)
    result = Result(name, summary=summary, tables={"pooled_onoff": [pooled]},
                    meta=dict(date=date, cell=cell, label=cm.data.get("label"),
                              run_name=run_label or name,
                              n_shuffle=n_shuffle, detect=base_det,
                              inputs=[rid for rid, _, _, _ in recs]))

    if save:
        out = cm.output_dir(name)
        title = f"{date}/{cell} {cm.data.get('label') or ''} [{name}]".strip()
        figpaths = save_figure(plots.flicker_onoff_figure(grp, label=title), out,
                               "flicker_onoff", formats=formats)
        save_figure(plots.flicker_cycle_grid(per_file), out, "flicker_cycle_grid", formats=formats)
        result.save_csv(out / "metrics.csv")
        result.save(out / "result")
        import matplotlib.pyplot as _plt
        _plt.close("all")
        rel = lambda p: _os.path.relpath(p, cm.dir)          # manifest paths relative to the cell
        files = {f"figure_{k}": rel(v) for k, v in figpaths.items()}
        files.update(metrics_csv=rel(out / "metrics.csv"), result_json=rel(out / "result.json"))
        params = {"n_shuffle": n_shuffle, **base_det}
        if abs_map:
            params["abs_per_trace"] = {_os.path.basename(str(k)): float(v)
                                       for k, v in abs_map.items() if v is not None}
        cm.record_output(name, files=files, label=run_label,
                         params=params,
                         inputs=[rid for rid, _, _, _ in recs],
                         summary=dict(verdict=verdict, flicker_hz=grp["freq"],
                                      on_p=on["p"], off_p=off["p"]))
        cm.save()
        store.update_index()
        _auto_mirror()
    return result


def run_cell_noise(store, date, cell, *, name="sta", save=True, run_label=None,
                   formats=("png", "pdf", "svg"), chan="Im_prime", ttl="TTL", detect=None,
                   abs_map=None, bins_per_update=6, include=None, filter_s=1.0, **noise_kw) -> Result:
    """
    Gaussian-noise temporal STA on a stored cell. DISPATCH: if the cell has seed-based
    gaussian_noise `.abf` recordings (June2026 Stage rig), regenerate each epoch's stimulus from its
    seed and reverse-correlate the abf spikes (the `_seed` branch); otherwise fall back to the legacy
    spike-CSV + stimulus-CSV path (2017-era cells, validated against Sara's MATLAB STA, peak ~22 ms).
    Writes the STA figure (PNG/PDF/SVG) + result.json into <cell>/outputs/<name>/ and records them.
    """
    import os as _os
    from .io.figures import save_figure
    from . import plots

    cm = store.cell(date, cell)
    seed_recs = [r for r in cm.data.get("recordings", [])
                 if str(r.get("file", "")).endswith(".abf")
                 and (r.get("stimulus") or {}).get("type") == "gaussian_noise"
                 and ((r.get("stimulus") or {}).get("params") or {}).get("seed") is not None]
    if seed_recs:
        return _run_cell_noise_seed(store, cm, date, cell, name=name, save=save, run_label=run_label,
                                    formats=formats, chan=chan, ttl=ttl, detect=detect,
                                    abs_map=abs_map, bins_per_update=bins_per_update, include=include,
                                    filter_s=filter_s)

    csvs = [str(cm.dir / r["file"]) for r in cm.data.get("recordings", [])
            if str(r.get("file", "")).endswith(".csv")]

    def pick(*subs):                                     # prefer the non-"_headers" file
        for sub in subs:
            cand = [c for c in csvs if sub in _os.path.basename(c).lower()]
            plain = [c for c in cand if "header" not in _os.path.basename(c).lower()]
            if plain or cand:
                return (plain or cand)[0]
        return None

    spike_csv = pick("spike")
    stim_csv = pick("stdev", "stim")
    if not spike_csv or not stim_csv:
        raise SystemExit(f"need a spike CSV + a stimulus CSV in {date}/{cell} "
                         f"(spike={spike_csv and _os.path.basename(spike_csv)}, "
                         f"stim={stim_csv and _os.path.basename(stim_csv)})")

    res = run_noise(spike_csv, stim_csv, **noise_kw)

    if save:
        out = cm.output_dir(name)
        label = f"{date}/{cell} {cm.data.get('label') or ''} [{name}]".strip()
        figpaths = save_figure(plots.noise_sta_figure(res.arrays, label=label, cone_isolation="S"),
                               out, "sta",
                               formats=formats)
        res.save(out / "result")
        import matplotlib.pyplot as _plt
        _plt.close("all")
        rel = lambda p: _os.path.relpath(p, cm.dir)
        files = {f"figure_{k}": rel(v) for k, v in figpaths.items()}
        files["result_json"] = rel(out / "result.json")
        cm.record_output(name, files=files, label=run_label,
                         params={"spike_csv": _os.path.basename(spike_csv),
                                 "stim_csv": _os.path.basename(stim_csv)},
                         inputs=[_os.path.basename(spike_csv), _os.path.basename(stim_csv)],
                         summary=res.summary[0])
        cm.save()
        store.update_index()
        _auto_mirror()
    return res


def _save_seed_run(cm, store, name, run_label, formats, fig, result, info, extra_params,
                   extra_figs=None):
    """Common save for a seed-based run: figure (PNG/PDF/SVG) + result.json → outputs/<name>/,
    recorded in the manifest (one record, all files) per the store-integrity rules.
    `extra_figs` ({suffix: figure}) are companion figures of the SAME run, saved as
    ``<name>_<suffix>.<fmt>`` and listed in the same record (``figure_<suffix>_<fmt>``)."""
    import os as _os
    from .io.figures import save_figure
    out = cm.output_dir(name)
    figpaths = save_figure(fig, out, name, formats=formats)
    extra_paths = {suffix: save_figure(f, out, f"{name}_{suffix}", formats=formats)
                   for suffix, f in (extra_figs or {}).items()}
    result.save(out / "result")
    import matplotlib.pyplot as _plt
    _plt.close("all")
    rel = lambda p: _os.path.relpath(p, cm.dir)
    files = {f"figure_{k}": rel(v) for k, v in figpaths.items()}
    for suffix, paths in extra_paths.items():
        files.update({f"figure_{suffix}_{k}": rel(v) for k, v in paths.items()})
    files["result_json"] = rel(out / "result.json")
    cm.record_output(name, files=files, label=run_label,
                     params={"source": "seed-manifest", "epochs": info, **extra_params},
                     inputs=[e["file"] for e in info], summary=result.summary[0])
    cm.save()
    store.update_index()
    _auto_mirror()


def _run_cell_noise_seed(store, cm, date, cell, *, name, save, run_label, formats, chan, ttl,
                         detect, abs_map, bins_per_update, include, filter_s=1.0) -> Result:
    """Seed-based full-field STA: regenerate each epoch's stimulus from its seed, reverse-correlate
    the abf spikes, average across epochs (see `sta_from_records`). `filter_s` = the recovered
    filter's length in seconds (the pipeline's STA node knob)."""
    from . import plots
    records, responses, info = load_seed_epochs(
        cm, chan=chan, ttl=ttl, detect=detect, abs_map=abs_map,
        bins_per_update=bins_per_update, include=include, want=("gaussian_noise",))
    if not records:
        raise SystemExit(f"no loadable seeded gaussian-noise recordings in {date}/{cell}")
    out = sta_from_records(records, responses, bins_per_update=bins_per_update,
                           filter_s=float(filter_s or 1.0))
    pk = int(np.argmax(np.abs(out["average"])))
    summary = [dict(n_epochs=int(out["n_epochs"]), filter_len=len(out["average"]),
                    peak_ms=float(out["time_ms"][pk]),
                    peak_sign="OFF" if out["average"][pk] < 0 else "ON")]
    arrays = dict(average=out["average"], time_ms=out["time_ms"], freqs=out["freqs"],
                  tuning=out["tuning"], per_epoch=out["per_epoch"])
    res = Result("noise", summary=summary, arrays=arrays,
                 meta=dict(source="seed-manifest", n_epochs=len(records),
                           bins_per_update=bins_per_update, filter_s=float(filter_s or 1.0),
                           epochs=info))
    if save:
        label = f"{date}/{cell} {cm.data.get('label') or ''} [{name}]".strip()
        _save_seed_run(cm, store, name, run_label, formats,
                       plots.noise_sta_figure(res.arrays, label=label,
                                              cone_isolation=records[0].get("cone_isolation")),
                       res, info,
                       {"bins_per_update": bins_per_update, "filter_s": float(filter_s or 1.0)})
    return res


def run_cell_checkerboard(store, date, cell, *, name="strf", save=True, run_label=None,
                          formats=("png", "pdf", "svg"), chan="Im_prime", ttl="TTL", detect=None,
                          abs_map=None, bins_per_update=1, filter_len=30, include=None) -> Result:
    """Checkerboard STRF on a stored cell: regenerate each epoch's spatiotemporal stimulus from its
    per-check seeds, reverse-correlate the abf spikes (pooled in time), and save the STRF figure
    (peak-lag map + peak kernel), the per-location STA grid (``<name>_grid``) and result.json —
    one output record. (June2026 Stage rig; timing from the sync bar, see analysis/sync.py.)"""
    from . import plots
    cm = store.cell(date, cell)
    records, responses, info = load_seed_epochs(
        cm, chan=chan, ttl=ttl, detect=detect, abs_map=abs_map,
        bins_per_update=bins_per_update, include=include, want=("checkerboard",))
    if not records:
        raise SystemExit(f"no loadable seeded checkerboard recordings in {date}/{cell}")
    strf = strf_from_records(records, responses, bins_per_update=bins_per_update, filter_len=filter_len)
    summary = [dict(peak_y=int(strf.peak_yx[0]), peak_x=int(strf.peak_yx[1]),
                    peak_time_ms=float(strf.peak_time_ms), n_epochs=len(records),
                    filter_len=len(strf.temporal))]
    arrays = dict(strf=strf.strf, spatial_rf=strf.spatial_rf, temporal=strf.temporal,
                  time_ms=strf.time_ms)
    res = Result("strf", summary=summary, arrays=arrays,
                 meta=dict(source="seed-manifest", n_epochs=len(records),
                           bins_per_update=bins_per_update, epochs=info))
    if save:
        label = f"{date}/{cell} {cm.data.get('label') or ''} [{name}]".strip()
        _save_seed_run(cm, store, name, run_label, formats, plots.strf_figure(strf, label=label),
                       res, info, {"bins_per_update": bins_per_update, "filter_len": filter_len},
                       extra_figs={"grid": plots.strf_grid_figure(strf, label=label)})
    return res


def _auto_mirror():
    """Back up the store to the configured mirror after a run (best-effort)."""
    from .dataio import auto_mirror, mirror_dir, mirror_store
    if auto_mirror() and mirror_dir() is not None:
        try:
            mirror_store()
        except Exception as e:
            print(f"(auto-mirror skipped: {e})")


# ---------------------------------------------------------------- checkerboard STRF
def run_strf(stimulus, response, n_y, n_x, *, paradigm=None, **kwargs) -> Result:
    """Spatiotemporal STRF from a checkerboard stimulus + binned response.

    stimulus: (n_checks, n_time) or (n_y, n_x, n_time); response: (n_time,).
    """
    paradigm = paradigm or CheckerboardParadigm(n_y=n_y, n_x=n_x, **kwargs)
    res = paradigm.analyze(stimulus, response)
    summary = [dict(peak_y=res.peak_yx[0], peak_x=res.peak_yx[1],
                    peak_time_ms=res.peak_time_ms, n_y=n_y, n_x=n_x,
                    filter_len=len(res.temporal))]
    arrays = dict(strf=res.strf, spatial_rf=res.spatial_rf,
                  temporal=res.temporal, time_ms=res.time_ms)
    return Result("strf", summary=summary, arrays=arrays,
                  meta=dict(n_y=n_y, n_x=n_x, normalize=paradigm.normalize))


# ============ seed-based reverse correlation (June2026 Stage rig) ==============================
# The stimulus is regenerated from its seed (io.stim.noise_from_record) rather than shipped as a CSV.
# Per the Neitz model: each recording IS one epoch whose spikes are already synced to its own
# stimulus, so we reverse-correlate per epoch and AVERAGE across epochs (full-field) / pool in time
# (checkerboard). The stimulus is piecewise-constant between updates; we upsample it by
# `bins_per_update` so the recovered filter has finer temporal sampling than the raw update rate.
#: stimulus.type → which analysis runs for it. sq_wave/flicker → periodic PSTH; full-field noise →
#: temporal STA; checkerboard → spatiotemporal STRF. Cone isolation is metadata, not a branch here.
_ANALYSIS_FOR_STIM = {
    "sq_wave": "flicker", "flicker": "flicker",
    "gaussian_noise": "sta", "checkerboard": "strf",
}


def analysis_for_stim_type(stim_type) -> str:
    """Map a recording's ``stimulus.type`` to its analysis kind: ``'flicker' | 'sta' | 'strf'``.
    Unknown / missing types fall back to ``'flicker'`` (the TTL-only path that needs no stim file)."""
    return _ANALYSIS_FOR_STIM.get(stim_type, "flicker")


def record_from_stimulus(stimulus) -> dict:
    """Rebuild a `noise_from_record`-ready record from a stored ``recording.stimulus`` dict.
    The import split ``stim_type`` out into ``.type``; merge it back into the params."""
    params = dict((stimulus or {}).get("params") or {})
    params["stim_type"] = (stimulus or {}).get("type")
    return params


def _update_rate(record) -> float:
    """Stimulus update rate (Hz) = frame rate / frames-per-update."""
    return float(record.get("refresh_rate_hz", 60)) / float(record.get("update_every_n_frames", 1))


def epoch_response(spike_times, t0, n_bins, bin_dt) -> np.ndarray:
    """Bin spike times (s) into `n_bins` bins of width `bin_dt` starting at `t0` → rate (spikes/s).
    `t0` is the epoch's stimulus-onset time (update 0); this is where the recording's TTL frame
    clock anchors the regenerated stimulus to the response."""
    edges = float(t0) + np.arange(int(n_bins) + 1) * float(bin_dt)
    counts, _ = np.histogram(np.asarray(spike_times, dtype=float), bins=edges)
    return counts.astype(float) / float(bin_dt)


def epoch_response_edges(spike_times, edges, bins_per_update=1) -> np.ndarray:
    """Bin spike times (s) onto a MEASURED update grid: `edges` (length ``n_updates + 1``) are the
    update start times from the sync bar (:func:`neitz.analysis.sync.update_clock`), each update
    subdivided into `bins_per_update` equal bins. Returns the rate (spikes/s) per bin, length
    ``n_updates * bins_per_update`` — the same layout :func:`epoch_response` produces for a
    uniform grid, so the reverse-correlation code does not care which one it gets."""
    edges = np.asarray(edges, dtype=float)
    bpu = max(1, int(bins_per_update))
    if bpu > 1:
        frac = np.arange(bpu) / bpu
        fine = (edges[:-1, None] + np.diff(edges)[:, None] * frac[None, :]).reshape(-1)
        edges = np.concatenate([fine, edges[-1:]])
    counts, _ = np.histogram(np.asarray(spike_times, dtype=float), bins=edges)
    return counts.astype(float) / np.diff(edges)


def _recorded(rec, v):
    """Slice a regenerated stimulus to the updates the recording actually covers (a truncated
    sweep sets ``rec['recorded_updates']`` in :func:`load_seed_epochs`)."""
    n = rec.get("recorded_updates")
    return v if not n else v[..., :int(n)]


def sta_from_records(records, responses, *, bins_per_update=6, filter_s=1.0, paradigm=None) -> dict:
    """Full-field Gaussian-noise STA from seed-based epochs (per-epoch reverse correlation, averaged).

    `records`: manifest dicts (each → `noise_from_record` → a `(1,1,n_updates)` linear stimulus).
    `responses`: per-epoch binned rate arrays aligned to the UPSAMPLED stimulus grid
    (length ``n_updates*bins_per_update``, e.g. from :func:`epoch_response`). Returns the
    ``NoiseParadigm.analyze`` dict (average filter, tuning, per_epoch, time_ms, n_epochs).
    """
    stimuli, up_rate = [], None
    for rec in records:
        # noise_from_record gives linear light v∈[0,1] (mean≈mu); reverse-correlate against the
        # CONTRAST (v − mean) so the STA has no spurious DC baseline (mean²·<r> term).
        s = np.asarray(_recorded(rec, noise_from_record(rec)), dtype=float).reshape(-1)
        stimuli.append(np.repeat(s - s.mean(), int(bins_per_update)))
        up_rate = _update_rate(rec) if up_rate is None else up_rate
    bin_rate = int(round(up_rate * bins_per_update))
    paradigm = paradigm or NoiseParadigm(bin_rate=bin_rate, filter_len=int(round(bin_rate * filter_s)))
    return paradigm.analyze(stimuli, responses)


def strf_from_records(records, responses, *, bins_per_update=1, filter_len=30, paradigm=None):
    """Checkerboard STRF from seed-based epochs: concatenate each epoch's (stimulus, response) in
    time and reverse-correlate (pools all frame↔spike pairs across epochs).

    `records` → `noise_from_record` → `(n_y, n_x, n_updates)`; `responses`: per-epoch binned rate
    aligned to the upsampled stimulus. Returns an :class:`STRFResult` whose lag axis (``time_ms``,
    ``peak_time_ms``) is one RESPONSE bin per tap: the record's update rate (``refresh_rate_hz /
    update_every_n_frames``, 59.94 Hz on the rig) times `bins_per_update` — not a fixed 60 Hz.
    """
    stim_parts, resp_parts, n_y, n_x, bin_rate = [], [], None, None, None
    for rec, resp in zip(records, responses):
        v = np.asarray(_recorded(rec, noise_from_record(rec)), dtype=float)   # (n_y,n_x,n_rec), linear
        n_y, n_x = v.shape[0], v.shape[1]
        if bin_rate is None:
            bin_rate = _update_rate(rec) * max(1, int(bins_per_update))
        v = v - v.mean()                                        # contrast (drop mean luminance)
        up = np.repeat(v, int(bins_per_update), axis=-1)        # hold each update across bins
        m = min(up.shape[-1], len(resp))
        stim_parts.append(up[..., :m])
        resp_parts.append(np.asarray(resp, dtype=float)[:m])
    stim = np.concatenate(stim_parts, axis=-1)                  # (n_y,n_x,T)
    response = np.concatenate(resp_parts)                       # (T,)
    paradigm = paradigm or CheckerboardParadigm(n_y=n_y, n_x=n_x, filter_len=filter_len,
                                                bin_rate=float(bin_rate or 60.0))
    return paradigm.analyze(stim, response)


def load_seed_epochs(cm, *, chan="Im_prime", ttl="TTL", detect=None, abs_map=None,
                     bins_per_update=6, include=None, want=("gaussian_noise", "checkerboard")):
    """Load a stored cell's seeded-noise epochs, ready for :func:`sta_from_records` /
    :func:`strf_from_records`. For each ``.abf`` recording whose stimulus is a seed-based noise of a
    wanted ``stim_type``: detect spikes on `chan`, find the stimulus onset ``t0`` from the TTL frame
    clock (:func:`~neitz.analysis.flicker.frame_clock_onset`), regenerate the stimulus
    (:func:`record_from_stimulus` → ``noise_from_record``), and bin the response to the update grid
    (:func:`epoch_response`). Returns ``(records, responses, info)``.

    STIMULUS TIMING (2026-11-12 rig convention, see :mod:`neitz.analysis.sync`): the sync bar's
    middle segment toggles once per noise update, so the update grid is MEASURED from the
    photodiode (:func:`~neitz.analysis.sync.update_clock`) — first on the `ttl` channel, then on
    every other analog channel — and the response is binned onto it with
    :func:`epoch_response_edges`. That absorbs duplicated frames (a 300 s gaussian ran ~1.6 s long on
    2026-11-12) and a sweep that stopped before the stimulus did (``recorded_updates`` is then set on
    the record so the regenerated stimulus is sliced to match). If no channel carries the update
    clock, ``t0`` = the envelope onset (:func:`~neitz.analysis.sync.stimulus_window`, else the legacy
    :func:`frame_clock_onset`, else 0.0) and the grid is uniform at ``update_every_n_frames /
    refresh_rate_hz``. ``info[i]['clock']`` says which (``'measured:<channel>'`` / ``'uniform'``).
    """
    from .spikes import detect_spikes
    from .analysis.flicker import frame_clock_onset
    from .analysis.sync import stimulus_window, update_clock
    records, responses, info = [], [], []
    for r in cm.data.get("recordings", []):
        f = str(r.get("file", ""))
        if not f.endswith(".abf"):
            continue
        rec_dict = record_from_stimulus(r.get("stimulus"))
        if rec_dict.get("stim_type") not in want or rec_dict.get("seed") is None:
            continue
        path = str(cm.dir / f)
        if include and path not in include and os.path.basename(path) not in include:
            continue
        rec = Recording.load(path)
        det = dict(detect or {})
        eff_abs = (abs_map or {}).get(path, (abs_map or {}).get(os.path.basename(path)))
        if eff_abs is not None:
            det["abs_threshold"] = float(eff_abs)
        st = detect_spikes(rec.channel(chan), rec.fs, **det)
        v = noise_from_record(rec_dict)
        n_updates = v.shape[-1]
        upd_frames = float(rec_dict.get("update_every_n_frames", 1))
        refresh = float(rec_dict.get("refresh_rate_hz", 60))
        update_dt = upd_frames / refresh
        ttl_sig = rec.channel(ttl)
        # -- measured update grid from the sync bar's middle segment (ttl first, then any other channel)
        edges, clock, sync_info = None, "uniform", {}
        names = [n for n in (getattr(rec, "channel_names", None) or []) if n != ttl
                 and str(n).lower() != str(chan).lower()]
        for cname in [ttl] + names:
            try:
                sig = ttl_sig if cname == ttl else rec.channel(cname)
                e, si = update_clock(sig, rec.fs, n_updates, upd_frames, refresh)
            except Exception:
                e, si = None, {}
            if e is not None and len(e) >= 2:
                edges, clock, sync_info = np.asarray(e, dtype=float), f"measured:{cname}", si
                break
        if edges is not None:
            t0 = float(edges[0])
        else:                                              # uniform grid from the envelope onset
            win = None
            try:
                win = stimulus_window(ttl_sig, rec.fs)
            except Exception:
                win = None
            if win is not None:
                t0 = float(win[0])
            else:
                try:
                    t0 = frame_clock_onset(ttl_sig, rec.fs)
                except Exception:
                    t0 = None
                t0 = 0.0 if t0 is None else float(t0)
            n_rec = int(min(n_updates, max(0, np.floor((len(ttl_sig) / rec.fs - t0) / update_dt))))
            edges = t0 + np.arange(n_rec + 1) * update_dt
        n_rec = len(edges) - 1
        if n_rec < n_updates:
            rec_dict["recorded_updates"] = int(n_rec)      # sweep ended before the stimulus did
        resp = epoch_response_edges(st.times, edges, bins_per_update)
        records.append(rec_dict)
        responses.append(resp)
        info.append(dict(file=os.path.basename(path), t0=round(t0, 4),
                         n_spikes=int(len(st.times)), stim_type=rec_dict["stim_type"],
                         clock=clock, n_updates=int(n_updates), n_updates_recorded=int(n_rec),
                         frame_dups=int(sync_info.get("frame_dups", 0) or 0)))
    return records, responses, info
