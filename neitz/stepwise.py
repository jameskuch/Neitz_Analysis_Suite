"""
neitz.stepwise — run an analysis pipeline ONE COMPONENT AT A TIME and look at what each produces.

The pipelines view can select any node; this module computes that node's output for the current
cell / file selection with the SAME functions the real run uses (`pipeline.CODE_REFS` names them
and the GUI shows their source beside the result). Nothing here is a second implementation of an
analysis: Detect calls `spikes.detect_spikes`, the STA node `run.load_seed_epochs` +
`run.sta_from_records`, the STRF node `run.strf_from_records`, the square-wave node
`FlickerParadigm.analyze_recording` + `group_from_trials`, the signal-flow blocks
`signal_flow.signal_at` / `evaluate`.

    out = node_output(graph, node_id, ctx)      # {"title", "summary", "notes", "blocks": [...]}

`ctx` is the live Analysis-View state::

    files      checked raw paths (one recording = one epoch)
    chan, ttl  signal / sync channel names
    detect     {polarity, method, k, abs_threshold, refractory_s}
    abs_map    {path: per-trace abs threshold}
    region     {"start": s | None, "end": s | None, "crop": bool}
    align_map  {path: offset seconds}
    load       path -> Recording-like (fs, duration, channel_names, channel(name));
               default Recording.load. The viewer passes its cache.
    n_shuffle  shuffles for the square-wave preview (default 100 — Run uses the node's value)
    fft_bin_ms bin for the pre-FFT conditioning blocks (default 5)
    filter_s   STA filter length (s), the STA node's `filter_s`

A block is one of::

    {"kind": "table",   "columns": [...], "rows": [[...], ...]}
    {"kind": "raster",  "epochs": [times...], "names": [...], "window": (s, e) | None}
    {"kind": "traces",  "title", "x_title", "y_title", "traces": [{x, y, name, width, color,
                        dash}], "vlines": [x...], "xrange": [lo, hi] | None}
    {"kind": "heatmap", "title", "z": [[...]], "mark": (y, x) | None}
    {"kind": "text",    "lines": [...]}
    {"kind": "error",   "error": "..."}

Pure numpy + the core package (no GUI deps) — tested in ``tests/test_stepwise.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import pipeline as pipe
from . import run
from . import signal_flow as sf
from .analysis import dsp
from .dataio.manifest import CellManifest
from .io.abf import Recording
from .spikes import detect_spikes, in_region
from .stimulus.flicker import FlickerParadigm

SEED_FAMILIES = ("gaussian_noise", "checkerboard")
#: response bins per stimulus update, as the run entry points use them (run_cell_noise /
#: run_cell_checkerboard defaults) — the preview MUST bin the way the run does.
BINS_PER_UPDATE = {"gaussian_noise": 6, "checkerboard": 1}
ZERO_PAD = 60          # revcorr / strf default: FFT zero padding in response bins


# ----------------------------------------------------------------------------------------------
# helpers shared by the node evaluators (memoized on the ctx dict under "_…" keys)
# ----------------------------------------------------------------------------------------------
def _loader(ctx):
    return ctx.get("load") or Recording.load


def _det(ctx):
    """The live detection dict without None entries (detect_spikes wants real numbers)."""
    return {k: v for k, v in (ctx.get("detect") or {}).items() if v is not None}


def recordings(ctx) -> list:
    """One dict per selected file: path, file, rec (or error), fs, duration, channels."""
    if "_recs" in ctx:
        return ctx["_recs"]
    out, load = [], _loader(ctx)
    for p in ctx.get("files") or []:
        if not p:
            continue
        try:
            rec = load(p)
            chan = ctx.get("chan") or (getattr(rec, "channel_names", None) or [None])[0]
            dur = getattr(rec, "duration", None)
            if dur is None:
                dur = len(rec.channel(chan)) / float(rec.fs)
            out.append(dict(path=p, file=os.path.basename(p), rec=rec, fs=float(rec.fs),
                            duration=float(dur),
                            channels=list(getattr(rec, "channel_names", None) or [])))
        except Exception as e:                                    # keep going; show the error
            out.append(dict(path=p, file=os.path.basename(p), rec=None, error=str(e)))
    ctx["_recs"] = out
    return out


def spikes(ctx) -> list:
    """Per-epoch detection with the live settings (per-trace abs honored), spike times shifted by
    the align offset — exactly what the Analysis View plots."""
    if "_spikes" in ctx:
        return ctx["_spikes"]
    det, amap, align = _det(ctx), ctx.get("abs_map") or {}, ctx.get("align_map") or {}
    chan = ctx.get("chan")
    out = []
    for r in recordings(ctx):
        if r.get("rec") is None:
            continue
        rec = r["rec"]
        d = dict(det)
        eff = amap.get(r["path"], amap.get(r["file"]))
        if eff is not None:
            d["abs_threshold"] = float(eff)
        st = detect_spikes(rec.channel(chan or r["channels"][0]), rec.fs, **d)
        try:
            off = float(align.get(r["path"], 0.0) or 0.0)
        except (TypeError, ValueError):
            off = 0.0
        out.append(dict(r, times=np.asarray(st.times, float) + off, n=int(len(st.times)),
                        rate=(len(st.times) / r["duration"]) if r["duration"] else float("nan"),
                        threshold=st.threshold, sigma=st.sigma, offset=off))
    ctx["_spikes"] = out
    return out


def region_window(ctx, epochs) -> tuple:
    """(start, end) of the analysis window — the region boxes, blank = the recordings' full span."""
    reg = ctx.get("region") or {}
    s, e = reg.get("start"), reg.get("end")
    if s is None:
        s = 0.0
    if e is None:
        e = max((ep["duration"] + ep.get("offset", 0.0) for ep in epochs), default=1.0)
    return float(s), float(e)


def region_spikes(ctx) -> list:
    """`spikes(ctx)` restricted to the region window (what every downstream node consumes)."""
    if "_region" in ctx:
        return ctx["_region"]
    eps = spikes(ctx)
    s, e = region_window(ctx, eps)
    out = [dict(ep, kept=in_region(ep["times"], s, e)) for ep in eps]
    ctx["_region"] = out
    ctx["_window"] = (s, e)
    return out


def _cell_of(path):
    """The CellManifest a raw file belongs to (<store>/<date>/<cell>/raw/<file>)."""
    return CellManifest.open(Path(path).parents[1])


def seed_epochs(ctx, want=SEED_FAMILIES, bins_per_update=None):
    """`run.load_seed_epochs` over the selected files — grouped by their cell, since a selection
    may span cells — returning (records, responses, info) exactly as the run gets them."""
    files = [p for p in (ctx.get("files") or []) if p]
    groups: dict = {}
    for p in files:
        groups.setdefault(str(Path(p).parents[1]), []).append(p)
    records, responses, info = [], [], []
    open_cell = ctx.get("cell_of") or _cell_of
    for cdir, paths in groups.items():
        try:
            cm = open_cell(paths[0])
        except Exception:
            continue
        r, s, i = run.load_seed_epochs(cm, chan=ctx.get("chan") or "Im_prime",
                                       ttl=ctx.get("ttl") or "TTL", detect=_det(ctx),
                                       abs_map=ctx.get("abs_map") or None,
                                       bins_per_update=bins_per_update, include=paths, want=want)
        records += r; responses += s; info += i
    return records, responses, info


def seed_result(ctx, family, filter_s=None) -> dict:
    """The live STA (`gaussian_noise`) or STRF (`checkerboard`) of the selected files, computed the
    way `run_cell_noise` / `run_cell_checkerboard` compute it (measured update clock per epoch,
    the run's bins-per-update, contrast = v − mean). Returns {"family", "n_epochs", "info",
    "bin_rate", "filter_len", and "sta": the NoiseParadigm dict | "strf": the STRFResult}."""
    bpu = BINS_PER_UPDATE[family]
    records, responses, info = seed_epochs(ctx, want=(family,), bins_per_update=bpu)
    if not records:
        raise ValueError(f"no seeded {family.replace('_', ' ')} recording among the checked files")
    if family == "gaussian_noise":
        fs_ = float(filter_s if filter_s is not None else (ctx.get("filter_s") or 1.0))
        out = run.sta_from_records(records, responses, bins_per_update=bpu, filter_s=fs_)
        bin_rate = int(round(run._update_rate(records[0]) * bpu))
        return dict(family=family, n_epochs=len(records), info=info, sta=out,
                    bin_rate=bin_rate, filter_len=int(len(out["average"])), filter_s=fs_)
    strf = run.strf_from_records(records, responses, bins_per_update=bpu)
    return dict(family=family, n_epochs=len(records), info=info, strf=strf,
                bin_rate=float(run._update_rate(records[0]) * bpu),
                filter_len=int(len(strf.temporal)))


def sync_table(info) -> dict:
    """The per-epoch stimulus-timing evidence: where each epoch's clock came from."""
    rows = []
    for i in info:
        rows.append([i.get("file"), f"{i.get('t0', 0):.4f}", i.get("clock", "?"),
                     f"{i.get('n_updates_recorded', 0)}/{i.get('n_updates', 0)}",
                     i.get("frame_dups", 0), i.get("n_spikes", 0)])
    return {"kind": "table", "columns": ["epoch", "stimulus t₀ (s)", "update clock",
                                         "updates recorded", "dup frames", "spikes"],
            "rows": rows}


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


# ----------------------------------------------------------------------------------------------
# node evaluators — one per component type
# ----------------------------------------------------------------------------------------------
def _source(graph, node, ctx):
    rows, n_ok = [], 0
    for r in recordings(ctx):
        if r.get("rec") is None:
            rows.append([r["file"], "—", "—", "—", f"⚠ {r['error']}"])
            continue
        n_ok += 1
        stim = (ctx.get("stim_of") or (lambda p: None))(r["path"]) or {}
        p = stim.get("params") or {}
        brief = stim.get("type") or "no stimulus record"
        if stim.get("type") in SEED_FAMILIES:
            brief += (f" · {p.get('checks_y', 1)}×{p.get('checks_x', 1)} · seed {p.get('seed')}"
                      f" · {p.get('n_updates')} updates @ {float(p.get('noise_update_hz') or 0):.2f} Hz")
        elif stim.get("type"):
            brief += f" · {p.get('flicker_hz', '?')} Hz"
        rows.append([r["file"], f"{r['duration']:.2f}", f"{r['fs']:.0f}", ", ".join(r["channels"]),
                     brief])
    return dict(summary=f"{n_ok} recording(s) = {n_ok} epoch(s); signal '{ctx.get('chan')}', "
                        f"sync '{ctx.get('ttl')}'",
                notes=["Each checked .abf is one epoch. Nothing is computed here — the files are "
                       "loaded and handed downstream."],
                blocks=[{"kind": "table",
                         "columns": ["file", "duration (s)", "fs (Hz)", "channels", "stimulus"],
                         "rows": rows}])


def _align(graph, node, ctx):
    mode = node.get("params", {}).get("mode", "auto")
    align = ctx.get("align_map") or {}
    rows = []
    for i, r in enumerate(recordings(ctx)):
        off = 0.0 if mode == "off" else float(align.get(r["path"], 0.0) or 0.0)
        how = ("reference" if i == 0 else "frame-sync nudge (Analysis View)" if off else "none")
        rows.append([r["file"], f"{off * 1000.0:+.2f}", how])
    return dict(summary=f"mode {mode} · {sum(1 for r in rows if float(r[1]) != 0)} epoch(s) shifted",
                notes=["The offsets are the Analysis View's trial-align nudges (⇄ auto seeds them from "
                       "each file's frame-sync onset, then refines by cross-correlation). Signal, "
                       "sync and spikes of an epoch move together.",
                       "Seeded noise needs no nudge: each epoch is binned on its OWN measured update "
                       "clock (see the STA / STRF node)."],
                blocks=[{"kind": "table", "columns": ["file", "offset (ms)", "source"], "rows": rows}])


def _detect(graph, node, ctx):
    eps = spikes(ctx)
    rows = [[e["file"], e["n"], f"{e['rate']:.2f}", _fmt(e["threshold"], 2), _fmt(e["sigma"], 3)]
            for e in eps]
    det = _det(ctx)
    return dict(summary=f"{sum(e['n'] for e in eps)} spikes in {len(eps)} epoch(s) · "
                        f"{det.get('polarity')} · {det.get('method')} k={det.get('k')} · "
                        f"refractory {1000 * float(det.get('refractory_s', 0.002)):g} ms",
                notes=["Detection uses the LIVE Analysis-View settings (the node's own fields are "
                       "the editable description; the live state wins at run time)."],
                blocks=[{"kind": "table",
                         "columns": ["file", "spikes", "rate (Hz)", "threshold", "σ̂ (MAD)"],
                         "rows": rows},
                        {"kind": "raster", "epochs": [e["times"] for e in eps],
                         "names": [e["file"] for e in eps], "window": None}])


def _region(graph, node, ctx):
    eps = region_spikes(ctx)
    s, e = ctx["_window"]
    rows = [[ep["file"], int(ep["kept"].size), int(ep["n"] - ep["kept"].size)] for ep in eps]
    kept = sum(r[1] for r in rows)
    return dict(summary=f"window {s:.2f}–{e:.2f} s · {kept} spikes kept, "
                        f"{sum(r[2] for r in rows)} dropped",
                notes=["Square-wave analyses and the signal-flow blocks consume ONLY the spikes inside "
                       "the window. The STA / STRF do not use it: each seeded epoch is binned on the "
                       "stimulus-update grid measured from the sync bar, so the pre-/post-stimulus "
                       "gray is excluded by construction."],
                blocks=[{"kind": "table", "columns": ["file", "in window", "dropped"], "rows": rows},
                        {"kind": "raster", "epochs": [ep["kept"] for ep in eps],
                         "names": [ep["file"] for ep in eps], "window": (s, e)}])


def _binned_rates(ctx):
    """The Analysis-View binned rate per epoch (fft bin), inside the region — the array the
    Smooth / Temporal-filter nodes condition before the FFT."""
    eps = region_spikes(ctx)
    s, e = ctx["_window"]
    bin_ms = float(ctx.get("fft_bin_ms") or 5.0)
    rate = 1000.0 / max(1.0, bin_ms)
    n = int(round((e - s) * rate))
    edges = np.linspace(s, e, n + 1) if n >= 8 else None
    out = []
    for ep in eps:
        if edges is None:
            continue
        cnt, _ = np.histogram(ep["kept"], bins=edges)
        out.append((ep["file"], cnt.astype(float) * rate))
    return out, rate, s


def _cond_traces(ctx, fn, label):
    rates, rate, s = _binned_rates(ctx)
    traces = []
    for i, (name, r) in enumerate(rates):
        t = s + (np.arange(len(r)) + 0.5) / rate
        if i == 0:
            traces.append({"x": t, "y": r, "name": f"{name} (before)", "width": 0.6, "color": "#bbb"})
        traces.append({"x": t, "y": fn(r, rate), "name": name, "width": 1.2})
    return dict(summary=f"{label} · {len(rates)} epoch(s) · {rate:g} Hz bins",
                notes=["Applied to the binned spike rate before the FFT / power spectrum (the "
                       "Analysis View's 'Pre-FFT conditioning'). The first epoch is also shown "
                       "unconditioned (gray) for comparison."],
                blocks=[{"kind": "traces", "title": label, "x_title": "time (s)",
                         "y_title": "rate (Hz)", "traces": traces}])


def _smooth(graph, node, ctx):
    p = node.get("params", {})
    win = float(p.get("window") or 0)
    if win < 2:
        return dict(summary="span < 2 → no-op", notes=[], blocks=[{"kind": "text", "lines": [
            "A span below 2 samples leaves the rate untouched."]}])
    return _cond_traces(ctx, lambda r, fs: dsp.smooth_1d(r, window=win, method=p.get("method", "moving"),
                                                        polyorder=int(p.get("polyorder", 2) or 2)),
                        f"smooth {p.get('method', 'moving')} span {int(win)}")


def _tfilter(graph, node, ctx):
    p = node.get("params", {})
    cut = float(p.get("cutoff_hz") or 0)
    if cut <= 0:
        return dict(summary="cutoff 0 → no-op", notes=[], blocks=[{"kind": "text", "lines": [
            "A cutoff of 0 Hz leaves the rate untouched."]}])
    return _cond_traces(ctx, lambda r, fs: dsp.apply_temporal_filter(
        r, ftype=p.get("ftype", "butterworth"), mode=p.get("mode", "lowpass"), cutoff_hz=cut, fs=fs,
        order=int(p.get("order", 2) or 2), taps=int(p.get("taps", 33) or 33)),
        f"{p.get('mode', 'lowpass')} {p.get('ftype', 'butterworth')} {cut:g} Hz")


def _sf_context(ctx):
    eps = region_spikes(ctx)
    s, e = ctx["_window"]
    return {"spikes": [ep["kept"] for ep in eps], "t0": s, "t1": e}


def _sf_transform(graph, node, ctx):
    sig = sf.signal_at(graph, node["id"], _sf_context(ctx))
    spec = sf.time_trace(sig, epoch_mode="overlay")
    return dict(summary=f"{sig.kind} · {sig.n_epochs} epoch(s)"
                        + (f" · fs {sig.fs:g} Hz · {max((len(e) for e in sig.epochs), default=0)} samples"
                           if sig.fs else ""),
                notes=["The signal leaving this block, drawn as a time trace (every epoch)."],
                blocks=[_spec_block(spec, pipe.COMPONENT_REGISTRY[node["type"]]["label"])])


def _sf_sink(graph, node, ctx):
    res = sf.evaluate(graph, _sf_context(ctx))
    spec = res.get(node["id"], {"kind": "error", "error": "not connected to a source", "traces": []})
    if spec.get("kind") == "error":
        return dict(summary="", notes=[], blocks=[{"kind": "error", "error": spec["error"]}])
    return dict(summary=f"{len(spec.get('traces', []))} trace(s)", notes=[],
                blocks=[_spec_block(spec, pipe.COMPONENT_REGISTRY[node["type"]]["label"])])


def _spec_block(spec, title):
    kind = "bars" if spec.get("kind") == "hist" else "traces"
    return {"kind": kind, "title": title, "x_title": spec.get("x_title", ""),
            "y_title": spec.get("y_title", ""),
            "traces": [{"x": np.asarray(t["x"]), "y": np.asarray(t["y"]), "name": t.get("name", ""),
                        "width": 1.2} for t in spec.get("traces", [])]}


def _flicker(graph, node, ctx):
    det = _det(ctx)
    par = FlickerParadigm(current_channel=ctx.get("chan") or "Im_prime",
                          ttl_channel=ctx.get("ttl") or "TTL",
                          **{k: v for k, v in det.items() if k in ("polarity", "method", "k",
                                                                   "abs_threshold", "refractory_s")})
    amap = ctx.get("abs_map") or {}
    rows, trials = [], []
    for r in recordings(ctx):
        if r.get("rec") is None:
            continue
        eff = amap.get(r["path"], amap.get(r["file"]))
        par.abs_threshold = float(eff) if eff is not None else det.get("abs_threshold")
        res = par.analyze_recording(r["rec"], name=r["file"])
        rows.append([r["file"], _fmt(res.freq, 2), res.n_in_region, _fmt(res.vector_strength, 3),
                     _fmt(res.rayleigh_p, 4)])
        st, fl = par.detect(r["rec"])
        if fl is not None:
            trials.append(dict(spikes=st.times, on=fl.on_edges, off=fl.off_edges,
                               dur=r["duration"], freq=fl.freq))
    n_sh = int(ctx.get("n_shuffle") or 100)
    blocks = [{"kind": "table", "columns": ["file", "Hz", "spikes in region", "vector strength",
                                            "Rayleigh p"], "rows": rows}]
    summary = f"{len(rows)} epoch(s)"
    if trials:
        grp = par.group_from_trials(trials, n_shuffle=n_sh)
        for which in ("on", "off"):
            g = grp[which]
            c = np.asarray(g["centers"]) * 1000.0
            blocks.append({"kind": "traces", "title": f"light {which.upper()} · {g['ratio']:.2f}× @ "
                                                      f"{g['peak_ms']:.0f} ms · p={g['p']:.3f}",
                           "x_title": "time from transition (ms)", "y_title": "spikes/s",
                           "traces": [{"x": c, "y": np.asarray(g["rate"]), "name": "observed", "width": 1.6},
                                      {"x": c, "y": np.asarray(g["band"]), "name": "null 99%",
                                       "width": 1.0, "dash": "dash", "color": "#d9534f"}],
                           "vlines": [0.0]})
        summary += (f" · {grp['freq']:.2f} Hz · ON {grp['on']['ratio']:.2f}× p={grp['on']['p']:.3f}"
                    f" · OFF {grp['off']['ratio']:.2f}× p={grp['off']['p']:.3f}")
    return dict(summary=summary,
                notes=[f"Preview shuffles: {n_sh} (Run uses the node's 'shuffles' = "
                       f"{node.get('params', {}).get('n_shuffle', 500)}). Pooled ON/OFF uses the "
                       "flicker detector's own steady window per file, not the region boxes."],
                blocks=blocks)


def _sta(graph, node, ctx):
    filt = node.get("params", {}).get("filter_s")
    res = seed_result(ctx, "gaussian_noise", filter_s=(float(filt) if filt else None))
    out, info = res["sta"], res["info"]
    t = np.asarray(out["time_ms"]); avg = np.asarray(out["average"]); per = np.asarray(out["per_epoch"])
    pk = int(np.argmax(np.abs(avg)))
    sign = "OFF" if avg[pk] < 0 else "ON"
    traces = []
    for i, row in enumerate(per):
        d = float(np.max(np.abs(row))) or 1.0
        traces.append({"x": t, "y": row / d, "name": info[i]["file"] if i < len(info) else f"epoch {i+1}",
                       "width": 0.6, "color": "#c8c8c8"})
    traces.append({"x": t, "y": avg, "name": "average STA", "width": 2.0, "color": "#1f77b4"})
    tune = {"kind": "traces", "title": "temporal tuning |FFT(STA)|", "x_title": "frequency (Hz)",
            "y_title": "amplitude", "xrange": [0, 40],
            "traces": [{"x": np.asarray(out["freqs"]), "y": np.asarray(out["tuning"]),
                        "name": "tuning", "width": 1.5, "color": "#7b3fa0"}]}
    n_rec = sum(1 for i in info if i.get("clock", "").startswith("measured"))
    notes = [f"filter {res['filter_s']:g} s = {res['filter_len']} taps at {res['bin_rate']} Hz "
             f"({BINS_PER_UPDATE['gaussian_noise']} bins per update)",
             f"FFT cross-correlation per epoch, stimulus and response zero-padded by {ZERO_PAD} "
             f"bins → " + ("no circular wrap-around for any lag kept"
                           if res["filter_len"] <= ZERO_PAD else
                           f"lags beyond {ZERO_PAD} bins ({1000 * ZERO_PAD / res['bin_rate']:.0f} ms) "
                           "include a small circular wrap term"),
             f"stimulus contrast = v − mean (linear light, regenerated from the seed); epochs "
             f"averaged, average ÷ max|·|",
             f"spikes ↔ frames: {n_rec}/{len(info)} epoch(s) binned on the update clock measured "
             f"from the sync bar (table) — automatic, no region or nudge involved"]
    if res["filter_len"] > ZERO_PAD:
        notes.append("⚠ filter longer than the zero padding (60 bins)")
    return dict(summary=f"{res['n_epochs']} epoch(s) · STA peak {t[pk]:.1f} ms {sign}",
                notes=notes, clock=f"{n_rec}/{len(info)} epochs on the measured sync-bar clock",
                blocks=[{"kind": "traces", "title": f"STA (linear filter) — peak {t[pk]:.1f} ms [{sign}]",
                         "x_title": "lag (ms)", "y_title": "filter (÷max)", "traces": traces,
                         "vlines": [float(t[pk])]},
                        tune, sync_table(info)])


def _strf(graph, node, ctx):
    res = seed_result(ctx, "checkerboard")
    strf, info = res["strf"], res["info"]
    py, px = strf.peak_yx
    temporal = np.asarray(strf.temporal)
    sign = "OFF" if temporal[int(np.argmax(np.abs(temporal)))] < 0 else "ON"
    n_rec = sum(1 for i in info if i.get("clock", "").startswith("measured"))
    return dict(summary=f"{res['n_epochs']} epoch(s) · peak check (y={py}, x={px}) @ "
                        f"{strf.peak_time_ms:.1f} ms {sign}",
                clock=f"{n_rec}/{len(info)} epochs on the measured sync-bar clock",
                # the whole cube, so a viewer can show ANY check's kernel (or a mean of several)
                cube=np.asarray(strf.strf, float), time_ms=np.asarray(strf.time_ms, float),
                peak_yx=(int(py), int(px)),
                notes=[f"{res['filter_len']} taps at {res['bin_rate']:.2f} Hz (one per update); "
                       f"zero-padded by {ZERO_PAD} bins (≥ the filter, so no circular wrap)",
                       "epochs concatenated in time (all frame↔spike pairs pooled), one FFT per check",
                       f"spikes ↔ frames: {n_rec}/{len(info)} epoch(s) on the measured update clock "
                       "(table) — automatic"],
                blocks=[{"kind": "heatmap", "title": f"spatial RF (peak-lag frame) — peak (y={py}, x={px})",
                         "z": np.asarray(strf.spatial_rf), "mark": (int(py), int(px))},
                        {"kind": "traces", "title": f"temporal filter @ peak — {strf.peak_time_ms:.1f} ms [{sign}]",
                         "x_title": "lag (ms)", "y_title": "filter",
                         "traces": [{"x": np.asarray(strf.time_ms), "y": temporal, "name": "peak check",
                                     "width": 2.0, "color": "#1f77b4"}],
                         "vlines": [float(strf.peak_time_ms)]},
                        sync_table(info)])


def _figures(graph, node, ctx):
    fam = pipe.pipeline_stim_family(graph) or "sq_wave"
    figs = pipe.COMPONENT_REGISTRY["figures"]["figures_by_stim"].get(fam, [])
    lines = [f"analysis family: {fam}", "files written by ▶ Run (outputs/<run name>/):"] + \
            [f"  • {f}" for f in figs] + \
            ["  • result.json (+ metrics.csv for sq wave)",
             f"4K PDF/SVG exports: {'on' if node.get('params', {}).get('export_4k', True) else 'off'}"]
    # existing runs of this family on the selected cell(s)
    seen = set()
    for p in ctx.get("files") or []:
        try:
            cdir = Path(p).parents[1]
        except (IndexError, ValueError):
            continue
        if str(cdir) in seen:
            continue
        seen.add(str(cdir))
        try:
            cm = (ctx.get("cell_of") or _cell_of)(p)
        except Exception:
            continue
        outs = cm.data.get("outputs", []) or []
        if outs:
            lines.append(f"existing runs in {cdir.name}: " + ", ".join(
                f"{o.get('analysis')} ({len(o.get('files') or {})} files)" for o in outs[-6:]))
    return dict(summary=f"{len(figs)} figure set(s) for {fam}",
                notes=["Nothing is written in step-through mode — press ▶ Run on current cell."],
                blocks=[{"kind": "text", "lines": lines}])


_EVAL = {
    "source": _source, "align": _align, "detect": _detect, "region": _region,
    "smooth": _smooth, "tfilter": _tfilter,
    "flicker": _flicker, "sta": _sta, "strf": _strf, "figures": _figures,
}


def family_output(ctx, family, filter_s=None) -> dict:
    """The STA (`gaussian_noise`) / STRF (`checkerboard`) node's step-through output for `ctx`
    without a graph — what the Analysis View's lower panels show for those pipelines."""
    ntype = {"gaussian_noise": "sta", "checkerboard": "strf"}[family]
    node = {"id": family, "type": ntype, "params": ({"filter_s": filter_s} if filter_s else {})}
    return node_output({"nodes": [node], "connections": []}, family, ctx)


def node_output(graph, node_id, ctx) -> dict:
    """Evaluate ONE node of `graph` for the live `ctx` (see the module docstring). Never raises:
    a failure comes back as an "error" block so the GUI can show it beside the node."""
    nodes = {n["id"]: n for n in (graph or {}).get("nodes", [])}
    node = nodes.get(node_id)
    if node is None:
        return dict(title=str(node_id), summary="", notes=[],
                    blocks=[{"kind": "error", "error": "no such node in the graph"}])
    t = node.get("type")
    label = pipe.COMPONENT_REGISTRY.get(t, {}).get("label", t)
    if t in sf.TRANSFORMS:
        fn = _sf_transform
    elif t in sf.SINKS:
        fn = _sf_sink
    else:
        fn = _EVAL.get(t)
    if fn is None:
        return dict(title=label, summary="", notes=[],
                    blocks=[{"kind": "text", "lines": ["no step-through view for this component"]}])
    try:
        out = fn(graph, node, ctx)
    except Exception as e:                                        # surface, don't crash the strip
        return dict(title=label, summary="", notes=[], blocks=[{"kind": "error", "error": str(e)}])
    out.setdefault("notes", [])
    out.setdefault("summary", "")
    out["title"] = label
    out["type"] = t
    return out
