"""Signal-flow engine for the Analysis Pipeline — the discrete, transparent DSP layer.

The philosophy: instead of a few monolithic "do the whole analysis" nodes, a kit of **small,
single-purpose blocks**, each a pure ``Signal -> Signal`` function whose math you can read, wired
into an arbitrary pathway and pointed at a **display sink**. This plugs into the part of the
pipeline that already works one-step-at-a-time (``source -> align -> detect -> region``): the
``region`` node emits spikes, and from there ::

    spikes ─▶ Bin ─▶ Smooth ─▶ Frequency filter ─▶ [FFT Power spectrum]

is a real signal-flow graph this module evaluates.

A :class:`Signal` is **per-epoch** (a list of 1-D arrays, one per recording/epoch) so single-trial
structure is preserved; each block maps over the epochs. Sinks reduce across epochs by a selectable
mode (``overlay`` all, ``average`` the mean, or operate on a ``pool``ed single series).

Two signal kinds:
  * ``events`` — spike times per epoch (fs is None); produced by the spike tap, consumed by Bin/ISI.
  * ``series`` — a uniformly sampled trace per epoch (has fs); everything downstream of Bin.

Pure numpy + scipy.signal (both in the conda base). No GUI deps — tested in
``tests/test_signal_flow.py``. The FFT power uses the SAME single-sided ``W = 2|X[k]|²/N²`` formula
as the validated spike-power graph, so a binned-rate pathway reproduces it exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .analysis import dsp


# =================================================================================================
# The Signal — per-epoch, kind = "events" (spike times) | "series" (sampled trace)
# =================================================================================================
@dataclass
class Signal:
    epochs: list                      # list[np.ndarray]: spike times (events) or samples (series)
    fs: float | None = None           # samples/sec for a series; None for events
    t0: float = 0.0                   # window start (s)
    t1: float | None = None           # window end (s) — set for events (binning span)
    units: str = ""
    label: str = ""
    kind: str = "series"              # "events" | "series"

    @property
    def n_epochs(self) -> int:
        return len(self.epochs)

    def map(self, fn, **kw) -> "Signal":
        """Apply ``fn`` to each epoch's array, returning a new Signal with the same metadata
        (overridable via kw)."""
        new = [np.asarray(fn(np.asarray(e, dtype=float)), dtype=float) for e in self.epochs]
        meta = dict(fs=self.fs, t0=self.t0, t1=self.t1, units=self.units,
                    label=self.label, kind=self.kind)
        meta.update(kw)
        return Signal(new, **meta)


def events_signal(spikes_per_epoch, t0=0.0, t1=None, label="spikes") -> Signal:
    """A spike-time (events) Signal from a list of per-epoch spike-time arrays."""
    eps = [np.asarray(s, dtype=float) for s in (spikes_per_epoch or [])]
    if t1 is None:
        t1 = max((float(e.max()) for e in eps if e.size), default=t0 + 1.0)
    return Signal(eps, fs=None, t0=float(t0), t1=float(t1), units="s",
                  label=label, kind="events")


def series_signal(traces_per_epoch, fs, t0=0.0, units="", label="") -> Signal:
    """A sampled-series Signal from per-epoch 1-D arrays at sample rate ``fs``."""
    eps = [np.asarray(x, dtype=float) for x in (traces_per_epoch or [])]
    n = max((e.size for e in eps), default=0)
    return Signal(eps, fs=float(fs), t0=float(t0), t1=float(t0 + n / fs if fs else t0),
                  units=units, label=label, kind="series")


# =================================================================================================
# Transform blocks — each a pure Signal -> Signal, mapping over epochs. The math is in the docstring
# and mirrored in the node's `math` field.
# =================================================================================================
def bin_spikes(sig: Signal, *, bin_ms=5.0, kernel="boxcar", unit="hz") -> Signal:
    """events -> series. Count spikes into Δ = bin_ms bins over [t0, t1]; rate r[n] = count/Δ (Hz)
    or raw count. 'gaussian' convolves the counts with a Gaussian of σ = one bin (smoother rate)."""
    if sig.kind != "events":
        raise ValueError("bin_spikes expects an events (spike-time) signal")
    dt = max(1e-4, float(bin_ms) / 1000.0)
    t0, t1 = sig.t0, (sig.t1 if sig.t1 is not None else sig.t0 + 1.0)
    n = max(1, int(round((t1 - t0) / dt)))
    edges = t0 + np.arange(n + 1) * dt
    out = []
    for ev in sig.epochs:
        cnt, _ = np.histogram(ev, bins=edges)
        r = cnt.astype(float)
        if kernel == "gaussian":
            k = np.exp(-0.5 * (np.arange(-3, 4)) ** 2)      # σ = 1 bin
            k /= k.sum()
            r = np.convolve(np.pad(r, 3, mode="reflect"), k, mode="valid")
        if unit == "hz":
            r = r / dt
        out.append(r)
    return series_signal(out, fs=1.0 / dt, t0=t0,
                         units=("Hz" if unit == "hz" else "count"),
                         label=f"{sig.label}→rate")


def resample(sig: Signal, *, mode="factor", factor=2.0, new_fs=None, method="poly") -> Signal:
    """series -> series at a new rate. Upsampling interpolates; downsampling anti-aliases first
    (scipy resample_poly = polyphase FIR), so it can't alias. `mode`='factor' uses `factor`
    (>1 up, <1 down); 'rate' targets `new_fs`. Reports the realized fs."""
    _require_series(sig, "resample")
    fs = sig.fs
    if mode == "rate" and new_fs:
        target = float(new_fs)
    else:
        target = fs * float(factor)
    target = max(1e-6, target)
    frac = Fraction(target / fs).limit_denominator(1000)
    up, down = max(1, frac.numerator), max(1, frac.denominator)
    from scipy.signal import resample_poly
    out = [resample_poly(e, up, down) if e.size > 1 else e for e in sig.epochs]
    real_fs = fs * up / down
    return series_signal(out, fs=real_fs, t0=sig.t0, units=sig.units,
                         label=f"{sig.label}→{real_fs:.4g}Hz")


def smooth(sig: Signal, *, method="moving", window=4, polyorder=2) -> Signal:
    """series -> series. Low-pass smoothing (reuses the tested dsp.smooth_1d): MATLAB moving
    average / Gaussian / Savitzky-Golay. Span is in samples."""
    _require_series(sig, "smooth")
    return sig.map(lambda e: dsp.smooth_1d(e, window=window, method=method, polyorder=polyorder),
                   label=f"{sig.label}→smooth")


def freq_filter(sig: Signal, *, ftype="butterworth", fmode="lowpass", cutoff_hz=30.0,
                order=2, taps=33) -> Signal:
    """series -> series. Zero-phase FIR low/high-pass (reuses the tested dsp.apply_temporal_filter);
    cutoff in Hz. High-pass = δ − low-pass (spectral inversion)."""
    _require_series(sig, "freq_filter")
    fs = sig.fs
    return sig.map(lambda e: dsp.apply_temporal_filter(
        e, ftype=ftype, mode=fmode, cutoff_hz=cutoff_hz, fs=fs, order=order, taps=taps),
        label=f"{sig.label}→{fmode}")


def detrend(sig: Signal, *, mode="mean") -> Signal:
    """series -> series. Remove the mean ('mean') or a least-squares line ('linear'). Do this
    before an FFT so the DC/ramp doesn't dominate the spectrum."""
    _require_series(sig, "detrend")
    if mode == "linear":
        from scipy.signal import detrend as _dt
        return sig.map(lambda e: _dt(e, type="linear") if e.size > 1 else e,
                       label=f"{sig.label}→detrend")
    return sig.map(lambda e: e - e.mean() if e.size else e, label=f"{sig.label}→demean")


def window(sig: Signal, *, wtype="hann") -> Signal:
    """series -> series. Multiply each epoch by a taper (Hann/Hamming/Blackman/Tukey) to cut
    spectral leakage before an FFT. 'boxcar' = no taper."""
    _require_series(sig, "window")
    from scipy.signal import get_window
    return sig.map(lambda e: e * get_window(wtype, e.size) if e.size > 1 else e,
                   label=f"{sig.label}→{wtype}")


def _require_series(sig: Signal, who: str):
    if sig.kind != "series":
        raise ValueError(f"{who} expects a sampled series (add a Bin block after spikes first)")


TRANSFORMS = {
    "sf_bin": bin_spikes,
    "sf_resample": resample,
    "sf_smooth": smooth,
    "sf_filter": freq_filter,
    "sf_detrend": detrend,
    "sf_window": window,
}


# =================================================================================================
# Sinks — a Signal -> a plot spec (dict of traces). `epoch_mode`: overlay every epoch, or average
# them (a single mean trace on a common grid).
# =================================================================================================
def _power_1d(x: np.ndarray, fs: float):
    """Single-sided FFT power W = 2|X[k]|²/N² of the mean-removed series; DC dropped. Matches the
    validated spike-power graph exactly."""
    r = np.asarray(x, dtype=float)
    if r.size < 4:
        return np.array([]), np.array([])
    r = r - r.mean()
    n = r.size
    X = np.fft.rfft(r)
    P = 2.0 * np.abs(X) ** 2 / (n ** 2)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    keep = f > 0
    return f[keep], P[keep]


def fft_power(sig: Signal, *, epoch_mode="average", fmax=None) -> dict:
    """series -> power spectrum. Per-epoch W = 2|X|²/N²; 'average' means the per-epoch spectra are
    interpolated to a common frequency grid and averaged."""
    _require_series(sig, "FFT Power")
    per = [_power_1d(e, sig.fs) for e in sig.epochs]
    per = [(f, p) for f, p in per if f.size]
    if not per:
        return {"kind": "fft", "x_title": "frequency (Hz)", "y_title": "power (W)", "traces": []}
    if epoch_mode == "average":
        # average only over the frequency band ALL epochs cover (no NaN edges to nan-mean over)
        lo = max(f[0] for f, _ in per)
        hi = min(f[-1] for f, _ in per)
        grid = np.linspace(lo, hi, max(len(f) for f, _ in per))
        mean = np.mean(np.vstack([np.interp(grid, f, p) for f, p in per]), axis=0)
        traces = [{"x": grid.tolist(), "y": mean.tolist(), "name": "epoch mean"}]
    else:                                             # overlay
        traces = [{"x": f.tolist(), "y": p.tolist(), "name": f"epoch {i+1}"}
                  for i, (f, p) in enumerate(per)]
    if fmax:
        for t in traces:
            xs = np.asarray(t["x"]); m = xs <= float(fmax)
            t["x"] = xs[m].tolist(); t["y"] = np.asarray(t["y"])[m].tolist()
    return {"kind": "fft", "x_title": "frequency (Hz)", "y_title": "power (W, R=1Ω)",
            "traces": traces}


def time_trace(sig: Signal, *, epoch_mode="overlay") -> dict:
    """series -> time-domain traces (t vs value). 'average' means across epochs on a common grid."""
    _require_series(sig, "Time trace")
    def t_of(e):
        return (sig.t0 + np.arange(e.size) / sig.fs)
    if epoch_mode == "average" and sig.n_epochs > 1:
        n = min((e.size for e in sig.epochs), default=0)
        if n:
            mean = np.mean([e[:n] for e in sig.epochs], axis=0)
            t = sig.t0 + np.arange(n) / sig.fs
            traces = [{"x": t.tolist(), "y": mean.tolist(), "name": "epoch mean"}]
        else:
            traces = []
    else:
        traces = [{"x": t_of(e).tolist(), "y": e.tolist(), "name": f"epoch {i+1}"}
                  for i, e in enumerate(sig.epochs)]
    return {"kind": "time", "x_title": "time (s)", "y_title": sig.units or "amplitude",
            "traces": traces}


def isi_hist(sig: Signal, *, bin_ms=2.0, epoch_mode="pool") -> dict:
    """events -> inter-spike-interval histogram (ms). Pools intervals across epochs by default."""
    if sig.kind != "events":
        raise ValueError("ISI expects an events (spike-time) signal — wire it from spikes, not a bin")
    isis = []
    for ev in sig.epochs:
        ev = np.sort(ev)
        if ev.size > 1:
            isis.append(np.diff(ev) * 1000.0)
    if not isis:
        return {"kind": "hist", "x_title": "ISI (ms)", "y_title": "count", "traces": []}
    allisi = np.concatenate(isis)
    hi = np.percentile(allisi, 99) if allisi.size else 1.0
    shown = allisi[allisi <= hi]
    dt = max(0.1, float(bin_ms))
    edges = np.arange(0, (hi if hi > 0 else 1.0) + dt, dt)
    cnt, e = np.histogram(shown, bins=edges)
    centers = 0.5 * (e[:-1] + e[1:])
    return {"kind": "hist", "x_title": "ISI (ms)", "y_title": "count",
            "traces": [{"x": centers.tolist(), "y": cnt.tolist(), "name": f"{allisi.size} intervals"}]}


SINKS = {
    "sf_fft": fft_power,
    "sf_time": time_trace,
    "sf_isi": isi_hist,
}

# node types this engine knows how to run (used by the UI to tell signal-flow nodes apart)
SIGNAL_NODE_TYPES = set(TRANSFORMS) | set(SINKS)


# =================================================================================================
# Evaluator — topo-sort the signal-flow portion of a pipeline graph and run it. `context` carries
# the live inputs: detected spikes per epoch (from the Analysis-View detection), the window, etc.
# =================================================================================================
def _param(node, key, default=None):
    return node.get("params", {}).get(key, default)


def evaluate(graph: dict, context: dict) -> dict:
    """Run every signal-flow pathway in `graph` and return ``{sink_node_id: plot_spec}``.

    `context` keys: ``spikes`` (list of per-epoch spike-time arrays, REQUIRED for spike pathways),
    ``t0``/``t1`` (window seconds), ``analog`` (optional list of per-epoch (trace, fs) for an analog
    tap), ``epoch_mode`` (default overlay/average for sinks that don't set their own).

    Nodes upstream of the signal layer (source/align/detect/region) are treated as the spike
    source: any signal-flow node whose input is fed by a non-signal node receives the context
    spikes as an events Signal. Blocks are pure; a bad block drops that pathway to an error spec
    rather than killing the whole evaluation.
    """
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    conns = graph.get("connections", [])
    incoming = {}                                     # node_id -> upstream node_id feeding it
    for c in conns:
        incoming[c["to_node"]] = c["from_node"]

    spikes = context.get("spikes")
    t0 = float(context.get("t0", 0.0) or 0.0)
    t1 = context.get("t1")
    default_mode = context.get("epoch_mode", "overlay")
    seed = (events_signal(spikes, t0=t0, t1=t1) if spikes is not None else None)

    memo: dict = {}                                   # node_id -> Signal (cached)

    def upstream_signal(nid, seen):
        src = incoming.get(nid)
        if src is None:
            return seed                               # dangling input → the spike source
        if src in nodes and nodes[src]["type"] in SIGNAL_NODE_TYPES:
            return signal_of(src, seen)               # another signal block
        return seed                                   # fed by source/align/detect/region → spikes

    def signal_of(nid, seen):
        if nid in memo:
            return memo[nid]
        if nid in seen:
            raise ValueError("cycle in the signal graph")
        seen = seen | {nid}
        node = nodes[nid]
        fn = TRANSFORMS[node["type"]]
        sig_in = upstream_signal(nid, seen)
        if sig_in is None:
            raise ValueError("no spike/signal source reaches this block")
        out = fn(sig_in, **_block_kwargs(node))
        memo[nid] = out
        return out

    results = {}
    for nid, node in nodes.items():
        if node["type"] not in SINKS:
            continue
        try:
            sink_in = upstream_signal(nid, set())
            if sink_in is None:
                raise ValueError("connect a spike source (region → … → this display)")
            kw = _sink_kwargs(node, default_mode)
            results[nid] = SINKS[node["type"]](sink_in, **kw)
        except Exception as e:
            results[nid] = {"kind": "error", "error": str(e), "traces": []}
    return results


def _block_kwargs(node) -> dict:
    """Map a node's params onto its transform function's kwargs (only known keys)."""
    t, p = node["type"], node.get("params", {})
    if t == "sf_bin":
        return {"bin_ms": p.get("bin_ms", 5.0), "kernel": p.get("kernel", "boxcar"),
                "unit": p.get("unit", "hz")}
    if t == "sf_resample":
        return {"mode": p.get("mode", "factor"), "factor": p.get("factor", 2.0),
                "new_fs": p.get("new_fs"), "method": p.get("method", "poly")}
    if t == "sf_smooth":
        return {"method": p.get("method", "moving"), "window": p.get("window", 4),
                "polyorder": p.get("polyorder", 2)}
    if t == "sf_filter":
        return {"ftype": p.get("ftype", "butterworth"), "fmode": p.get("fmode", "lowpass"),
                "cutoff_hz": p.get("cutoff_hz", 30.0), "order": p.get("order", 2),
                "taps": p.get("taps", 33)}
    if t == "sf_detrend":
        return {"mode": p.get("mode", "mean")}
    if t == "sf_window":
        return {"wtype": p.get("wtype", "hann")}
    return {}


def _sink_kwargs(node, default_mode) -> dict:
    t, p = node["type"], node.get("params", {})
    if t == "sf_fft":
        return {"epoch_mode": p.get("epoch_mode", default_mode), "fmax": p.get("fmax")}
    if t == "sf_time":
        return {"epoch_mode": p.get("epoch_mode", default_mode)}
    if t == "sf_isi":
        return {"bin_ms": p.get("bin_ms", 2.0), "epoch_mode": "pool"}
    return {}
