"""
Analysis pipelines — a saveable, editable graph of processing components that lives OVER the
tested CLI/`run` orchestration.

A pipeline is a declarative description of an analysis: a set of **component nodes** (each with
typed input/output ports and internal parameters) wired by **connections**. It is NOT a new
analysis engine — running a pipeline extracts its node parameters and dispatches to the existing,
regression-tested `run_cell_flicker` / `run_cell_noise` / `run_cell_checkerboard` in `run.py`
(the same way the GUI's "Run analysis" always has). The graph is the human-editable face; the
terminal *analysis* node picks which run function fires.

The canonical "Run analysis" flow is itself a pipeline (`default_flicker_pipeline`):

    Cell recordings ─▶ Align frame-syncs ─▶ Detect spikes ─▶ Select region ─▶ Flicker ON/OFF ─▶ Figures

Storage: one JSON file per pipeline under `<store_root>/pipelines/<slug>.json` (the same
`~/Documents/ephysdataio` data structure as everything else). "Save as" is a plain copy under a
new name.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import re
from datetime import datetime
from pathlib import Path

from .dataio.config import data_root

# ---------------------------------------------------------------------------------------------
# Component registry: the palette. Each entry defines a component's ports and its parameter
# schema. Ports carry a `type` so the editor can validate a connection (out.type == in.type).
# Param schema entries: {type: number|choice|bool|text, default, label, options?, min?, max?, step?}.
# `terminal` marks the analysis components — the pipeline's terminal node decides which run_cell_*
# fires and under which stimulus family.
# ---------------------------------------------------------------------------------------------
PORT_RECORDINGS = "recordings"      # a set of loaded .abf recordings (+ their frame-sync/TTL)
PORT_SPIKES = "spikes"              # detected spike trains per recording
PORT_RESULT = "result"             # an analysis Result (metrics + figure specs)
PORT_OUTPUTS = "outputs"           # written files (figures / csv / json)
PORT_SIGNAL = "signal"             # a per-epoch sampled 1-D series (signal-flow DSP layer)
PORT_DISPLAY = "display"           # a rendered plot from a sink (FFT / time / ISI …)

COMPONENT_REGISTRY: dict = {
    "source": {
        "label": "Cell recordings",
        "category": "source",
        "color": "#2d7d46",
        "help": "The selected cell's .abf recordings (from the Analysis View selection).",
        "desc": ("The pipeline's input: the recordings checked in the Analysis View for the "
                 "current cell. Each .abf carries the signal channel (Im_prime / Vm_sec), the "
                 "frame-clock TTL, and its sample rate. One recording = one epoch."),
        "math": ["No computation — this node hands the selected epochs downstream.",
                 "signal xᵢ(t), frame-clock TTLᵢ(t), sample rate fs   for each epoch i"],
        "inputs": [],
        "outputs": [{"name": "recordings", "type": PORT_RECORDINGS}],
        "params": {},
    },
    "align": {
        "label": "Align frame-syncs",
        "category": "processing",
        "color": "#2f6fb0",
        "help": "Nudge each file so its frame-sync (trial start) lines up before pooling.",
        "desc": ("Shifts every epoch in time so its frame-clock (trial start) lines up before "
                 "the epochs are pooled/averaged. 'auto' seeds each offset from the TTL's first "
                 "onset, then refines to sample precision by FFT cross-correlating the two "
                 "frame-sync waveforms. The signal, TTL, and spikes all move together."),
        "math": ["seed:   τ₀ᵢ = first TTL onset of epoch i",
                 "refine: τᵢ* = argmaxτ  Σₜ syncᵢ(t)·sync_ref(t+τ)     (cross-correlation peak)",
                 "apply:  t → t − τᵢ*   for signal, TTL and spikes of epoch i"],
        "inputs": [{"name": "recordings", "type": PORT_RECORDINGS}],
        "outputs": [{"name": "aligned", "type": PORT_RECORDINGS}],
        "params": {
            "mode": {"type": "choice", "default": "auto", "label": "mode",
                     "options": ["auto", "off"]},
        },
    },
    "detect": {
        "label": "Detect spikes",
        "category": "processing",
        "color": "#8e44ad",
        "help": "Threshold-detect spikes on the signal channel (per-trace thresholds honored).",
        "desc": ("Finds spikes on the signal channel by threshold crossing. 'mad' sets the "
                 "threshold from the robust noise level (median absolute deviation); 'matlab' "
                 "reproduces Sara's spikeDetectorOnline.m (500 Hz high-pass, max/3 threshold, "
                 "4σ noise gate). Polarity picks the crossing direction; a refractory period "
                 "suppresses double-counts."),
        "math": ["MAD(x) = median(|x − median(x)|),   σ̂ = 1.4826·MAD(x)",
                 "θ = k·σ̂        (mad)      θ = max|x|/3   (matlab)",
                 "spike at t where x(t) crosses ∓θ (polarity), gated by a refractory Δt",
                 "neg → −x,   pos → x,   abs → |x|  before thresholding"],
        "inputs": [{"name": "recordings", "type": PORT_RECORDINGS}],
        "outputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "params": {
            "polarity": {"type": "choice", "default": "neg", "label": "polarity",
                         "options": ["neg", "pos", "abs"]},
            "method": {"type": "choice", "default": "mad", "label": "algorithm",
                       "options": ["mad", "abs", "mad_floor", "matlab"]},
            "k": {"type": "number", "default": 6, "label": "k·MAD", "min": 2, "max": 15, "step": 0.5},
            "refractory_ms": {"type": "number", "default": 2, "label": "refractory (ms)",
                              "min": 0, "step": 0.5},
            "abs_threshold": {"type": "number", "default": None, "label": "abs threshold"},
        },
    },
    "region": {
        "label": "Select region",
        "category": "processing",
        "color": "#c0803a",
        "help": "Restrict the analysis to the [start, end] window (drops the adapting block).",
        "desc": ("Restricts the analysis to a [start, end] time window — typically to drop the "
                 "pre-stimulus adapting block and keep only the steady-state response. Blank "
                 "start/end use the auto-detected stimulus window: the sync-bar envelope for a "
                 "seeded noise / checkerboard file (exactly the stimulus frames — the 2 s "
                 "pre-stimulus gray is excluded), the flicker detector's steady region for a square "
                 "wave. 'crop' hard-trims; otherwise the window is a mask. The square-wave analysis "
                 "and the signal-flow blocks consume only the spikes inside it; the STA / STRF do "
                 "NOT use it — each seeded epoch is binned on the update grid measured from its "
                 "own sync bar, so the window is implicit there."),
        "math": ["keep spikes with  t_start ≤ tⱼ ≤ t_end",
                 "blank ⇒ auto window: envelope (t_on, t_off) of the sync bar for seeded noise,",
                 "        flicker steady region (t₀, t₁) for a square wave"],
        "inputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "outputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "params": {
            "start_s": {"type": "number", "default": None, "label": "start (s)"},
            "end_s": {"type": "number", "default": None, "label": "end (s)"},
            "crop": {"type": "bool", "default": False, "label": "crop to region"},
        },
    },

    # ── Signal-flow DSP kit (neitz/signal_flow.py) — discrete blocks, each a pure Signal→Signal
    #    with the math on its face. Bin turns the region's spikes into a sampled rate; the rest are
    #    signal→signal; the display sinks (FFT/time/ISI) render a pathway. A "signal" is per-epoch. ──
    "sf_bin": {
        "label": "Spike binning",
        "category": "transform",
        "color": "#2f8f8f",
        "help": "Spikes → a sampled rate: count spikes into Δ-wide bins, r = count/Δ (Hz).",
        "desc": ("Turns spike times into a uniformly-sampled rate trace — the entry point to the "
                 "signal kit. Counts the spikes falling in each Δ = bin-width window over the "
                 "region; the result is one rate value per bin (Hz), or raw counts. 'gaussian' "
                 "smooths the counts with a σ = 1-bin bell for a continuous rate. The bin width sets "
                 "the new sample rate fs = 1/Δ (and the FFT's Nyquist)."),
        "math": ["r[n] = (# spikes in [t₀+nΔ, t₀+(n+1)Δ)) / Δ    (Hz;  Δ = bin_ms/1000)",
                 "fs = 1/Δ   →   spectrum Nyquist = fs/2",
                 "gaussian: r ← r ∗ g,  g[k] ∝ exp(−k²/2),  σ = 1 bin"],
        "inputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "outputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "params": {
            "bin_ms": {"type": "number", "default": 5, "label": "bin (ms)", "min": 0.1, "step": 0.5},
            "kernel": {"type": "choice", "default": "boxcar", "label": "kernel",
                       "options": ["boxcar", "gaussian"]},
            "unit": {"type": "choice", "default": "hz", "label": "unit", "options": ["hz", "count"]},
        },
    },
    "sf_resample": {
        "label": "Resample (up/down)",
        "category": "transform",
        "color": "#3a7fb0",
        "help": "Change the sample rate — up (interpolate) or down (anti-alias + decimate).",
        "desc": ("Changes a signal's sample rate. Upsampling interpolates (more bins per period — "
                 "Sara's 'upsample to 6/15 bins per frame' trick); downsampling first low-passes to "
                 "the new Nyquist so it can't alias, then decimates. Polyphase FIR (scipy "
                 "resample_poly). 'factor' scales fs (×2 up, ×0.5 down); 'rate' targets a new fs. "
                 "The realized fs is reported on the wire."),
        "math": ["target fs = fs·factor   (factor) |   = new_fs   (rate)",
                 "up/down = rational approx of target/fs;  y = polyphase-FIR resample(x, up, down)",
                 "downsample anti-aliases at the new fs/2 before decimation (no aliasing)"],
        "inputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "outputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "params": {
            "mode": {"type": "choice", "default": "factor", "label": "mode",
                     "options": ["factor", "rate"]},
            "factor": {"type": "number", "default": 2, "label": "factor (×fs)", "min": 0.05, "step": 0.5},
            "new_fs": {"type": "number", "default": None, "label": "new fs (Hz)"},
        },
    },
    "sf_smooth": {
        "label": "Smooth",
        "category": "transform",
        "color": "#3a8f6f",
        "help": "Low-pass smooth a signal: moving-average / gaussian / Savitzky-Golay (dsp.smooth_1d).",
        "desc": ("Low-pass smoothing of a signal — the discrete, in-pathway version of the same "
                 "tested smoother. 'moving' is MATLAB smooth(y, span) (odd span, ends shrink); "
                 "'gaussian' weights by a bell of FWHM = span; 'savgol' fits a local polynomial "
                 "(preserves peak height/width). Span is in samples of THIS signal."),
        "math": ["moving:   ỹ[i] = mean(y[i−w … i+w]),  w = (span−1)/2 (shrinks at the ends)",
                 "gaussian: ỹ = y ∗ g,  σ = span/2.3548 (FWHM)",
                 "savgol:   local degree-p least-squares fit over `span` samples"],
        "inputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "outputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "params": {
            "method": {"type": "choice", "default": "moving", "label": "method",
                       "options": ["moving", "gaussian", "savgol"]},
            "window": {"type": "number", "default": 4, "label": "span (samples)", "min": 0, "step": 1},
            "polyorder": {"type": "number", "default": 2, "label": "savgol order", "min": 1,
                          "max": 6, "step": 1},
        },
    },
    "sf_filter": {
        "label": "Frequency filter",
        "category": "transform",
        "color": "#5a7fb0",
        "help": "In-line zero-phase FIR low/high-pass (Hz); dsp.apply_temporal_filter.",
        "desc": ("Applies a zero-phase FIR frequency filter to the signal, in-line in the pathway. "
                 "Low-pass keeps slow components, high-pass removes drift/DC; cutoff in Hz (relative "
                 "to THIS signal's fs). Types trade sharpness for ringing. The realized "
                 "(truncated-kernel) response is what runs."),
        "math": ["fc = cutoff_Hz / fs   (cycles/sample)",
                 "butterworth:  |H(f)| = 1/√(1+(f/fc)^{2n});  windowed-sinc for the others",
                 "high-pass = δ − low-pass (spectral inversion);  ỹ = y ∗ h (reflect-padded, zero phase)"],
        "inputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "outputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "params": {
            "fmode": {"type": "choice", "default": "lowpass", "label": "mode",
                      "options": ["lowpass", "highpass"]},
            "cutoff_hz": {"type": "number", "default": 30, "label": "cutoff (Hz)", "min": 0.1, "step": 1},
            "ftype": {"type": "choice", "default": "butterworth", "label": "type",
                      "options": ["gaussian", "butterworth", "ideal", "hamming", "hanning", "blackman"]},
            "order": {"type": "number", "default": 2, "label": "order", "min": 1, "max": 10, "step": 1},
            "taps": {"type": "number", "default": 33, "label": "taps", "min": 3, "max": 129, "step": 2},
        },
    },
    "sf_detrend": {
        "label": "Detrend / demean",
        "category": "transform",
        "color": "#7a8f3a",
        "help": "Remove the mean or a least-squares line — do this before an FFT.",
        "desc": ("Removes a constant ('mean') or a straight-line trend ('linear') from each epoch. "
                 "Run it before an FFT so the DC term or a slow ramp doesn't dominate and drag the "
                 "spectrum's low end (the power graph already drops the DC bin, but a ramp leaks)."),
        "math": ["mean:   y ← y − ⟨y⟩",
                 "linear: y ← y − (a·t + b),  (a,b) = least-squares fit"],
        "inputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "outputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "params": {
            "mode": {"type": "choice", "default": "mean", "label": "mode",
                     "options": ["mean", "linear"]},
        },
    },
    "sf_window": {
        "label": "Window (taper)",
        "category": "transform",
        "color": "#8f7a3a",
        "help": "Taper each epoch (Hann/Hamming/…) to cut FFT spectral leakage.",
        "desc": ("Multiplies each epoch by a taper that goes to ~0 at the ends, so a finite segment "
                 "doesn't leak a sinc skirt across the spectrum. Use before an FFT on a non-periodic "
                 "segment. 'boxcar' = no taper (rectangular)."),
        "math": ["y[n] ← y[n]·w[n],   w = Hann / Hamming / Blackman / Tukey / boxcar",
                 "Hann: w[n] = 0.5(1 − cos(2πn/(N−1)))"],
        "inputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "outputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "params": {
            "wtype": {"type": "choice", "default": "hann", "label": "window",
                      "options": ["hann", "hamming", "blackman", "tukey", "boxcar"]},
        },
    },
    "sf_fft": {
        "label": "FFT Power spectrum",
        "category": "display",
        "color": "#7d5bb0",
        "help": "Display sink: single-sided power W = 2|X[k]|²/N² of the signal.",
        "desc": ("A display sink — wire a pathway into it to SEE that pathway's power spectrum. "
                 "Single-sided FFT power, the same W = 2|X[k]|²/N² formula as the validated "
                 "spike-power graph, computed per epoch (mean removed, DC dropped). 'overlay' draws "
                 "every epoch; 'average' interpolates them to a common grid and means them."),
        "math": ["X[k] = rfft(y − ⟨y⟩),   W[k] = 2·|X[k]|² / N²   (single-sided, R=1Ω)",
                 "f[k] = k·fs/N,   DC (k=0) dropped",
                 "average: mean of the per-epoch W over the shared frequency band"],
        "inputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "outputs": [{"name": "display", "type": PORT_DISPLAY}],
        "params": {
            "epoch_mode": {"type": "choice", "default": "average", "label": "epochs",
                           "options": ["overlay", "average"]},
            "fmax": {"type": "number", "default": None, "label": "max freq (Hz)"},
        },
    },
    "sf_time": {
        "label": "Time trace",
        "category": "display",
        "color": "#b0863a",
        "help": "Display sink: the signal in the time domain (t vs value).",
        "desc": ("A display sink showing the signal itself against time — inspect what a pathway "
                 "actually looks like at any stage (the binned rate, after smoothing, after "
                 "filtering). 'overlay' draws each epoch; 'average' means them on a common grid."),
        "math": ["t[n] = t₀ + n/fs,   plot (t, y)"],
        "inputs": [{"name": "signal", "type": PORT_SIGNAL}],
        "outputs": [{"name": "display", "type": PORT_DISPLAY}],
        "params": {
            "epoch_mode": {"type": "choice", "default": "overlay", "label": "epochs",
                           "options": ["overlay", "average"]},
        },
    },
    "sf_isi": {
        "label": "ISI histogram",
        "category": "display",
        "color": "#b05b7d",
        "help": "Display sink: inter-spike-interval histogram (wire from spikes, not a bin).",
        "desc": ("A display sink for the inter-spike-interval distribution (ms). Wire it straight "
                 "from a spike source (Detect/Select-region), NOT from a Bin — it needs event times. "
                 "Intervals are pooled across epochs and shown up to the 99th percentile."),
        "math": ["ISIⱼ = 1000·(tⱼ₊₁ − tⱼ)  ms,   pooled over epochs → histogram"],
        "inputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "outputs": [{"name": "display", "type": PORT_DISPLAY}],
        "params": {
            "bin_ms": {"type": "number", "default": 2, "label": "bin (ms)", "min": 0.1, "step": 0.5},
        },
    },

    "smooth": {
        "label": "Smooth (pre-FFT)",
        "category": "processing",
        "color": "#3a8f8f",
        "help": "Moving-average / gaussian / Savitzky-Golay smoothing of the binned rate "
                "before the transform (MATLAB smooth(y, span)).",
        "desc": ("Low-pass smooths the binned spike-rate trace before it is transformed — the "
                 "operation Sara ran on the online traces (smooth(YData, 4)). 'moving' is the "
                 "MATLAB moving average (odd span, windows shrink at the ends); 'gaussian' "
                 "weights by a bell of FWHM = span; 'savgol' fits a local polynomial "
                 "(preserves peak height/width). Insert it right before the analysis/FFT node."),
        "math": ["moving:   ỹ[i] = (1/(2w+1)) Σ_{j=−w}^{w} y[i+j],   w = min((span−1)/2, i, n−1−i)",
                 "gaussian: ỹ = y ∗ g,   g[k] ∝ exp(−k²/2σ²),   σ = span / 2.3548 (FWHM)",
                 "savgol:   ỹ[i] = local degree-p least-squares fit over `span` samples",
                 "an even span is reduced to span−1 (odd, symmetric — the MATLAB rule)"],
        "inputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "outputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "params": {
            "method": {"type": "choice", "default": "moving", "label": "method",
                       "options": ["moving", "gaussian", "savgol"]},
            "window": {"type": "number", "default": 4, "label": "span (samples)",
                       "min": 0, "max": 999, "step": 1},
            "polyorder": {"type": "number", "default": 2, "label": "savgol order",
                          "min": 1, "max": 6, "step": 1},
        },
    },
    "tfilter": {
        "label": "Temporal filter",
        "category": "processing",
        "color": "#5a7fb0",
        "help": "Zero-phase FIR low-/high-pass in time (Hz). Time-domain port of the "
                "benaqTools spatial filter.",
        "desc": ("A zero-phase FIR low-pass or high-pass applied to the binned rate in the time "
                 "domain — the 1-D (time) port of the benaqTools spatial filter. The cutoff is "
                 "in Hz (converted to cycles/sample with the bin rate). Low-pass keeps slow "
                 "trends; high-pass removes drift/DC. Types trade sharpness for ringing; the "
                 "realized (truncated-kernel) response is what runs."),
        "math": ["fc = cutoff_Hz / fs   (cycles/sample)",
                 "butterworth:  |H(f)| = 1 / √(1 + (f/fc)^{2n})",
                 "windowed-sinc: h[k] = 2fc·sinc(2fc·k)·w[k]   (hamming/hann/blackman/ideal)",
                 "high-pass = δ − low-pass   (spectral inversion, exact for symmetric FIRs)",
                 "ỹ = y ∗ h   (reflect-padded, centered ⇒ zero phase)"],
        "inputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "outputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "params": {
            "ftype": {"type": "choice", "default": "butterworth", "label": "type",
                      "options": ["gaussian", "butterworth", "ideal", "hamming",
                                  "hanning", "blackman"]},
            "mode": {"type": "choice", "default": "lowpass", "label": "mode",
                     "options": ["lowpass", "highpass"]},
            "cutoff_hz": {"type": "number", "default": 30, "label": "cutoff (Hz)",
                          "min": 0.1, "step": 1},
            "order": {"type": "number", "default": 2, "label": "order (butter)",
                      "min": 1, "max": 10, "step": 1},
            "taps": {"type": "number", "default": 33, "label": "taps", "min": 3,
                     "max": 129, "step": 2},
        },
    },
    "flicker": {
        "label": "Sq wave ON/OFF",
        "category": "analysis",
        "color": "#b0392f",
        "help": "Square-wave (sq wave) cycle/transition PSTH + pooled ON/OFF shift test.",
        "desc": ("The square-wave (flicker) analysis: folds spikes over the stimulus period into "
                 "a cycle PSTH, measures how tightly they lock to the cycle (vector strength), "
                 "and tests whether the ON and OFF transitions drive a transient response — the "
                 "pooled ON/OFF ratio against a spike-shuffled null. The analysis internals stay "
                 "named 'flicker' (output folder, run function) for data compatibility."),
        "math": ["period  T = 1/f   (stimulus frequency f from the frame clock)",
                 "cycle PSTH:  r(φ) = (1/N_cyc)·(count of spikes with phase φ)/Δφ,  φ = (t mod T)/T",
                 "vector strength:  VS = | (1/N) Σⱼ e^{i2πφⱼ} |   ∈ [0,1]",
                 "ON/OFF ratio = mean rate in the transition window / baseline rate",
                 "p = fraction of `shuffles` circular/jitter nulls with ratio ≥ observed"],
        "inputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "outputs": [{"name": "result", "type": PORT_RESULT}],
        "terminal": "sq_wave",
        "params": {
            "n_shuffle": {"type": "number", "default": 500, "label": "shuffles", "min": 0, "step": 100},
        },
    },
    "sta": {
        "label": "Reverse-correlation STA",
        "category": "analysis",
        "color": "#b0392f",
        "help": "Temporal spike-triggered average from the seed-regenerated Gaussian noise.",
        "desc": ("Temporal spike-triggered average: the mean stimulus contrast in the window "
                 "preceding each spike — the cell's linear temporal filter. Per epoch: the "
                 "Gaussian-noise stimulus is regenerated from its seed (bit-identical to MATLAB) "
                 "as linear light v, the contrast v − mean is held across 6 response bins per "
                 "update, the spikes are binned on the update grid MEASURED from the sync bar's "
                 "middle segment (so spikes and frames are aligned automatically — no region or "
                 "nudge involved), and the filter is the FFT cross-correlation of response and "
                 "stimulus, both zero-padded by 60 bins (no circular wrap-around for lags up to "
                 "60 bins). Epochs are averaged and the average divided by max|·|. The STA's FFT "
                 "gives the temporal modulation-transfer function (tuning)."),
        "math": ["per epoch:  r[n] = spikes in update bin n (6 bins per update, measured edges)",
                 "            s[n] = v[n] − ⟨v⟩   (linear light, held across the 6 bins)",
                 "            STAᵢ = Re IFFT( FFT([r, 0₆₀]) · conj FFT([s, 0₆₀]) )[0 : filter_len]",
                 "STA = ⟨STAᵢ⟩ᵢ / max|⟨STAᵢ⟩ᵢ|,   filter_len = filter_s × (6 × update rate)",
                 "peak lag = the cell's response latency;  temporal MTF = |FFT{ STA(τ) }|"],
        "inputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "outputs": [{"name": "result", "type": PORT_RESULT}],
        "terminal": "gaussian_noise",
        "params": {
            "filter_s": {"type": "number", "default": 1.0, "label": "filter (s)", "min": 0.1, "step": 0.1},
        },
    },
    "strf": {
        "label": "Spatiotemporal STRF",
        "category": "analysis",
        "color": "#b0392f",
        "help": "Spatiotemporal reverse correlation from the seed-regenerated checkerboard.",
        "desc": ("Spatiotemporal receptive field: reverse-correlation of the spikes against the "
                 "seed-regenerated checkerboard, giving response as a function of space (x, y) and "
                 "time lag τ. The spatial slice at the peak lag is the receptive-field map."),
        "math": ["STRF(x, y, τ) = (1/N) Σⱼ S(x, y, tⱼ − τ),   S = checkerboard contrast (v − ⟨v⟩)",
                 "epochs concatenated in time; spikes binned on the MEASURED update clock (1 bin/update)",
                 "one FFT cross-correlation per check, zero-padded by 60 bins, 30 taps (0.5 s)",
                 "spatial RF = STRF(·, ·, τ_peak);   temporal kernel = STRF(x₀, y₀, ·)"],
        "inputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "outputs": [{"name": "result", "type": PORT_RESULT}],
        "terminal": "checkerboard",
        "params": {},
    },
    "figures": {
        "label": "Figures & exports",
        "category": "output",
        "color": "#555b66",
        "help": "Write PNG/PDF/SVG figures (+ 4K exports) into the cell's outputs. Which figures "
                "depends on the analysis (see the breakout panel).",
        "desc": ("Writes the analysis figures into the cell's outputs/ folder (PNG for the "
                 "gallery, PDF+SVG at 4K when '4K exports' is on). WHICH figures are produced is "
                 "set by the analysis feeding this node — the square-wave, STA, and STRF paths "
                 "each emit their own set (listed here). Files are recorded as one integrity-safe "
                 "manifest entry."),
        "math": ["PNG 1920×1080 (gallery)  ·  PDF/SVG 3840×2160 (4K) when enabled",
                 "one record_output(analysis, files=…) per run — replaces by name, no duplicates"],
        # which figures each stimulus family generates through this node (drives the breakout list)
        "figures_by_stim": {
            "sq_wave": ["window_4k (signal + frame-sync, 250 ms stim zoom)",
                        "power_4k (spike-power spectrum)",
                        "analog_framesync_4k (color signal+spikes / B&W / frame-syncs)",
                        "framesync_separated_4k (per-epoch frame-sync)",
                        "cycle grid (per-epoch cycle PSTH)",
                        "assumptions_4k (detection settings + per-file counts)"],
            "gaussian_noise": ["sta (temporal linear filter)",
                               "temporal MTF (|FFT| of the filter)",
                               "assumptions (detection settings + counts)"],
            "checkerboard": ["strf spatial RF (peak-lag frame)",
                             "strf temporal kernel",
                             "assumptions (detection settings + counts)"],
        },
        "inputs": [{"name": "result", "type": PORT_RESULT}],
        "outputs": [{"name": "outputs", "type": PORT_OUTPUTS}],
        "params": {
            "export_4k": {"type": "bool", "default": True, "label": "4K exports"},
        },
    },
}

# terminal component type -> stimulus family it runs under (mirrors run.analysis_for_stim_type)
_TERMINALS = {t: c["terminal"] for t, c in COMPONENT_REGISTRY.items() if c.get("terminal")}

# ---------------------------------------------------------------------------------------------
# The code behind each component: the functions that ACTUALLY run for it, in call order, as
# "module:qualname" references. The GUI shows their live source (inspect.getsource) next to the
# node — never a hand-copied listing that could drift from what runs. A "viewer:" reference names
# the Dash GUI module, which this package never imports; the GUI passes itself via `extra`.
# ---------------------------------------------------------------------------------------------
CODE_REFS = {
    "source": ["neitz.io:load_recording", "neitz.io.abf:Recording.load", "neitz.io.abf:Recording.channel"],
    "align": ["viewer:set_align_seed", "viewer:fine_align_offset",
              "neitz.analysis.flicker:detect_flicker"],
    "detect": ["neitz.spikes:detect_spikes", "neitz.spikes:detect_spikes_matlab"],
    "region": ["neitz.spikes:in_region", "viewer:stim_window", "neitz.analysis.sync:stimulus_window",
               "neitz.analysis.flicker:detect_flicker"],
    "sf_bin": ["neitz.signal_flow:bin_spikes"],
    "sf_resample": ["neitz.signal_flow:resample"],
    "sf_smooth": ["neitz.signal_flow:smooth", "neitz.analysis.dsp:smooth_1d"],
    "sf_filter": ["neitz.signal_flow:freq_filter", "neitz.analysis.dsp:apply_temporal_filter"],
    "sf_detrend": ["neitz.signal_flow:detrend"],
    "sf_window": ["neitz.signal_flow:window"],
    "sf_fft": ["neitz.signal_flow:fft_power", "neitz.signal_flow:_power_1d"],
    "sf_time": ["neitz.signal_flow:time_trace"],
    "sf_isi": ["neitz.signal_flow:isi_hist"],
    "smooth": ["viewer:pre_fft", "neitz.analysis.dsp:smooth_1d"],
    "tfilter": ["viewer:pre_fft", "neitz.analysis.dsp:apply_temporal_filter"],
    "flicker": ["neitz.run:run_cell_flicker", "neitz.stimulus.flicker:FlickerParadigm.analyze_recording",
                "neitz.analysis.flicker:detect_flicker", "neitz.analysis.flicker:vector_strength",
                "neitz.analysis.flicker:cycle_psth", "neitz.stimulus.flicker:FlickerParadigm.group_from_trials",
                "neitz.analysis.flicker:shift_test"],
    "sta": ["neitz.run:run_cell_noise", "neitz.run:_run_cell_noise_seed", "neitz.run:load_seed_epochs",
            "neitz.analysis.sync:update_clock", "neitz.run:epoch_response_edges",
            "neitz.io.stim:noise_from_record", "neitz.run:sta_from_records",
            "neitz.stimulus.noise:NoiseParadigm.analyze", "neitz.analysis.revcorr:average_filter",
            "neitz.analysis.revcorr:reverse_correlation", "neitz.analysis.revcorr:normalize_filter",
            "neitz.analysis.revcorr:temporal_tuning"],
    "strf": ["neitz.run:run_cell_checkerboard", "neitz.run:load_seed_epochs",
             "neitz.analysis.sync:update_clock", "neitz.run:epoch_response_edges",
             "neitz.io.stim:noise_from_record", "neitz.run:strf_from_records",
             "neitz.stimulus.checkerboard:CheckerboardParadigm.analyze",
             "neitz.analysis.strf:spatiotemporal_revcorr", "neitz.analysis.strf:spatial_rf"],
    "figures": ["neitz.run:_save_seed_run", "neitz.plots:noise_sta_figure", "neitz.plots:strf_figure",
                "neitz.plots:strf_grid_figure", "neitz.plots:flicker_onoff_figure",
                "neitz.plots:flicker_cycle_grid", "viewer:export_window_figures",
                "neitz.io.figures:save_figure"],
}


def ordered_nodes(pipe) -> list:
    """The nodes in EXECUTION order: along the connections from the source(s) (a stable Kahn
    topological sort, original order breaking ties), unconnected / cyclic leftovers last."""
    nodes = {n["id"]: n for n in pipe.get("nodes", [])}
    out_edges, indeg, wired = {}, {nid: 0 for nid in nodes}, set()
    for c in pipe.get("connections", []):
        if c.get("from_node") in nodes and c.get("to_node") in nodes:
            out_edges.setdefault(c["from_node"], []).append(c["to_node"])
            indeg[c["to_node"]] += 1
            wired.update((c["from_node"], c["to_node"]))
    order, ready = [], [nid for nid in nodes if indeg[nid] == 0 and nid in wired]
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for t in out_edges.get(nid, []):
            indeg[t] -= 1
            if indeg[t] == 0:
                ready.append(t)
    order += [nid for nid in nodes if nid not in order]
    return [nodes[nid] for nid in order]


def _fmt_params(node) -> str:
    p = node.get("params") or {}
    return ", ".join(f"{k}={v!r}" for k, v in p.items()) or "no parameters"


def _driver_lines(pipe, fam, context) -> list:
    """The call ▶ Run makes for this pipeline (viewer._run_one → neitz.run), with the live settings
    filled in when `context` has them, else placeholders."""
    c = context or {}
    rk = run_kwargs(pipe)
    date, cell = repr(c.get("date") or "<date>"), repr(c.get("cell") or "<cell>")
    files = c.get("files") or []
    inc = ("[" + ", ".join(repr(f) for f in files) + "]") if files else "None"
    det = c.get("detect") or {"polarity": "neg", "method": "mad", "k": 6.0,
                              "abs_threshold": None, "refractory_s": 0.002}
    amap = c.get("abs_map") or None
    chan, ttl = repr(c.get("chan") or "Im_prime"), repr(c.get("ttl") or "TTL")
    lines = ["from neitz.dataio import DataStore"]
    if fam == "gaussian_noise":
        lines += ["from neitz.run import run_cell_noise", "",
                  "store = DataStore()                      # ~/Documents/ephysdataio (or $EPHYSDATAIO_ROOT)",
                  f"result = run_cell_noise(store, {date}, {cell}, name={(c.get('run_name') or 'sta')!r},",
                  f"                        chan={chan}, ttl={ttl},",
                  f"                        detect={det!r},",
                  f"                        abs_map={amap!r},",
                  f"                        include={inc},      # the checked epochs; None = every seeded one",
                  f"                        filter_s={rk['filter_s']!r})        # the STA node's filter length (s)"]
    elif fam == "checkerboard":
        lines += ["from neitz.run import run_cell_checkerboard", "",
                  "store = DataStore()                      # ~/Documents/ephysdataio (or $EPHYSDATAIO_ROOT)",
                  f"result = run_cell_checkerboard(store, {date}, {cell}, name={(c.get('run_name') or 'strf')!r},",
                  f"                               chan={chan}, ttl={ttl},",
                  f"                               detect={det!r},",
                  f"                               abs_map={amap!r},",
                  f"                               include={inc})"]
    else:
        lines += ["from neitz.run import run_cell_flicker", "",
                  "store = DataStore()                      # ~/Documents/ephysdataio (or $EPHYSDATAIO_ROOT)",
                  f"result = run_cell_flicker(store, {date}, {cell}, name={(c.get('run_name') or 'flicker')!r},",
                  f"                          n_shuffle={rk['n_shuffle']!r}, include={inc},",
                  f"                          detect={det!r},",
                  f"                          abs_map={amap!r})",
                  "# (the viewer then also writes the 4K window exports from the live Analysis-View graphs)"]
    lines += ["print(result.summary)"]
    return lines


def pipeline_script(pipe, extra=None, context=None) -> str:
    """The WHOLE pipeline as ONE readable script: a header (the components in execution order with
    their canvas parameters, the live settings), the exact driver call ▶ Run makes, then every
    function that runs, grouped by component in execution order — each function once, at the first
    component that calls it, copied verbatim from the package sources (`component_code`, with
    file:line). `context` (optional) carries the live Analysis-View state: date, cell, files
    (basenames), chan, ttl, detect, abs_map, region, run_name."""
    nodes = ordered_nodes(pipe)
    name = pipe.get("name") or "untitled"
    fam = pipeline_stim_family(pipe)
    c = context or {}
    lines = ['"""', f"{name} — the whole pipeline as one script "
                   f"({datetime.now().strftime('%Y-%m-%d %H:%M')}).", "",
             "Components in execution order (parameters as set on the canvas):"]
    for i, n in enumerate(nodes, 1):
        spec = COMPONENT_REGISTRY.get(n["type"], {})
        lines.append(f"  {i}. {spec.get('label', n['type'])}  [{n['type']}]  —  {_fmt_params(n)}")
    if c:
        lines += ["", "Live Analysis-View settings this would run with:"]
        for key in ("date", "cell", "files", "chan", "ttl", "detect", "abs_map", "region"):
            if key in c:
                lines.append(f"  {key}: {c[key]!r}")
        if fam in ("gaussian_noise", "checkerboard"):
            lines.append("  (the region boxes do not enter the STA / STRF: each epoch is binned on the "
                         "update clock measured from its sync bar)")
    lines += ["", "Every function below is the code the viewer's ▶ Run executes, copied verbatim from "
                  "the package", "sources (file:line noted), each listed once at the first component "
                  "that calls it.", '"""', ""]
    bar = "# " + "=" * 94
    lines += [bar, "# HOW THIS PIPELINE IS RUN   (viewer._run_one → neitz.run)", bar]
    lines += _driver_lines(pipe, fam, c)
    seen = {}
    for i, n in enumerate(nodes, 1):
        spec = COMPONENT_REGISTRY.get(n["type"], {})
        lines += ["", "", bar, f"# {i}. {spec.get('label', n['type'])}   [{n['type']}]   {_fmt_params(n)}",
                  f"# {spec.get('help', '')}", bar]
        code = component_code(n["type"], extra)
        if not code:
            lines.append("# (no code listed for this component)")
        for cc in code:
            if cc["ref"] in seen:
                lines.append(f"# {cc['name']}  → already listed under component {seen[cc['ref']]}")
                continue
            seen[cc["ref"]] = i
            where = f"  ({cc['file']}:{cc['line']})" if cc["file"] else ""
            lines += [f"# --- {cc['name']}{where} ---"]
            lines += cc["source"].rstrip("\n").split("\n")
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _resolve_ref(ref, extra=None):
    mod_name, _, qual = ref.partition(":")
    mod = (extra or {}).get(mod_name)
    if mod is None:
        mod = importlib.import_module(mod_name)
    obj = mod
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def component_code(ctype, extra=None) -> list:
    """``[{ref, name, file, line, source}]`` for the functions that run for component `ctype`, in
    call order, read with `inspect` from the live modules — so it is the code that actually runs.
    `extra` maps a pseudo-module name (e.g. ``"viewer"``) to a module object this package cannot
    import. A reference that cannot be resolved yields a one-line ``source`` saying why."""
    out = []
    for ref in CODE_REFS.get(ctype, []):
        name = ref.partition(":")[2]
        try:
            obj = _resolve_ref(ref, extra)
            src = inspect.getsource(obj)
            file = inspect.getsourcefile(obj) or ""
            line = inspect.getsourcelines(obj)[1]
        except Exception as e:                                    # a missing GUI module, say
            out.append({"ref": ref, "name": name, "file": "", "line": 0,
                        "source": f"# source unavailable here: {e}"})
            continue
        out.append({"ref": ref, "name": name, "file": os.path.basename(file), "line": int(line),
                    "source": src})
    return out


def component_defaults(ctype) -> dict:
    """Default param values for a component type."""
    spec = COMPONENT_REGISTRY.get(ctype, {})
    return {k: v.get("default") for k, v in spec.get("params", {}).items()}


# ---------------------------------------------------------------------------------------------
# Pipeline model: plain dicts (JSON-native), plus helpers. A node = {id, type, params, x, y,
# label}. A connection = {from_node, from_port, to_node, to_port}.
# ---------------------------------------------------------------------------------------------
def new_node(ctype, node_id, x=40.0, y=40.0, params=None, label=None) -> dict:
    spec = COMPONENT_REGISTRY.get(ctype, {})
    p = component_defaults(ctype)
    if params:
        p.update({k: v for k, v in params.items() if k in p})
    return {"id": node_id, "type": ctype, "params": p, "x": float(x), "y": float(y),
            "label": label or spec.get("label", ctype)}


def empty_pipeline(name="untitled") -> dict:
    return {"version": 1, "name": name, "nodes": [], "connections": [],
            "created": datetime.now().isoformat(timespec="seconds")}


def _chain(name, steps) -> dict:
    """Build a left-to-right linear pipeline from a list of component types."""
    nodes, conns, x = [], [], 30.0
    prev = None
    for i, ctype in enumerate(steps):
        nid = f"{ctype}{i}"
        nodes.append(new_node(ctype, nid, x=x, y=60.0))
        if prev is not None:
            out_port = COMPONENT_REGISTRY[prev[1]]["outputs"][0]["name"]
            in_port = COMPONENT_REGISTRY[ctype]["inputs"][0]["name"]
            conns.append({"from_node": prev[0], "from_port": out_port,
                          "to_node": nid, "to_port": in_port})
        prev = (nid, ctype)
        x += 210.0
    return {"version": 1, "name": name, "nodes": nodes, "connections": conns,
            "created": datetime.now().isoformat(timespec="seconds")}


def default_flicker_pipeline() -> dict:
    """The canonical 'Run analysis' flow for square-wave (sq wave / flicker) cells."""
    return _chain("Sq wave ON/OFF", ["source", "align", "detect", "region", "flicker", "figures"])


def default_sta_pipeline() -> dict:
    return _chain("Gaussian-noise STA", ["source", "detect", "region", "sta", "figures"])


def default_strf_pipeline() -> dict:
    return _chain("Checkerboard STRF", ["source", "detect", "region", "strf", "figures"])


DEFAULT_PIPELINES = {
    "Sq wave ON/OFF": default_flicker_pipeline,
    "Gaussian-noise STA": default_sta_pipeline,
    "Checkerboard STRF": default_strf_pipeline,
}


def terminal_node(pipe) -> dict | None:
    """The analysis (terminal) node — the one whose type carries a stimulus family."""
    for n in pipe.get("nodes", []):
        if n.get("type") in _TERMINALS:
            return n
    return None


def pipeline_stim_family(pipe) -> str | None:
    """Which stimulus family this pipeline analyzes (sq_wave / gaussian_noise / checkerboard)."""
    n = terminal_node(pipe)
    return _TERMINALS.get(n["type"]) if n else None


def node_of_type(pipe, ctype) -> dict | None:
    return next((n for n in pipe.get("nodes", []) if n.get("type") == ctype), None)


def validate(pipe) -> list:
    """Return a list of human-readable problems (empty = OK). Cheap structural checks."""
    problems = []
    nodes = pipe.get("nodes", [])
    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)):
        problems.append("duplicate node ids")
    idset = set(ids)
    if not any(n["type"] == "source" for n in nodes):
        problems.append("no 'Cell recordings' source node")
    if terminal_node(pipe) is None:
        problems.append("no analysis node (flicker / sta / strf)")
    for c in pipe.get("connections", []):
        if c["from_node"] not in idset or c["to_node"] not in idset:
            problems.append("connection references a missing node")
    return problems


# ---------------------------------------------------------------------------------------------
# Storage: one JSON per pipeline under <store_root>/pipelines/. Names are free text; the file
# slug is sanitized. On first use the three default pipelines are materialized.
# ---------------------------------------------------------------------------------------------
def _slug(name) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()) or "untitled"
    return s[:80]


def pipelines_dir(root=None) -> Path:
    d = (Path(root) if root else data_root()) / "pipelines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_defaults(root=None) -> None:
    d = pipelines_dir(root)
    if any(d.glob("*.json")):
        return
    for name, factory in DEFAULT_PIPELINES.items():
        save_pipeline(factory(), root=root)


def list_pipelines(root=None) -> list:
    """Sorted list of saved pipeline names (materializing the defaults on first use)."""
    _ensure_defaults(root)
    names = []
    for p in pipelines_dir(root).glob("*.json"):
        try:
            names.append(json.loads(p.read_text()).get("name") or p.stem)
        except Exception:
            names.append(p.stem)
    return sorted(names, key=str.lower)


def load_pipeline(name, root=None) -> dict | None:
    d = pipelines_dir(root)
    # prefer an exact name match inside the file; fall back to the slug filename
    for p in d.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if (data.get("name") or p.stem) == name:
            return data
    f = d / f"{_slug(name)}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return None
    return None


def save_pipeline(pipe, root=None) -> Path:
    pipe = dict(pipe)
    pipe.setdefault("version", 1)
    pipe["saved"] = datetime.now().isoformat(timespec="seconds")
    f = pipelines_dir(root) / f"{_slug(pipe.get('name'))}.json"
    f.write_text(json.dumps(pipe, indent=2, default=str))
    return f


def delete_pipeline(name, root=None) -> bool:
    d = pipelines_dir(root)
    hit = False
    for p in list(d.glob("*.json")):
        try:
            nm = json.loads(p.read_text()).get("name") or p.stem
        except Exception:
            nm = p.stem
        if nm == name:
            p.unlink()
            hit = True
    return hit


def copy_pipeline(src_name, new_name, root=None) -> dict | None:
    """'Save as from existing' — duplicate a pipeline under a new name."""
    src = load_pipeline(src_name, root=root)
    if src is None:
        return None
    dup = dict(src)
    dup["name"] = new_name
    dup["created"] = datetime.now().isoformat(timespec="seconds")
    save_pipeline(dup, root=root)
    return dup


# ---------------------------------------------------------------------------------------------
# Execution: extract node params from the graph and dispatch to the tested run_cell_* functions.
# `context` carries the live Analysis-View state (checked files, channels, per-trace maps, …) so a
# run honors what the user set up — the GUI already threads these; the pipeline just re-packages
# the graph's node params on top.
# ---------------------------------------------------------------------------------------------
def detect_from_pipeline(pipe, fallback=None) -> dict:
    """Spike-detection dict (run_cell_* `detect=`) built from the pipeline's Detect node."""
    fb = dict(fallback or {})
    n = node_of_type(pipe, "detect")
    if n is None:
        return fb
    p = n["params"]
    refr = p.get("refractory_ms")
    det = {
        "polarity": p.get("polarity", fb.get("polarity", "neg")),
        "method": p.get("method", fb.get("method", "mad")),
        "k": float(p["k"]) if p.get("k") is not None else fb.get("k"),
        "abs_threshold": (float(p["abs_threshold"]) if p.get("abs_threshold") not in (None, "")
                          else fb.get("abs_threshold")),
        "refractory_s": (float(refr) / 1000.0) if refr not in (None, "") else fb.get("refractory_s", 0.002),
    }
    return det


def region_from_pipeline(pipe) -> dict:
    """{start_s, end_s, crop} from the Select-region node (values may be None = auto)."""
    n = node_of_type(pipe, "region")
    if n is None:
        return {"start_s": None, "end_s": None, "crop": False}
    p = n["params"]
    return {"start_s": p.get("start_s"), "end_s": p.get("end_s"), "crop": bool(p.get("crop"))}


def run_kwargs(pipe) -> dict:
    """Assorted run knobs read off the graph (n_shuffle, 4K export flag, align mode, STA filter_s)."""
    fl = node_of_type(pipe, "flicker")
    fig = node_of_type(pipe, "figures")
    align = node_of_type(pipe, "align")
    sta = node_of_type(pipe, "sta")
    try:
        filter_s = float(sta["params"].get("filter_s") or 1.0) if sta else 1.0
    except (TypeError, ValueError):
        filter_s = 1.0
    return {
        "n_shuffle": int(fl["params"].get("n_shuffle", 500)) if fl else 500,
        "export_4k": bool(fig["params"].get("export_4k", True)) if fig else True,
        "align_mode": align["params"].get("mode", "auto") if align else "off",
        "filter_s": filter_s,                          # the STA node's filter length (s)
    }


def smooth_from_pipeline(pipe) -> dict | None:
    """{method, window, polyorder} from the Smooth node, or None if the graph has none.

    Feeds neitz.analysis.dsp.smooth_1d — applied to the binned rate before the FFT/power."""
    n = node_of_type(pipe, "smooth")
    if n is None:
        return None
    p = n["params"]
    try:
        win = float(p.get("window") or 0)
    except (TypeError, ValueError):
        win = 0.0
    if win < 2:
        return None                        # a span < 2 is a no-op
    return {"method": p.get("method", "moving"), "window": win,
            "polyorder": int(p.get("polyorder", 2) or 2)}


def tfilter_from_pipeline(pipe) -> dict | None:
    """{ftype, mode, cutoff_hz, order, taps} from the Temporal-filter node, or None.

    Feeds neitz.analysis.dsp.apply_temporal_filter — applied to the binned rate before FFT."""
    n = node_of_type(pipe, "tfilter")
    if n is None:
        return None
    p = n["params"]
    try:
        cut = float(p.get("cutoff_hz") or 0)
    except (TypeError, ValueError):
        cut = 0.0
    if cut <= 0:
        return None
    return {"ftype": p.get("ftype", "butterworth"), "mode": p.get("mode", "lowpass"),
            "cutoff_hz": cut, "order": int(p.get("order", 2) or 2),
            "taps": int(p.get("taps", 33) or 33)}
