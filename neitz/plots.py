"""
neitz.plots — figure generators for the analyses (returned as matplotlib Figures so
they can be saved via neitz.io.figures.save_figure as PNG + PDF + SVG).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def flicker_onoff_figure(group, label=""):
    """Pooled transition-triggered ON/OFF PSTH for one cell (2 panels)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, which in zip(axes, ("on", "off")):
        r = group[which]
        c = r["centers"] * 1000
        ax.step(c, r["rate"], where="mid", color="tab:blue", lw=1.5, label="observed")
        ax.step(c, r["band"], where="mid", color="tab:red", ls="--", lw=1.0, label="null 99%")
        ax.axhline(r["baseline"], color="k", ls=":", lw=0.8)
        ax.axvspan(10, 150, color="gold", alpha=0.12)
        ax.axvline(0, color="gray", lw=0.6)
        sig = "SIG" if r["p"] < 0.01 else "n.s."
        ax.set_title(f"light {which.upper()}  {r['ratio']:.2f}x @ {r['peak_ms']:.0f} ms  "
                     f"p={r['p']:.3f} [{sig}]", fontsize=9)
        ax.set_xlabel("time from transition (ms)")
        ax.set_ylabel("spikes/s")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{label} — flicker ON/OFF  ({group['freq']:.1f} Hz, {group['n_trials']} trials)")
    fig.tight_layout()
    return fig


def flicker_cycle_grid(per_file, ncol=5):
    """Per-recording cycle-PSTH grid. `per_file` = list of FlickerResult.
    The grid fits the count: ≤5 epochs → a single row of exactly n (4 → 1×4, no empty 5th);
    >5 → `ncol`-wide rows (10 → 2×5)."""
    n = len(per_file)
    ncol = min(n, ncol) or 1                  # don't pad out to a 5th column when there are fewer
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
    for ax, res in zip(axes.flat, per_file):
        if res.cycle_rate is not None:
            ax.bar(res.cycle_phase, res.cycle_rate, width=1.0 / len(res.cycle_phase),
                   align="edge", alpha=0.8)
            ax.axvspan(0, 0.5, color="gold", alpha=0.12)
        ax.set_title(f"{res.name}\n{res.freq:.1f}Hz VS={res.vector_strength:.2f}", fontsize=8)
        ax.set_xlabel("phase"); ax.set_ylabel("spikes/s")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def _iso_label(cone_isolation):
    """'S' -> 'S-iso', 'achromatic' -> 'achromatic', None/'' -> ''."""
    iso = str(cone_isolation or "").strip()
    if not iso:
        return ""
    return iso if iso.lower() == "achromatic" else f"{iso}-iso"


def noise_sta_figure(arrays, label="", cone_isolation=None):
    """Gaussian-noise reverse correlation: the temporal STA (linear filter) + its
    tuning. `arrays` = the dict from run_noise (average, time_ms, freqs, tuning,
    per_epoch). Used to validate against Sara's MATLAB STA (peak ~22 ms).
    `cone_isolation` ('S', 'L', 'achromatic', …) only labels the panel."""
    avg = np.asarray(arrays["average"], float)
    t = np.asarray(arrays["time_ms"], float)
    freqs = np.asarray(arrays["freqs"], float)
    tuning = np.asarray(arrays["tuning"], float)
    pk = int(np.argmax(np.abs(avg)))
    per = arrays.get("per_epoch")
    n_ep = len(per) if per is not None else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    if per is not None:                                  # faint per-epoch filters (÷own max)
        for row in np.asarray(per, float):
            d = np.max(np.abs(row)) or 1.0
            ax1.plot(t, row / d, color="0.8", lw=0.4)
    ax1.plot(t, avg, color="tab:blue", lw=1.8, label="average STA")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.axvline(t[pk], color="tab:red", ls="--", lw=1.0)
    iso = _iso_label(cone_isolation)
    ax1.set_title(f"{iso + ' ' if iso else ''}STA (linear filter) — peak {t[pk]:.1f} ms "
                  f"[{'OFF' if avg[pk] < 0 else 'ON'}]", fontsize=9)
    ax1.set_xlabel("time (ms)"); ax1.set_ylabel("filter (÷max)")
    ax1.legend(fontsize=7)

    ax2.plot(freqs, tuning, color="tab:purple", lw=1.5)
    ax2.set_xlim(0, 40)
    ax2.set_title("temporal tuning (|FFT| of STA)", fontsize=9)
    ax2.set_xlabel("frequency (Hz)"); ax2.set_ylabel("amplitude")

    fig.suptitle(f"{label} — gaussian-noise reverse correlation"
                 + (f"  ({n_ep} epochs)" if n_ep else ""))
    fig.tight_layout()
    return fig


def strf_figure(res, label=""):
    """Checkerboard STRF: spatial receptive field (peak-lag map) + temporal filter at the peak check.
    `res` is a CheckerboardParadigm STRFResult (spatial_rf, temporal, peak_yx, peak_time_ms, time_ms).
    """
    spatial = np.asarray(res.spatial_rf, float)
    temporal = np.asarray(res.temporal, float)
    t = np.asarray(res.time_ms, float)
    py, px = res.peak_yx

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    vmax = float(np.max(np.abs(spatial))) or 1.0
    im = ax1.imshow(spatial, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal", origin="upper")
    ax1.plot(px, py, "kx", ms=11, mew=2.2)                 # mark the peak check
    ax1.set_title(f"spatial RF — peak check (y={py}, x={px})", fontsize=9)
    ax1.set_xlabel("x check"); ax1.set_ylabel("y check")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2.plot(t, temporal, color="tab:blue", lw=1.8)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.axvline(res.peak_time_ms, color="tab:red", ls="--", lw=1.0)
    ax2.set_title(f"temporal filter @ peak — {res.peak_time_ms:.1f} ms "
                  f"[{'OFF' if temporal[int(np.argmax(np.abs(temporal)))] < 0 else 'ON'}]", fontsize=9)
    ax2.set_xlabel("time (ms)"); ax2.set_ylabel("filter")

    fig.suptitle(f"{label} — checkerboard STRF ({spatial.shape[0]}×{spatial.shape[1]} checks)")
    fig.tight_layout()
    return fig


def strf_grid_figure(res, label="", strong=4.0, moderate=2.5):
    """Checkerboard STRF as a GRID of per-location spike-triggered averages: one small panel per
    check laid out like the stimulus (row 0 at the top, column 0 at the left), every panel on the
    same scale (the cube is ÷ its global max). The peak check is boxed in red; a check whose
    |peak| exceeds `strong` × the median check is drawn red, `moderate` × blue, the rest grey.
    `res` is a CheckerboardParadigm STRFResult (strf (n_y, n_x, flen), time_ms, peak_yx)."""
    cube = np.asarray(res.strf, float)
    t = np.asarray(res.time_ms, float)
    n_y, n_x, _ = cube.shape
    flat = cube.reshape(-1, cube.shape[-1])
    strength = np.abs(flat).max(axis=1)
    med = float(np.median(strength)) or 1.0
    vmax = float(np.abs(cube).max()) or 1.0
    py, px = res.peak_yx
    cw = max(0.45, min(1.15, 18.0 / n_x))                 # panel size: readable, but bounded for big grids
    ch = max(0.4, min(0.95, 11.0 / n_y))
    fig, axes = plt.subplots(n_y, n_x, figsize=(n_x * cw + 0.6, n_y * ch + 0.8),
                             sharex=True, sharey=True, squeeze=False)
    small = cw < 0.7
    for y in range(n_y):
        for x in range(n_x):
            ax = axes[y, x]
            k = cube[y, x]
            s = float(np.abs(k).max()) / med
            ax.plot(t, k, lw=0.9 if not small else 0.6,
                    color=("tab:red" if s > strong else "tab:blue" if s > moderate else "0.55"))
            ax.axhline(0, color="k", lw=0.3)
            ax.axvline(res.peak_time_ms, color="0.85", lw=0.4)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylim(-vmax * 1.05, vmax * 1.05)
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if (y, x) == (py, px):
                for sp in ax.spines.values():
                    sp.set_color("tab:red"); sp.set_linewidth(1.8)
            if not small:
                ax.text(0.03, 0.78, f"{y},{x}", transform=ax.transAxes, fontsize=5.5, color="0.4")
    for x in range(n_x):
        axes[-1, x].set_xlabel(f"x{x}", fontsize=7 if not small else 5)
    for y in range(n_y):
        axes[y, 0].set_ylabel(f"y{y}", fontsize=7 if not small else 5)
    fig.suptitle(f"{label} — STA at every checkerboard location ({n_y}×{n_x} checks, lags "
                 f"0–{t[-1]:.0f} ms, shared scale); red > {strong:g}× median check, blue > "
                 f"{moderate:g}×, box = peak check ({py},{px}) @ {res.peak_time_ms:.0f} ms", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig
