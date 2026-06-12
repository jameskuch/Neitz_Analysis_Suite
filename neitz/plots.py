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
    """Per-recording cycle-PSTH grid. `per_file` = list of FlickerResult."""
    n = len(per_file)
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
