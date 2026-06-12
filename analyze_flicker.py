"""
analyze_flicker.py — per-file flicker light-response (cycle-PSTH + vector strength).

Thin client of neitz.stimulus.FlickerParadigm. Stimulus timing is read from the TTL
channel; the question is whether spikes entrain to the flicker frequency.

Usage:
    /Users/j/miniconda3/bin/python analyze_flicker.py
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neitz.io.abf import Recording
from neitz.stimulus import FlickerParadigm

ABF_GLOBS = ["data/ipRGC barak/**/*.abf", "ipRGC barak/**/*.abf"]
PARADIGM = FlickerParadigm(current_channel="Im_prime", ttl_channel="TTL",
                           polarity="neg", method="mad", k=6.0)
OUT_FIG = "flicker_analysis.png"
OUT_CSV = "flicker_metrics.csv"


def find_files():
    for g in ABF_GLOBS:
        fs = sorted(glob.glob(g, recursive=True))
        if fs:
            return fs
    return []


def main():
    files = find_files()
    if not files:
        raise SystemExit(f"No .abf files matched {ABF_GLOBS}")

    results = []
    print(f"{'file':24s}{'fold':10s}{'flick_Hz':9s}{'#sp(reg)':9s}{'VS':7s}{'Rayleigh_p'}")
    print("-" * 72)
    for f in files:
        res = PARADIGM.analyze_recording(Recording.load(f), name=os.path.basename(f))
        fold = os.path.basename(os.path.dirname(f))
        results.append((res, fold))
        fhz = f"{res.freq:.2f}" if np.isfinite(res.freq) else "  -"
        vs = f"{res.vector_strength:.3f}" if np.isfinite(res.vector_strength) else "  -"
        pp = f"{res.rayleigh_p:.2e}" if np.isfinite(res.rayleigh_p) else "  -"
        print(f"{res.name:24s}{fold:10s}{fhz:>8s} {res.n_in_region:>8d} {vs:>6s} {pp:>11s}")

    with open(OUT_CSV, "w") as fh:
        fh.write("file,folder,flicker_hz,n_in_region,vector_strength,rayleigh_p\n")
        for res, fold in results:
            fh.write(f"{res.name},{fold},{res.freq:.3f},{res.n_in_region},"
                     f"{res.vector_strength:.4f},{res.rayleigh_p:.4f}\n")
    print(f"\nWrote {OUT_CSV}")

    n = len(results)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, (res, fold) in zip(axes.flat, results):
        if res.cycle_rate is not None:
            ax.bar(res.cycle_phase, res.cycle_rate, width=1.0 / len(res.cycle_phase),
                   align="edge", alpha=0.8)
            ax.axvspan(0, 0.5, color="gold", alpha=0.12)
        ax.set_title(f"{res.name.replace('.abf', '')}\n{res.freq:.1f}Hz  VS={res.vector_strength:.2f}",
                     fontsize=8)
        ax.set_xlabel("flicker phase"); ax.set_ylabel("spikes/s")
    for ax in axes.flat[n:]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=120)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
