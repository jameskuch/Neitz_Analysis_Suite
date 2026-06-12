"""
flicker_onoff_pooled.py — pooled, significance-tested ON/OFF classification per cell.

Thin client of neitz.stimulus.FlickerParadigm.analyze_group: pools all trials of
each cell (grouped by subfolder) and runs the transition-triggered ON/OFF PSTHs with
the jitter-null significance test.

Usage:
    /Users/j/miniconda3/bin/python flicker_onoff_pooled.py
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neitz.io.abf import Recording
from neitz.stimulus import FlickerParadigm

STORE = os.path.expanduser(os.environ.get("EPHYSDATAIO_ROOT", "~/Documents/ephysdataio"))
ABF_GLOBS = [os.path.join(STORE, "2026-06-02", "**", "*.abf")]
PARADIGM = FlickerParadigm(current_channel="Im_prime", ttl_channel="TTL",
                           polarity="neg", method="mad", k=6.0)
N_SHUFFLE = 1000
ALPHA = 0.01
OUT_FIG = "flicker_onoff_pooled.png"
OUT_CSV = "flicker_onoff_pooled_metrics.csv"


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

    cells = {}
    for f in files:
        cells.setdefault(os.path.basename(os.path.dirname(os.path.dirname(f))), []).append(f)  # cell folder

    rng = np.random.default_rng(0)
    results = []
    print(f"{'cell':10s}{'trials':7s}{'Hz':6s}{'ON ratio(p)':18s}{'OFF ratio(p)':18s}class")
    print("-" * 78)
    for cell, fs in cells.items():
        recs = [Recording.load(f) for f in sorted(fs)]
        grp = PARADIGM.analyze_group(recs, n_shuffle=N_SHUFFLE, rng=rng)
        if grp["n_trials"] == 0:
            continue
        on, off = grp["on"], grp["off"]
        cls = ("ON-OFF" if on["p"] < ALPHA and off["p"] < ALPHA else
               "ON" if on["p"] < ALPHA else "OFF" if off["p"] < ALPHA else "no sig.")
        results.append((cell, grp, cls))
        print(f"{cell:10s}{grp['n_trials']:<7d}{grp['freq']:<6.2f}"
              f"{on['ratio']:5.2f} (p={on['p']:.3f})   {off['ratio']:5.2f} (p={off['p']:.3f})   {cls}")

    with open(OUT_CSV, "w") as fh:
        fh.write("cell,n_trials,flicker_hz,on_ratio,on_peak_ms,on_p,off_ratio,off_peak_ms,off_p,class\n")
        for cell, grp, cls in results:
            on, off = grp["on"], grp["off"]
            fh.write(f"{cell},{grp['n_trials']},{grp['freq']:.2f},"
                     f"{on['ratio']:.3f},{on['peak_ms']:.0f},{on['p']:.4f},"
                     f"{off['ratio']:.3f},{off['peak_ms']:.0f},{off['p']:.4f},{cls}\n")
    print(f"\nWrote {OUT_CSV}")

    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.4 * n), squeeze=False)
    for i, (cell, grp, cls) in enumerate(results):
        for j, which in enumerate(("on", "off")):
            ax = axes[i][j]
            r = grp[which]
            c = r["centers"] * 1000
            ax.step(c, r["rate"], where="mid", color="tab:blue", lw=1.5, label="observed")
            ax.step(c, r["band"], where="mid", color="tab:red", ls="--", lw=1.0, label="null 99%")
            ax.axhline(r["baseline"], color="k", ls=":", lw=0.8)
            ax.axvspan(10, 150, color="gold", alpha=0.12)
            ax.axvline(0, color="gray", lw=0.6)
            sig = "SIG" if r["p"] < ALPHA else "n.s."
            ax.set_title(f"{cell}  {which.upper()}  {r['ratio']:.2f}x @ {r['peak_ms']:.0f}ms  "
                         f"p={r['p']:.3f} [{sig}]", fontsize=9)
            ax.set_xlabel("time from transition (ms)"); ax.set_ylabel("spikes/s")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=120)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
