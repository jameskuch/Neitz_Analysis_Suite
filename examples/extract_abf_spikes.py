"""
extract_abf_spikes.py — convert .abf recordings to the siso-spikes.csv format.

Thin client of the neitz package: detects action currents (escaped spikes) on the
current channel and writes a binary spike-train CSV in the exact siso-spikes.csv
layout (col 0 = time @ OUT_RATE Hz, one binary column per recording = "epoch"),
in the siso-spikes.csv layout.

Usage:
    /Users/j/miniconda3/bin/python extract_abf_spikes.py
"""
import os
import glob
import numpy as np

from neitz.io.abf import Recording
from neitz.spikes import detect_spikes

# ---- config ----
STORE = os.path.expanduser(os.environ.get("EPHYSDATAIO_ROOT", "~/Documents/ephysdataio"))
ABF_GLOBS = [os.path.join(STORE, "2026-06-02", "c01", "raw", "*.abf")]   # c01 = ipRGC
SPIKE_CH = "Im_prime"
OUT_RATE = 10000                 # 10 kHz, matches siso-spikes.csv
OUT_CSV = "barak-siso-spikes.csv"
DETECT = dict(polarity="neg", method="mad", k=6.0, refractory_s=0.002)
TRUNCATE_TO_COMMON = True        # files differ in length -> truncate to shortest


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

    print(f"Converting {len(files)} recording(s) -> {OUT_CSV}\n")
    cols, durations = [], []
    for f in files:
        rec = Recording.load(f)
        st = detect_spikes(rec.channel(SPIKE_CH), rec.fs, **DETECT)
        cols.append(st.binary(OUT_RATE, duration=rec.duration))
        durations.append(rec.duration)
        print(f"  {os.path.basename(f):28s} dur={rec.duration:6.2f}s  "
              f"spikes={len(st):4d}  rate={st.rate(rec.duration):5.2f} Hz")

    lengths = [len(c) for c in cols]
    if len(set(lengths)) > 1 and TRUNCATE_TO_COMMON:
        n = min(lengths)
        print(f"\n  NOTE: files differ in length ({min(lengths)}..{max(lengths)} samples); "
              f"truncating all columns to {n} ({n / OUT_RATE:.2f}s).")
        cols = [c[:n] for c in cols]

    mat = np.column_stack(cols)
    t = np.arange(mat.shape[0]) / OUT_RATE
    np.savetxt(OUT_CSV, np.column_stack([t, mat]),
               fmt=["%.6g"] + ["%d"] * mat.shape[1], delimiter=",")
    print(f"\nWrote {OUT_CSV}: {mat.shape[0]} rows x {mat.shape[1] + 1} cols "
          f"(time + {mat.shape[1]} epochs); total spikes {int(mat.sum())}")


if __name__ == "__main__":
    main()
