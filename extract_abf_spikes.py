"""
extract_abf_spikes.py

Detect spikes (action currents) in the Barak ipRGC voltage-clamp .abf files and
write them out in the EXACT same CSV format as SaraipRGC/siso-spikes.csv so they
can be fed to the extract_siso4.py pipeline.

Target format (no header):
    col 0       = time vector at OUT_RATE Hz   (e.g. 0, 1e-4, 2e-4, ...)
    cols 1..N   = binary spike trains (0/1), one column per recording/epoch
                  (Neitz.extract_spike_times_from_matrix reads spikes as t[col > 0])

IMPORTANT context (read before trusting the output):
  * These recordings are VOLTAGE CLAMP (protocol 'v_calmp_guassian'); channel 0 is
    membrane current Im_prime (pA). The "spikes" are biphasic action currents that
    escape the clamp -- large (~100-200 pA) relative to the ~2-3 pA noise, so they
    threshold cleanly. We detect the inward (negative-going) peak of each event.
  * Each file is ONE continuous sweep (~79-92 s) at 20 kHz -- NOT 15 epochs x 10 s
    like the siso data. Here each FILE becomes one column ("epoch").
  * The actual Gaussian stimulus shown during these recordings is NOT in the repo
    yet (pending from Barak). Without it the reverse-correlation filter cannot be
    computed -- this script only formats the RESPONSE side.

Usage:
    /Users/j/miniconda3/bin/python extract_abf_spikes.py
"""

import os
import glob
import numpy as np
import pyabf
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

# Which recordings to convert. Default = the "good" ipRGC subfolder cells
# (real voltage clamp at ~-59 mV with large action currents). Swap to the parent
# folder glob if you want the 0009-0013 set.
ABF_FILES = sorted(glob.glob('ipRGC barak/ipRGC/*.abf'))
# ABF_FILES = sorted(glob.glob('ipRGC barak/*.abf'))

SPIKE_CH      = 0          # channel index of Im_prime (membrane current, pA)
OUT_RATE      = 10000      # output sample rate (Hz) -- 10 kHz matches siso-spikes.csv
OUT_CSV       = 'barak-siso-spikes.csv'

# --- spike detection ---
POLARITY      = 'neg'      # 'neg' (inward/downward), 'pos' (outward), or 'abs'
THRESH_MODE   = 'mad'      # 'mad' -> THRESH_K * robust sigma ; 'pA' -> THRESH_PA
THRESH_K      = 6.0        # used when THRESH_MODE == 'mad'  (6-8 is the clean plateau)
THRESH_PA     = 20.0       # used when THRESH_MODE == 'pA'
REFRACTORY_S  = 0.002      # min spacing between detected spikes (s) -> kills double-counts

# --- combining unequal-length files into one rectangular CSV ---
# Files differ slightly in length (some 79 s, one 92 s). To keep one rectangular
# CSV we truncate every column to the shortest common length and warn.
TRUNCATE_TO_COMMON = True

SAVE_PLOT     = True
PLOT_FILE     = 'barak-siso-spikes_check.png'
PLOT_WIN_S    = (20.0, 25.0)   # window to show in the verification plot


# ============================================================
# Helpers
# ============================================================

def detect_spike_times(im, fs):
    """Return spike times (s) detected on a current trace `im` sampled at `fs` Hz."""
    med = np.median(im)
    sigma = np.median(np.abs(im - med)) * 1.4826   # robust std (MAD)

    if POLARITY == 'neg':
        sig = -(im - med)
    elif POLARITY == 'pos':
        sig = (im - med)
    elif POLARITY == 'abs':
        sig = np.abs(im - med)
    else:
        raise ValueError(f"POLARITY must be 'neg','pos','abs'; got {POLARITY!r}")

    if THRESH_MODE == 'mad':
        height = THRESH_K * sigma
    elif THRESH_MODE == 'pA':
        height = THRESH_PA
    else:
        raise ValueError(f"THRESH_MODE must be 'mad' or 'pA'; got {THRESH_MODE!r}")

    peaks, _ = find_peaks(sig, height=height, distance=max(1, int(fs * REFRACTORY_S)))
    return peaks / fs, sigma, height


def binary_train(spike_times_s, n_out, out_rate):
    """Make a length-n_out binary (0/1) train with a 1 at each spike's nearest bin."""
    train = np.zeros(n_out, dtype=np.int8)
    idx = np.round(spike_times_s * out_rate).astype(int)
    idx = idx[(idx >= 0) & (idx < n_out)]
    train[idx] = 1
    return train


# ============================================================
# Main
# ============================================================

def main():
    if not ABF_FILES:
        raise SystemExit("No .abf files matched ABF_FILES glob.")

    print(f"Converting {len(ABF_FILES)} recording(s) -> {OUT_CSV}\n")

    columns = []     # binary trains (each at OUT_RATE)
    labels = []      # file basenames
    durations = []   # seconds
    detect_info = []

    for f in ABF_FILES:
        abf = pyabf.ABF(f)
        abf.setSweep(0, channel=SPIKE_CH)
        im = abf.sweepY
        fs = abf.dataRate
        dur = abf.sweepLengthSec

        st, sigma, height = detect_spike_times(im, fs)

        n_out = int(round(dur * OUT_RATE))
        train = binary_train(st, n_out, OUT_RATE)

        columns.append(train)
        labels.append(os.path.basename(f))
        durations.append(dur)
        detect_info.append((len(st), len(st) / dur, sigma, height))

        print(f"  {os.path.basename(f):28s} dur={dur:6.2f}s  "
              f"sigma={sigma:5.2f}pA thr={height:6.1f}pA  "
              f"spikes={len(st):4d}  rate={len(st)/dur:5.2f} Hz")

    # ---- assemble rectangular matrix ----
    lengths = [len(c) for c in columns]
    if len(set(lengths)) > 1:
        if TRUNCATE_TO_COMMON:
            n = min(lengths)
            longest = max(lengths)
            print(f"\n  NOTE: files differ in length ({min(lengths)}..{longest} samples). "
                  f"Truncating all columns to the common {n} samples "
                  f"({n/OUT_RATE:.2f}s). Set TRUNCATE_TO_COMMON=False to write per-file CSVs.")
            columns = [c[:n] for c in columns]
        else:
            # write one CSV per file instead of a combined matrix
            for lab, col in zip(labels, columns):
                t = np.arange(len(col)) / OUT_RATE
                out = lab.replace('.abf', '_spikes.csv')
                np.savetxt(out, np.column_stack([t, col]),
                           fmt=['%.6g', '%d'], delimiter=',')
                print(f"    wrote {out}  ({len(col)} rows)")
            return
    else:
        n = lengths[0]

    spike_matrix = np.column_stack(columns)          # (n, num_files)
    t = np.arange(n) / OUT_RATE                       # 10 kHz time base
    data = np.column_stack([t, spike_matrix])

    fmt = ['%.6g'] + ['%d'] * spike_matrix.shape[1]
    np.savetxt(OUT_CSV, data, fmt=fmt, delimiter=',')
    print(f"\nWrote {OUT_CSV}: {data.shape[0]} rows x {data.shape[1]} cols "
          f"(time + {spike_matrix.shape[1]} epochs)")
    print(f"Total spikes across columns: {int(spike_matrix.sum())}")

    # ---- verification plot ----
    if SAVE_PLOT:
        f0 = ABF_FILES[0]
        abf = pyabf.ABF(f0); abf.setSweep(0, channel=SPIKE_CH)
        im = abf.sweepY; tt = abf.sweepX; fs = abf.dataRate
        i0, i1 = int(PLOT_WIN_S[0]*fs), int(PLOT_WIN_S[1]*fs)
        st, _, _ = detect_spike_times(im, fs)
        st = st[(st >= PLOT_WIN_S[0]) & (st < PLOT_WIN_S[1])]

        plt.figure(figsize=(14, 4))
        plt.plot(tt[i0:i1], im[i0:i1], lw=0.5)
        plt.plot(st, np.full_like(st, im[i0:i1].min()), 'rv', ms=6)
        plt.title(f"{labels[0]}  Im_prime  {PLOT_WIN_S[0]:.0f}-{PLOT_WIN_S[1]:.0f}s  "
                  f"(red = detected spikes, {POLARITY} {THRESH_MODE})")
        plt.xlabel('time (s)'); plt.ylabel('pA')
        plt.tight_layout(); plt.savefig(PLOT_FILE, dpi=110)
        print(f"Saved verification plot: {PLOT_FILE}")


if __name__ == '__main__':
    main()
