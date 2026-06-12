"""
analyze_flicker.py

Light-sensitivity / flicker-following analysis for the Barak ipRGC voltage-clamp
.abf recordings (black/white full-field square-wave flicker).

Unlike the s-iso gaussian pipeline (extract_siso4.py = reverse correlation), these
recordings test whether a cell is LIGHT SENSITIVE by asking: do the spikes entrain
to the flicker frequency?  The stimulus timing is taken straight from the TTL frame
-sync channel (ch2) -- no external stimulus file is needed.

Per file it:
  1. detects spikes (escaped action currents) on ch0 (Im_prime), same as
     extract_abf_spikes.py: inward peak, 6*MAD, 2 ms refractory.
  2. recovers the square-wave flicker from the TTL: a ~120 Hz carrier modulated by a
     low-frequency (~2 Hz) black/white envelope. White = carrier present (envelope
     high); black = envelope low. White-onset rising edges define the cycle.
  3. restricts analysis to the steady flicker region (constant pre/post blocks excluded).
  4. quantifies entrainment:
        - vector strength (VS) at the flicker freq + Rayleigh p
        - F1/F0 modulation from the cycle-PSTH
        - mean firing rate in the white vs black phase (uses the actual TTL state per
          spike, so it is robust to non-50% duty cycle)
  5. prints a light-sensitive / not verdict and draws a cycle-PSTH per cell.

Constant pre/post blocks (e.g. the unidentified 0-15 s block) are reported separately,
not folded into the flicker analysis.

Usage:
    /Users/j/miniconda3/bin/python analyze_flicker.py
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

# Process both the ipRGC flicker cell and the "loose" light-responsive cell.
# (The loose 0009-0013 files were moved into a notipRGC/ subfolder.)
ABF_FILES = sorted(glob.glob('ipRGC barak/**/*.abf', recursive=True))

CURRENT_CH    = 0          # Im_prime (pA)
TTL_CH        = 2          # frame-sync photodiode

# --- spike detection (matches extract_abf_spikes.py) ---
POLARITY      = 'neg'      # inward action currents
THRESH_K      = 6.0        # k * robust-sigma (MAD)
REFRACTORY_S  = 0.002

# --- flicker / TTL ---
ENV_BLOCK_S   = 0.010      # block size to collapse the carrier into a white/black envelope
PERIOD_TOL    = 0.25       # accept cycles whose length is within +/-25% of the median
N_PHASE_BINS  = 25         # cycle-PSTH resolution

# --- verdict thresholds ---
MIN_FLICKER_SPIKES = 20    # need at least this many spikes in the flicker region to judge
RAYLEIGH_P    = 0.01       # entrainment significant if Rayleigh p < this
RATE_RATIO    = 2.0        # OR white/black rate ratio beyond this (either direction)

OUT_FIG       = 'flicker_analysis.png'
OUT_CSV       = 'flicker_metrics.csv'


# ============================================================
# Helpers
# ============================================================

def detect_spike_times(im, fs):
    med = np.median(im)
    sigma = np.median(np.abs(im - med)) * 1.4826
    sig = -(im - med) if POLARITY == 'neg' else (im - med)
    peaks, _ = find_peaks(sig, height=THRESH_K * sigma,
                          distance=max(1, int(fs * REFRACTORY_S)))
    return peaks / fs


def ttl_envelope(ttl, fs):
    """Collapse the carrier into a white/black envelope; return (env, t_env, on_bool)."""
    mid = (np.percentile(ttl, 95) + np.percentile(ttl, 5)) / 2
    on = (ttl > mid).astype(float)
    blk = max(1, int(ENV_BLOCK_S * fs))
    nb = len(on) // blk
    env = on[:nb * blk].reshape(nb, blk).mean(1)
    t_env = (np.arange(nb) + 0.5) * blk / fs
    hi = np.percentile(env, 90)
    env_on = env > 0.5 * hi
    return env, t_env, env_on, blk / fs


def find_flicker_cycles(t_env, env_on):
    """White-onset times (rising edges) and the regular flicker period."""
    rises = np.where(np.diff(env_on.astype(int)) == 1)[0] + 1
    if len(rises) < 4:
        return None, None, None
    onsets = t_env[rises]
    iti = np.diff(onsets)
    period = np.median(iti)
    # keep only the run of regularly spaced onsets (drops stray edges in constant blocks)
    good = np.abs(iti - period) < PERIOD_TOL * period
    if good.sum() < 3:
        return None, None, None
    idx = np.where(good)[0]
    onsets = onsets[idx[0]: idx[-1] + 2]   # include the closing onset
    return onsets, period, onsets[-1] + period


def vector_strength(spike_t, t0, freq):
    """VS and Rayleigh p for spikes locked to `freq` starting at phase 0 = t0."""
    if len(spike_t) == 0:
        return 0.0, 1.0, 0.0
    theta = 2 * np.pi * freq * (spike_t - t0)
    vs = np.abs(np.mean(np.exp(1j * theta)))
    mean_phase = np.angle(np.mean(np.exp(1j * theta))) % (2 * np.pi)
    n = len(spike_t)
    p = np.exp(-n * vs ** 2)            # Rayleigh test (large-n approx)
    return vs, p, mean_phase


# ============================================================
# Per-file analysis
# ============================================================

def analyze_file(path):
    abf = pyabf.ABF(path); fs = abf.dataRate
    abf.setSweep(0, channel=CURRENT_CH); im = abf.sweepY
    abf.setSweep(0, channel=TTL_CH); ttl = abf.sweepY
    T = len(im) / fs

    st = detect_spike_times(im, fs)
    env, t_env, env_on, dt_env = ttl_envelope(ttl, fs)
    onsets, period, flick_end = find_flicker_cycles(t_env, env_on)

    res = dict(file=os.path.basename(path), folder=os.path.basename(os.path.dirname(path)),
               dur=T, n_spikes=len(st), flicker_hz=np.nan, vs=np.nan, rayleigh_p=np.nan,
               f1f0=np.nan, white_rate=np.nan, black_rate=np.nan, rate_ratio=np.nan,
               n_flick_spikes=0, verdict='no flicker detected', psth=None, period=None)

    if onsets is None:
        return res, st, env, t_env

    t0, t1 = onsets[0], flick_end
    freq = 1.0 / period
    res['flicker_hz'] = freq
    res['period'] = period

    # spikes inside the steady flicker region
    in_reg = (st >= t0) & (st < t1)
    sp = st[in_reg]
    res['n_flick_spikes'] = len(sp)

    # white/black phase rate using the ACTUAL ttl envelope state at each spike
    on_interp = np.interp(sp, t_env, env_on.astype(float)) > 0.5
    reg_mask = (t_env >= t0) & (t_env < t1)
    white_time = env_on[reg_mask].sum() * dt_env
    black_time = (~env_on[reg_mask]).sum() * dt_env
    res['white_rate'] = on_interp.sum() / white_time if white_time > 0 else np.nan
    res['black_rate'] = (~on_interp).sum() / black_time if black_time > 0 else np.nan
    if res['black_rate'] and res['black_rate'] > 0:
        res['rate_ratio'] = res['white_rate'] / res['black_rate']

    # vector strength at flicker freq
    vs, p, mean_phase = vector_strength(sp, t0, freq)
    res['vs'], res['rayleigh_p'] = vs, p

    # cycle-PSTH (fold onto one period)
    if len(sp):
        phase = ((sp - t0) % period) / period
        counts, edges = np.histogram(phase, bins=N_PHASE_BINS, range=(0, 1))
        n_cycles = (t1 - t0) / period
        rate = counts / (n_cycles * (period / N_PHASE_BINS))   # spikes/s
        res['psth'] = (edges[:-1], rate)
        # F1/F0 modulation from the psth
        F = np.fft.rfft(rate - rate.mean())
        f0 = rate.mean()
        f1 = np.abs(F[1]) / (N_PHASE_BINS / 2)
        res['f1f0'] = f1 / f0 if f0 > 0 else np.nan

    # verdict
    if res['n_flick_spikes'] < MIN_FLICKER_SPIKES:
        res['verdict'] = f"too few spikes ({res['n_flick_spikes']})"
    else:
        entrained = (p < RAYLEIGH_P) or \
                    (np.isfinite(res['rate_ratio']) and
                     (res['rate_ratio'] > RATE_RATIO or res['rate_ratio'] < 1 / RATE_RATIO))
        res['verdict'] = 'LIGHT-SENSITIVE (entrained)' if entrained else 'not entrained'

    return res, st, env, t_env


# ============================================================
# Main
# ============================================================

def main():
    if not ABF_FILES:
        raise SystemExit("No .abf files matched ABF_FILES glob.")

    results = []
    print(f"{'file':24s} {'fold':6s} {'flick_Hz':8s} {'#sp(flick)':10s} "
          f"{'VS':6s} {'Rayleigh_p':11s} {'white/blk':9s} {'verdict'}")
    print('-' * 110)
    for f in ABF_FILES:
        res, st, env, t_env = analyze_file(f)
        results.append(res)
        fhz = f"{res['flicker_hz']:.2f}" if np.isfinite(res['flicker_hz']) else '  -  '
        vs = f"{res['vs']:.3f}" if np.isfinite(res['vs']) else '  -  '
        pp = f"{res['rayleigh_p']:.2e}" if np.isfinite(res['rayleigh_p']) else '  -  '
        rr = f"{res['rate_ratio']:.2f}" if np.isfinite(res['rate_ratio']) else '  -  '
        print(f"{res['file']:24s} {res['folder']:6s} {fhz:>8s} {res['n_flick_spikes']:>10d} "
              f"{vs:>6s} {pp:>11s} {rr:>9s} {res['verdict']}")

    # ---- summary CSV ----
    cols = ['file', 'folder', 'dur', 'n_spikes', 'flicker_hz', 'n_flick_spikes',
            'vs', 'rayleigh_p', 'f1f0', 'white_rate', 'black_rate', 'rate_ratio', 'verdict']
    with open(OUT_CSV, 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for r in results:
            fh.write(','.join(str(r.get(c, '')) for c in cols) + '\n')
    print(f"\nWrote {OUT_CSV}")

    # ---- cycle-PSTH grid ----
    n = len(results)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, r in zip(axes.flat, results):
        if r['psth'] is not None:
            x, rate = r['psth']
            ax.bar(x, rate, width=1.0 / N_PHASE_BINS, align='edge', alpha=0.8)
            ax.axvspan(0, 0.5, color='gold', alpha=0.12)   # nominal white half
        ax.set_title(f"{r['file'].replace('.abf','')}\n"
                     f"{r['flicker_hz']:.1f}Hz  VS={r['vs']:.2f}  {r['verdict']}",
                     fontsize=8)
        ax.set_xlabel('flicker phase'); ax.set_ylabel('spikes/s')
    for ax in axes.flat[n:]:
        ax.axis('off')
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=120)
    print(f"Wrote {OUT_FIG}")


if __name__ == '__main__':
    main()
