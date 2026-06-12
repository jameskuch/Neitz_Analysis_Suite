"""
flicker_onoff_psth.py

Transition-triggered PSTHs for the Barak flicker recordings, to classify cells as
ON / OFF / ON-OFF.  Unlike analyze_flicker.py (which folds spikes onto the flicker
cycle), this aligns spikes SEPARATELY to:
    * light-ON  edges  (black -> white, envelope rising)
    * light-OFF edges  (white -> black, envelope falling)
in real time over a short window, so a transient onset response (ON cell), a
transient offset response (OFF cell), or both (ON-OFF cell) show up as distinct
peaks with a visible latency.

Stimulus timing comes from the TTL channel (ch2); response is the escaped action
currents on ch0 (same detector as extract_abf_spikes.py / analyze_flicker.py).

Usage:
    /Users/j/miniconda3/bin/python flicker_onoff_psth.py
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

STORE = os.path.expanduser(os.environ.get("EPHYSDATAIO_ROOT", "~/Documents/ephysdataio"))
ABF_FILES = sorted(glob.glob(os.path.join(STORE, "2026-06-02", "**", "*.abf"), recursive=True))

CURRENT_CH   = 0
TTL_CH       = 2
THRESH_K     = 6.0
REFRACTORY_S = 0.002

ENV_BLOCK_S  = 0.010
PERIOD_TOL   = 0.25

BIN_MS       = 10.0        # PSTH bin width
PRE_FRAC     = 0.20        # window before a transition, as fraction of the cycle period
POST_FRAC    = 0.80        # window after a transition (capped below the next transition)
LATENCY_MS   = (10, 150)   # window used to measure the transient ON/OFF response
RESP_RATIO   = 1.5         # peak/baseline above this in the latency window => "responds"

OUT_FIG      = 'flicker_onoff_psth.png'
OUT_CSV      = 'flicker_onoff_metrics.csv'


# ============================================================
# Helpers (shared logic with analyze_flicker.py)
# ============================================================

def detect_spike_times(im, fs):
    med = np.median(im)
    sigma = np.median(np.abs(im - med)) * 1.4826
    peaks, _ = find_peaks(-(im - med), height=THRESH_K * sigma,
                          distance=max(1, int(fs * REFRACTORY_S)))
    return peaks / fs


def ttl_edges(ttl, fs):
    """Return (on_edges, off_edges, period) in seconds, restricted to the steady flicker run."""
    mid = (np.percentile(ttl, 95) + np.percentile(ttl, 5)) / 2
    on = (ttl > mid).astype(float)
    blk = max(1, int(ENV_BLOCK_S * fs))
    nb = len(on) // blk
    env = on[:nb * blk].reshape(nb, blk).mean(1)
    t_env = (np.arange(nb) + 0.5) * blk / fs
    env_on = env > 0.5 * np.percentile(env, 90)

    d = np.diff(env_on.astype(int))
    rises = np.where(d == 1)[0] + 1
    falls = np.where(d == -1)[0] + 1
    if len(rises) < 4:
        return None, None, None

    onsets = t_env[rises]
    iti = np.diff(onsets)
    period = np.median(iti)
    good = np.abs(iti - period) < PERIOD_TOL * period
    if good.sum() < 3:
        return None, None, None
    idx = np.where(good)[0]
    t0, t1 = onsets[idx[0]], onsets[idx[-1] + 1]

    on_edges = t_env[rises]
    off_edges = t_env[falls]
    on_edges = on_edges[(on_edges >= t0) & (on_edges <= t1)]
    off_edges = off_edges[(off_edges >= t0) & (off_edges <= t1)]
    return on_edges, off_edges, period


def triggered_psth(spike_t, edges, pre_s, post_s, bin_s):
    """Average firing rate (spikes/s) in [-pre, +post] around each edge."""
    if len(edges) == 0:
        return None, None
    bins = np.arange(-pre_s, post_s + bin_s, bin_s)
    counts = np.zeros(len(bins) - 1)
    for e in edges:
        rel = spike_t - e
        rel = rel[(rel >= -pre_s) & (rel < post_s)]
        counts += np.histogram(rel, bins=bins)[0]
    rate = counts / (len(edges) * bin_s)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, rate


def transient_index(centers, rate, baseline):
    """Peak rate in the latency window relative to baseline."""
    if centers is None or baseline <= 0:
        return np.nan, np.nan
    m = (centers * 1000 >= LATENCY_MS[0]) & (centers * 1000 <= LATENCY_MS[1])
    if not m.any():
        return np.nan, np.nan
    peak = rate[m].max()
    peak_t = centers[m][np.argmax(rate[m])] * 1000
    return peak / baseline, peak_t


# ============================================================
# Main
# ============================================================

def main():
    results = []
    fig_rows = []

    print(f"{'file':22s}{'fold':10s}{'flick_Hz':9s}{'base':6s}"
          f"{'ON pk/base@ms':16s}{'OFF pk/base@ms':16s}{'class'}")
    print('-' * 95)

    for f in ABF_FILES:
        abf = pyabf.ABF(f); fs = abf.dataRate
        abf.setSweep(0, channel=CURRENT_CH); im = abf.sweepY
        abf.setSweep(0, channel=TTL_CH); ttl = abf.sweepY

        st = detect_spike_times(im, fs)
        on_e, off_e, period = ttl_edges(ttl, fs)

        r = dict(file=os.path.basename(f), folder=os.path.basename(os.path.dirname(os.path.dirname(f))))  # cell folder
        if on_e is None:
            r.update(flicker_hz=np.nan, on_ratio=np.nan, off_ratio=np.nan, cls='no flicker')
            results.append(r); fig_rows.append((f, None))
            print(f"{r['file']:22s}{r['folder']:10s}{'  -':9s}{'':6s}{'':16s}{'':16s}no flicker")
            continue

        pre_s = PRE_FRAC * period
        post_s = min(POST_FRAC * period, period - pre_s)   # don't run into the next edge

        t0, t1 = on_e[0], (off_e[-1] if len(off_e) else on_e[-1])
        in_reg = st[(st >= t0 - pre_s) & (st <= t1 + post_s)]
        baseline = len(in_reg) / ((t1 - t0) if t1 > t0 else 1.0)   # mean rate in flicker region

        c_on, rate_on = triggered_psth(st, on_e, pre_s, post_s, BIN_MS / 1000)
        c_off, rate_off = triggered_psth(st, off_e, pre_s, post_s, BIN_MS / 1000)

        on_ratio, on_t = transient_index(c_on, rate_on, baseline)
        off_ratio, off_t = transient_index(c_off, rate_off, baseline)

        on_resp = np.isfinite(on_ratio) and on_ratio >= RESP_RATIO
        off_resp = np.isfinite(off_ratio) and off_ratio >= RESP_RATIO
        cls = ('ON-OFF' if on_resp and off_resp else
               'ON' if on_resp else 'OFF' if off_resp else 'no transient')
        r.update(flicker_hz=1.0 / period, baseline=baseline,
                 on_ratio=on_ratio, on_t=on_t, off_ratio=off_ratio, off_t=off_t, cls=cls)
        results.append(r)
        fig_rows.append((f, (c_on, rate_on, c_off, rate_off, baseline, period)))

        print(f"{r['file']:22s}{r['folder']:10s}{1.0/period:6.2f} Hz {baseline:5.1f} "
              f"{on_ratio:6.2f}@{on_t:5.0f}ms   {off_ratio:6.2f}@{off_t:5.0f}ms   {cls}")

    # ---- CSV ----
    cols = ['file', 'folder', 'flicker_hz', 'baseline', 'on_ratio', 'on_t',
            'off_ratio', 'off_t', 'cls']
    with open(OUT_CSV, 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for r in results:
            fh.write(','.join(str(r.get(c, '')) for c in cols) + '\n')
    print(f"\nWrote {OUT_CSV}")

    # ---- figure: ON (blue) vs OFF (red) triggered PSTH per cell ----
    n = len(fig_rows); ncol = 5; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, (f, data), r in zip(axes.flat, fig_rows, results):
        if data is not None:
            c_on, rate_on, c_off, rate_off, baseline, period = data
            ax.step(c_on * 1000, rate_on, where='mid', color='tab:blue', label='light ON')
            ax.step(c_off * 1000, rate_off, where='mid', color='tab:red', label='light OFF')
            ax.axhline(baseline, color='k', ls=':', lw=0.8)
            ax.axvline(0, color='gray', lw=0.6)
            ax.legend(fontsize=6, loc='upper right')
        ax.set_title(f"{r['file'].replace('.abf','')}  {r.get('cls','')}\n"
                     f"ON {r.get('on_ratio',np.nan):.1f}x  OFF {r.get('off_ratio',np.nan):.1f}x",
                     fontsize=8)
        ax.set_xlabel('time from transition (ms)'); ax.set_ylabel('spikes/s')
    for ax in axes.flat[n:]:
        ax.axis('off')
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=120)
    print(f"Wrote {OUT_FIG}")


if __name__ == '__main__':
    main()
