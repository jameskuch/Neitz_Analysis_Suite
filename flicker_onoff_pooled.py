"""
flicker_onoff_pooled.py

Pooled, significance-tested ON/OFF classification for the Barak flicker recordings.

Like Sara's s-iso analysis averaged 15 epochs of one cell, this pools ALL trials of
each cell (grouped by subfolder: ipRGC vs notipRGC) before judging the response.
Per cell it builds transition-triggered PSTHs aligned to light-ON and light-OFF edges
across every trial, then tests significance with a circular-shift null that preserves
each trial's firing rate and spike-train structure but destroys stimulus locking.

A cell is called ON / OFF / ON-OFF only if its peak firing in the latency window
exceeds the shuffle null (p < ALPHA) for that edge type.

Stimulus timing comes from the TTL channel (ch2); spikes are the escaped action
currents on ch0 (same detector as the other scripts).

Usage:
    /Users/j/miniconda3/bin/python flicker_onoff_pooled.py
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

ABF_FILES = sorted(glob.glob('ipRGC barak/**/*.abf', recursive=True))

CURRENT_CH   = 0
TTL_CH       = 2
THRESH_K     = 6.0
REFRACTORY_S = 0.002

ENV_BLOCK_S  = 0.010
PERIOD_TOL   = 0.25

BIN_MS       = 10.0
PRE_MS       = 100.0
LATENCY_MS   = (10, 150)    # transient-response window after a transition
N_SHUFFLE    = 1000
ALPHA        = 0.01         # significance level for ON/OFF call
RNG_SEED     = 0

OUT_FIG      = 'flicker_onoff_pooled.png'
OUT_CSV      = 'flicker_onoff_pooled_metrics.csv'


# ============================================================
# Helpers
# ============================================================

def detect_spike_times(im, fs):
    med = np.median(im)
    sigma = np.median(np.abs(im - med)) * 1.4826
    peaks, _ = find_peaks(-(im - med), height=THRESH_K * sigma,
                          distance=max(1, int(fs * REFRACTORY_S)))
    return peaks / fs


def ttl_edges(ttl, fs):
    """ON edges, OFF edges (s), the flicker period, and the steady-region (t0,t1)."""
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
        return None
    onsets = t_env[rises]
    period = np.median(np.diff(onsets))
    good = np.abs(np.diff(onsets) - period) < PERIOD_TOL * period
    if good.sum() < 3:
        return None
    idx = np.where(good)[0]
    t0, t1 = onsets[idx[0]], onsets[idx[-1] + 1]
    on_e = t_env[rises]; off_e = t_env[falls]
    on_e = on_e[(on_e >= t0) & (on_e <= t1)]
    off_e = off_e[(off_e >= t0) & (off_e <= t1)]
    return dict(on=on_e, off=off_e, period=period, t0=t0, t1=t1)


def pooled_rate(trials, which, bins, bin_s):
    """Pooled triggered firing rate over all edges of all trials."""
    counts = np.zeros(len(bins) - 1)
    n_edges = 0
    for tr in trials:
        edges = tr[which]
        sp = tr['spikes']
        for e in edges:
            counts += np.histogram(sp - e, bins=bins)[0]
        n_edges += len(edges)
    rate = counts / (n_edges * bin_s) if n_edges else counts
    return rate, n_edges


def latency_peak(centers, rate, baseline):
    m = (centers * 1000 >= LATENCY_MS[0]) & (centers * 1000 <= LATENCY_MS[1])
    peak = rate[m].max()
    peak_t = centers[m][np.argmax(rate[m])] * 1000
    return peak, peak / baseline if baseline > 0 else np.nan, peak_t


# ============================================================
# Per-cell pooled analysis
# ============================================================

def analyze_cell(name, paths, rng):
    trials = []
    for p in paths:
        abf = pyabf.ABF(p); fs = abf.dataRate
        abf.setSweep(0, channel=CURRENT_CH); im = abf.sweepY
        abf.setSweep(0, channel=TTL_CH); ttl = abf.sweepY
        e = ttl_edges(ttl, fs)
        if e is None:
            continue
        st = detect_spike_times(im, fs)
        trials.append(dict(spikes=st, on=e['on'], off=e['off'],
                           period=e['period'], t0=e['t0'], t1=e['t1'],
                           dur=len(im) / fs))
    if not trials:
        return None

    period = np.median([t['period'] for t in trials])
    pre_s = PRE_MS / 1000
    post_s = min(0.9 * period, period - pre_s + 0.05)
    bin_s = BIN_MS / 1000
    bins = np.arange(-pre_s, post_s + bin_s, bin_s)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # baseline = pooled mean rate inside the flicker regions
    tot_sp = sum(int(((t['spikes'] >= t['t0']) & (t['spikes'] <= t['t1'])).sum()) for t in trials)
    tot_time = sum(t['t1'] - t['t0'] for t in trials)
    baseline = tot_sp / tot_time if tot_time else np.nan

    out = dict(cell=name, n_trials=len(trials), period=period, baseline=baseline,
               n_on=sum(len(t['on']) for t in trials),
               n_off=sum(len(t['off']) for t in trials),
               n_spikes=sum(len(t['spikes']) for t in trials), centers=centers)

    for which in ('on', 'off'):
        rate, n_edges = pooled_rate(trials, which, bins, bin_s)
        _, ratio, peak_t = latency_peak(centers, rate, baseline)

        # circular-shift null: shift each trial's spikes by a random offset, recompute peak ratio
        null_ratios = np.empty(N_SHUFFLE)
        null_rates = np.empty((N_SHUFFLE, len(centers)))
        for k in range(N_SHUFFLE):
            shifted = []
            for t in trials:
                off = rng.uniform(0, t['dur'])
                sp = (t['spikes'] + off) % t['dur']
                shifted.append(dict(spikes=np.sort(sp), on=t['on'], off=t['off']))
            nr, _ = pooled_rate(shifted, which, bins, bin_s)
            null_rates[k] = nr
            _, nratio, _ = latency_peak(centers, nr, baseline)
            null_ratios[k] = nratio
        p = (np.sum(null_ratios >= ratio) + 1) / (N_SHUFFLE + 1)
        band = np.percentile(null_rates, 100 * (1 - ALPHA), axis=0)

        out[which] = dict(rate=rate, ratio=ratio, peak_t=peak_t, p=p, band=band,
                          null_hi=np.percentile(null_ratios, 100 * (1 - ALPHA)))
    return out


# ============================================================
# Main
# ============================================================

def main():
    rng = np.random.default_rng(RNG_SEED)

    # group files by subfolder = cell
    cells = {}
    for f in ABF_FILES:
        cells.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)

    results = []
    print(f"{'cell':10s}{'trials':7s}{'flick_Hz':9s}{'base':6s}{'#spk':6s}"
          f"{'ON ratio(p)':18s}{'OFF ratio(p)':18s}{'class'}")
    print('-' * 95)
    for name, paths in cells.items():
        r = analyze_cell(name, sorted(paths), rng)
        if r is None:
            continue
        results.append(r)
        on_sig = r['on']['p'] < ALPHA
        off_sig = r['off']['p'] < ALPHA
        cls = ('ON-OFF' if on_sig and off_sig else
               'ON' if on_sig else 'OFF' if off_sig else 'no sig. response')
        r['cls'] = cls
        print(f"{name:10s}{r['n_trials']:<7d}{r['period'] and 1/r['period']:<9.2f}"
              f"{r['baseline']:<6.1f}{r['n_spikes']:<6d}"
              f"{r['on']['ratio']:5.2f} (p={r['on']['p']:.3f})   "
              f"{r['off']['ratio']:5.2f} (p={r['off']['p']:.3f})   {cls}")

    # ---- CSV ----
    with open(OUT_CSV, 'w') as fh:
        fh.write('cell,n_trials,flicker_hz,baseline,n_spikes,n_on,n_off,'
                 'on_ratio,on_peak_ms,on_p,off_ratio,off_peak_ms,off_p,class\n')
        for r in results:
            fh.write(f"{r['cell']},{r['n_trials']},{1/r['period']:.2f},{r['baseline']:.2f},"
                     f"{r['n_spikes']},{r['n_on']},{r['n_off']},"
                     f"{r['on']['ratio']:.3f},{r['on']['peak_t']:.0f},{r['on']['p']:.4f},"
                     f"{r['off']['ratio']:.3f},{r['off']['peak_t']:.0f},{r['off']['p']:.4f},"
                     f"{r['cls']}\n")
    print(f"\nWrote {OUT_CSV}")

    # ---- figure: per cell, ON and OFF triggered PSTH with shuffle band ----
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.4 * n), squeeze=False)
    for i, r in enumerate(results):
        for j, which in enumerate(('on', 'off')):
            ax = axes[i][j]
            c = r['centers'] * 1000
            ax.step(c, r[which]['rate'], where='mid', color='tab:blue', lw=1.5, label='observed')
            ax.step(c, r[which]['band'], where='mid', color='tab:red', lw=1.0, ls='--',
                    label=f'shuffle {100*(1-ALPHA):.0f}th pct')
            ax.axhline(r['baseline'], color='k', ls=':', lw=0.8, label='baseline')
            ax.axvspan(*LATENCY_MS, color='gold', alpha=0.12)
            ax.axvline(0, color='gray', lw=0.6)
            sig = 'SIG' if r[which]['p'] < ALPHA else 'n.s.'
            ax.set_title(f"{r['cell']}  light {which.upper()}  "
                         f"{r[which]['ratio']:.2f}x @ {r[which]['peak_t']:.0f}ms  "
                         f"p={r[which]['p']:.3f} [{sig}]", fontsize=9)
            ax.set_xlabel('time from transition (ms)'); ax.set_ylabel('spikes/s')
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=120)
    print(f"Wrote {OUT_FIG}")


if __name__ == '__main__':
    main()
