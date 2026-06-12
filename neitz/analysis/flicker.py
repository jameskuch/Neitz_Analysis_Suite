"""
neitz.analysis.flicker — periodic-stimulus (flicker) analysis.

Recovers the square-wave flicker timing from a TTL frame-sync channel and provides
cycle-PSTH, transition-triggered PSTH (ON/OFF), and vector-strength tools, plus a
circular-shift significance test. Validated in analyze_flicker.py / flicker_onoff_pooled.py.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Flicker:
    on_edges: np.ndarray     # light-ON transition times (s)
    off_edges: np.ndarray    # light-OFF transition times (s)
    period: float            # s
    t0: float                # start of steady flicker region (s)
    t1: float                # end of steady flicker region (s)

    @property
    def freq(self):
        return 1.0 / self.period if self.period else float("nan")


def detect_flicker(ttl, fs, *, env_block_s=0.010, period_tol=0.25) -> "Flicker | None":
    """Recover the square-wave flicker from a TTL frame-sync channel.

    Collapses the carrier (e.g. 120 Hz) into a white/black envelope, finds the
    regular low-frequency square wave, and returns its ON/OFF edges and the steady
    region. Returns None if no regular flicker is present.
    """
    ttl = np.asarray(ttl, dtype=float)
    mid = (np.percentile(ttl, 95) + np.percentile(ttl, 5)) / 2
    on = (ttl > mid).astype(float)
    blk = max(1, int(env_block_s * fs))
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
    period = float(np.median(np.diff(onsets)))
    good = np.abs(np.diff(onsets) - period) < period_tol * period
    if good.sum() < 3:
        return None
    idx = np.where(good)[0]
    t0, t1 = float(onsets[idx[0]]), float(onsets[idx[-1] + 1])
    on_e = t_env[rises]; off_e = t_env[falls]
    on_e = on_e[(on_e >= t0) & (on_e <= t1)]
    off_e = off_e[(off_e >= t0) & (off_e <= t1)]
    return Flicker(on_edges=on_e, off_edges=off_e, period=period, t0=t0, t1=t1)


def vector_strength(spike_t, t0, freq) -> tuple:
    """VS and Rayleigh p for spikes locked to `freq` (phase 0 at t0)."""
    spike_t = np.asarray(spike_t, float)
    if len(spike_t) == 0:
        return 0.0, 1.0
    theta = 2 * np.pi * freq * (spike_t - t0)
    vs = float(np.abs(np.mean(np.exp(1j * theta))))
    p = float(np.exp(-len(spike_t) * vs ** 2))
    return vs, p


def triggered_psth(spike_t, edges, pre_s, post_s, bin_s) -> tuple:
    """Mean firing rate (spikes/s) in [-pre, +post] around each edge."""
    spike_t = np.asarray(spike_t, float)
    if len(edges) == 0:
        return None, None
    bins = np.arange(-pre_s, post_s + bin_s, bin_s)
    counts = np.zeros(len(bins) - 1)
    for e in edges:
        counts += np.histogram(spike_t - e, bins=bins)[0]
    rate = counts / (len(edges) * bin_s)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, rate


def cycle_psth(spike_t, t0, t1, period, n_bins=25) -> tuple:
    """Fold spikes in [t0,t1] onto one cycle; return (phase_edges, rate)."""
    spike_t = np.asarray(spike_t, float)
    sp = spike_t[(spike_t >= t0) & (spike_t < t1)]
    phase = ((sp - t0) % period) / period
    counts, edges = np.histogram(phase, bins=n_bins, range=(0, 1))
    n_cycles = (t1 - t0) / period
    rate = counts / (n_cycles * (period / n_bins)) if n_cycles else counts
    return edges[:-1], rate


def shift_test(trials, which, *, pre_s, post_s, bin_s, latency_ms=(10, 150),
               n_shuffle=1000, rng=None) -> dict:
    """
    Pooled transition-triggered PSTH + circular-shift significance test.

    `trials` is a list of dicts: {'spikes': array(s), 'on': edges, 'off': edges,
    'dur': trial_duration_s}. `which` is 'on' or 'off'. Returns observed rate,
    latency-window peak ratio over baseline, the shuffle band, and p-value.
    """
    rng = rng or np.random.default_rng(0)
    bins = np.arange(-pre_s, post_s + bin_s, bin_s)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def pooled(trs):
        counts = np.zeros(len(centers)); n_edges = 0
        for t in trs:
            sp = t["spikes"]
            for e in t[which]:
                counts += np.histogram(sp - e, bins=bins)[0]
            n_edges += len(t[which])
        return counts / (n_edges * bin_s) if n_edges else counts

    # baseline = pooled mean rate across all spikes / total duration
    tot_sp = sum(len(t["spikes"]) for t in trials)
    tot_t = sum(t["dur"] for t in trials)
    baseline = tot_sp / tot_t if tot_t else float("nan")

    rate = pooled(trials)
    m = (centers * 1000 >= latency_ms[0]) & (centers * 1000 <= latency_ms[1])
    obs_ratio = rate[m].max() / baseline if baseline > 0 else float("nan")
    peak_t = centers[m][np.argmax(rate[m])] * 1000

    null_ratios = np.empty(n_shuffle)
    null_rates = np.empty((n_shuffle, len(centers)))
    for k in range(n_shuffle):
        shifted = [dict(spikes=np.sort((t["spikes"] + rng.uniform(0, t["dur"])) % t["dur"]),
                        **{which: t[which]}) for t in trials]
        nr = pooled(shifted)
        null_rates[k] = nr
        null_ratios[k] = nr[m].max() / baseline if baseline > 0 else np.nan
    p = float((np.sum(null_ratios >= obs_ratio) + 1) / (n_shuffle + 1))

    return dict(centers=centers, rate=rate, baseline=baseline, ratio=obs_ratio,
                peak_ms=peak_t, p=p, band=np.percentile(null_rates, 99, axis=0))
