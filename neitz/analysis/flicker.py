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
    t0 = float(onsets[idx[0]])
    # End at the CLOSE of the last full cycle, not at its opening onset. onsets[idx[-1]+1] is the
    # last confirmed-periodic onset — it *begins* a cycle that runs one `period` longer. Stopping at
    # that onset (the old behaviour) clipped the final complete frame into the "excluded" block.
    # Extend by `period`, clamped to the envelope end so we never claim data past the recording.
    t1 = min(float(onsets[idx[-1] + 1]) + period, float(t_env[-1]))
    on_e = t_env[rises]; off_e = t_env[falls]
    on_e = on_e[(on_e >= t0) & (on_e <= t1)]
    off_e = off_e[(off_e >= t0) & (off_e <= t1)]
    return Flicker(on_edges=on_e, off_edges=off_e, period=period, t0=t0, t1=t1)


def frame_clock_onset(ttl, fs, *, min_run=8, tol=0.35):
    """Stimulus-onset time (s) for a *non-periodic* stimulus (Gaussian noise) from its TTL frame
    clock: the first TTL rising edge that begins a SUSTAINED, regular pulse train (the projector
    frame clock running during the stimulus). Returns the edge time, or ``None`` if no such run is
    found (e.g. a flat / absent frame clock).

    Unlike :func:`detect_flicker` (which needs a low-frequency square wave), noise updates every
    frame, so there is no envelope square wave — we just locate where the frame clock *starts*.
    `min_run` consecutive inter-pulse gaps must sit within `tol` of the train's median period.

    NOTE (June2026): these noise trials run NO adapting carrier before the stimulus (per James), so
    the first sustained frame-clock run IS the stimulus onset — exactly what this returns.
    """
    ttl = np.asarray(ttl, dtype=float)
    mid = (np.percentile(ttl, 95) + np.percentile(ttl, 5)) / 2.0
    on = (ttl > mid).astype(int)
    rises = np.where(np.diff(on) == 1)[0] + 1
    if len(rises) < min_run + 1:
        return None
    t = rises / float(fs)
    gaps = np.diff(t)
    period = float(np.median(gaps))
    if period <= 0:
        return None
    for i in range(len(gaps) - min_run + 1):                # first run of steady gaps
        if np.all(np.abs(gaps[i:i + min_run] - period) < tol * period):
            return float(t[i])
    return None


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
               n_shuffle=1000, rng=None, null="jitter", jitter_s=None) -> dict:
    """
    Pooled transition-triggered PSTH + significance test for a time-locked response.

    `trials` is a list of dicts: {'spikes': array(s), 'on': edges, 'off': edges,
    'dur': trial_duration_s}. `which` is 'on' or 'off'.

    null : 'jitter' (default) — jitter each spike independently by ±jitter_s. This
           breaks precise time-locking while preserving the firing rate, so a real
           locked response exceeds the null. Correct for a PERIODIC stimulus.
           'shift' — circularly shift each trial's whole train. For a perfectly
           periodic stimulus this only ROTATES the triggered PSTH (the peak height
           is preserved), so its p-value has a floor ~(window/period) and low power.
           Kept for reference/back-compatibility.
    jitter_s : half-width of the per-spike jitter window (s); defaults to half the
           median inter-edge interval (~half the flicker period).

    Returns observed rate, latency-window peak ratio over baseline, peak latency,
    the null 99th-percentile band, and the p-value.
    """
    rng = rng or np.random.default_rng(0)
    bins = np.arange(-pre_s, post_s + bin_s, bin_s)
    centers = 0.5 * (bins[:-1] + bins[1:])

    if jitter_s is None:
        gaps = [np.median(np.diff(t[which])) for t in trials if len(t[which]) > 1]
        jitter_s = (float(np.median(gaps)) / 2.0) if gaps else max(post_s, 0.1)

    def pooled(trs):
        counts = np.zeros(len(centers)); n_edges = 0
        for t in trs:
            sp = t["spikes"]
            for e in t[which]:
                counts += np.histogram(sp - e, bins=bins)[0]
            n_edges += len(t[which])
        return counts / (n_edges * bin_s) if n_edges else counts

    baseline = (sum(len(t["spikes"]) for t in trials)
                / sum(t["dur"] for t in trials)) if sum(t["dur"] for t in trials) else float("nan")

    rate = pooled(trials)
    m = (centers * 1000 >= latency_ms[0]) & (centers * 1000 <= latency_ms[1])
    obs_ratio = rate[m].max() / baseline if baseline > 0 else float("nan")
    peak_t = centers[m][np.argmax(rate[m])] * 1000

    def perturb(t):
        sp = t["spikes"]
        if null == "jitter":
            sp = sp + rng.uniform(-jitter_s, jitter_s, sp.shape)
        else:  # 'shift' — whole-train circular rotation
            sp = (sp + rng.uniform(0, t["dur"])) % t["dur"]
        return dict(spikes=np.sort(sp), **{which: t[which]})

    null_ratios = np.empty(n_shuffle)
    null_rates = np.empty((n_shuffle, len(centers)))
    for k in range(n_shuffle):
        nr = pooled([perturb(t) for t in trials])
        null_rates[k] = nr
        null_ratios[k] = nr[m].max() / baseline if baseline > 0 else np.nan
    p = float((np.sum(null_ratios >= obs_ratio) + 1) / (n_shuffle + 1))

    return dict(centers=centers, rate=rate, baseline=baseline, ratio=obs_ratio,
                peak_ms=peak_t, p=p, band=np.percentile(null_rates, 99, axis=0),
                null=null, jitter_s=jitter_s)
