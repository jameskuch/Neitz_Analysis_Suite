"""
FlickerParadigm — square-wave (black/white) light-response analysis.

Bundles spike detection + flicker recovery from the TTL + the response analyses
(cycle PSTH, vector strength, pooled transition-triggered ON/OFF with significance)
so scripts, GUI, and notebooks share one entry point. Stimulus timing comes from
the TTL channel — no external stimulus file needed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from ..spikes import detect_spikes
from ..analysis import flicker as flk
from .base import StimulusMeta


@dataclass
class FlickerResult:
    name: str
    fs: float
    duration: float
    n_spikes: int
    flicker: object               # flk.Flicker or None
    n_in_region: int
    vector_strength: float
    rayleigh_p: float
    cycle_phase: object           # np.ndarray or None
    cycle_rate: object            # np.ndarray or None

    @property
    def freq(self):
        return self.flicker.freq if self.flicker else float("nan")


@dataclass
class FlickerParadigm:
    current_channel: str = "Im_prime"
    ttl_channel: str = "TTL"
    polarity: str = "neg"
    method: str = "mad"
    k: float = 6.0
    abs_threshold: float = None
    refractory_s: float = 0.002
    n_phase_bins: int = 25
    meta: StimulusMeta = field(default_factory=lambda: StimulusMeta(kind="flicker"))

    @property
    def _det(self) -> dict:
        return dict(polarity=self.polarity, method=self.method, k=self.k,
                    abs_threshold=self.abs_threshold, refractory_s=self.refractory_s)

    def detect(self, rec):
        """(SpikeTrain, Flicker|None) for one recording."""
        st = detect_spikes(rec.channel(self.current_channel), rec.fs, **self._det)
        fl = flk.detect_flicker(rec.channel(self.ttl_channel), rec.fs)
        return st, fl

    def analyze_recording(self, rec, name: str = None) -> FlickerResult:
        st, fl = self.detect(rec)
        if fl is None:
            return FlickerResult(name or "", rec.fs, rec.duration, len(st), None,
                                 0, float("nan"), float("nan"), None, None)
        inreg = st.times[(st.times >= fl.t0) & (st.times <= fl.t1)]
        vs, p = flk.vector_strength(inreg, fl.t0, fl.freq)
        phase, rate = flk.cycle_psth(inreg, fl.t0, fl.t1, fl.period, self.n_phase_bins)
        return FlickerResult(name or "", rec.fs, rec.duration, len(st), fl,
                             len(inreg), vs, p, phase, rate)

    def trials_from(self, recs) -> list:
        """shift_test trial dicts from recordings (skips files with no flicker)."""
        trials = []
        for rec in recs:
            st, fl = self.detect(rec)
            if fl is None:
                continue
            trials.append(dict(spikes=st.times, on=fl.on_edges, off=fl.off_edges,
                               dur=rec.duration, freq=fl.freq))
        return trials

    def analyze_group(self, recs, *, pre_s=0.1, post_s=None, bin_s=0.01,
                      n_shuffle=1000, rng=None) -> dict:
        """Pool all trials of a cell; transition-triggered ON/OFF + significance."""
        return self.group_from_trials(self.trials_from(recs), pre_s=pre_s, post_s=post_s,
                                      bin_s=bin_s, n_shuffle=n_shuffle, rng=rng)

    def group_from_trials(self, trials, *, pre_s=0.1, post_s=None, bin_s=0.01,
                          n_shuffle=1000, rng=None) -> dict:
        """Pooled ON/OFF from pre-built trial dicts — lets a caller detect each file with
        its OWN threshold (per-trace abs) and then pool the SAME spike trains here."""
        if not trials:
            return dict(n_trials=0, freq=float("nan"), on=None, off=None)
        freq = float(np.median([t["freq"] for t in trials]))
        if post_s is None:
            post_s = 0.8 / freq if freq else 0.4
        rng = rng or np.random.default_rng(0)
        on = flk.shift_test(trials, "on", pre_s=pre_s, post_s=post_s, bin_s=bin_s,
                            n_shuffle=n_shuffle, rng=rng)
        off = flk.shift_test(trials, "off", pre_s=pre_s, post_s=post_s, bin_s=bin_s,
                             n_shuffle=n_shuffle, rng=rng)
        return dict(n_trials=len(trials), freq=freq, on=on, off=off)
