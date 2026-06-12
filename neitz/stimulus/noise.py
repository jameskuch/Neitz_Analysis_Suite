"""
NoiseParadigm — Gaussian-noise reverse correlation.

Cone-isolation flavour (S/L/M/LM-iso) is metadata only; the math is identical.
The same engine extends to the subdivided-checkerboard STRF (run per check).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from ..analysis import revcorr
from .base import StimulusMeta


@dataclass
class NoiseParadigm:
    bin_rate: int = 360          # response bin rate (Hz)
    filter_len: int = 360        # samples of the recovered filter (1 s at 360 Hz)
    zero_pad: int = 60           # matches Sara's MTFanalysis
    normalize: str = "max"       # 'max' (matches MATLAB plot) | 'std' | None
    meta: StimulusMeta = field(default_factory=lambda: StimulusMeta(kind="gaussian_noise"))

    def analyze(self, stimuli, responses) -> dict:
        """
        stimuli / responses: per-epoch arrays (n_epochs x n, or list of 1-D).
        Returns dict: per_epoch, average, average_raw, freqs, tuning, time_ms, n_epochs.
        """
        out = revcorr.average_filter(stimuli, responses, self.filter_len,
                                     zero_pad=self.zero_pad, normalize=self.normalize)
        out["freqs"], out["tuning"] = revcorr.temporal_tuning(out["average"], self.bin_rate)
        out["time_ms"] = 1000.0 * np.arange(self.filter_len) / self.bin_rate
        return out
