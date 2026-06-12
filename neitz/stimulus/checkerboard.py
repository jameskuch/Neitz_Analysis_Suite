"""
CheckerboardParadigm — subdivided pseudo-random Gaussian -> spatiotemporal STRF.

The stimulus area is an (n_y x n_x) grid of checks, each running an independent
Gaussian-noise sequence. Reverse-correlating every check with the spike response
gives a spatiotemporal receptive field: a temporal filter per check, plus a
spatial map (peak-lag frame) and the temporal filter at the strongest check.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from ..analysis import strf as strf_engine
from .base import StimulusMeta


@dataclass
class STRFResult:
    strf: np.ndarray          # (n_y, n_x, filter_len)
    spatial_rf: np.ndarray    # (n_y, n_x) peak-lag map
    temporal: np.ndarray      # (filter_len,) filter at the strongest check
    peak_yx: tuple            # (y, x) of the strongest check
    peak_time_ms: float       # lag of the temporal peak
    time_ms: np.ndarray       # (filter_len,) lag axis


@dataclass
class CheckerboardParadigm:
    n_y: int
    n_x: int
    bin_rate: int = 60        # stimulus frame rate (Hz)
    filter_len: int = 30      # STRF taps (0.5 s at 60 Hz)
    zero_pad: int = 60
    normalize: str = "max"    # 'max' | 'std' | None
    meta: StimulusMeta = field(default_factory=lambda: StimulusMeta(kind="checkerboard"))

    def analyze(self, stimulus, response) -> STRFResult:
        """
        stimulus : (n_checks, n_time) OR (n_y, n_x, n_time)
        response : (n_time,) binned spike response
        """
        stim = np.asarray(stimulus, dtype=float)
        if stim.ndim == 3:
            stim = stim.reshape(stim.shape[0] * stim.shape[1], stim.shape[2])

        flat = strf_engine.spatiotemporal_revcorr(stim, response, self.filter_len, self.zero_pad)
        flat = strf_engine.normalize_strf(flat, self.normalize)
        cube = strf_engine.reshape_strf(flat, self.n_y, self.n_x)

        pc = strf_engine.peak_check(flat)
        py, px = divmod(pc, self.n_x)
        temporal = flat[pc]
        lag = int(np.argmax(np.abs(temporal)))
        time_ms = 1000.0 * np.arange(self.filter_len) / self.bin_rate
        return STRFResult(strf=cube, spatial_rf=strf_engine.spatial_rf(cube),
                          temporal=temporal, peak_yx=(py, px),
                          peak_time_ms=float(time_ms[lag]), time_ms=time_ms)
