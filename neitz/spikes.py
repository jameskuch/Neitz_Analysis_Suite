"""
neitz.spikes — spike / action-current detection, decoupled from any file format.

Works on any 1-D signal (extracellular voltage OR voltage-clamp current). This is
the detector validated in extract_abf_spikes.py: robust (MAD) or absolute threshold,
selectable polarity, with a refractory period.

    from neitz.spikes import detect_spikes
    st = detect_spikes(im, fs, polarity="neg", k=6.0)   # escaped action currents
    st.times          # spike times (s)
    st.binary(10_000) # 0/1 train at 10 kHz  (siso-spikes.csv style)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks


@dataclass
class SpikeTrain:
    times: np.ndarray          # spike times (s)
    fs: float                  # source sample rate (Hz)
    amplitudes: np.ndarray = None
    polarity: str = None
    threshold: float = None
    sigma: float = None

    def __len__(self):
        return len(self.times)

    def rate(self, duration: float) -> float:
        return len(self.times) / duration if duration else float("nan")

    def binary(self, out_rate: float, n_out: int = None, duration: float = None) -> np.ndarray:
        """0/1 train at out_rate Hz with a 1 at each spike's nearest sample."""
        if n_out is None:
            if duration is None:
                duration = self.times.max() if len(self.times) else 0.0
            n_out = int(round(duration * out_rate))
        train = np.zeros(n_out, dtype=np.int8)
        idx = np.round(self.times * out_rate).astype(int)
        idx = idx[(idx >= 0) & (idx < n_out)]
        train[idx] = 1
        return train


def detect_spikes(signal, fs, *, polarity: str = "neg", method: str = "mad",
                  k: float = 6.0, abs_threshold: float = None,
                  refractory_s: float = 0.002) -> SpikeTrain:
    """
    polarity : 'neg' (downward), 'pos' (upward), or 'abs' (either)
    method   :
      'mad'        -> threshold = k * robust-sigma (noise-adaptive, per recording)
      'abs'        -> threshold = abs_threshold (fixed pA)
      'mad_floor'  -> threshold = max(k * robust-sigma, abs_threshold). k·MAD still
                      adapts to each file's noise, but a spike must ALSO clear the
                      absolute floor — rejects small proximal-cell events while
                      keeping noise-adaptive detection.
    """
    signal = np.asarray(signal, dtype=float)
    med = np.median(signal)
    sigma = np.median(np.abs(signal - med)) * 1.4826

    if polarity == "neg":
        sig = -(signal - med)
    elif polarity == "pos":
        sig = signal - med
    elif polarity == "abs":
        sig = np.abs(signal - med)
    else:
        raise ValueError("polarity must be 'neg', 'pos', or 'abs'")

    if method == "mad":
        height = k * sigma
    elif method == "abs":
        if abs_threshold is None:
            raise ValueError("method='abs' requires abs_threshold")
        height = float(abs_threshold)
    elif method == "mad_floor":
        height = max(k * sigma, float(abs_threshold or 0.0))
    else:
        raise ValueError("method must be 'mad', 'abs', or 'mad_floor'")

    peaks, _ = find_peaks(sig, height=height, distance=max(1, int(fs * refractory_s)))
    return SpikeTrain(times=peaks / fs, fs=fs, amplitudes=sig[peaks],
                      polarity=polarity, threshold=height, sigma=sigma)
