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


# --- Sara's MATLAB detector (spikeDetectorOnline.m, MHT/AIW), ported faithfully -----
HIGHPASS_SPIKES_HZ = 500.0     # spikeDetectorOnline: high-pass to keep only spikes
NOISE_SIGMA_FACTOR = 4.0       # AIW noise gate: mean spike must clear mean+4σ of non-spikes


def _highpass_fft(x, fs, cut_hz):
    """FFT high-pass — port of highPassFilter.m: zero the lowest `keep` FFT bins and
    their conjugate mirror, keep the rest. keep = round(cut / (fs/L))."""
    x = np.asarray(x, dtype=float)
    L = x.size
    keep = int(round(cut_hz * L / fs))
    F = np.fft.fft(x)
    if keep > 0:
        F[:keep] = 0
        F[-(keep + 1):] = 0           # matches MATLAB's end-FreqKeepPts:end
    return np.real(np.fft.ifft(F))


def _local_maxima(x):
    """Indices of local maxima — port of getPeaks(x, +1): find(diff(diff(x)>0)<0)+1."""
    rising = (np.diff(x) > 0).astype(np.int8)
    return np.where(np.diff(rising) < 0)[0] + 1


def detect_spikes_matlab(signal, fs, *, polarity: str = "neg", refractory_s: float = 0.002,
                         highpass_hz: float = HIGHPASS_SPIKES_HZ,
                         thresh: float = None) -> SpikeTrain:
    """Port of Sara's spikeDetectorOnline.m. 500 Hz FFT high-pass → remove baseline (median)
    → orient by `polarity` ('neg' flips, 'pos' as-is, 'abs' rectifies — the original MATLAB
    auto-flips, but here the GUI's neg/pos/abs drives it) → threshold = `thresh` if given, else
    1/3 of the max deflection → positive local maxima above threshold → AIW noise gate (mean
    spike must be >= mean(non-spike) + 4·std(non-spike), else NO spikes)."""
    trace = _highpass_fft(signal, fs, highpass_hz)
    trace = trace - np.median(trace)
    if polarity == "neg":
        trace = -trace                                # downward action currents → positive
    elif polarity == "abs":
        trace = np.abs(trace)                         # either direction
    # 'pos' → use the trace as-is
    if thresh is None:
        thresh = np.max(trace) / 3.0                  # the technique's built-in default
    ind = _local_maxima(trace)
    ind = ind[trace[ind] > 0]                          # positive deflections only
    amps = trace[ind]
    keep = amps > thresh
    peak_times, peaks = ind[keep], amps[keep]
    if peaks.size:                                     # AIW: reject if not clearly above noise
        mask = np.ones(trace.size, dtype=bool)
        mask[peak_times] = False
        nonspike = trace[mask]
        if np.mean(peaks) < np.mean(nonspike) + NOISE_SIGMA_FACTOR * np.std(nonspike, ddof=1):
            peak_times = np.array([], dtype=int)
            peaks = np.array([], dtype=float)
    return SpikeTrain(times=peak_times / fs, fs=fs, amplitudes=peaks,
                      polarity=polarity, threshold=float(thresh), sigma=None)


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
      'matlab'     -> Sara's spikeDetectorOnline.m (500 Hz high-pass, 4σ noise gate). Respects
                      polarity (neg/pos/abs); threshold = abs_threshold if given, else max/3.
                      Ignores k.
    """
    signal = np.asarray(signal, dtype=float)
    if method == "matlab":
        return detect_spikes_matlab(
            signal, fs, polarity=polarity, refractory_s=refractory_s,
            thresh=(float(abs_threshold) if abs_threshold is not None else None))
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
