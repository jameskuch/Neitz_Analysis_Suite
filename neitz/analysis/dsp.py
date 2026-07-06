"""1-D DSP for the analysis pipeline: trace **smoothing** + **temporal filtering**.

Two operations the pipeline can apply to a 1-D signal (e.g. the binned spike rate) before
the FFT / power spectrum:

``smooth_1d``
    Moving-average / gaussian / Savitzky-Golay smoothing. The default ``"moving"`` method
    reproduces MATLAB's ``smooth(y, span)`` — odd span, symmetric windows that shrink toward
    each end — which is the smoothing Sara used on the online traces before the transform
    (``SaraipRGC/iprgc4jim.m``: ``set(gco,'YData', smooth(get(gco,'YData'), 4))``).

``temporal_filter_1d`` / ``apply_temporal_filter``
    Zero-phase FIR low-/high-pass filtering — the **time-domain port** of ``benaqTools_py``'s
    ``spatial_filter.py`` (which designs the same kernels in cycles/pixel for a GPU pass).
    Here the cutoff is given in **Hz** and converted to cycles/sample with the sample rate;
    high-pass uses spectral inversion (``delta - lowpass``), exact for these symmetric FIRs.
    Every reported response is the REALIZED response of the truncated kernel (rfft), not the
    ideal target — what you see is what runs.

Pure numpy (Savitzky-Golay borrows scipy, imported lazily). No GUI deps — tested in
``tests/test_dsp.py``.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------------------------
SMOOTH_METHODS = ("moving", "gaussian", "savgol")


def _moving_average(y: np.ndarray, span: int) -> np.ndarray:
    """MATLAB ``smooth(y, span)`` 'moving' method: odd span, windows shrink at the ends.

    yy[i] = mean(y[i-w : i+w+1]) with w = min((span-1)//2, i, n-1-i). An even span is
    reduced to span-1 (MATLAB forces an odd, symmetric window). O(n) via a prefix sum.
    """
    n = y.size
    span = int(span)
    if span % 2 == 0:                       # MATLAB forces an odd span for 'moving'
        span -= 1
    if span < 3:
        return y.astype(float).copy()
    half = (span - 1) // 2
    csum = np.concatenate(([0.0], np.cumsum(y, dtype=float)))
    out = np.empty(n, dtype=float)
    for i in range(n):
        w = min(half, i, n - 1 - i)
        out[i] = (csum[i + w + 1] - csum[i - w]) / (2 * w + 1)
    return out


def _reflect_convolve(y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve with a symmetric kernel, reflect-padded so the output length == input and
    edge transients are suppressed. The kernel is centered (zero-phase)."""
    r = (len(kernel) - 1) // 2
    if r <= 0:
        return y.astype(float).copy()
    yp = np.pad(y, r, mode="reflect")
    return np.convolve(yp, kernel, mode="valid")


def _gaussian_smooth(y: np.ndarray, span: float) -> np.ndarray:
    """Gaussian-weighted moving average; ``span`` is the full-width at half-maximum (samples)."""
    sigma = max(1e-6, float(span) / 2.354820045)     # FWHM -> sigma
    half = int(max(1, round(3.0 * sigma)))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return _reflect_convolve(y, k)


def smooth_1d(y, window: float = 4, method: str = "moving", polyorder: int = 2) -> np.ndarray:
    """Smooth a 1-D signal.

    Parameters
    ----------
    y : array_like
        The 1-D signal.
    window : number
        Span in **samples**. ``moving`` uses an odd span (MATLAB rule); ``gaussian`` treats
        it as the FWHM; ``savgol`` as the (odd) frame length. ``< 2`` is a no-op.
    method : {"moving", "gaussian", "savgol"}
        ``moving`` = MATLAB ``smooth``; ``gaussian`` = gaussian-weighted; ``savgol`` =
        Savitzky-Golay (polynomial ``polyorder``, preserves peaks/widths better).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("smooth_1d expects a 1-D array")
    if y.size < 3 or window is None or float(window) < 2:
        return y.copy()
    method = str(method).lower()
    if method in ("moving", "movmean", "mean"):
        return _moving_average(y, window)
    if method == "gaussian":
        return _gaussian_smooth(y, window)
    if method in ("savgol", "sgolay"):
        from scipy.signal import savgol_filter
        w = int(window)
        if w % 2 == 0:
            w += 1
        w = min(w, y.size if y.size % 2 == 1 else y.size - 1)
        if w < 3:
            return y.copy()
        po = min(int(polyorder), w - 1)
        return savgol_filter(y, w, po)
    raise ValueError(f"unknown smoothing method {method!r}")


# ---------------------------------------------------------------------------------------------
# Temporal filtering — 1-D zero-phase FIR (time-domain port of spatial_filter.py)
# ---------------------------------------------------------------------------------------------
FILTER_TYPES = ("gaussian", "butterworth", "ideal", "hamming", "hanning", "blackman")
FILTER_MODES = ("lowpass", "highpass")
MAX_TAPS = 129         # time kernels can be longer than the GPU spatial cap (33)
_NFFT = 2048           # dense grid for butterworth sampling / realized responses


def _odd_taps(taps: int) -> int:
    t = int(max(3, min(MAX_TAPS, taps)))
    return t if t % 2 == 1 else t + 1


def _windowed_sinc(fc: float, taps: int, window: str) -> np.ndarray:
    n = np.arange(taps) - (taps - 1) / 2.0
    k = 2.0 * fc * np.sinc(2.0 * fc * n)
    if window == "hamming":
        k *= np.hamming(taps)
    elif window == "hanning":
        k *= np.hanning(taps)
    elif window == "blackman":
        k *= np.blackman(taps)
    # "ideal" = rectangular window (no taper)
    return k


@lru_cache(maxsize=256)
def design_lowpass_1d(ftype: str, fc: float, order: int, taps: int) -> tuple:
    """1-D zero-phase low-pass kernel, DC gain exactly 1.

    ``fc`` is in cycles/sample (0 < fc <= 0.5 Nyquist); ``taps`` forced odd. Returned as a
    tuple so ``lru_cache`` can hold it.
    """
    taps = _odd_taps(taps)
    fc = float(min(0.5, max(1e-4, fc)))
    r = (taps - 1) // 2

    if ftype == "gaussian":
        # half-power at fc: exp(-2 pi^2 s^2 fc^2) = 1/sqrt(2)
        sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * fc)
        n = np.arange(-r, r + 1, dtype=np.float64)
        k = np.exp(-0.5 * (n / max(sigma, 1e-6)) ** 2)
    elif ftype == "butterworth":
        f = np.linspace(0.0, 0.5, _NFFT // 2 + 1)
        H = 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * max(1, int(order))))
        h = np.fft.irfft(H, n=_NFFT)
        h = np.roll(h, _NFFT // 2)               # center the zero-phase kernel
        k = h[_NFFT // 2 - r:_NFFT // 2 + r + 1]
        k = k * np.hanning(taps)                 # taper truncation ripple
    elif ftype in ("ideal", "hamming", "hanning", "blackman"):
        k = _windowed_sinc(fc, taps, ftype)
    else:
        raise ValueError(f"unknown filter type {ftype!r}")

    s = k.sum()
    if abs(s) < 1e-12:
        k = np.zeros(taps)
        k[r] = 1.0
    else:
        k = k / s
    return tuple(float(v) for v in k)


def temporal_kernel(ftype: str, mode: str, fc_cps: float, order: int, taps: int) -> np.ndarray:
    """Final 1-D kernel for the requested mode, ``fc_cps`` in cycles/sample."""
    k = np.array(design_lowpass_1d(str(ftype), float(fc_cps), int(order), int(taps)),
                 dtype=np.float64)
    if mode == "highpass":
        hp = -k
        hp[(len(k) - 1) // 2] += 1.0             # delta - lowpass (spectral inversion)
        return hp
    return k


def realized_response(kernel_1d: np.ndarray, npts: int = 129):
    """(freqs cyc/sample, |H(f)|) of the truncated kernel — the ACTUAL response."""
    H = np.abs(np.fft.rfft(kernel_1d, n=_NFFT))[: _NFFT // 2 + 1]
    f = np.linspace(0.0, 0.5, _NFFT // 2 + 1)
    idx = np.linspace(0, len(f) - 1, npts).astype(int)
    return f[idx], H[idx]


def apply_temporal_filter(y, *, ftype: str = "butterworth", mode: str = "lowpass",
                          cutoff_hz: float, fs: float, order: int = 2,
                          taps: int = 33) -> np.ndarray:
    """Zero-phase FIR filter a 1-D signal sampled at ``fs`` Hz. ``cutoff_hz`` is the -3 dB
    (or brick-wall) frequency; it is converted to cycles/sample = cutoff_hz / fs."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("apply_temporal_filter expects a 1-D array")
    if not (fs and fs > 0):
        return y.copy()
    fc = float(cutoff_hz) / float(fs)            # Hz -> cycles/sample
    fc = min(0.5, max(1e-4, fc))
    k = temporal_kernel(ftype, mode, fc, order, _odd_taps(taps))
    if y.size <= len(k):                         # too short to filter meaningfully
        return y.copy()
    return _reflect_convolve(y, k)


def temporal_filter_design(ftype: str, mode: str, cutoff_hz: float, fs: float,
                           order: int, taps: int) -> dict:
    """Full design bundle for UI plots (frequencies in **Hz**). ``fs`` sets the Nyquist."""
    taps = _odd_taps(int(taps))
    fs = float(fs) if fs and fs > 0 else 1.0
    fc_cps = min(0.5, max(1e-4, float(cutoff_hz) / fs))
    k = temporal_kernel(ftype, mode, fc_cps, order, taps)
    f_cps, H = realized_response(k)
    return {
        "kernel": [round(float(v), 6) for v in k],
        "taps": taps,
        "freq_hz": [round(float(v * fs), 4) for v in f_cps],
        "response": [round(float(v), 5) for v in H],
        "cutoff_hz": round(float(fc_cps * fs), 4),
        "nyquist_hz": round(float(0.5 * fs), 3),
        "fc_cps": round(float(fc_cps), 6),
    }
