"""
neitz.analysis.strf — spatiotemporal reverse correlation (checkerboard STRF).

The subdivided-checkerboard paradigm runs an independent pseudo-random Gaussian
sequence in each spatial check. This is the SAME reverse correlation as the 1-D
temporal filter (neitz.analysis.revcorr), just run per check — vectorised here with
one FFT across all checks.

    strf = spatiotemporal_revcorr(stim, response, filter_len)   # stim (n_checks, n_time)
    cube = reshape_strf(strf, n_y, n_x)                         # (n_y, n_x, filter_len)
    yx   = peak_check(strf)                                      # strongest check (flat index)
"""
from __future__ import annotations
import numpy as np


def spatiotemporal_revcorr(stimulus, response, filter_len, zero_pad=60) -> np.ndarray:
    """
    stimulus : (n_checks, n_time)   per-check stimulus sequences
    response : (n_time,)            binned spike response
    Returns the per-check linear filter, shape (n_checks, filter_len).
    """
    stimulus = np.asarray(stimulus, dtype=float)
    response = np.asarray(response, dtype=float)
    if stimulus.ndim != 2:
        raise ValueError("stimulus must be (n_checks, n_time)")
    n_checks, n_time = stimulus.shape
    if response.shape[0] != n_time:
        raise ValueError(f"length mismatch: stim {n_time} vs response {response.shape[0]}")

    s = np.concatenate([stimulus, np.zeros((n_checks, zero_pad))], axis=1)
    r = np.concatenate([response, np.zeros(zero_pad)])
    cross = np.fft.fft(r)[None, :] * np.conj(np.fft.fft(s, axis=1))
    return np.real(np.fft.ifft(cross, axis=1))[:, :filter_len]


def reshape_strf(strf_flat, n_y, n_x) -> np.ndarray:
    """(n_checks, filter_len) -> (n_y, n_x, filter_len)."""
    strf_flat = np.asarray(strf_flat)
    n_checks, flen = strf_flat.shape
    if n_y * n_x != n_checks:
        raise ValueError(f"{n_y}x{n_x} != {n_checks} checks")
    return strf_flat.reshape(n_y, n_x, flen)


def peak_check(strf_flat) -> int:
    """Flat index of the check with the largest |filter| peak."""
    strf_flat = np.asarray(strf_flat)
    return int(np.argmax(np.max(np.abs(strf_flat), axis=1)))


def spatial_rf(strf, method="peakframe") -> np.ndarray:
    """
    Spatial receptive-field map. `strf` is (n_checks, flen) or (n_y, n_x, flen).
    'peakframe' = the time slice at the global peak lag; 'std' = std over time.
    """
    strf = np.asarray(strf)
    flat = strf.reshape(-1, strf.shape[-1])
    if method == "std":
        m = flat.std(axis=1)
    else:
        pc = peak_check(flat)
        lag = int(np.argmax(np.abs(flat[pc])))
        m = flat[:, lag]
    return m.reshape(strf.shape[0], strf.shape[1]) if strf.ndim == 3 else m


def normalize_strf(strf, method="max") -> np.ndarray:
    strf = np.asarray(strf, dtype=float)
    if method in (None, "none"):
        return strf
    if method == "max":
        d = np.max(np.abs(strf))
    elif method == "std":
        d = np.std(strf)
    else:
        raise ValueError("method must be 'max', 'std', or None")
    return strf / d if d else strf
