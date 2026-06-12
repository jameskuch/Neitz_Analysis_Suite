"""
neitz.analysis.revcorr — reverse correlation (spike-triggered average / linear filter).

This is the engine shared by every noise paradigm. The cone-isolation flavour
(S/L/M/LM-iso) does NOT change the math — it is only stimulus metadata. The SAME
engine extends to the subdivided-checkerboard spatiotemporal map: a 1-D stimulus
gives a temporal filter; an (n_checks, time) stimulus gives an STRF, one filter per
check (run reverse_correlation per column).

Matches Sara's MTFanalysis / extract_siso4.py exactly:
    filter = real(ifft( fft(response) * conj(fft(stimulus)) ))[:filter_len]
with the normalization gotcha made explicit (Sara's graphDataOnline divides the
DISPLAY by max(abs); her internal analysis.linearFilter divides by std).
"""

from __future__ import annotations
import numpy as np


def reverse_correlation(stimulus, response, filter_len, zero_pad=60) -> np.ndarray:
    """Unnormalized linear filter for one epoch (one stimulus/response pair)."""
    stimulus = np.asarray(stimulus, dtype=float)
    response = np.asarray(response, dtype=float)
    if len(stimulus) != len(response):
        raise ValueError(f"length mismatch: {len(stimulus)} vs {len(response)}")
    s = np.concatenate([stimulus, np.zeros(zero_pad)])
    r = np.concatenate([response, np.zeros(zero_pad)])
    rsr = np.fft.fft(r) * np.conj(np.fft.fft(s))
    return np.real(np.fft.ifft(rsr))[:filter_len]


def normalize_filter(filt, method="max", ddof=1) -> np.ndarray:
    """method: 'max' (÷max|.|, matches MATLAB plot), 'std' (÷std), or None/'none'."""
    filt = np.asarray(filt, dtype=float)
    if method in (None, "none"):
        return filt
    if method == "max":
        denom = np.max(np.abs(filt))
    elif method == "std":
        denom = np.std(filt, ddof=ddof)
    else:
        raise ValueError("method must be 'max', 'std', or None")
    return filt / denom if denom else filt


def average_filter(stimuli, responses, filter_len, *, zero_pad=60,
                   normalize="max") -> dict:
    """
    Average the per-epoch reverse-correlation filters across epochs (Sara-style),
    then normalize the average. `stimuli`/`responses` are (n_epochs, n) arrays or
    lists of 1-D arrays.

    Returns dict with: per_epoch (n_epochs, filter_len), average (normalized),
    average_raw, freqs, tuning (|fft| of normalized average).
    """
    stimuli = [np.asarray(s, float) for s in stimuli]
    responses = [np.asarray(r, float) for r in responses]
    n = len(stimuli)
    per = np.array([reverse_correlation(stimuli[i], responses[i], filter_len, zero_pad)
                    for i in range(n)])
    avg_raw = per.mean(axis=0)
    avg = normalize_filter(avg_raw, normalize)
    tuning = np.abs(np.fft.rfft(avg))
    # frequency axis assumes the filter spans `filter_len` bins at the response's bin rate;
    # caller passes bin_rate via temporal_tuning if they want real Hz.
    return dict(per_epoch=per, average=avg, average_raw=avg_raw,
                tuning=tuning, n_epochs=n)


def temporal_tuning(filt, bin_rate) -> tuple:
    """Return (freqs_Hz, amplitude) for a filter sampled at bin_rate Hz."""
    filt = np.asarray(filt, dtype=float)
    amp = np.abs(np.fft.rfft(filt))
    freqs = np.fft.rfftfreq(len(filt), d=1.0 / bin_rate)
    return freqs, amp
