"""
neitz — a clean backbone for Neitz-lab electrophysiology analysis.

Layered, path-explicit, GUI-agnostic. Replaces the monolithic Neitz.py.

    from neitz.io.abf import Recording
    from neitz.spikes import detect_spikes
    from neitz.analysis import revcorr, flicker

Layers:
    io/         load recordings by explicit path (no baked-in data dirs)
    spikes      threshold-based spike / action-current detection
    analysis/   revcorr (STA / linear filter / STRF) and flicker (PSTH / VS) engines
    stimulus/   paradigm abstraction (gaussian noise, flicker, future checkerboard)
"""

__version__ = "0.1.0"
