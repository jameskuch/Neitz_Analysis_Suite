"""
neitz.stimulus — paradigm abstraction.

A Paradigm bundles (a) how to obtain the stimulus for a recording and (b) which
analysis to run. The point is that adding a new stimulus (a new cone isolation, a
new flicker rate, the subdivided checkerboard) is a new Paradigm, not a new copy of
the loading/detection/plotting plumbing.

Two paradigms exist conceptually today:

  * NoiseParadigm        — Gaussian noise -> reverse correlation (neitz.analysis.revcorr).
                           Cone isolation (S/L/M/LM-iso) is METADATA; the math is identical.
                           Stimulus values come from a CSV / MATLAB seed (exact sequence needed).
  * FlickerParadigm      — square-wave flicker -> cycle / transition-triggered PSTH
                           (neitz.analysis.flicker). Stimulus timing is read from the TTL,
                           so no external stimulus file is required.

  * (future) CheckerboardParadigm — subdivided pseudo-random Gaussian per check ->
                           spatiotemporal STRF. Same reverse-correlation engine, run
                           per check; stimulus is (n_checks, time).

This module defines the lightweight interface; concrete paradigm classes are wired
up in the next milestone alongside the Dash GUI.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class StimulusMeta:
    """What was shown — independent of the analysis math."""
    kind: str                       # 'gaussian_noise' | 'flicker' | 'checkerboard'
    cone_isolation: str = None      # 'S' | 'L' | 'M' | 'LM' | 'achromatic' | None
    frame_rate: float = 60.0
    extras: dict = field(default_factory=dict)


class Paradigm(Protocol):
    """A stimulus+analysis pairing. Concrete classes implement `analyze`."""
    meta: StimulusMeta

    def analyze(self, recording, **kwargs) -> dict:
        ...
