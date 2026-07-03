"""
neitz.stimulus.reproduce — regenerate the exact Stage/LightCrafter stimulus noise
from its seed, so no per-frame stimulus values need to be stored or shipped.

The June2026 MATLAB stimuli draw Gaussian noise with MATLAB's ``mt19937ar`` stream
and the INVERSE-CDF transform (deliberately NOT ``randn``), specifically so the
sequence is reproducible here:

    MATLAB:  stream = RandStream('mt19937ar', 'Seed', seed);
             noiseVals = mu + sigma .* norminv(rand(stream, n_y, n_x, n_updates));
             noiseVals = min(max(noiseVals, 0), 1);

MATLAB ``rand(mt19937ar, ...)`` is bit-identical to numpy's legacy ``RandomState``,
and MATLAB ``norminv`` == ``scipy.special.ndtri``, so the two reproduce each other
to machine precision (verified ~3e-16 on a 32x40x5 array). MATLAB fills arrays
column-major, hence ``order='F'`` on the reshape.

The returned value is the LINEAR stimulus ``v`` (fraction of maximum light). Because
the projector is gamma-corrected, the light reaching the retina is linear in ``v`` —
so ``v`` is exactly the stimulus to reverse-correlate against. :func:`gamma_adjust`
recovers the (8-bit-bound) DAC value that was actually sent, for reconstructing what
was displayed; it is NOT used in the STA/STRF.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

# LightCrafter 4500 Video-mode display gamma (measured; rig calibration power fit).
GAMMA = 2.2056


def reproduce_noise(seed, n_y, n_x, n_updates, mu=0.5, sigma=0.3):
    """Regenerate the ``(n_y, n_x, n_updates)`` linear noise tensor ``v`` in [0, 1].

    Exactly matches the MATLAB stimulus: ``mt19937ar`` uniforms -> inverse-CDF
    normal -> ``mu + sigma*z`` -> clip to [0, 1], filled column-major.

    Full-field stimuli use ``n_y = n_x = 1``; an M x N checkerboard uses
    ``n_y = checks_y``, ``n_x = checks_x``. ``n_updates`` is the number of distinct
    noise frames (``ceil(stim_frames / update_every_n_frames)``).
    """
    n_y, n_x, n_updates = int(n_y), int(n_x), int(n_updates)
    u = np.random.RandomState(int(seed)).rand(n_y * n_x * n_updates)   # == MATLAB rand(stream, ...)
    v = mu + sigma * ndtri(u)                                          # == MATLAB mu + sigma*norminv(u)
    v = v.reshape((n_y, n_x, n_updates), order="F")                   # MATLAB column-major layout
    return np.clip(v, 0.0, 1.0)


def gamma_adjust(v, gamma=GAMMA):
    """DAC fraction actually sent to the projector: ``v ** (1/gamma)``, in [0, 1].

    The 8-bit code sent is ``round(255 * gamma_adjust(v))``. Use this only to
    reconstruct what was displayed; reverse correlation uses the linear ``v`` from
    :func:`reproduce_noise`.
    """
    v = np.clip(np.asarray(v, dtype=float), 0.0, 1.0)
    return v ** (1.0 / gamma)


def expand_to_frames(noise_updates, update_every_n_frames, stim_frames=None):
    """Upsample a per-update tensor ``(..., n_updates)`` to per-frame
    ``(..., n_frames)`` by holding each update for ``update_every_n_frames`` frames
    (the stimulus is piecewise-constant between updates). If ``stim_frames`` is
    given the result is truncated (or edge-padded) to exactly that many frames.
    """
    frames = np.repeat(np.asarray(noise_updates), int(update_every_n_frames), axis=-1)
    if stim_frames is not None:
        sf = int(stim_frames)
        if frames.shape[-1] >= sf:
            frames = frames[..., :sf]
        else:
            pad = np.repeat(frames[..., -1:], sf - frames.shape[-1], axis=-1)
            frames = np.concatenate([frames, pad], axis=-1)
    return frames
