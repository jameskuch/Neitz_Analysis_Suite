"""
neitz.analysis.sync — stimulus timing from a photodiode on the Stage sync bar.

Rig convention (June2026 Stage / LightCrafter 4500, bar spec ``triple/4``, 2026-09-16): the
rightmost W/8 of the canvas carries three photodiode segments. Top and bottom are the STIMULUS
ENVELOPE (solid blue exactly while stimulus frames are on screen, dark in pre-/post-stim, ITI
and held screens). The middle is the BASE FREQUENCY: blue for one noise update, dark for the
next — one toggle per update, i.e. a square wave at ``flicker_hz``. A DLP renders a solid
colour as a burst of bit-plane exposures per frame, so a photodiode on a lit segment sees a
few short pulses per frame and sits at its idle level on a dark one.

Measured on 2026-11-12 (Clampex ``TTL`` + ``I_MTest 3``, both INVERTED: dark ≈ +4.9 V, light
≈ 0): a 16x10 checkerboard updating every frame gives exactly 600 middle-segment bursts for
1199 updates with every period 2 frames; a 300 s full-field gaussian (update every 7 frames)
shows ~80 half-cycles one frame too long — duplicated frames the client-side ``frame_sync``
never reported. So the update grid should come from the diode, not from ``t0 + k / refresh``.
Everything here is polarity-agnostic: "idle" is whatever level the first half-second sits at.

    t_on, t_off = stimulus_window(sig, fs)                       # envelope: first/last light
    edges, info = update_clock(sig, fs, n_updates, update_every_n_frames, refresh_hz)
"""
from __future__ import annotations
import numpy as np


def _light_mask(sig, fs, *, idle_s=0.5, min_swing_sigma=10.0):
    """Boolean "segment is lit" mask (polarity-agnostic), or ``None`` for a flat channel.

    The idle (dark) level is the median of the first `idle_s` seconds — the recording always
    starts inside the pre-stimulus phase, where the whole bar is dark. The channel counts as
    active only if its 1st–99th percentile swing clears `min_swing_sigma` robust sigmas of the
    idle noise, so a flat / unconnected channel returns ``None`` instead of noise crossings.
    """
    x = np.asarray(sig, dtype=float)
    if x.size < 10:
        return None
    n_idle = max(10, min(x.size, int(idle_s * fs)))
    idle_seg = x[:n_idle]
    idle = float(np.median(idle_seg))
    sigma = 1.4826 * float(np.median(np.abs(idle_seg - idle)))
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if (hi - lo) <= min_swing_sigma * (sigma + 1e-12):
        return None
    thr = 0.5 * (lo + hi)
    return (x < thr) if idle > thr else (x > thr)


def stimulus_window(sig, fs, **kw):
    """``(t_on, t_off)`` in seconds: the first and last moment the photodiode sees light, i.e.
    the stimulus envelope of a lit sync-bar segment. ``None`` if the channel is flat."""
    light = _light_mask(sig, fs, **kw)
    if light is None:
        return None
    idx = np.flatnonzero(light)
    if idx.size == 0:
        return None
    return float(idx[0]) / float(fs), float(idx[-1] + 1) / float(fs)


def _bursts(light, fs, gap_s):
    """Group lit samples into bursts separated by dark gaps longer than `gap_s`.
    Returns ``(starts, ends)`` in seconds (end = first dark sample after the burst)."""
    d = np.diff(light.astype(np.int8))
    rises = np.flatnonzero(d == 1) + 1
    falls = np.flatnonzero(d == -1) + 1
    if light[0]:
        rises = np.concatenate([[0], rises])
    if light[-1]:
        falls = np.concatenate([falls, [light.size]])
    if rises.size == 0:
        return np.array([]), np.array([])
    gap = int(round(gap_s * fs))
    keep = np.concatenate([[True], (rises[1:] - falls[:-1]) > gap])
    starts = rises[keep]
    ends = np.concatenate([falls[:-1][keep[1:]], [falls[-1]]])
    return starts / float(fs), ends / float(fs)


def update_clock(sig, fs, n_updates, update_every_n_frames=1, refresh_hz=60.0, *,
                 gap_s=None, period_tol=0.05, max_interp_frac=0.35):
    """Per-update timing from the MIDDLE sync segment (blue one update, dark the next).

    Returns ``(edges, info)``: `edges` are the update start times in seconds, length
    ``n_recorded + 1`` (the last entry closes the final update), so update ``k`` spans
    ``edges[k]..edges[k+1]``; `info` reports ``n_bursts`` (lit phases seen), ``n_lit`` (after
    reconstruction), ``expected`` (= ceil(n_updates/2)), ``n_interpolated`` (lit onsets that had
    to be filled in), ``truncated`` (the recording ended before the stimulus did) and
    ``frame_dups`` (lit phases held one frame longer = duplicated frames). ``(None, info)`` when
    the channel does not look like the middle segment for this stimulus: flat, a burst period
    that is not ~2 updates, the wrong number of lit phases, or too many to interpolate.

    The lit onsets are measured directly as burst onsets. They are then snapped to the square
    wave's period (``2 x update_every_n_frames`` frames): a burst that starts less than half a
    period after the previous one is a fragment of the same lit phase (a frame with sparse
    bit-plane pulses split it) and is dropped; a gap of ``k > 1`` periods means ``k - 1`` lit
    phases were not seen as separate bursts (a bright full-field update scatters enough light
    onto the diode to fill the dark phase between them — seen on 2026-11-12's gaussian epochs)
    and their onsets are interpolated. The dark update after each lit one starts when the lit one
    ends: ``update_every_n_frames`` frames after its onset, or one or two frames later when the
    burst is measurably that much longer (a duplicated frame); a burst that is longer still was
    merged with a filled dark phase, so the nominal length is used.
    """
    light = _light_mask(sig, fs)
    n_updates = int(n_updates)
    upd_frames = int(update_every_n_frames)
    expected = int(np.ceil(n_updates / 2.0))
    info = dict(n_bursts=0, n_lit=0, expected=expected, n_interpolated=0, truncated=False,
                frame_dups=0, period_s=None)
    if light is None:
        return None, info
    t_frame = 1.0 / float(refresh_hz)
    t_upd = upd_frames * t_frame
    period = 2.0 * t_upd
    if gap_s is None:
        gap_s = max(0.010, 0.5 * t_upd)
    starts, ends = _bursts(light, fs, gap_s)
    n = int(starts.size)
    info["n_bursts"] = n
    if n < 2:
        return None, info
    # the square wave's period must show up as the most common onset spacing
    d_all = np.diff(starts)
    near = d_all[np.abs(d_all - period) < period_tol * period]
    if near.size < max(2, 0.5 * min(n - 1, expected - 1)):
        return None, info
    info["period_s"] = float(np.median(near))
    # -- snap the lit onsets to the period: drop fragments, interpolate missing phases
    widths = ends - starts
    lit, lit_w, n_interp = [float(starts[0])], [float(widths[0])], 0
    for s_i, w_i in zip(starts[1:], widths[1:]):
        d = s_i - lit[-1]
        # whole periods elapsed: a lit phase can only run LONG (a duplicated frame), never short,
        # so round down with a quarter-period of slack rather than to nearest (one-frame updates
        # put a duplicated frame at exactly 1.5 periods, which must still count as ONE)
        k = int(np.floor(d / period + 0.25))
        if k == 0:                                       # fragment of the current lit phase
            lit_w[-1] = max(lit_w[-1], float(s_i + w_i - lit[-1]))
            continue
        prev = lit[-1]
        if d / period - k < -0.15:                       # arrived EARLY: the dark phase before it was
            s_i = prev + k * period                      # partly filled by scatter -> snap to the grid
            w_i = max(float(w_i) - (s_i - float(prev + d)), t_upd)
        for j in range(1, k):                            # lit phases hidden by a filled dark phase
            lit.append(prev + (s_i - prev) * j / k)
            lit_w.append(t_upd)
            n_interp += 1
        lit.append(float(s_i))
        lit_w.append(float(w_i))
    # a long FINAL burst (dark phases filled by scatter right up to the end of the stimulus) hides
    # its later lit onsets too: lay them out at the period inside it
    n_hidden = int(np.floor((lit_w[-1] - t_upd) / period + 0.25))
    n_hidden = min(n_hidden, max(0, expected - len(lit)) + 1)   # never past the stimulus' own count
    for j in range(1, n_hidden + 1):
        lit.append(lit[-1] + period)
        lit_w.append(t_upd)
        n_interp += 1
    lit = np.asarray(lit)
    lit_w = np.asarray(lit_w)
    n_lit = int(lit.size)
    info["n_lit"] = n_lit
    info["n_interpolated"] = n_interp
    if n_lit > 1 and n_interp > max_interp_frac * n_lit:
        return None, info
    n_samples = light.size
    truncated = n_lit < expected and (n_samples / float(fs) - ends[-1]) < 2.0 * t_upd
    info["truncated"] = bool(truncated)
    if n_lit > expected + 1 or (n_lit < expected - 1 and not truncated):
        return None, info
    # -- lit length in frames. The onset-to-onset interval is the truth for the whole cycle
    # (2 x upd_frames, plus any duplicated frames); the burst width says how many of the extra
    # frames fell in the lit half. Interpolated / merged cycles have no extra by construction.
    tail = float(np.median(lit_w - np.floor(lit_w / t_frame) * t_frame))     # pulse span inside a frame
    k_width = np.round((lit_w - tail) / t_frame).astype(int) + 1
    cycle = np.round(np.diff(lit) / t_frame).astype(int)                     # frames per cycle, measured
    extra = np.concatenate([np.maximum(0, cycle - 2 * upd_frames), [2]])     # last cycle: trust the width
    m_lit = np.clip(k_width - upd_frames, 0, extra)
    k_lit = upd_frames + m_lit
    info["frame_dups"] = int(np.sum(m_lit))
    dark = lit + k_lit * t_frame
    edges = np.empty(2 * n_lit + 1)
    edges[0:2 * n_lit:2] = lit
    edges[1:2 * n_lit:2] = dark
    edges[-1] = dark[-1] + t_upd                                             # closes the last dark update
    if truncated:                                        # the last lit update may be cut short
        n_rec = min(n_updates, max(0, 2 * n_lit - 2))
    else:
        n_rec = min(n_updates, 2 * n_lit)
    return edges[:n_rec + 1], info
