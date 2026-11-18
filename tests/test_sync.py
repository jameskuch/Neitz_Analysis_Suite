"""neitz.analysis.sync — stimulus timing from the sync-bar photodiode (2026-11-12 rig convention).

Synthetic signals mimic what Clampex recorded on 2026-11-12: the diode is INVERTED (dark ≈ 4.9 V,
light ≈ 0.1 V), a lit frame is a burst of short bit-plane pulses, the middle segment is lit for one
noise update and dark for the next, and the whole bar is dark before/after the stimulus.
"""
import numpy as np
import pytest
from neitz.analysis.sync import stimulus_window, update_clock
from neitz.run import epoch_response_edges

FS = 20000.0
REFRESH = 59.9435
T_FRAME = 1.0 / REFRESH
DARK, LIGHT = 4.9, 0.1


def middle_segment(n_updates, upd_frames=1, *, t_on=2.0, post_s=1.5, dup_update=None,
                   lit_every=False, rng_noise=0.01, filled_dark=(), split_lit=()):
    """Inverted photodiode trace of the MIDDLE sync segment. Returns (signal, true_edges) where
    true_edges[k] is the start of update k (length n_updates + 1). `dup_update` holds that lit
    update one frame longer (a duplicated frame); `lit_every` lights every update (an envelope
    segment, which must NOT be mistaken for the update clock)."""
    t_upd = upd_frames * T_FRAME
    edges = [t_on]
    for k in range(n_updates):
        extra = T_FRAME if (dup_update is not None and k == dup_update) else 0.0
        edges.append(edges[-1] + t_upd + extra)
    edges = np.asarray(edges)
    n = int(round((edges[-1] + post_s) * FS))
    x = np.full(n, DARK) + np.random.RandomState(0).randn(n) * rng_noise
    for k in range(n_updates):
        if not lit_every and k % 2 == 1 and k not in filled_dark:
            continue
        n_frames = int(round((edges[k + 1] - edges[k]) / T_FRAME))
        for f in range(n_frames):
            f0 = edges[k] + f * T_FRAME
            if k in split_lit and f == n_frames // 2:           # one frame with sparse pulses
                offs = (0.0005,)
            else:
                offs = (0.0005, 0.0045, 0.0085, 0.0125)          # 4 bit-plane pulses, 2.5 ms each
            for off in offs:
                a, b = int((f0 + off) * FS), int((f0 + off + 0.0025) * FS)
                x[a:b] = LIGHT
    return x, edges


def test_stimulus_window_finds_envelope():
    x, edges = middle_segment(40, 1)
    t_on, t_off = stimulus_window(x, FS)
    assert abs(t_on - edges[0]) < 0.002
    assert abs(t_off - (edges[38] + 0.015)) < 0.005            # last lit update (k=38) ends its last pulse ~15 ms in


def test_stimulus_window_flat_channel_is_none():
    x = np.full(int(5 * FS), DARK) + np.random.RandomState(1).randn(int(5 * FS)) * 0.01
    assert stimulus_window(x, FS) is None


@pytest.mark.parametrize("upd_frames,n_updates", [(1, 61), (7, 24)])
def test_update_clock_recovers_update_edges(upd_frames, n_updates):
    x, true = middle_segment(n_updates, upd_frames)
    edges, info = update_clock(x, FS, n_updates, upd_frames, REFRESH)
    assert edges is not None and len(edges) == n_updates + 1
    assert info["n_bursts"] == int(np.ceil(n_updates / 2)) and info["frame_dups"] == 0
    assert np.max(np.abs(edges[:-1] - true[:-1])) < 0.002        # every update start within 2 ms
    assert not info["truncated"]


def test_update_clock_tracks_a_duplicated_frame():
    """A lit update held one frame longer shows as a longer burst: the dark update after it (and
    everything that follows) shifts by one frame, and the duplication is counted."""
    n = 41
    x, true = middle_segment(n, 1, dup_update=10)
    edges, info = update_clock(x, FS, n, 1, REFRESH)
    assert edges is not None and len(edges) == n + 1
    assert info["frame_dups"] == 1
    assert np.max(np.abs(edges[:-1] - true[:-1])) < 0.002
    assert abs((edges[11] - edges[10]) - 2 * T_FRAME) < 0.002    # update 10 lasted two frames


def test_update_clock_truncated_recording():
    """The sweep stops before the stimulus ends: the measured grid covers only the recorded updates."""
    n = 60
    x, true = middle_segment(n, 1)
    cut = int((true[30] + 0.004) * FS)                           # stop inside update 30
    edges, info = update_clock(x[:cut], FS, n, 1, REFRESH)
    assert edges is not None and info["truncated"]
    assert 25 <= len(edges) - 1 <= 30
    assert np.max(np.abs(edges[:-1] - true[:len(edges) - 1])) < 0.002


def test_update_clock_rejects_envelope_and_flat_channels():
    n = 40
    env, _ = middle_segment(n, 1, lit_every=True)                # envelope segment: lit every update
    assert update_clock(env, FS, n, 1, REFRESH)[0] is None
    flat = np.full(int(4 * FS), DARK)
    assert update_clock(flat, FS, n, 1, REFRESH)[0] is None
    mid, _ = middle_segment(n, 1)
    assert update_clock(mid, FS, 4 * n, 1, REFRESH)[0] is None    # wrong stimulus for this trace


def test_epoch_response_edges_nonuniform_bins():
    edges = np.array([1.0, 1.1, 1.3, 1.4])                      # update widths 0.1, 0.2, 0.1
    spikes = np.array([1.05, 1.15, 1.25, 1.35, 1.35, 9.0])
    r = epoch_response_edges(spikes, edges, bins_per_update=1)
    assert np.allclose(r, [1 / 0.1, 2 / 0.2, 2 / 0.1])
    r2 = epoch_response_edges(spikes, edges, bins_per_update=2)
    assert r2.shape == (6,) and np.allclose(r2, [0, 1 / 0.05, 1 / 0.1, 1 / 0.1, 0, 2 / 0.05])   # 1.05 -> [1.05,1.1)


def test_load_seed_epochs_prefers_measured_clock(monkeypatch, tmp_path):
    """With a middle-segment channel present the loader bins the response on the MEASURED update
    grid (clock='measured:<channel>'); the envelope-only TTL falls back to the uniform grid."""
    from neitz import run
    from neitz.dataio import DataStore
    n_updates = 61
    mid, true = middle_segment(n_updates, 1)
    env, _ = middle_segment(n_updates, 1, lit_every=True)
    sig = np.random.RandomState(0).randn(len(mid)) * 0.5
    for s in np.arange(2.0, true[-1], 0.03):
        sig[int(s * FS)] = -50.0

    class FakeRec:
        fs = FS
        channel_names = ["Im_prime", "TTL", "I_MTest 3"]
        def channel(self, name):
            n = str(name).lower()
            return sig if "im_" in n else env if n == "ttl" else mid
    monkeypatch.setattr(run.Recording, "load", staticmethod(lambda p: FakeRec()))
    cm = DataStore(root=tmp_path).cell("2026-11-12", "c01")
    cm.dir.mkdir(parents=True, exist_ok=True)
    cm.data["recordings"] = [{"id": "r1", "file": "raw/x.abf", "stimulus": {
        "type": "checkerboard", "params": {"seed": 5, "checks_y": 2, "checks_x": 2, "n_updates": n_updates,
                                           "check_seeds": [[1, 2], [3, 4]], "seed_scheme": "gaussCheckSeeds/1",
                                           "update_every_n_frames": 1, "refresh_rate_hz": REFRESH,
                                           "mu": 0.5, "sigma": 0.3}}}]
    records, responses, info = run.load_seed_epochs(cm, chan="Im_prime", ttl="TTL", bins_per_update=1,
                                                    want=("checkerboard",))
    assert len(records) == 1 and info[0]["clock"] == "measured:I_MTest 3"
    assert responses[0].shape == (n_updates,) and abs(info[0]["t0"] - true[0]) < 0.002
    assert info[0]["n_updates_recorded"] == n_updates and "recorded_updates" not in records[0]

    # envelope-only rig: uniform grid from the envelope onset
    class FakeRec2(FakeRec):
        channel_names = ["Im_prime", "TTL"]
        def channel(self, name):
            return sig if "im_" in str(name).lower() else env
    monkeypatch.setattr(run.Recording, "load", staticmethod(lambda p: FakeRec2()))
    records, responses, info = run.load_seed_epochs(cm, chan="Im_prime", ttl="TTL", bins_per_update=1,
                                                    want=("checkerboard",))
    assert info[0]["clock"] == "uniform" and abs(info[0]["t0"] - true[0]) < 0.002
    assert responses[0].shape == (n_updates,)


def test_update_clock_interpolates_filled_dark_phases():
    """2026-11-12 gaussian epochs: a bright full-field update scatters light onto the diode, so a
    dark phase reads as lit and two (or more) lit phases merge into one burst. The square wave's
    period is known, so the hidden lit onsets are interpolated and the grid stays complete."""
    n, upd = 48, 7
    x, true = middle_segment(n, upd, filled_dark=(5, 13, 15))      # dark updates 5, 13 and 15 lit by scatter
    edges, info = update_clock(x, FS, n, upd, REFRESH)
    assert edges is not None and len(edges) == n + 1
    assert info["n_bursts"] == 24 - 3 and info["n_lit"] == 24 and info["n_interpolated"] == 3
    assert np.max(np.abs(edges[:-1] - true[:-1])) < 0.004          # interpolated onsets stay within a frame/4
    assert info["frame_dups"] == 0                                   # merged bursts are NOT counted as dups


def test_update_clock_drops_split_lit_fragments():
    """A frame with sparse pulses inside a lit phase splits the burst in two; the fragment that
    starts less than half a period after the phase onset is folded back into it."""
    n, upd = 40, 7
    x, true = middle_segment(n, upd, split_lit=(4, 12))
    edges, info = update_clock(x, FS, n, upd, REFRESH)
    assert edges is not None and len(edges) == n + 1 and info["n_lit"] == 20
    assert np.max(np.abs(edges[:-1] - true[:-1])) < 0.002
