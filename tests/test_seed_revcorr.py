"""Seed-based reverse correlation (June2026 Stage rig) — synthetic recovery tests.

No real .abf / TTL needed: we regenerate the stimulus from a seed (the pinned, MATLAB-exact
`noise_from_record`), synthesize a LINEAR response through a known filter, and assert the STA/STRF
recovers that filter. This validates the reverse-correlation-against-`v` math end to end; the only
piece these don't exercise is the real-abf TTL→epoch-start binding (validated when a seeded-noise
cell is imported).
"""
import numpy as np
from neitz.io.stim import noise_from_record
from neitz.run import (sta_from_records, strf_from_records, record_from_stimulus, epoch_response,
                       analysis_for_stim_type)


def test_analysis_dispatch():
    assert analysis_for_stim_type("gaussian_noise") == "sta"
    assert analysis_for_stim_type("checkerboard") == "strf"
    assert analysis_for_stim_type("sq_wave") == "flicker"
    assert analysis_for_stim_type("flicker") == "flicker"        # legacy value
    assert analysis_for_stim_type(None) == "flicker"             # unknown → TTL-only path
    assert analysis_for_stim_type("jitter") == "flicker"


def _ff_record(seed, n_updates=4000):
    return {"stim_type": "gaussian_noise", "cone_isolation": "S", "seed": seed,
            "checks_y": 1, "checks_x": 1, "n_updates": n_updates,
            "update_every_n_frames": 1, "refresh_rate_hz": 60, "mu": 0.5, "sigma": 0.3}


def test_sta_recovers_injected_filter():
    """Full-field STA: a causal filter with a single tap at lag L is recovered with its peak at L,
    averaged across several epochs (each a different seed)."""
    bpu, L = 6, 5
    records = [_ff_record(s) for s in (1, 2, 3)]
    responses = []
    for rec in records:
        stim = np.repeat(noise_from_record(rec).reshape(-1), bpu)   # upsampled stimulus
        h = np.zeros(30); h[L] = 1.0                                # delta filter at lag L
        responses.append(np.convolve(stim, h)[:len(stim)])         # linear causal response
    out = sta_from_records(records, responses, bins_per_update=bpu)
    avg = np.asarray(out["average"])
    assert int(out["n_epochs"]) == 3
    assert int(np.argmax(np.abs(avg))) == L                        # peak at the injected lag
    # the recovered filter is dominated by that one tap
    assert np.abs(avg[L]) > 5 * np.median(np.abs(avg))


def test_strf_recovers_injected_check_and_lag():
    """Checkerboard STRF: only one check drives the response (temporal tap at lag L); the recovered
    STRF peaks at that check and lag, pooling epochs in time."""
    L = 3
    records = [{"stim_type": "checkerboard", "cone_isolation": "S", "seed": s,
                "checks_y": 2, "checks_x": 2, "n_updates": 4000,
                "update_every_n_frames": 1, "refresh_rate_hz": 60, "mu": 0.5, "sigma": 0.3}
               for s in (7, 8)]
    responses = []
    for rec in records:
        v = noise_from_record(rec)                                 # (2,2,N)
        drive = v[1, 0, :]                                         # only check (y=1, x=0)
        h = np.zeros(20); h[L] = 1.0
        responses.append(np.convolve(drive, h)[:v.shape[-1]])
    res = strf_from_records(records, responses, bins_per_update=1, filter_len=20)
    assert res.peak_yx == (1, 0)                                   # correct check
    assert abs(res.peak_time_ms - 1000.0 * L / 60.0) < 1000.0 / 60.0 + 1e-6   # correct lag (±1 frame)


def test_strf_figure_renders():
    """The STRF plotter produces a 2-panel figure (spatial RF + temporal) from an STRFResult."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from neitz import plots
    records = [{"stim_type": "checkerboard", "seed": s, "checks_y": 2, "checks_x": 3,
                "n_updates": 1500, "update_every_n_frames": 1, "refresh_rate_hz": 60,
                "mu": 0.5, "sigma": 0.3} for s in (4, 5)]
    responses = []
    for rec in records:
        v = noise_from_record(rec)
        h = np.zeros(20); h[4] = 1.0
        responses.append(np.convolve(v[0, 2, :], h)[:v.shape[-1]])
    res = strf_from_records(records, responses, bins_per_update=1, filter_len=20)
    fig = plots.strf_figure(res, label="synthetic")
    assert len(fig.axes) >= 2 and np.asarray(res.spatial_rf).shape == (2, 3)
    plt.close(fig)


def test_record_from_stimulus_roundtrip():
    """The stored recording.stimulus (type split out) rebuilds a noise_from_record-ready record."""
    stored = {"type": "checkerboard",
              "params": {"seed": 2, "checks_y": 32, "checks_x": 40, "n_updates": 5,
                         "cone_isolation": "S", "mu": 0.5, "sigma": 0.3},
              "source": "stim-manifest"}
    rec = record_from_stimulus(stored)
    assert rec["stim_type"] == "checkerboard" and rec["seed"] == 2
    assert noise_from_record(rec).shape == (32, 40, 5)


def test_load_seed_epochs_assembles_from_abf(monkeypatch, tmp_path):
    """Integration of the store loader with a MOCKED Recording (synthetic signal + frame-clock TTL):
    only seeded noise is loaded, spikes are detected, t0 comes from the TTL frame clock, and the
    response is binned to the update grid. (The only untested-here piece is the REAL rig's TTL
    convention.)"""
    from neitz import run
    from neitz.dataio import DataStore

    fs, dur, t_on = 20000, 12.0, 2.0
    tt = np.arange(int(fs * dur)) / fs
    ttl = np.where(tt >= t_on, (np.mod((tt - t_on) * 60.0, 1.0) < 0.5).astype(float), 0.0)  # 60 Hz clock
    sig = np.random.RandomState(0).randn(len(tt)) * 0.5
    for s in np.arange(3.0, 10.0, 0.02):                       # big negative spikes on the signal
        sig[int(s * fs)] = -50.0

    class FakeRec:
        fs = 20000.0
        def channel(self, name):
            return sig if "im" in str(name).lower() else ttl
    monkeypatch.setattr(run.Recording, "load", staticmethod(lambda p: FakeRec()))

    cm = DataStore(root=tmp_path).cell("2026-07-03", "c01")
    cm.dir.mkdir(parents=True, exist_ok=True)
    cm.data["recordings"] = [
        {"id": "r1", "file": "raw/noise.abf",
         "stimulus": {"type": "gaussian_noise",
                      "params": {"seed": 5, "checks_y": 1, "checks_x": 1, "n_updates": 300,
                                 "update_every_n_frames": 1, "refresh_rate_hz": 60,
                                 "mu": 0.5, "sigma": 0.3}}},
        {"id": "r2", "file": "raw/flicker.abf",                # NOT seeded noise → skipped
         "stimulus": {"type": "sq_wave", "params": {"flicker_hz": 2}}},
    ]
    records, responses, info = run.load_seed_epochs(
        cm, chan="Im_prime", ttl="TTL", detect={"polarity": "neg", "method": "mad", "k": 6})
    assert len(records) == 1 and records[0]["seed"] == 5          # only the seeded-noise recording
    assert responses[0].shape[0] == 300 * 6                       # n_updates × default bins_per_update
    assert abs(info[0]["t0"] - t_on) < 0.05 and info[0]["n_spikes"] > 100


def test_epoch_response_bins_from_t0():
    """Spikes are binned into the stimulus-update grid starting at the epoch onset t0."""
    t0, bin_dt = 2.0, 0.1
    spikes = np.array([2.05, 2.05, 2.25, 9.9])                     # 2 in bin0, 1 in bin2, 1 out
    r = epoch_response(spikes, t0, n_bins=5, bin_dt=bin_dt)
    assert r.shape == (5,)
    assert r[0] == 2 / bin_dt and r[2] == 1 / bin_dt and r[1] == 0 and r[3] == 0
