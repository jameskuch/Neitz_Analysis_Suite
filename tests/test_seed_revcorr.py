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


def test_strf_time_axis_follows_refresh_rate_and_bins_per_update():
    """The lag axis is one RESPONSE bin per tap: the record's update rate (59.94 Hz on the rig, not a
    fixed 60) times bins_per_update. With two bins per update an injected lag of L updates comes back
    at tap 2L and the SAME milliseconds as at one bin per update."""
    L, bpu, hz = 3, 2, 59.94
    records = [{"stim_type": "checkerboard", "cone_isolation": "achromatic", "seed": 11,
                "checks_y": 2, "checks_x": 2, "n_updates": 4000,
                "update_every_n_frames": 1, "refresh_rate_hz": hz, "mu": 0.5, "sigma": 0.3}]
    v = noise_from_record(records[0])
    drive = np.repeat(v[0, 1, :], bpu)                             # check (0,1) on the 2-bin grid
    h = np.zeros(20); h[L * bpu] = 1.0
    resp = np.convolve(drive, h)[:drive.size]
    res = strf_from_records(records, [resp], bins_per_update=bpu, filter_len=20)
    assert np.allclose(res.time_ms, np.arange(20) * 1000.0 / (hz * bpu))
    assert res.peak_yx == (0, 1)
    assert abs(res.peak_time_ms - 1000.0 * L / hz) < 1e-9          # 50.05 ms, not 50.00
    # one bin per update at the rig rate: tap k sits at k / 59.94 s
    res1 = strf_from_records(records, [resp[::bpu]], bins_per_update=1, filter_len=20)
    assert abs(res1.time_ms[1] - 1000.0 / hz) < 1e-9
    assert abs(res1.peak_time_ms - 1000.0 * L / hz) < 1e-9


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


def test_strf_grid_figure_renders():
    """The per-location STA grid draws one panel per check, laid out like the stimulus."""
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
    fig = plots.strf_grid_figure(res, label="synthetic")
    assert len(fig.axes) == 2 * 3
    assert res.peak_yx == (0, 2)
    plt.close(fig)


def _synthetic_checkerboard_cell(tmp_path, monkeypatch, *, n_updates=400, drive_yx=(1, 2), lag=2):
    """A stored cell with ONE per-check-seeded 2x3 checkerboard epoch whose .abf is mocked: an
    inverted middle-segment sync trace (lit one update, dark the next) on 'I_MTest 3' and a
    current channel whose spikes follow check `drive_yx` with a delay of `lag` updates."""
    from neitz import run
    from neitz.dataio import DataStore
    fs, refresh = 20000.0, 59.9435
    t_frame = 1.0 / refresh
    seeds = [[11, 12, 13], [21, 22, 23]]
    rec = {"stim_type": "checkerboard", "seed": 9, "checks_y": 2, "checks_x": 3, "n_updates": n_updates,
           "check_seeds": seeds, "seed_scheme": "gaussCheckSeeds/1", "update_every_n_frames": 1,
           "refresh_rate_hz": refresh, "mu": 0.5, "sigma": 0.3}
    v = noise_from_record(rec)                                        # (2, 3, n_updates)
    t_on = 2.0
    edges = t_on + np.arange(n_updates + 1) * t_frame
    n = int((edges[-1] + 1.5) * fs)
    sync = np.full(n, 4.9)
    for k in range(0, n_updates, 2):                                  # lit updates: 4 pulses each
        for off in (0.0005, 0.0045, 0.0085, 0.0125):
            a, b = int((edges[k] + off) * fs), int((edges[k] + off + 0.0025) * fs)
            sync[a:b] = 0.1
    im = np.random.RandomState(1).randn(n) * 0.5
    dy, dx = drive_yx
    for k in range(lag, n_updates):                                   # spike when the driving check was bright
        if v[dy, dx, k - lag] > 0.65:
            im[int((edges[k] + 0.002) * fs)] = -60.0

    class FakeRec:
        channel_names = ["Im_prime", "TTL", "I_MTest 3"]
        def __init__(self): self.fs = fs
        def channel(self, name):
            nm = str(name).lower()
            return im if "im_" in nm else np.full(n, 4.9) if nm == "ttl" else sync
    monkeypatch.setattr(run.Recording, "load", staticmethod(lambda p: FakeRec()))
    monkeypatch.setattr(run, "_auto_mirror", lambda: None)           # never touch a real mirror from a test
    ds = DataStore(root=tmp_path)
    cm = ds.new_cell("2026-11-12", label="synthetic")
    (cm.dir / "raw").mkdir(parents=True, exist_ok=True)
    (cm.dir / "raw" / "2026_11_12_0001.abf").write_bytes(b"ABF2 dummy")
    cm.data["recordings"] = [{"id": "2026_11_12_0001", "file": "raw/2026_11_12_0001.abf",
                              "stimulus": {"type": "checkerboard", "source": "stim-manifest",
                                           "params": {k: val for k, val in rec.items() if k != "stim_type"}}}]
    cm.save()
    return ds, cm, (dy, dx), lag * t_frame * 1000.0


def test_run_cell_checkerboard_saves_strf_and_grid(tmp_path, monkeypatch):
    """End to end through the store: per-check seeds -> measured sync clock -> STRF; the run writes
    BOTH figures (strf + strf_grid) and result.json under outputs/strf/ and records every file in
    ONE output record (store-integrity rules), with no orphan files in the folder."""
    import os
    from neitz.run import run_cell_checkerboard
    ds, cm, (dy, dx), lag_ms = _synthetic_checkerboard_cell(tmp_path, monkeypatch)
    res = run_cell_checkerboard(ds, "2026-11-12", cm.data["cell"], name="strf", save=True)
    sm = res.summary[0]
    assert (sm["peak_y"], sm["peak_x"]) == (dy, dx)
    assert abs(sm["peak_time_ms"] - lag_ms) < 1000.0 / 59.9435 + 1e-6
    assert res.meta["epochs"][0]["clock"] == "measured:I_MTest 3"

    cm = ds.cell("2026-11-12", cm.data["cell"])
    outs = [o for o in cm.data["outputs"] if o["analysis"] == "strf"]
    assert len(outs) == 1, "one record per analysis"
    files = outs[0]["files"]
    for key in ("figure_png", "figure_grid_png", "figure_grid_pdf", "figure_grid_svg", "result_json"):
        assert key in files and (cm.dir / files[key]).exists(), key
    assert all(v.startswith("outputs/strf/") for v in files.values())
    on_disk = {f for f in os.listdir(cm.dir / "outputs" / "strf") if f.endswith(".png")}
    assert on_disk == {"strf.png", "strf_grid.png"}, "no orphan / missing PNGs"
