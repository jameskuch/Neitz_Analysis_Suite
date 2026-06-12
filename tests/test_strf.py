"""Spatiotemporal STRF — synthetic receptive-field recovery."""
import numpy as np
from neitz.analysis import strf as S
from neitz.stimulus import CheckerboardParadigm, STRFResult


def _planted(n_y=4, n_x=5, T=8000, flen=30, target=(2, 3), lag=8, seed=0):
    """White-noise checkerboard + a response driven by ONE check's temporal filter."""
    rng = np.random.default_rng(seed)
    n_checks = n_y * n_x
    stim = rng.standard_normal((n_checks, T))
    tgt = target[0] * n_x + target[1]
    true_filter = np.zeros(flen)
    true_filter[lag] = 1.0                       # impulse at `lag`
    resp = np.zeros(T)
    for tau in range(flen):                      # response = filter (*) stim[target]
        resp[tau:] += true_filter[tau] * stim[tgt, :T - tau]
    return stim, resp, tgt, lag, (n_y, n_x, flen)


def test_engine_recovers_planted_check_and_lag():
    stim, resp, tgt, lag, (ny, nx, flen) = _planted()
    strf = S.spatiotemporal_revcorr(stim, resp, flen)
    assert strf.shape == (ny * nx, flen)
    assert S.peak_check(strf) == tgt
    assert abs(int(np.argmax(np.abs(strf[tgt]))) - lag) <= 1
    # the planted check dominates the others
    peaks = np.max(np.abs(strf), axis=1)
    assert peaks[tgt] > 5 * np.median(np.delete(peaks, tgt))


def test_checkerboard_paradigm():
    stim, resp, tgt, lag, (ny, nx, flen) = _planted()
    res = CheckerboardParadigm(n_y=ny, n_x=nx, filter_len=flen).analyze(stim, resp)
    assert isinstance(res, STRFResult)
    assert res.strf.shape == (ny, nx, flen)
    assert res.spatial_rf.shape == (ny, nx)
    assert res.peak_yx == (tgt // nx, tgt % nx)
    # spatial map peaks at the planted check
    assert np.unravel_index(np.argmax(np.abs(res.spatial_rf)), res.spatial_rf.shape) == res.peak_yx
    assert np.isclose(np.max(np.abs(res.strf)), 1.0)        # default 'max' normalization


def test_accepts_3d_stimulus():
    stim, resp, tgt, lag, (ny, nx, flen) = _planted()
    cube = stim.reshape(ny, nx, -1)
    res = CheckerboardParadigm(n_y=ny, n_x=nx, filter_len=flen).analyze(cube, resp)
    assert res.peak_yx == (tgt // nx, tgt % nx)


def test_run_strf_saveable(tmp_path):
    from neitz.run import run_strf, Result
    stim, resp, tgt, lag, (ny, nx, flen) = _planted()
    res = run_strf(stim, resp, ny, nx, filter_len=flen)
    assert res.kind == "strf"
    assert res.summary[0]["peak_y"] == tgt // nx and res.summary[0]["peak_x"] == tgt % nx
    res.save(tmp_path / "strf")
    loaded = Result.load(tmp_path / "strf")
    assert loaded.arrays["strf"].shape == (ny, nx, flen)
