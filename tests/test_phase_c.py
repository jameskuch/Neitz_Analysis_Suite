"""Phase C — plots + store-driven cell run."""
import os
import numpy as np
import pytest

from neitz.io.figures import save_figure
from neitz import plots


def _synthetic_group():
    """A minimal analyze_group-style dict for plotting."""
    centers = np.linspace(-0.1, 0.4, 50)
    band = np.full(50, 3.0)
    def side(peak):
        rate = np.full(50, 2.0); rate[20] = peak
        return dict(centers=centers, rate=rate, band=band, baseline=2.0,
                    ratio=peak / 2.0, peak_ms=50.0, p=0.3)
    return {"n_trials": 5, "freq": 2.0, "on": side(2.4), "off": side(2.2)}


def test_flicker_onoff_figure_saves(tmp_path):
    fig = plots.flicker_onoff_figure(_synthetic_group(), label="2026-06-02/c01 ipRGC")
    paths = save_figure(fig, tmp_path, "flicker_onoff")
    assert set(paths) == {"png", "pdf", "svg"}
    for p in paths.values():
        assert os.path.exists(p) and os.path.getsize(p) > 0


# --- integration against the real store (skips if absent) ---
_STORE = os.path.expanduser(os.environ.get("EPHYSDATAIO_ROOT", "~/Documents/ephysdataio"))
_CELL = os.path.join(_STORE, "2026-06-02", "c01", "manifest.json")


@pytest.mark.skipif(not os.path.exists(_CELL), reason="ephysdataio store not present")
def test_run_cell_flicker_writes_outputs(tmp_path):
    from neitz.dataio import DataStore
    from neitz.run import run_cell_flicker
    ds = DataStore(_STORE)
    res = run_cell_flicker(ds, "2026-06-02", "c01", n_shuffle=50, save=True)
    assert res.kind == "flicker"
    assert len(res.summary) == 5                         # ipRGC has 5 recordings
    pooled = res.tables["pooled_onoff"][0]
    assert pooled["n_trials"] == 5 and "verdict" in pooled
    # outputs written + recorded in the manifest
    cm = ds.cell("2026-06-02", "c01")
    out = [o for o in cm.data["outputs"] if o["analysis"] == "flicker"]
    assert out, "flicker output not recorded in manifest"
    files = out[-1]["files"]
    assert "figure_pdf" in files and "figure_svg" in files and "metrics_csv" in files
    for rel in files.values():
        assert (cm.dir / rel).exists()                   # paths resolve under the cell


def test_noise_sta_figure_saves(tmp_path):
    t = np.arange(360)
    avg = np.sin(t / 20.0) * np.exp(-t / 100.0)
    arrays = dict(average=avg, time_ms=1000.0 * t / 360.0,
                  freqs=np.fft.rfftfreq(360, d=1 / 360.0),
                  tuning=np.abs(np.fft.rfft(avg)),
                  per_epoch=np.random.RandomState(0).randn(3, 360))
    fig = plots.noise_sta_figure(arrays, label="2017-01-18/c01 S-iso")
    paths = save_figure(fig, tmp_path, "sta")
    assert set(paths) == {"png", "pdf", "svg"}
    for p in paths.values():
        assert os.path.exists(p) and os.path.getsize(p) > 0


_SISO = os.path.join(_STORE, "2017-01-18", "c01", "manifest.json")


@pytest.mark.skipif(not os.path.exists(_SISO), reason="S-iso cell not present")
def test_run_cell_noise_sta_peak():
    """Gaussian-noise STA validates against Sara's MATLAB ground truth (~22.2 ms, OFF)."""
    from neitz.dataio import DataStore
    from neitz.run import run_cell_noise
    ds = DataStore(_STORE)
    res = run_cell_noise(ds, "2017-01-18", "c01", save=False)   # don't pollute the store
    s = res.summary[0]
    assert s["n_epochs"] == 15
    assert abs(s["peak_ms"] - 22.22) < 0.5                       # matches Sara/MATLAB
    assert s["peak_sign"] == "OFF"
    assert res.arrays["average"].shape == (360,)
