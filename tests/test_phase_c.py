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
