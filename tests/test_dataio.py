"""Data store, manifest, and figure saving."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neitz.dataio import DataStore, proper_name, is_proper_name
from neitz.io.figures import save_figure


def test_proper_name():
    assert is_proper_name("2026_06_02_0040.abf")
    assert not is_proper_name("siso-spikes.csv")
    assert proper_name("2026_06_02_0040.abf", "2026-06-02") == "2026_06_02_0040.abf"
    assert proper_name("siso-spikes.csv", "2017-01-18") == "2017_01_18_siso-spikes.csv"


def test_store_cell_recording_roundtrip(tmp_path):
    ds = DataStore(root=tmp_path)
    cm = ds.new_cell("2026-06-02", label="ipRGC", cell_type="ipRGC")
    assert cm.data["cell"] == "c01"                       # forced numbering
    assert cm.data["label"] == "ipRGC"                    # free-text label

    src = tmp_path / "2026_06_02_0040.abf"
    src.write_bytes(b"abc")
    rec = cm.add_recording(src, stimulus={"type": "flicker", "params": {"flicker_hz": 2.0}},
                           channels={"signal": "Im_prime", "ttl": "TTL"})
    assert rec["id"] == "2026_06_02_0040"
    assert (cm.dir / "raw" / "2026_06_02_0040.abf").exists()   # copied into raw/
    cm.set_stimulus(rec["id"], "flicker", {"flicker_hz": 2.0})
    cm.save()

    # de-dup: re-adding the same file does not copy again
    cm.add_recording(src)
    assert len(list((cm.dir / "raw").iterdir())) == 1

    reloaded = ds.cell("2026-06-02", "c01")
    assert reloaded.get_stimulus("2026_06_02_0040")["params"]["flicker_hz"] == 2.0

    cm2 = ds.new_cell("2026-06-02", label="notipRGC")     # auto-increments
    assert cm2.data["cell"] == "c02"

    idx = ds.update_index()
    assert {c["cell"] for c in idx} == {"c01", "c02"}


def test_record_output(tmp_path):
    ds = DataStore(root=tmp_path)
    cm = ds.new_cell("2026-06-02", label="x")
    out = cm.output_dir("flicker_onoff")
    assert out.exists() and out.name == "flicker_onoff"
    cm.record_output("flicker_onoff", files={"figure_png": "outputs/flicker_onoff/f.png"},
                     summary={"verdict": "no sig."}, params={"k": 6})
    o = cm.data["outputs"][0]
    assert o["analysis"] == "flicker_onoff" and o["summary"]["verdict"] == "no sig."
    assert o["created"]                                   # timestamp set


def test_save_figure_multiformat(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save_figure(fig, tmp_path / "out", "demo")
    assert set(paths) == {"png", "pdf", "svg"}
    for p in paths.values():
        assert os.path.exists(p)
