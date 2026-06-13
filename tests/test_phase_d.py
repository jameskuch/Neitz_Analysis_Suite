"""Phase D — GUI manifest helpers/callbacks (against a tmp store)."""
import viewer
from neitz.dataio import DataStore


def test_parse_params():
    assert viewer.parse_params("a=1, b=2.5, c=hi") == {"a": 1.0, "b": 2.5, "c": "hi"}
    assert viewer.parse_params("") == {}


def test_pick_cell_and_save_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("EPHYSDATAIO_ROOT", str(tmp_path))
    ds = DataStore(str(tmp_path))
    cm = ds.new_cell("2026-06-02", label="testcell")
    src = tmp_path / "2026_06_02_0001.abf"
    src.write_bytes(b"x")
    cm.add_recording(src)
    cm.save(); ds.update_index()

    # the cell appears in the dropdown options
    labels = [o["label"] for o in viewer.store_cell_options()]
    assert any("testcell" in s for s in labels)

    # pick_cell (now multi-cell) lists the cell's files and returns sel as a list.
    # A single "date|cell" string is accepted and wrapped.
    opts, files, sel, stype, sparams = viewer.pick_cell("2026-06-02|c01")
    assert len(opts) == 1                                 # the cell's one recording is listed
    assert isinstance(files, list)                        # checked = openable files only
    assert sel == [{"date": "2026-06-02", "cell": "c01"}]
    assert stype is None                                  # no stimulus yet

    # save_meta writes stimulus into the manifest (the GUI metadata-query flow)
    msg = viewer.save_meta(1, sel, "flicker", "flicker_hz=2, frame_rate=60")
    assert "saved" in msg
    reloaded = ds.cell("2026-06-02", "c01")
    stim = reloaded.get_stimulus("2026_06_02_0001")
    assert stim["type"] == "flicker" and stim["params"]["flicker_hz"] == 2.0
