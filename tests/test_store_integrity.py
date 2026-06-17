"""Store / manifest integrity — output records must stay associated with their cell and must NOT
duplicate. Guards the file↔manifest tracking invariants in CLAUDE.md ("let's not let that happen
again"). Pure-Python (no .abf needed), so it runs everywhere."""
from neitz.dataio import DataStore


def test_record_output_replaces_same_analysis(tmp_path):
    """A re-run with the same analysis name REPLACES its record (no accruing duplicates/zombies)."""
    cm = DataStore(root=tmp_path).cell("2099-01-01", "c01")
    cm.record_output("flicker", files={"figure_png": "outputs/flicker/x.png"}, params={"k": 6})
    cm.record_output("flicker", files={"figure_png": "outputs/flicker/x.png"}, params={"k": 8})
    cm.record_output("4_epochs", files={"figure_png": "outputs/4_epochs/y.png"}, params={})

    outs = cm.data["outputs"]
    names = [o["analysis"] for o in outs]
    assert names.count("flicker") == 1, "same-named run must REPLACE, not append"
    assert names.count("4_epochs") == 1, "a distinct run name stays a separate record"
    kept = next(o for o in outs if o["analysis"] == "flicker")
    assert kept["params"]["k"] == 8, "the kept record is the latest run"


def test_record_output_keeps_files_label_and_parent(tmp_path):
    """Every output stays under outputs/<analysis>/ (associated with the parent) with its files + label."""
    cm = DataStore(root=tmp_path).cell("2099-01-02", "c02")
    cm.record_output("light_response", label="light response 4 Hz",
                     files={"fig": "outputs/light_response/a.png",
                            "pdf": "outputs/light_response/a.pdf"}, params={})
    rec = cm.data["outputs"][-1]
    assert rec["files"] == {"fig": "outputs/light_response/a.png",
                            "pdf": "outputs/light_response/a.pdf"}
    assert rec.get("label") == "light response 4 Hz", "friendly run name preserved"
    for v in rec["files"].values():                       # files live under the analysis folder
        assert v.startswith("outputs/light_response/"), f"output detached from its analysis: {v}"


def test_manifest_persists_outputs_across_reload(tmp_path):
    """Saving then reloading the cell keeps the output record (no loss on round-trip)."""
    ds = DataStore(root=tmp_path)
    cm = ds.cell("2099-01-03", "c03")
    cm.record_output("sta", files={"fig": "outputs/sta/sta.png"}, params={})
    cm.save()
    again = DataStore(root=tmp_path).cell("2099-01-03", "c03")
    names = [o["analysis"] for o in again.data.get("outputs", [])]
    assert names == ["sta"], "output record must survive save+reload, attached to the cell"
