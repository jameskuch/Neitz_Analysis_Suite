"""neitz.io.stim.epoch_groups — group recordings into epochs by stim_signature."""
from neitz.io import stim


def _rec(rec_id, sig, stim_type="sq_wave", cone="achromatic"):
    return {"id": rec_id,
            "stimulus": {"type": stim_type,
                         "params": {"stim_signature": sig, "cone_isolation": cone},
                         "source": "stim-manifest"}}


def test_repeated_stimulus_groups_as_n_epochs():
    # the rig scenario: same 2 Hz square wave x3 -> ONE group, 3 epochs
    g = stim.epoch_groups([_rec("0001", "A"), _rec("0002", "A"), _rec("0003", "A")])
    assert len(g) == 1
    assert g[0]["n_epochs"] == 3
    assert g[0]["stim_signature"] == "A"
    assert g[0]["recording_ids"] == ["0001", "0002", "0003"]
    assert g[0]["stim_type"] == "sq_wave"
    assert g[0]["cone_isolation"] == "achromatic"


def test_distinct_stimuli_are_separate_groups():
    g = stim.epoch_groups([_rec("1", "A"), _rec("2", "A"),
                           _rec("3", "B"), _rec("4", "B"), _rec("5", "B")])
    assert [x["n_epochs"] for x in g] == [2, 3]
    assert [x["stim_signature"] for x in g] == ["A", "B"]


def test_non_consecutive_same_signature_stays_separate():
    # A, B, A -> three groups: grouping is by CONSECUTIVE runs, not global identity
    g = stim.epoch_groups([_rec("1", "A"), _rec("2", "B"), _rec("3", "A")])
    assert [x["stim_signature"] for x in g] == ["A", "B", "A"]
    assert all(x["n_epochs"] == 1 for x in g)


def test_missing_signature_never_merges():
    # no stimulus / no params / explicit None -> each its own group, no accidental merge
    recs = [{"id": "1"},
            {"id": "2", "stimulus": {"type": "sq_wave", "params": {}}},
            _rec("3", None)]
    g = stim.epoch_groups(recs)
    assert len(g) == 3
    assert all(x["stim_signature"] is None and x["n_epochs"] == 1 for x in g)


def test_empty():
    assert stim.epoch_groups([]) == []
