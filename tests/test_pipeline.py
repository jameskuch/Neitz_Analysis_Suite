"""Regression tests for the analysis-pipeline model, storage, and run-param extraction."""
import tempfile

import pytest

from neitz import pipeline as pl


def test_default_flicker_pipeline_shape():
    p = pl.default_flicker_pipeline()
    types = [n["type"] for n in p["nodes"]]
    assert types == ["source", "align", "detect", "region", "flicker", "figures"]
    # linear chain: one connection between each adjacent pair
    assert len(p["connections"]) == len(p["nodes"]) - 1
    assert pl.validate(p) == []
    assert pl.pipeline_stim_family(p) == "sq_wave"


def test_stim_family_by_terminal():
    assert pl.pipeline_stim_family(pl.default_sta_pipeline()) == "gaussian_noise"
    assert pl.pipeline_stim_family(pl.default_strf_pipeline()) == "checkerboard"


def test_validate_flags_problems():
    p = pl.empty_pipeline("bad")
    probs = pl.validate(p)
    assert any("source" in x for x in probs)
    assert any("analysis" in x for x in probs)


def test_detect_and_region_extraction():
    p = pl.default_flicker_pipeline()
    det_node = pl.node_of_type(p, "detect")
    det_node["params"].update({"polarity": "pos", "k": 8, "refractory_ms": 3, "abs_threshold": 25})
    det = pl.detect_from_pipeline(p)
    assert det["polarity"] == "pos"
    assert det["k"] == 8.0
    assert det["refractory_s"] == pytest.approx(0.003)
    assert det["abs_threshold"] == 25.0

    reg_node = pl.node_of_type(p, "region")
    reg_node["params"].update({"start_s": 1.5, "end_s": 40.0, "crop": True})
    reg = pl.region_from_pipeline(p)
    assert reg == {"start_s": 1.5, "end_s": 40.0, "crop": True}

    assert pl.run_kwargs(p)["n_shuffle"] == 500


def test_storage_roundtrip_and_copy():
    with tempfile.TemporaryDirectory() as d:
        # first listing materializes the three defaults
        names = pl.list_pipelines(root=d)
        assert set(pl.DEFAULT_PIPELINES).issubset(set(names))

        p = pl.load_pipeline("Sq wave ON/OFF", root=d)
        assert p is not None
        pl.node_of_type(p, "detect")["params"]["k"] = 9
        p["name"] = "my-flicker"
        pl.save_pipeline(p, root=d)
        assert "my-flicker" in pl.list_pipelines(root=d)
        assert pl.load_pipeline("my-flicker", root=d)["nodes"][2]["params"]["k"] == 9

        # save-as from existing
        dup = pl.copy_pipeline("my-flicker", "my-flicker copy", root=d)
        assert dup["name"] == "my-flicker copy"
        assert "my-flicker copy" in pl.list_pipelines(root=d)

        assert pl.delete_pipeline("my-flicker copy", root=d)
        assert "my-flicker copy" not in pl.list_pipelines(root=d)


def test_new_node_uses_defaults_and_overrides():
    n = pl.new_node("detect", "d1", params={"k": 10, "bogus": 1})
    assert n["params"]["k"] == 10
    assert "bogus" not in n["params"]          # unknown params dropped
    assert n["params"]["polarity"] == "neg"    # default kept


def test_flicker_node_relabeled_sq_wave():
    # display label renamed; the internal type / stimulus family stay 'flicker' / 'sq_wave'
    assert pl.COMPONENT_REGISTRY["flicker"]["label"] == "Sq wave ON/OFF"
    assert pl.COMPONENT_REGISTRY["flicker"]["terminal"] == "sq_wave"


def test_registry_has_desc_and_math_for_every_component():
    for ctype, spec in pl.COMPONENT_REGISTRY.items():
        assert spec.get("desc"), f"{ctype} missing desc"
        assert isinstance(spec.get("math"), list) and spec["math"], f"{ctype} missing math"


def test_smooth_and_tfilter_extraction():
    p = pl.default_flicker_pipeline()
    # no smooth / tfilter node in the default chain
    assert pl.smooth_from_pipeline(p) is None
    assert pl.tfilter_from_pipeline(p) is None

    p["nodes"].append(pl.new_node("smooth", "sm0", params={"window": 6, "method": "gaussian"}))
    sm = pl.smooth_from_pipeline(p)
    assert sm == {"method": "gaussian", "window": 6.0, "polyorder": 2}

    # a span < 2 is a no-op (None)
    pl.node_of_type(p, "smooth")["params"]["window"] = 1
    assert pl.smooth_from_pipeline(p) is None

    p["nodes"].append(pl.new_node("tfilter", "tf0",
                                  params={"cutoff_hz": 25, "mode": "highpass", "taps": 41}))
    tf = pl.tfilter_from_pipeline(p)
    assert tf["cutoff_hz"] == 25.0 and tf["mode"] == "highpass" and tf["taps"] == 41
    assert pl.pipeline_stim_family(p) == "sq_wave"    # new processing nodes don't change dispatch
