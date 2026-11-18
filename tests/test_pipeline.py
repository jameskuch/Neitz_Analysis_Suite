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


def test_component_code_is_the_live_source_of_what_runs():
    """Every component lists the functions that run for it; each resolves to real source read from
    the live module (a `viewer:` reference needs the GUI module passed in `extra`)."""
    import viewer
    for ctype in pl.COMPONENT_REGISTRY:
        refs = pl.CODE_REFS.get(ctype, [])
        assert refs, f"{ctype} lists no code"
        code = pl.component_code(ctype, extra={"viewer": viewer})
        assert [c["ref"] for c in code] == refs
        for c in code:
            assert c["file"] and c["line"] > 0, f"{c['ref']} did not resolve"
            assert c["source"].lstrip().startswith(("def ", "class ", "@")), c["ref"]
    # the STA chain really ends in the FFT cross-correlation
    names = [c["name"] for c in pl.component_code("sta")]
    assert names[:3] == ["run_cell_noise", "_run_cell_noise_seed", "load_seed_epochs"]
    assert "reverse_correlation" in names
    # an unresolvable reference is reported inline, never raised
    pl.CODE_REFS["__probe"] = ["nosuch.module:fn"]
    try:
        bad = pl.component_code("__probe")
        assert bad[0]["file"] == "" and bad[0]["source"].startswith("# source unavailable")
    finally:
        del pl.CODE_REFS["__probe"]


def test_run_kwargs_carries_the_sta_filter_length():
    p = pl.default_sta_pipeline()
    assert pl.run_kwargs(p)["filter_s"] == 1.0
    pl.node_of_type(p, "sta")["params"]["filter_s"] = 0.5
    assert pl.run_kwargs(p)["filter_s"] == 0.5
    pl.node_of_type(p, "sta")["params"]["filter_s"] = "junk"
    assert pl.run_kwargs(p)["filter_s"] == 1.0
    assert pl.run_kwargs(pl.default_flicker_pipeline())["filter_s"] == 1.0


def test_ordered_nodes_follows_the_connections():
    p = pl.default_sta_pipeline()
    p["nodes"].reverse()                                   # storage order scrambled
    assert [n["id"] for n in pl.ordered_nodes(p)] == ["source0", "detect1", "region2", "sta3", "figures4"]
    # an unconnected node comes last; a missing-node connection is ignored
    p["nodes"].append(pl.new_node("sf_isi", "lonely"))
    p["connections"].append({"from_node": "ghost", "from_port": "x", "to_node": "sta3", "to_port": "spikes"})
    assert [n["id"] for n in pl.ordered_nodes(p)][-1] == "lonely"


def test_pipeline_script_is_one_listing_in_execution_order_each_function_once():
    import viewer
    p = pl.default_sta_pipeline()
    pl.node_of_type(p, "sta")["params"]["filter_s"] = 0.5
    ctx = {"date": "2026-11-12", "cell": "c04", "files": ["2026_11_12_0039.abf"], "chan": "Im_prime",
           "ttl": "TTL", "detect": {"polarity": "neg", "method": "mad", "k": 6.0},
           "abs_map": {}, "region": {"start": 2.69, "end": 195.0}, "run_name": None}
    txt = pl.pipeline_script(p, extra={"viewer": viewer}, context=ctx)
    lines = txt.split("\n")
    assert lines[0] == '"""' and "Gaussian-noise STA — the whole pipeline as one script" in lines[1]
    assert "  4. Reverse-correlation STA  [sta]  —  filter_s=0.5" in txt
    assert "  files: ['2026_11_12_0039.abf']" in txt
    # the driver is the real run call with the live settings
    assert "result = run_cell_noise(store, '2026-11-12', 'c04', name='sta'," in txt
    assert "include=['2026_11_12_0039.abf']" in txt and "filter_s=0.5)" in txt
    # components in execution order, every function once, listed where first used
    order = [txt.index(f"# {i}. ") for i in range(1, 6)]
    assert order == sorted(order)
    assert txt.count("def reverse_correlation(") == 1 and txt.count("def detect_spikes(") == 1
    assert txt.index("def detect_spikes(") < txt.index("def run_cell_noise(") < txt.index("def reverse_correlation(")
    assert "# --- reverse_correlation  (revcorr.py:" in txt
    # the figures node shares nothing with the sta node, but a reused ref is a pointer, not a copy
    pl.CODE_REFS["figures"].insert(0, "neitz.run:load_seed_epochs")
    try:
        txt2 = pl.pipeline_script(p, extra={"viewer": viewer})
        assert txt2.count("def load_seed_epochs(") == 1
        assert "# load_seed_epochs  → already listed under component 4" in txt2
    finally:
        pl.CODE_REFS["figures"].pop(0)
    # no context → placeholders, still valid-looking python
    bare = pl.pipeline_script(pl.default_flicker_pipeline(), extra={"viewer": viewer})
    assert "run_cell_flicker(store, '<date>', '<cell>'" in bare and "n_shuffle=500" in bare
    compile(bare.split("# =====")[0], "<header>", "exec")        # the docstring header parses
