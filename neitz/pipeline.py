"""
Analysis pipelines — a saveable, editable graph of processing components that lives OVER the
tested CLI/`run` orchestration.

A pipeline is a declarative description of an analysis: a set of **component nodes** (each with
typed input/output ports and internal parameters) wired by **connections**. It is NOT a new
analysis engine — running a pipeline extracts its node parameters and dispatches to the existing,
regression-tested `run_cell_flicker` / `run_cell_noise` / `run_cell_checkerboard` in `run.py`
(the same way the GUI's "Run analysis" always has). The graph is the human-editable face; the
terminal *analysis* node picks which run function fires.

The canonical "Run analysis" flow is itself a pipeline (`default_flicker_pipeline`):

    Cell recordings ─▶ Align frame-syncs ─▶ Detect spikes ─▶ Select region ─▶ Flicker ON/OFF ─▶ Figures

Storage: one JSON file per pipeline under `<store_root>/pipelines/<slug>.json` (the same
`~/Documents/ephysdataio` data structure as everything else). "Save as" is a plain copy under a
new name.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .dataio.config import data_root

# ---------------------------------------------------------------------------------------------
# Component registry: the palette. Each entry defines a component's ports and its parameter
# schema. Ports carry a `type` so the editor can validate a connection (out.type == in.type).
# Param schema entries: {type: number|choice|bool|text, default, label, options?, min?, max?, step?}.
# `terminal` marks the analysis components — the pipeline's terminal node decides which run_cell_*
# fires and under which stimulus family.
# ---------------------------------------------------------------------------------------------
PORT_RECORDINGS = "recordings"      # a set of loaded .abf recordings (+ their frame-sync/TTL)
PORT_SPIKES = "spikes"              # detected spike trains per recording
PORT_RESULT = "result"             # an analysis Result (metrics + figure specs)
PORT_OUTPUTS = "outputs"           # written files (figures / csv / json)

COMPONENT_REGISTRY: dict = {
    "source": {
        "label": "Cell recordings",
        "category": "source",
        "color": "#2d7d46",
        "help": "The selected cell's .abf recordings (from the Analysis View selection).",
        "inputs": [],
        "outputs": [{"name": "recordings", "type": PORT_RECORDINGS}],
        "params": {},
    },
    "align": {
        "label": "Align frame-syncs",
        "category": "processing",
        "color": "#2f6fb0",
        "help": "Nudge each file so its frame-sync (trial start) lines up before pooling.",
        "inputs": [{"name": "recordings", "type": PORT_RECORDINGS}],
        "outputs": [{"name": "aligned", "type": PORT_RECORDINGS}],
        "params": {
            "mode": {"type": "choice", "default": "auto", "label": "mode",
                     "options": ["auto", "off"]},
        },
    },
    "detect": {
        "label": "Detect spikes",
        "category": "processing",
        "color": "#8e44ad",
        "help": "Threshold-detect spikes on the signal channel (per-trace thresholds honored).",
        "inputs": [{"name": "recordings", "type": PORT_RECORDINGS}],
        "outputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "params": {
            "polarity": {"type": "choice", "default": "neg", "label": "polarity",
                         "options": ["neg", "pos", "abs"]},
            "method": {"type": "choice", "default": "mad", "label": "algorithm",
                       "options": ["mad", "abs", "mad_floor", "matlab"]},
            "k": {"type": "number", "default": 6, "label": "k·MAD", "min": 2, "max": 15, "step": 0.5},
            "refractory_ms": {"type": "number", "default": 2, "label": "refractory (ms)",
                              "min": 0, "step": 0.5},
            "abs_threshold": {"type": "number", "default": None, "label": "abs threshold"},
        },
    },
    "region": {
        "label": "Select region",
        "category": "processing",
        "color": "#c0803a",
        "help": "Restrict the analysis to the [start, end] window (drops the adapting block).",
        "inputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "outputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "params": {
            "start_s": {"type": "number", "default": None, "label": "start (s)"},
            "end_s": {"type": "number", "default": None, "label": "end (s)"},
            "crop": {"type": "bool", "default": False, "label": "crop to region"},
        },
    },
    "flicker": {
        "label": "Flicker ON/OFF",
        "category": "analysis",
        "color": "#b0392f",
        "help": "Square-wave cycle/transition PSTH + pooled ON/OFF shift test.",
        "inputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "outputs": [{"name": "result", "type": PORT_RESULT}],
        "terminal": "sq_wave",
        "params": {
            "n_shuffle": {"type": "number", "default": 500, "label": "shuffles", "min": 0, "step": 100},
        },
    },
    "sta": {
        "label": "Reverse-correlation STA",
        "category": "analysis",
        "color": "#b0392f",
        "help": "Temporal spike-triggered average from the seed-regenerated Gaussian noise.",
        "inputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "outputs": [{"name": "result", "type": PORT_RESULT}],
        "terminal": "gaussian_noise",
        "params": {
            "filter_s": {"type": "number", "default": 1.0, "label": "filter (s)", "min": 0.1, "step": 0.1},
        },
    },
    "strf": {
        "label": "Spatiotemporal STRF",
        "category": "analysis",
        "color": "#b0392f",
        "help": "Spatiotemporal reverse correlation from the seed-regenerated checkerboard.",
        "inputs": [{"name": "spikes", "type": PORT_SPIKES}],
        "outputs": [{"name": "result", "type": PORT_RESULT}],
        "terminal": "checkerboard",
        "params": {},
    },
    "figures": {
        "label": "Figures & exports",
        "category": "output",
        "color": "#555b66",
        "help": "Write PNG/PDF/SVG figures (+ 4K exports for flicker) into the cell's outputs.",
        "inputs": [{"name": "result", "type": PORT_RESULT}],
        "outputs": [{"name": "outputs", "type": PORT_OUTPUTS}],
        "params": {
            "export_4k": {"type": "bool", "default": True, "label": "4K exports"},
        },
    },
}

# terminal component type -> stimulus family it runs under (mirrors run.analysis_for_stim_type)
_TERMINALS = {t: c["terminal"] for t, c in COMPONENT_REGISTRY.items() if c.get("terminal")}


def component_defaults(ctype) -> dict:
    """Default param values for a component type."""
    spec = COMPONENT_REGISTRY.get(ctype, {})
    return {k: v.get("default") for k, v in spec.get("params", {}).items()}


# ---------------------------------------------------------------------------------------------
# Pipeline model: plain dicts (JSON-native), plus helpers. A node = {id, type, params, x, y,
# label}. A connection = {from_node, from_port, to_node, to_port}.
# ---------------------------------------------------------------------------------------------
def new_node(ctype, node_id, x=40.0, y=40.0, params=None, label=None) -> dict:
    spec = COMPONENT_REGISTRY.get(ctype, {})
    p = component_defaults(ctype)
    if params:
        p.update({k: v for k, v in params.items() if k in p})
    return {"id": node_id, "type": ctype, "params": p, "x": float(x), "y": float(y),
            "label": label or spec.get("label", ctype)}


def empty_pipeline(name="untitled") -> dict:
    return {"version": 1, "name": name, "nodes": [], "connections": [],
            "created": datetime.now().isoformat(timespec="seconds")}


def _chain(name, steps) -> dict:
    """Build a left-to-right linear pipeline from a list of component types."""
    nodes, conns, x = [], [], 30.0
    prev = None
    for i, ctype in enumerate(steps):
        nid = f"{ctype}{i}"
        nodes.append(new_node(ctype, nid, x=x, y=60.0))
        if prev is not None:
            out_port = COMPONENT_REGISTRY[prev[1]]["outputs"][0]["name"]
            in_port = COMPONENT_REGISTRY[ctype]["inputs"][0]["name"]
            conns.append({"from_node": prev[0], "from_port": out_port,
                          "to_node": nid, "to_port": in_port})
        prev = (nid, ctype)
        x += 210.0
    return {"version": 1, "name": name, "nodes": nodes, "connections": conns,
            "created": datetime.now().isoformat(timespec="seconds")}


def default_flicker_pipeline() -> dict:
    """The canonical 'Run analysis' flow for square-wave (flicker) cells."""
    return _chain("Flicker ON/OFF", ["source", "align", "detect", "region", "flicker", "figures"])


def default_sta_pipeline() -> dict:
    return _chain("Gaussian-noise STA", ["source", "detect", "region", "sta", "figures"])


def default_strf_pipeline() -> dict:
    return _chain("Checkerboard STRF", ["source", "detect", "region", "strf", "figures"])


DEFAULT_PIPELINES = {
    "Flicker ON/OFF": default_flicker_pipeline,
    "Gaussian-noise STA": default_sta_pipeline,
    "Checkerboard STRF": default_strf_pipeline,
}


def terminal_node(pipe) -> dict | None:
    """The analysis (terminal) node — the one whose type carries a stimulus family."""
    for n in pipe.get("nodes", []):
        if n.get("type") in _TERMINALS:
            return n
    return None


def pipeline_stim_family(pipe) -> str | None:
    """Which stimulus family this pipeline analyzes (sq_wave / gaussian_noise / checkerboard)."""
    n = terminal_node(pipe)
    return _TERMINALS.get(n["type"]) if n else None


def node_of_type(pipe, ctype) -> dict | None:
    return next((n for n in pipe.get("nodes", []) if n.get("type") == ctype), None)


def validate(pipe) -> list:
    """Return a list of human-readable problems (empty = OK). Cheap structural checks."""
    problems = []
    nodes = pipe.get("nodes", [])
    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)):
        problems.append("duplicate node ids")
    idset = set(ids)
    if not any(n["type"] == "source" for n in nodes):
        problems.append("no 'Cell recordings' source node")
    if terminal_node(pipe) is None:
        problems.append("no analysis node (flicker / sta / strf)")
    for c in pipe.get("connections", []):
        if c["from_node"] not in idset or c["to_node"] not in idset:
            problems.append("connection references a missing node")
    return problems


# ---------------------------------------------------------------------------------------------
# Storage: one JSON per pipeline under <store_root>/pipelines/. Names are free text; the file
# slug is sanitized. On first use the three default pipelines are materialized.
# ---------------------------------------------------------------------------------------------
def _slug(name) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()) or "untitled"
    return s[:80]


def pipelines_dir(root=None) -> Path:
    d = (Path(root) if root else data_root()) / "pipelines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_defaults(root=None) -> None:
    d = pipelines_dir(root)
    if any(d.glob("*.json")):
        return
    for name, factory in DEFAULT_PIPELINES.items():
        save_pipeline(factory(), root=root)


def list_pipelines(root=None) -> list:
    """Sorted list of saved pipeline names (materializing the defaults on first use)."""
    _ensure_defaults(root)
    names = []
    for p in pipelines_dir(root).glob("*.json"):
        try:
            names.append(json.loads(p.read_text()).get("name") or p.stem)
        except Exception:
            names.append(p.stem)
    return sorted(names, key=str.lower)


def load_pipeline(name, root=None) -> dict | None:
    d = pipelines_dir(root)
    # prefer an exact name match inside the file; fall back to the slug filename
    for p in d.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if (data.get("name") or p.stem) == name:
            return data
    f = d / f"{_slug(name)}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return None
    return None


def save_pipeline(pipe, root=None) -> Path:
    pipe = dict(pipe)
    pipe.setdefault("version", 1)
    pipe["saved"] = datetime.now().isoformat(timespec="seconds")
    f = pipelines_dir(root) / f"{_slug(pipe.get('name'))}.json"
    f.write_text(json.dumps(pipe, indent=2, default=str))
    return f


def delete_pipeline(name, root=None) -> bool:
    d = pipelines_dir(root)
    hit = False
    for p in list(d.glob("*.json")):
        try:
            nm = json.loads(p.read_text()).get("name") or p.stem
        except Exception:
            nm = p.stem
        if nm == name:
            p.unlink()
            hit = True
    return hit


def copy_pipeline(src_name, new_name, root=None) -> dict | None:
    """'Save as from existing' — duplicate a pipeline under a new name."""
    src = load_pipeline(src_name, root=root)
    if src is None:
        return None
    dup = dict(src)
    dup["name"] = new_name
    dup["created"] = datetime.now().isoformat(timespec="seconds")
    save_pipeline(dup, root=root)
    return dup


# ---------------------------------------------------------------------------------------------
# Execution: extract node params from the graph and dispatch to the tested run_cell_* functions.
# `context` carries the live Analysis-View state (checked files, channels, per-trace maps, …) so a
# run honors what the user set up — the GUI already threads these; the pipeline just re-packages
# the graph's node params on top.
# ---------------------------------------------------------------------------------------------
def detect_from_pipeline(pipe, fallback=None) -> dict:
    """Spike-detection dict (run_cell_* `detect=`) built from the pipeline's Detect node."""
    fb = dict(fallback or {})
    n = node_of_type(pipe, "detect")
    if n is None:
        return fb
    p = n["params"]
    refr = p.get("refractory_ms")
    det = {
        "polarity": p.get("polarity", fb.get("polarity", "neg")),
        "method": p.get("method", fb.get("method", "mad")),
        "k": float(p["k"]) if p.get("k") is not None else fb.get("k"),
        "abs_threshold": (float(p["abs_threshold"]) if p.get("abs_threshold") not in (None, "")
                          else fb.get("abs_threshold")),
        "refractory_s": (float(refr) / 1000.0) if refr not in (None, "") else fb.get("refractory_s", 0.002),
    }
    return det


def region_from_pipeline(pipe) -> dict:
    """{start_s, end_s, crop} from the Select-region node (values may be None = auto)."""
    n = node_of_type(pipe, "region")
    if n is None:
        return {"start_s": None, "end_s": None, "crop": False}
    p = n["params"]
    return {"start_s": p.get("start_s"), "end_s": p.get("end_s"), "crop": bool(p.get("crop"))}


def run_kwargs(pipe) -> dict:
    """Assorted run knobs read off the graph (n_shuffle, 4K export flag, align mode)."""
    fl = node_of_type(pipe, "flicker")
    fig = node_of_type(pipe, "figures")
    align = node_of_type(pipe, "align")
    return {
        "n_shuffle": int(fl["params"].get("n_shuffle", 500)) if fl else 500,
        "export_4k": bool(fig["params"].get("export_4k", True)) if fig else True,
        "align_mode": align["params"].get("mode", "auto") if align else "off",
    }
