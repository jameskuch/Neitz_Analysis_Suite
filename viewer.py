"""
viewer.py — interactive ABF viewer + spike / flicker analysis (Dash GUI).

Pure-Python (Dash/Plotly) front end on the neitz backbone.

Single file  -> inspect mode: signal (decimated, zoomable) + detected spikes,
    frame-sync below (x linked), 240 Hz adapting blocks shaded & excluded, all
    analysis inside the (editable) flicker region, + spike-train FFT with the
    stimulus square-wave frequency marked, + acquisition metadata.

Multiple files -> group mode: the checked files are one group; each file's signal
    AND its frame sync are overlaid in matching colors/opacity (spikes hidden) so
    you can compare frame syncs across trials (rig debugging); FFT shows each
    file's spectrum faint + the bold GROUP-AVERAGE spectrum.

Settings (polarity / threshold / refractory) persist across files and sessions.
File picking uses native macOS dialogs via osascript; tkinter fallback.

Run:  /Users/j/miniconda3/bin/python viewer.py   ->  http://127.0.0.1:8050
"""

from __future__ import annotations
import os
import io
import json
import glob
import base64
import shutil
import platform
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless: render sparkline thumbnails to PNG bytes
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update, ALL

from neitz.io import load_recording
from neitz.spikes import detect_spikes
from neitz.analysis import flicker as flk
from neitz.dataio import DataStore
from neitz.run import run_cell_flicker, run_cell_noise

# default browse location is the managed data store (~/Documents/ephysdataio)
EPHYS_ROOT = os.path.expanduser(os.environ.get("EPHYSDATAIO_ROOT", "~/Documents/ephysdataio"))
DEFAULT_GLOBS = [os.path.join(EPHYS_ROOT, "**", "*.abf"),
                 os.path.join(EPHYS_ROOT, "**", "*.csv")]
DEFAULT_DIR = EPHYS_ROOT if os.path.isdir(EPHYS_ROOT) else os.getcwd()
_last_dir = DEFAULT_DIR          # remembered folder; native dialogs open here
# master password gating destructive deletes in the explorer (override via env)
ADMIN_PASSWORD = os.environ.get("NEITZ_ADMIN_PASSWORD", "neitz")
PALETTE = pc.qualitative.Plotly
BIN_RATE = 200      # Hz — bin spikes to this rate before FFT
FMAX = 60           # Hz — FFT display limit
PERSIST = dict(persistence=True, persistence_type="local")


# ---- discovery & native dialogs --------------------------------------------
def discover_abf(root=None):
    globs = ["**/*.abf"] if root else DEFAULT_GLOBS
    base = root or "."
    found = []
    for g in globs:
        found += glob.glob(os.path.join(base, g), recursive=True)
    seen, out = set(), []
    for f in sorted(found):
        if f not in seen:
            seen.add(f); out.append(f)
    return out


def file_options(files):
    """Checklist options labelled by basename (value = full path)."""
    return [{"label": " " + os.path.basename(f), "value": f} for f in files]


def _osascript(script):
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=600)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def native_choose_file():
    if platform.system() == "Darwin":
        return _osascript(f'POSIX path of (choose file with prompt "Select an ABF or spike CSV" '
                          f'default location (POSIX file "{_last_dir}") of type {{"abf", "csv"}})')
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        p = filedialog.askopenfilename(filetypes=[("ABF/CSV", "*.abf *.csv"), ("All", "*.*")])
        root.destroy(); return p or None
    except Exception:
        return None


def native_choose_folder():
    if platform.system() == "Darwin":
        return _osascript(f'POSIX path of (choose folder with prompt "Select a folder" '
                          f'default location (POSIX file "{_last_dir}"))')
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        p = filedialog.askdirectory()
        root.destroy(); return p or None
    except Exception:
        return None


# ---- cache & helpers -------------------------------------------------------
_CACHE: dict[str, dict] = {}


def get_recording(path):
    if path not in _CACHE:
        _CACHE[path] = dict(rec=load_recording(path), chans={}, flicker={})
    return _CACHE[path]["rec"]


def get_channel(path, name):
    get_recording(path)
    chans = _CACHE[path]["chans"]
    if name not in chans:
        chans[name] = _CACHE[path]["rec"].channel(name)
    return chans[name]


def get_flicker(path, ttl_name):
    rec = get_recording(path)
    fc = _CACHE[path]["flicker"]
    if ttl_name not in fc:
        if getattr(rec, "protocol", "") == "csv-spikes":
            fc[ttl_name] = None          # spike CSVs have no frame-sync / flicker
        else:
            try:
                fc[ttl_name] = flk.detect_flicker(get_channel(path, ttl_name), rec.fs)
            except Exception:
                fc[ttl_name] = None
    return fc[ttl_name]


def minmax_decimate(t, y, n_target=4000):
    n = len(y)
    if n <= n_target * 2:
        return t, y
    binw = n // n_target
    nb = n // binw
    yb = y[:nb * binw].reshape(nb, binw)
    tb = t[:nb * binw].reshape(nb, binw)
    out_t = np.repeat(tb[:, 0], 2)
    out_y = np.empty(nb * 2)
    out_y[0::2], out_y[1::2] = yb.min(1), yb.max(1)
    return out_t, out_y


def binned_rate(times, t0, t1, bin_rate=BIN_RATE):
    n = int(round((t1 - t0) * bin_rate))
    if n < 8:
        return None
    edges = np.linspace(t0, t1, n + 1)
    cnt, _ = np.histogram(times, bins=edges)
    return cnt.astype(float) * bin_rate


def power_w(rate, bin_rate=BIN_RATE):
    """Single-sided FFT power per frequency bin, in watts (R = 1 Ω):

        W = 2 * |X[k]|^2 / N^2

    X[k] = rfft of the mean-removed binned-rate signal, N = number of samples.
    Clipped to FMAX.
    """
    r = np.asarray(rate, dtype=float)
    r = r - r.mean()
    n = len(r)
    x = np.fft.rfft(r)
    p = 2.0 * np.abs(x) ** 2 / (n ** 2)
    f = np.fft.rfftfreq(n, d=1.0 / bin_rate)
    keep = f <= FMAX
    return f[keep], p[keep]


def power_db(pw):
    """Power (W) -> dB, with empty bins floored at 80 dB below the spectrum's peak so
    they don't drag the y-axis to -inf."""
    pw = np.asarray(pw, dtype=float)
    pk = float(pw.max()) if pw.size else 0.0
    floor = pk * 1e-8 if pk > 0 else 1e-20
    return 10.0 * np.log10(np.maximum(pw, floor))


def blank_fig(msg=""):
    """Empty placeholder figure (cleared graph when nothing is selected)."""
    f = go.Figure()
    f.update_layout(margin=dict(l=40, r=20, t=20, b=20),
                    annotations=[dict(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                                      showarrow=False, font=dict(size=16, color="#999"))])
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False)
    return f


# ---- data-store (manifest) helpers -----------------------------------------
def store_cell_options(sort="date_desc", recent=None):
    try:
        idx = DataStore().index()
    except Exception:
        idx = []
    if sort == "recent":                             # most-recently-opened first, then newest date
        order = {v: i for i, v in enumerate(recent or [])}
        key = lambda c: f"{c['date']}|{c['cell']}"
        seen = [c for c in idx if key(c) in order]
        seen.sort(key=lambda c: order[key(c)])
        rest = sorted([c for c in idx if key(c) not in order],
                      key=lambda c: (c["date"], c["cell"]), reverse=True)
        idx = seen + rest
    elif sort == "date_asc":
        idx = sorted(idx, key=lambda c: (c["date"], c["cell"]))
    elif sort == "label":
        idx = sorted(idx, key=lambda c: ((c.get("label") or "").lower(), c["date"]))
    elif sort == "type":
        idx = sorted(idx, key=lambda c: ((c.get("cell_type") or "~").lower(), c["date"], c["cell"]))
    else:                                            # date_desc (newest first)
        idx = sorted(idx, key=lambda c: (c["date"], c["cell"]), reverse=True)
    return [{"label": f"{c['date']} / {c['cell']} — {c.get('label') or ''}".strip(" —"),
             "value": f"{c['date']}|{c['cell']}"} for c in idx]


def parse_params(text):
    """'flicker_hz=2, frame_rate=60' -> {'flicker_hz': 2.0, 'frame_rate': 60.0}."""
    out = {}
    for part in (text or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def cell_data_files(cm):
    return [str(cm.dir / r["file"]) for r in cm.data.get("recordings", [])
            if str(r.get("file", "")).endswith((".abf", ".csv"))]


_LOADABLE: dict = {}


def loadable(path):
    """True if `path` opens as a valid time-series recording (finite fs > 0).

    Filters out Sara-style stimulus/header CSVs (e.g. siso-DLP.csv) whose column 0
    is a phase label, not a time vector, which would otherwise break the viewer.
    """
    if path not in _LOADABLE:
        try:
            rec = get_recording(path)
            _LOADABLE[path] = bool(np.isfinite(rec.fs) and rec.fs > 0)
        except Exception:
            _LOADABLE[path] = False
    return _LOADABLE[path]


def cell_default_files(cm):
    """(all data files, files to check by default). Prefers spike-train CSVs, then
    any openable recording; never auto-checks a file that won't open."""
    files = cell_data_files(cm)
    openable = [f for f in files if loadable(f)]
    spikes = [f for f in openable if "spike" in os.path.basename(f).lower()]
    return files, (spikes or openable)


def _img_datauri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def output_gallery(date, cell):
    """Clickable thumbnails of every PNG under the cell's outputs/ (newest first)."""
    cm = DataStore().cell(date, cell)
    outdir = cm.dir / "outputs"
    pngs = sorted(outdir.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if outdir.exists() else []
    if not pngs:
        return [html.Span("no output images yet — run an analysis on this cell",
                          style={"color": "#888", "fontSize": "12px"})]
    thumbs = []
    for p in pngs:
        rel = p.relative_to(outdir)
        thumbs.append(html.Div([
            html.Img(src=_img_datauri(p), id={"type": "out-thumb", "src": str(p)}, n_clicks=0,
                     style={"height": "150px", "border": "1px solid #ccc", "cursor": "pointer",
                            "display": "block", "background": "white"}),
            html.Div(str(rel), style={"fontSize": "10px", "maxWidth": "240px", "wordBreak": "break-all"}),
        ], style={"margin": "4px"}))
    return thumbs


_MODAL_SHOWN = {"display": "flex", "position": "fixed", "top": 0, "left": 0,
                "width": "100%", "height": "100%", "background": "rgba(0,0,0,0.88)",
                "zIndex": 3000, "alignItems": "center", "justifyContent": "center"}


def card(title, children, opened=True):
    """A collapsible, titled compartment (native <details>/<summary>) for the sidebar."""
    return html.Details(open=opened, style={
        "border": "1px solid #dcdcdc", "borderRadius": "6px",
        "marginBottom": "8px", "background": "white"}, children=[
        html.Summary(title, style={"fontWeight": "bold", "fontSize": "12px", "color": "#234",
                                   "padding": "6px 8px", "cursor": "pointer",
                                   "userSelect": "none"}),
        html.Div(children, style={"padding": "8px", "borderTop": "1px solid #eee"}),
    ])


_FIELD = {"marginBottom": "6px"}                    # stacked label+control block
_LBL = {"fontSize": "11px", "fontWeight": "bold", "color": "#444", "display": "block"}


def ov(**pos):
    """A semi-transparent control overlay pinned to a graph corner (absolute)."""
    s = {"position": "absolute", "zIndex": 20, "background": "rgba(255,255,255,0.85)",
         "padding": "0 4px", "borderRadius": "4px", "fontSize": "10px",
         "display": "flex", "alignItems": "center", "gap": "3px",
         "boxShadow": "0 0 3px rgba(0,0,0,0.18)"}
    s.update(pos)
    return s


_OVL = {"fontSize": "10px", "color": "#444", "fontWeight": "bold"}    # inline label inside an overlay
_OVI = {"fontSize": "10px", "height": "16px", "padding": "0 3px", "boxSizing": "border-box",
        "textAlign": "right"}                                        # compact overlay textbox
# region start/end overlays: same fixed height; start flush with the plot's left (l margin),
# end flush with the plot's right (r margin). Shared so a callback can hide them when cropping.
_START_OV = ov(top="34px", left="78px", height="20px")
_END_OV = ov(top="34px", right="20px", height="20px")
_END_FIELDS = {"display": "flex", "alignItems": "center", "gap": "3px"}   # end (s) label+input


# ============================================================
# Data Explorer (pop-out): dates -> cell thumbnails -> file preview + manifest JSON
# ============================================================
_SPARK_CACHE: dict = {}                             # (path, mtime) -> sparkline data-URI


def sparkline_datauri(path, width_in=2.6, height_in=0.72):
    """Tiny decimated waveform PNG (data-URI) for a raw recording; cached by mtime."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        key = (path, 0.0)
    if key in _SPARK_CACHE:
        return _SPARK_CACHE[key]
    uri = None
    try:
        rec = get_recording(path)
        y = rec.channel(rec.channel_names[0])
        step = max(1, len(y) // 1500)
        ys = y[::step]
        fig = Figure(figsize=(width_in, height_in), dpi=64)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.plot(ys, color="#27408b", linewidth=0.5)
        ax.margins(x=0, y=0.05)
        buf = io.BytesIO(); FigureCanvasAgg(fig).print_png(buf)
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        uri = None
    _SPARK_CACHE[key] = uri
    return uri


_BIGWAVE_CACHE: dict = {}                           # (path, mtime) -> full waveform data-URI


def big_waveform_datauri(path):
    """Larger labelled waveform PNG (data-URI) for the enlarge modal; cached by mtime."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        key = (path, 0.0)
    if key in _BIGWAVE_CACHE:
        return _BIGWAVE_CACHE[key]
    uri = None
    try:
        rec = get_recording(path)
        ch = rec.channel_names[0]
        y = rec.channel(ch)
        fs = rec.fs if (np.isfinite(rec.fs) and rec.fs > 0) else 1.0
        t = np.arange(len(y)) / fs
        step = max(1, len(y) // 6000)
        fig = Figure(figsize=(11, 4.2), dpi=110)
        ax = fig.add_subplot(111)
        ax.plot(t[::step], y[::step], color="#27408b", linewidth=0.6)
        ax.set_xlabel("time (s)"); ax.set_ylabel(f"{ch} ({rec.units(ch)})")
        ax.set_title(os.path.basename(path), fontsize=10)
        ax.margins(x=0)
        fig.tight_layout()
        buf = io.BytesIO(); FigureCanvasAgg(fig).print_png(buf)
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        uri = None
    _BIGWAVE_CACHE[key] = uri
    return uri


def _latest_output_path(cm):
    outdir = cm.dir / "outputs"
    if outdir.exists():
        pngs = sorted(outdir.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pngs:
            return str(pngs[0])
    return None


def _latest_output_datauri(cm):
    p = _latest_output_path(cm)
    return _img_datauri(p) if p else None


_THUMB_IMG = {"height": "62px", "border": "1px solid #ccc", "background": "white",
              "display": "block", "marginBottom": "2px"}


def _date_rows_data():
    """Per-date rows for the rail: {date, label (joined cell labels/types), n cells}."""
    try:
        idx = DataStore().index()
    except Exception:
        idx = []
    by = {}
    for c in idx:
        d = by.setdefault(c["date"], {"date": c["date"], "labels": [], "n": 0})
        d["n"] += 1
        lab = c.get("label") or c.get("cell_type") or ""
        if lab and lab not in d["labels"]:
            d["labels"].append(lab)
    rows = []
    for d in by.values():
        d["label"] = ", ".join(d["labels"])
        rows.append(d)
    return rows


_RAIL_DATE = {"flex": "0 0 70px"}
_RAIL_N = {"flex": "0 0 26px", "textAlign": "center"}
_RAIL_TRASH = {"flex": "0 0 22px"}
_RAILHDR = {"background": "#222", "color": "#bcd", "border": "none", "cursor": "pointer",
            "fontSize": "10px", "padding": "3px", "borderRadius": "3px", "fontWeight": "bold"}
_RAILQ = {"fontSize": "10px", "padding": "1px 3px", "boxSizing": "border-box", "minWidth": 0,
          "border": "1px solid #2a2a35", "background": "#0d0d12", "color": "white",
          "borderRadius": "3px"}


def explorer_dates_body(active=None, sort=None, search=None):
    """Filtered + sorted date rows: [ experiment label | date | #cells | 🗑 ]."""
    rows = _date_rows_data()
    s = search or {}
    ql = (s.get("label") or "").lower(); qd = (s.get("date") or "").lower()
    qc = (s.get("cells") or "").strip()
    rows = [r for r in rows
            if ql in r["label"].lower() and qd in r["date"].lower()
            and (not qc or qc in str(r["n"]))]
    sort = sort or {"col": "date", "dir": "desc"}
    keyf = {"label": lambda r: (r["label"].lower(), r["date"]),
            "date": lambda r: r["date"],
            "cells": lambda r: (r["n"], r["date"])}.get(sort.get("col"), lambda r: r["date"])
    rows.sort(key=keyf, reverse=(sort.get("dir") == "desc"))
    out = []
    for r in rows:
        is_active = (r["date"] == active)
        out.append(html.Div([
            html.Div([
                html.Div(r["label"] or "—", title=r["label"],
                         style={"flex": "1", "minWidth": 0, "overflow": "hidden",
                                "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                html.Div(r["date"], style=_RAIL_DATE),
                html.Div(str(r["n"]), style=_RAIL_N),
            ], id={"type": "exp-date", "date": r["date"]}, n_clicks=0,
               style={"flex": "1", "minWidth": 0, "display": "flex", "gap": "4px",
                      "alignItems": "center", "cursor": "pointer"}),
            html.Button("🗑", id={"type": "del-date", "date": r["date"]}, n_clicks=0,
                        title=f"delete all of {r['date']}",
                        style=dict(_RAIL_TRASH, border="none", background="none", color="#e66",
                                   cursor="pointer", fontSize="12px", padding="0")),
        ], style={"padding": "6px 8px", "fontSize": "11px", "color": "white",
                  "display": "flex", "alignItems": "center", "gap": "4px",
                  "borderBottom": "1px solid #2a2a35",
                  "background": ("#3367d6" if is_active else "transparent")}))
    return out or [html.Div("no dates", style={"color": "#999", "padding": "10px",
                                               "fontSize": "11px"})]


def delete_date(date):
    """Move an entire date folder (all its cells) to a reversible .trash/ and reindex."""
    ds = DataStore()
    cells = [c for c in ds.index() if c["date"] == date]
    if not cells:
        return None
    date_dir = ds.cell(date, cells[0]["cell"]).dir.parent      # <root>/<date>
    trash = date_dir.parent / ".trash" / f"{date}_ALL"
    trash.parent.mkdir(parents=True, exist_ok=True)
    if trash.exists():
        trash = date_dir.parent / ".trash" / f"{date}_ALL_dup"
    shutil.move(str(date_dir), str(trash))
    for dct in (_CACHE, _LOADABLE, _SPARK_CACHE, _BIGWAVE_CACHE):   # drop cached entries for the date
        for k in [kk for kk in dct if str(date) in str(kk) or
                  (isinstance(kk, tuple) and kk and str(date) in str(kk[0]))]:
            dct.pop(k, None)
    ds.update_index()
    return trash


def explorer_day_cards(date):
    """Center grid for a date: one card per cell (latest output + waveform)."""
    ds = DataStore()
    cells = [c for c in ds.index() if c["date"] == date]
    cards = []
    for c in cells:
        cm = ds.cell(date, c["cell"])
        files = cell_data_files(cm)
        out = _latest_output_datauri(cm)
        spark = sparkline_datauri(files[0]) if files else None
        _full = {"width": "100%", "maxWidth": "100%", "height": "auto", "display": "block",
                 "background": "white"}
        imgs = []
        if spark:                                    # waveform on top
            imgs.append(html.Img(src=spark, className="gprev",
                                 **({"data-ps": "wave|" + files[0]} if files else {}),
                                 style=dict(_full, border="1px solid #ccc", marginBottom="4px")))
        if out:                                      # processed output beneath, sized to the card
            imgs.append(html.Img(src=out, className="gprev",
                                 **({"data-ps": "img|" + _latest_output_path(cm)} if _latest_output_path(cm) else {}),
                                 style=dict(_full, border="1px solid #ddd")))
        if not imgs:
            imgs.append(html.Div("no preview", style={"color": "#999", "fontSize": "11px",
                                                      "height": "62px"}))
        label = c.get("label") or ""
        ctype = c.get("cell_type")
        cards.append(html.Div(
            imgs + [
                html.Div(c["cell"], style={"fontWeight": "bold", "fontSize": "13px"}),
                html.Div(label, style={"fontSize": "11px", "color": "#444",
                                       "wordBreak": "break-word"}),
                html.Div((f"{ctype} · " if ctype else "")
                         + f"{c.get('n_recordings', 0)} rec · {c.get('n_outputs', 0)} out",
                         style={"fontSize": "10px", "color": "#777"}),
            ],
            id={"type": "exp-cell", "cell": c["cell"]}, n_clicks=0,
            style={"width": "210px", "border": "1px solid #ccc", "borderRadius": "6px",
                   "padding": "8px", "background": "white", "cursor": "pointer",
                   "overflow": "hidden", "boxShadow": "0 1px 3px rgba(0,0,0,0.2)"}))
    return cards or [html.Div("no cells on this date", style={"color": "#ccc"})]


def explorer_file_options(date, cell):
    """Center checklist for a cell: each raw recording as a checkbox + waveform thumb."""
    cm = DataStore().cell(date, cell)
    opts = []
    for r in cm.data.get("recordings", []):
        if not str(r.get("file", "")).endswith((".abf", ".csv")):
            continue
        p = str(cm.dir / r["file"])
        spark = sparkline_datauri(p)
        stim = (r.get("stimulus") or {}).get("type")
        thumb = html.Div([
            html.Img(src=spark, className="gprev", style=dict(_THUMB_IMG, width="220px"),
                     **{"data-ps": "wave|" + p}) if spark
            else html.Div("—", style={"height": "62px", "color": "#999"}),
            html.Div(os.path.basename(p), style={"fontSize": "11px", "wordBreak": "break-all"}),
            html.Div(f"stim: {stim}" if stim else "stim: —",
                     style={"fontSize": "10px", "color": "#777"}),
        ], style={"display": "inline-block", "verticalAlign": "top"})
        opts.append({"label": thumb, "value": p})
    return opts


def _json_tree(obj, key=None, top=False):
    """Recursive collapsible tree for a JSON-able object (manifest.json)."""
    klab = "" if key is None else f"{key}: "
    if isinstance(obj, dict):
        return html.Details(open=top, children=[
            html.Summary(f"{klab}{{{len(obj)} keys}}",
                         style={"cursor": "pointer", "fontSize": "12px", "color": "#226"}),
            html.Div([_json_tree(v, k) for k, v in obj.items()],
                     style={"marginLeft": "12px"})])
    if isinstance(obj, list):
        return html.Details(open=top, children=[
            html.Summary(f"{klab}[{len(obj)} items]",
                         style={"cursor": "pointer", "fontSize": "12px", "color": "#226"}),
            html.Div([_json_tree(v, i) for i, v in enumerate(obj)],
                     style={"marginLeft": "12px"})])
    return html.Div([html.Span(klab, style={"color": "#999"}),
                     html.Span(json.dumps(obj), style={"color": "#063"})],
                    style={"fontFamily": "monospace", "fontSize": "12px", "marginLeft": "2px"})


def explorer_detail(date, cell):
    """Right pane for a cell: raw-trace thumbnails + manifest JSON + output figures,
    everything click-to-enlarge; output figures get a per-figure delete button."""
    cm = DataStore().cell(date, cell)

    # raw analog recordings — click the waveform to enlarge (task 1)
    raw_thumbs = []
    for p in cell_data_files(cm):
        if not loadable(p):
            continue
        spark = sparkline_datauri(p)
        if not spark:
            continue
        raw_thumbs.append(html.Div([
            html.Img(src=spark, id={"type": "raw-thumb", "src": p}, n_clicks=0, className="gprev",
                     **{"data-ps": "wave|" + p},
                     style={"width": "150px", "border": "1px solid #ccc", "cursor": "pointer",
                            "background": "white", "display": "block"}),
            html.Div(os.path.basename(p), style={"fontSize": "11px", "maxWidth": "150px",
                                                 "overflow": "hidden", "textOverflow": "ellipsis",
                                                 "whiteSpace": "nowrap"}),
        ], style={"margin": "3px"}))

    # processed output figures — click to enlarge, 🗑 to delete that figure (task 2)
    outdir = cm.dir / "outputs"
    pngs = sorted(outdir.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if outdir.exists() else []
    out_thumbs = []
    for p in pngs:
        rel = p.relative_to(outdir)
        out_thumbs.append(html.Div([
            html.Img(src=_img_datauri(str(p)), className="gprev",
                     id={"type": "out-thumb", "src": str(p)}, n_clicks=0,
                     **{"data-ps": "img|" + str(p)},
                     style={"height": "110px", "border": "1px solid #ccc", "cursor": "pointer",
                            "background": "white", "display": "block"}),
            html.Div([
                html.Span(str(rel), style={"fontSize": "11px", "flex": "1", "overflow": "hidden",
                                           "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                html.Button("🗑", id={"type": "del-output", "src": str(p)}, n_clicks=0,
                            title="delete this figure",
                            style={"fontSize": "12px", "padding": "0 4px", "color": "#b00",
                                   "border": "none", "background": "none", "cursor": "pointer"}),
            ], style={"display": "flex", "alignItems": "center", "maxWidth": "180px"}),
        ], style={"margin": "3px"}))

    return [
        html.Div(f"{date} / {cell}", style={"fontWeight": "bold", "fontSize": "15px",
                                            "marginBottom": "4px"}),
        html.Div("raw recordings (click to enlarge)",
                 style={"fontWeight": "bold", "fontSize": "13px", "color": "#555"}),
        html.Div(raw_thumbs or [html.Span("none", style={"color": "#999", "fontSize": "13px"})],
                 style={"display": "flex", "flexWrap": "wrap", "marginBottom": "6px"}),
        html.Div("manifest.json", style={"fontWeight": "bold", "fontSize": "13px",
                                         "color": "#555", "marginTop": "6px"}),
        html.Div(_json_tree(cm.data, top=True),
                 style={"maxHeight": "32vh", "overflowY": "auto", "border": "1px solid #eee",
                        "padding": "6px", "background": "#fbfbfb"}),
        html.Div("output figures (click to enlarge · 🗑 to delete)",
                 style={"fontWeight": "bold", "fontSize": "13px", "color": "#555",
                        "marginTop": "8px"}),
        html.Div(out_thumbs or [html.Span("none yet", style={"color": "#999",
                                                             "fontSize": "13px"})],
                 style={"display": "flex", "flexWrap": "wrap"}),
    ]


def explorer_breadcrumb(date, cell):
    parts = [html.Span(f"📂 {date}", style={"fontWeight": "bold"})]
    if cell:
        parts += [html.Span("  ›  ", style={"color": "#888"}),
                  html.Span(cell, style={"fontWeight": "bold"})]
    return parts


def delete_files(date, cell, paths):
    """Move the given files to a reversible .trash/ and drop them from the manifest.

    Files are MOVED (not unlinked) into ~/Documents/ephysdataio/.trash/<date>_<cell>/
    so a delete can be undone by hand. Refuses anything outside the cell's folder.
    Returns (removed_names, trash_dir).
    """
    cm = DataStore().cell(date, cell)
    cell_dir = cm.dir.resolve()
    trash = Path(EPHYS_ROOT) / ".trash" / f"{date}_{cell}"
    trash.mkdir(parents=True, exist_ok=True)
    removed = []
    for p in paths:
        pp = Path(p).resolve()
        if cell_dir not in pp.parents:               # safety: only inside this cell
            continue
        rel = str(pp.relative_to(cell_dir))
        if pp.exists():
            dest = trash / pp.name
            if dest.exists():
                dest = trash / f"{pp.stem}__dup{pp.suffix}"
            shutil.move(str(pp), str(dest))
        cm.data["recordings"] = [r for r in cm.data.get("recordings", [])
                                 if r.get("file") != rel]
        for d in (_CACHE, _LOADABLE, _SPARK_CACHE):  # drop any cached copies
            for key in [kk for kk in d if (kk == p or (isinstance(kk, tuple) and kk and kk[0] == p))]:
                d.pop(key, None)
        removed.append(pp.name)
    cm.save()
    DataStore().update_index()
    return removed, trash


_EXPLORER_SHOWN = {"display": "flex", "position": "fixed", "top": 0, "left": 0,
                   "width": "100%", "height": "100%", "background": "rgba(18,18,26,0.97)",
                   "zIndex": 2500, "flexDirection": "column", "padding": "10px",
                   "boxSizing": "border-box"}

_DEL_SHOWN = {"display": "flex", "position": "fixed", "top": 0, "left": 0,
              "width": "100%", "height": "100%", "background": "rgba(0,0,0,0.6)",
              "zIndex": 2800, "alignItems": "center", "justifyContent": "center"}


# ============================================================
# App
# ============================================================
app = Dash(__name__)
app.title = "Neitz ABF Viewer"
_files = discover_abf()

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "display": "flex", "gap": "10px",
           "height": "100vh", "padding": "8px", "boxSizing": "border-box"},
    children=[

    # ================= LEFT SIDEBAR: controls (≤ 1/3 width, scrollable) =========
    html.Div(style={"flex": "0 0 20%", "maxWidth": "20%", "minWidth": "230px",
                    "height": "100%", "overflowY": "auto", "paddingRight": "6px",
                    "boxSizing": "border-box"}, children=[

        # upper-left nav: click to jump to the Data Explorer
        html.Div([
            html.Div("Data Explorer ↱", id="open-explorer", n_clicks=0,
                     title="open the Data Explorer",
                     style={"fontWeight": "bold", "fontSize": "15px", "cursor": "pointer",
                            "color": "#3367d6"}),
            html.Div("Neitz ABF Viewer", style={"fontSize": "10px", "color": "#888"}),
        ], style={"marginBottom": "8px"}),

        # ---- compartment: data store (cell select + files + stimulus) ----
        card("Data store", [

            # cell(s) label + inline sort control on the same row (sort = dropdown order)
            html.Div([
                html.Label("cell(s)", style=dict(_LBL, marginBottom=0)),
                dcc.RadioItems(id="cell-sort", value="recent", inline=True,
                               options=[{"label": "opened recently", "value": "recent"},
                                        {"label": "↓date", "value": "date_desc"},
                                        {"label": "↑date", "value": "date_asc"},
                                        {"label": "label", "value": "label"},
                                        {"label": "type", "value": "type"}],
                               labelStyle={"fontSize": "10px", "marginLeft": "5px"},
                               inputStyle={"marginRight": "2px"}, **PERSIST),
            ], style={"display": "flex", "alignItems": "baseline",
                      "justifyContent": "space-between", "flexWrap": "wrap"}),
            html.Div(dcc.Dropdown(id="cell-select", options=store_cell_options(), multi=True,
                                  placeholder="pick date(s) / cell(s)…", maxHeight=380,
                                  optionHeight=34, style={"width": "100%"}),
                     id="cell-select-wrap", style={"marginBottom": "6px"}),

            # files for the selected cell(s) — multi-column listbox (saves height)
            html.Label("files", style=_LBL),
            html.Div(dcc.Checklist(id="file", options=[], value=[],   # blank until a cell is picked
                                   labelStyle={"display": "block", "width": "150px",
                                               "boxSizing": "border-box", "fontSize": "11px",
                                               "whiteSpace": "nowrap", "overflow": "hidden",
                                               "textOverflow": "ellipsis"},
                                   inputStyle={"marginRight": "4px", "verticalAlign": "middle"},
                                   # flexbox column-wrap: each label is atomic (never split
                                   # across columns the way CSS multicol did)
                                   style={"display": "flex", "flexFlow": "column wrap",
                                          "alignContent": "flex-start", "maxHeight": "150px"}),
                     style={"maxHeight": "158px", "overflowX": "auto", "overflowY": "hidden",
                            "border": "1px solid #ccc", "padding": "4px", "background": "white"}),
            html.Div(id="meta", style={"fontSize": "10px", "color": "#333", "lineHeight": "1.45",
                                       "background": "#f6f6f6", "padding": "6px",
                                       "borderRadius": "4px", "marginTop": "6px",
                                       "marginBottom": "8px"}),

            # stimulus metadata — moved below the files
            html.Div([html.Label("stimulus type", style=_LBL),
                      dcc.Dropdown(id="stim-type", style={"width": "100%"},
                                   options=[{"label": "sq wave", "value": "flicker"},
                                            {"label": "gaussian_noise", "value": "gaussian_noise"},
                                            {"label": "checkerboard", "value": "checkerboard"},
                                            {"label": "(none)", "value": "(none)"}])],
                     style=_FIELD),
            html.Div([html.Label("stimulus params (k=v, …)", style=_LBL),
                      dcc.Input(id="stim-params", type="text", debounce=True,
                                placeholder="flicker_hz=2, frame_rate=60",
                                style={"width": "100%", "boxSizing": "border-box"})],
                     style=_FIELD),
            html.Div([
                html.Button("Save metadata", id="save-meta", n_clicks=0),
                html.Button("▶ Run analysis", id="run-cell", n_clicks=0,
                            style={"marginLeft": "4px"}),
                html.Button("⤓ Backup mirror", id="backup-mirror", n_clicks=0,
                            style={"marginLeft": "4px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
            html.Div([html.Label("run by the cell's stimulus type · run name keeps a variant "
                                 "(blank = auto)", style=dict(_LBL, fontWeight="normal")),
                      dcc.Input(id="run-name", type="text", value="", debounce=True,
                                placeholder="auto (sq wave / sta)",
                                style={"width": "100%", "boxSizing": "border-box"})],
                     style={"marginTop": "6px"}),
            html.Div(id="store-msg", style={"fontSize": "11px", "color": "#070",
                                            "marginTop": "6px"}),
        ]),

        # ---- compartment: channels & spike detection ----
        card("Channels & spike detection", [
            html.Div([                                   # signal + TTL channel, side by side
                html.Div([html.Label("signal channel", style=_LBL),
                          dcc.Dropdown(id="chan", style={"width": "100%"})],
                         style={"flex": "1", "minWidth": 0}),
                html.Div([html.Label("TTL channel", style=_LBL),
                          dcc.Dropdown(id="ttl", style={"width": "100%"})],
                         style={"flex": "1", "minWidth": 0, "marginLeft": "8px"}),
            ], style={"display": "flex", "marginBottom": "6px"}),
            html.Div([html.Label("polarity", style=_LBL),
                      dcc.RadioItems(id="polarity",
                                     options=[{"label": p, "value": p} for p in ("neg", "pos", "abs")],
                                     value="neg", inline=True, **PERSIST)], style=_FIELD),
            html.Div([html.Label("threshold", style=_LBL),
                      dcc.RadioItems(id="method",
                                     options=[{"label": "k·MAD", "value": "mad"},
                                              {"label": "absolute", "value": "abs"},
                                              {"label": "k·MAD ≥ floor", "value": "mad_floor"}],
                                     value="mad", inline=True, **PERSIST)], style=_FIELD),
            html.Div([html.Label("k (MAD)", style=_LBL),
                      dcc.Slider(id="k", min=2, max=15, step=0.5, value=6,
                                 marks={2: "2", 6: "6", 10: "10", 15: "15"},
                                 tooltip={"placement": "bottom"}, **PERSIST)], style=_FIELD),
            html.Div([
                html.Div([html.Label("abs thresh (all)", style=_LBL),
                          dcc.Input(id="absth", type="number", value=20, debounce=True,
                                    style={"width": "85px"}, **PERSIST)]),
                html.Div([html.Label("refractory (ms)", style=_LBL),
                          dcc.Input(id="refr", type="number", value=2, debounce=True,
                                    style={"width": "85px"}, **PERSIST)],
                         style={"marginLeft": "10px"}),
            ], style={"display": "flex"}),
            dcc.Checklist(id="absth-sync",
                          options=[{"label": " sync — use one abs threshold for all traces",
                                    "value": "sync"}],
                          value=["sync"], style={"marginTop": "6px"},
                          labelStyle={"fontSize": "11px"}, **PERSIST),
            html.Div(id="absth-editor", style={"display": "none", "marginTop": "4px"}),
            html.Button("🎯 auto abs (per trace)", id="auto-absth", n_clicks=0,
                        style={"marginTop": "6px", "fontSize": "11px", "width": "100%"}),
            html.Div(id="absth-msg", style={"fontSize": "10px", "color": "#666",
                                            "marginTop": "3px", "wordBreak": "break-all"}),
        ]),

        # (the region & display controls now live as overlays ON the graphs, right)

        # ---- compartment: cell outputs (click to enlarge) ----
        card("Cell outputs (click to enlarge)", [
            html.Div(id="outputs-gallery",
                     style={"display": "flex", "flexWrap": "wrap", "gap": "6px",
                            "maxHeight": "300px", "overflowY": "auto",
                            "border": "1px solid #eee", "padding": "4px", "background": "#fafafa"}),
        ]),
    ]),

    # ================= RIGHT PANEL: graphs (80% width, full height) =============
    html.Div(style={"flex": "1 1 0", "minWidth": 0, "height": "100%",
                    "display": "flex", "flexDirection": "column"}, children=[
        html.Div(id="readout", style={"fontWeight": "bold", "fontSize": "12px",
                                      "padding": "2px 0", "flex": "0 0 auto"}),
        # signal + frame-sync (frame-sync row enlarged) — gets the lion's share of height.
        # Region & display controls float in the corners, hugging the graph.
        html.Div(style={"flex": "3 1 0", "minHeight": 0, "position": "relative"}, children=[
            dcc.Graph(id="time", style={"height": "100%"}, config={"responsive": True},
                      figure=blank_fig("pick a cell, then check file(s) to display")),
            # top-left (flush with the plot's left): region start — hidden when cropping
            html.Div([html.Span("start (s)", style=_OVL),
                      dcc.Input(id="region-start", type="number", debounce=True,
                                style=dict(_OVI, width="60px"))],
                     id="start-box", style=_START_OV),
            # top-right (flush with the plot's right): region end + crop. The end label+input
            # (end-fields) hide when cropping; the crop checkbox stays.
            html.Div([html.Div([html.Span("end (s)", style=_OVL),
                                dcc.Input(id="region-end", type="number", debounce=True,
                                          style=dict(_OVI, width="60px"))],
                               id="end-fields", style=_END_FIELDS),
                      dcc.Checklist(id="region-mode",       # checkbox to the RIGHT of "crop"
                                    options=[{"label": "crop", "value": "crop"}], value=[],
                                    inline=True,
                                    labelStyle={"fontSize": "10px", "marginLeft": "4px",
                                                "display": "inline-flex", "alignItems": "center",
                                                "flexDirection": "row-reverse"},
                                    inputStyle={"marginLeft": "3px"}, **PERSIST)],
                     id="end-box", style=_END_OV),
            # bottom-LEFT, same level as "stagger frame sync %". "show detected spikes" (default
            # ON) HIDES when "show binned spikes" is checked, and keeps its value for when it
            # reappears.
            html.Div([
                dcc.Checklist(id="disp-show",
                              options=[{"label": " show detected spikes", "value": "show_spikes"}],
                              value=["show_spikes"], inline=True,
                              labelStyle={"fontSize": "10px", "marginRight": "8px",
                                          "display": "inline-flex", "alignItems": "center"},
                              inputStyle={"marginRight": "3px"}),
                dcc.Checklist(id="disp-binned",
                              options=[{"label": " show binned spikes", "value": "spike_train"}],
                              value=[], inline=True,
                              labelStyle={"fontSize": "10px", "display": "inline-flex",
                                          "alignItems": "center"},
                              inputStyle={"marginRight": "3px"}),
            ], style=ov(bottom="calc(38% + 25px)", right="6px", height="20px")),   # up 1.25×height, under Im_prime
            # just above the frame-sync x-axis, right-aligned with "bin (ms)": stagger %
            html.Div([html.Span("stagger frame sync %", style=_OVL),
                      dcc.Input(id="stagger-pct", type="number", value=0, min=0, max=100, step=5,
                                debounce=True, style=dict(_OVI, width="48px"), **PERSIST)],
                     style=ov(bottom="26px", right="8px")),
        ]),
        # bottom strip (less tall): FFT at half width + ISI histogram at the other half
        html.Div(style={"flex": "1 1 0", "minHeight": 0, "display": "flex", "gap": "6px"},
                 children=[
            html.Div(dcc.Graph(id="fft", style={"height": "100%"}, config={"responsive": True},
                               figure=blank_fig("")),
                     style={"flex": "1 1 0", "minWidth": 0}),
            # ISI histogram with the spike-train bin control floated inside, below the toolbar
            html.Div([dcc.Graph(id="isi", style={"height": "100%"}, config={"responsive": True},
                                figure=blank_fig("")),
                      html.Div([html.Span("bin (ms)", style=_OVL),
                                dcc.Input(id="train-bin", type="number", value=0, min=0,
                                          debounce=True, style=dict(_OVI, width="48px"), **PERSIST)],
                               style=ov(top="34px", right="8px"))],
                     style={"flex": "1 1 0", "minWidth": 0, "position": "relative"}),
        ]),
    ]),

    # ---- invisible state + overlays ----
    dcc.Store(id="sel-cell"),
    dcc.Store(id="gallery-trigger"),
    dcc.Store(id="absth-map"),                            # {file path: per-trace abs threshold}
    dcc.Store(id="absth-seed"),                           # {file path: seed value for the editor}
    dcc.Store(id="recent-cells", storage_type="local"),   # most-recently-opened date|cell list
    dcc.Store(id="rail-sort", data={"col": "date", "dir": "desc"}),   # explorer rail sort
    dcc.Store(id="store-rev", data=0),                    # bumped when the store changes (rail refresh)
    dcc.Store(id="exp-date"),                             # explorer: selected date
    dcc.Store(id="exp-cell"),                             # explorer: selected cell (within date)
    dcc.Store(id="last-folder", storage_type="local"),   # remembers data folder across sessions
    dcc.Interval(id="once", interval=300, max_intervals=1),
    dcc.Store(id="kb-dummy"),                             # clientside keydown wiring sink
    dcc.Input(id="hover-sink", type="text", value="",     # clientside writes the hovered "kind|path"
              style={"display": "none"}),
    # ---- full-screen pop-out for an output image ----
    html.Div(id="output-modal", style={"display": "none"}, children=[
        # backdrop fills the screen BEHIND the image; clicking it (off the image) closes
        html.Div(id="modal-backdrop", n_clicks=0,
                 style={"position": "absolute", "top": 0, "left": 0, "width": "100%",
                        "height": "100%", "zIndex": 0, "cursor": "zoom-out"}),
        html.Button("✕ close", id="modal-close", n_clicks=0,
                    style={"position": "absolute", "top": "12px", "right": "16px", "zIndex": 2,
                           "fontSize": "15px", "padding": "4px 10px"}),
        html.Img(id="modal-img", style={"maxWidth": "94vw", "maxHeight": "92vh", "zIndex": 1,
                                        "position": "relative",
                                        "boxShadow": "0 0 24px #000", "background": "white"}),
    ]),

    # ================= DATA EXPLORER pop-out (dates → cells → files + JSON) =====
    html.Div(id="explorer-modal", children=[
        # header bar — left nav back to Analysis, Import centered, Open selected at right
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px",
                        "color": "white", "marginBottom": "8px", "flex": "0 0 auto",
                        "position": "relative"}, children=[
            html.Div("↰ Analysis", id="exp-close", n_clicks=0, title="back to the analysis view",
                     style={"fontWeight": "bold", "fontSize": "16px", "cursor": "pointer",
                            "color": "#9fc0ff"}),
            html.Button("←  back to day", id="exp-back", n_clicks=0,
                        style={"display": "none", "fontSize": "12px"}),
            html.Span(id="exp-breadcrumb", style={"fontSize": "13px"}),
            html.Span(id="exp-msg", style={"fontSize": "12px", "color": "#7fdc7f"}),
            html.Div(style={"flex": "1"}),
            html.Button("📥 Import data…", id="import-data", n_clicks=0,
                        style={"fontWeight": "bold", "position": "absolute", "left": "50%",
                               "transform": "translateX(-50%)"}),
            html.Button("📈 Open selected in viewer", id="exp-open-viewer", n_clicks=0,
                        style={"fontWeight": "bold"}),
        ]),
        # body: left rail (full height) · middle (browser over a hover-preview) · right detail (full height)
        html.Div(style={"flex": "1 1 0", "minHeight": 0, "display": "flex", "gap": "8px"},
                 children=[
            # left rail: a sortable + searchable 4-column table of dates
            html.Div(style={"flex": "0 0 380px", "minHeight": 0, "display": "flex",
                            "flexDirection": "column", "background": "#15151d",
                            "border": "1px solid #2a2a35", "borderRadius": "6px",
                            "overflow": "hidden"}, children=[
                # sortable column headers
                html.Div([
                    html.Button("experiment ⇅", id={"type": "rail-sort", "col": "label"},
                                n_clicks=0, style=dict(_RAILHDR, flex="1", textAlign="left")),
                    html.Button("date ⇅", id={"type": "rail-sort", "col": "date"}, n_clicks=0,
                                style=dict(_RAILHDR, **_RAIL_DATE)),
                    html.Button("# ⇅", id={"type": "rail-sort", "col": "cells"}, n_clicks=0,
                                style=dict(_RAILHDR, **_RAIL_N)),
                    html.Span(style=_RAIL_TRASH),
                ], style={"display": "flex", "gap": "4px", "padding": "4px 6px"}),
                # per-column search
                html.Div([
                    dcc.Input(id="rail-q-label", type="text", placeholder="search…", debounce=True,
                              style=dict(_RAILQ, flex="1")),
                    dcc.Input(id="rail-q-date", type="text", placeholder="date", debounce=True,
                              style=dict(_RAILQ, **_RAIL_DATE)),
                    dcc.Input(id="rail-q-cells", type="text", placeholder="#", debounce=True,
                              style=dict(_RAILQ, **_RAIL_N)),
                    html.Span(style=_RAIL_TRASH),
                ], style={"display": "flex", "gap": "4px", "padding": "0 6px 4px"}),
                # the rows (rebuilt by a callback)
                html.Div(id="exp-dates", style={"flex": "1 1 0", "overflowY": "auto"}),
            ]),
            # middle column: browser (top 2/3) over the hover preview (bottom 1/3)
            html.Div(style={"flex": "1 1 0", "minWidth": 0, "display": "flex",
                            "flexDirection": "column", "gap": "6px"}, children=[
                html.Div(style={"flex": "2 1 0", "minHeight": 0, "overflowY": "auto",
                                "background": "#23232c", "borderRadius": "6px", "padding": "10px"},
                         children=[
                    html.Div(id="exp-cards",
                             style={"display": "flex", "flexWrap": "wrap", "gap": "18px",
                                    "alignItems": "flex-start"}),
                    # files as a desktop-icon grid (wrap into rows/columns)
                    dcc.Checklist(id="exp-files", options=[], value=[], inline=True,
                                  labelStyle={"display": "inline-flex", "alignItems": "flex-start",
                                              "verticalAlign": "top", "background": "white",
                                              "borderRadius": "5px", "padding": "5px", "margin": "5px"},
                                  inputStyle={"marginRight": "5px", "marginTop": "2px"}),
                    html.Div(html.Button("", id="del-files", n_clicks=0, style={"display": "none"}),
                             id="del-files-wrap", style={"marginTop": "6px"}),
                ]),
                # bottom 1/3 of the middle column: hover preview (any graph you hover lands here)
                html.Div(id="exp-prev",
                         style={"flex": "1 1 0", "minHeight": 0, "overflow": "hidden",
                                "background": "white", "borderRadius": "6px", "padding": "6px",
                                "display": "flex", "alignItems": "center", "position": "relative",
                                "justifyContent": "center", "textAlign": "center"},
                         children=[
                    html.Img(id="exp-prev-img", style={"width": "100%", "height": "100%",
                                                       "objectFit": "contain", "display": "none"}),
                    html.Span("hover any graph to preview it here", id="exp-prev-hint",
                              style={"color": "#999", "fontSize": "12px"}),
                ]),
            ]),
            # right: detail, full window height (larger base font)
            html.Div(id="exp-detail",
                     style={"flex": "0 0 30%", "minHeight": 0, "overflowY": "auto", "fontSize": "13px",
                            "background": "white", "borderRadius": "6px", "padding": "10px"}),
        ]),
    ], style={"display": "none"}),

    # ---- delete confirmation (warning only — no password) ----
    dcc.Store(id="del-targets"),
    html.Div(id="del-modal", style={"display": "none"}, children=[
        html.Div(style={"background": "white", "borderRadius": "8px", "padding": "18px",
                        "maxWidth": "560px", "boxShadow": "0 0 40px #000"}, children=[
            html.Div("⚠️  Delete from the data store", style={
                "fontWeight": "bold", "fontSize": "15px", "color": "#b00", "marginBottom": "6px"}),
            html.Div("These are MOVED to a reversible .trash/ folder and removed from the "
                     "manifest (and the mirror on next backup). Confirm to proceed.",
                     style={"fontSize": "12px", "color": "#444", "marginBottom": "8px"}),
            html.Div(id="del-list", style={"fontSize": "12px", "fontFamily": "monospace",
                                           "maxHeight": "220px", "overflowY": "auto",
                                           "background": "#f6f6f6", "padding": "8px",
                                           "borderRadius": "4px", "marginBottom": "10px"}),
            html.Div([
                html.Button("🗑 Delete", id="del-confirm", n_clicks=0,
                            style={"fontWeight": "bold", "color": "white", "background": "#b00",
                                   "border": "none", "padding": "6px 12px", "borderRadius": "4px"}),
                html.Button("Cancel", id="del-cancel", n_clicks=0, style={"marginLeft": "8px"}),
            ]),
            html.Div(id="del-msg", style={"fontSize": "12px", "marginTop": "8px"}),
        ]),
    ]),
])


# (no auto-restore on load — the viewer starts fully blank; pick a cell to populate "files")


# ---- load first file -> channels, metadata, auto-fill region ---------------
@app.callback(Output("chan", "options"), Output("chan", "value"),
              Output("ttl", "options"), Output("ttl", "value"), Output("meta", "children"),
              Output("region-start", "value"), Output("region-end", "value"),
              Input("file", "value"), prevent_initial_call=False)
def load_meta(files):
    files = [f for f in (files or []) if f]
    if not files:
        return [], None, [], None, "No file selected.", None, None
    rec, used, err = None, None, None
    for f in files:                                  # first file that opens as a real recording
        try:
            r = get_recording(f)
            if np.isfinite(r.fs) and r.fs > 0:
                rec, used = r, f
                break
        except Exception as e:
            err = e
    if rec is None:
        msg = ("Selected file(s) aren't openable time-series recordings"
               + (f" — {err}" if err else "") + ".")
        return [], None, [], None, msg, None, None
    opts = [{"label": n, "value": n} for n in rec.channel_names]
    ttl_default = next((n for n in rec.channel_names if "ttl" in n.lower()), rec.channel_names[-1])
    fl = get_flicker(used, ttl_default)
    rstart = round(fl.t0, 2) if fl else 0.0
    rend = round(fl.t1, 2) if fl else round(rec.duration, 2)
    md = rec.metadata()
    fields = "  ·  ".join(str(md[k]) for k in
                          ["file", "protocol", "sample rate", "duration", "channels",
                           "recorded", "creator"] if k in md)
    n = len(files)
    txt = (f"{n} file{'s' if n != 1 else ''} selected (region from {os.path.basename(used)})"
           f"   ·   {fields}")
    return opts, rec.channel_names[0], opts, ttl_default, txt, rstart, rend


# ---- crop hides the start/end region boxes (the crop checkbox stays) ----------
@app.callback(Output("start-box", "style"), Output("end-fields", "style"),
              Input("region-mode", "value"), prevent_initial_call=False)
def toggle_region_boxes(mode):
    if "crop" in (mode or []):
        return dict(_START_OV, display="none"), {"display": "none"}
    return _START_OV, _END_FIELDS


# ---- "show binned spikes" hides "show detected spikes" (which keeps its value) ----
@app.callback(Output("disp-show", "style"), Input("disp-binned", "value"),
              prevent_initial_call=False)
def toggle_disp_show(binned):
    if "spike_train" in (binned or []):
        return {"display": "none"}
    return {"display": "inline-block"}


# ---- render time + fft -----------------------------------------------------
@app.callback(Output("time", "figure"), Output("fft", "figure"), Output("isi", "figure"),
              Output("readout", "children"),
              Input("file", "value"), Input("chan", "value"), Input("ttl", "value"),
              Input("polarity", "value"), Input("method", "value"), Input("k", "value"),
              Input("absth", "value"), Input("refr", "value"),
              Input("region-start", "value"), Input("region-end", "value"),
              Input("disp-show", "value"), Input("disp-binned", "value"),
              Input("stagger-pct", "value"),
              Input("region-mode", "value"), Input("train-bin", "value"),
              Input("absth-map", "data"), Input("time", "relayoutData"), prevent_initial_call=True)
def render(files, chan, ttl_name, polarity, method, k, absth, refr, rstart, rend,
           disp_show, disp_binned, stagger_pct, region_mode, train_bin, absth_map, relayout):
    files = [f for f in (files or []) if f]
    if not files:
        return blank_fig("No file selected"), blank_fig(""), blank_fig(""), "No file selected."
    files = [f for f in files if loadable(f)]          # drop stimulus/header CSVs that won't open
    if not files:
        return (blank_fig("Selected file(s) aren't time-series recordings"),
                blank_fig(""), blank_fig(""),
                "Selected file(s) aren't openable time-series recordings.")
    if not chan:
        return no_update, no_update, no_update, no_update

    det = dict(polarity=polarity, method=method, k=float(k),
               abs_threshold=float(absth) if absth is not None else None,
               refractory_s=(float(refr) / 1000.0) if refr else 0.002)
    amap = absth_map or {}                          # per-trace absolute thresholds
    multi = len(files) > 1
    try:
        stagger_frac = max(0.0, min(100.0, float(stagger_pct))) / 100.0
    except (TypeError, ValueError):
        stagger_frac = 0.0
    stagger = stagger_frac > 0
    spike_train = "spike_train" in (disp_binned or [])   # "show binned spikes" (spike-train view)
    show_spikes = "show_spikes" in (disp_show or [])     # "show detected spikes" (default on)
    crop = "crop" in (region_mode or [])   # checkbox: show only the analysis region
    tbin = float(train_bin) if train_bin else 0.0

    rec0 = get_recording(files[0])
    fs0 = rec0.fs
    y0 = get_channel(files[0], chan)
    t0_full, t1_full = 0.0, len(y0) / fs0

    rs = float(rstart) if rstart is not None else t0_full
    re_ = float(rend) if rend is not None else t1_full

    x0, x1 = t0_full, t1_full
    try:
        trig = ctx.triggered_id
    except Exception:
        trig = None
    if trig == "time" and relayout and "xaxis.range[0]" in relayout:
        x0, x1 = float(relayout["xaxis.range[0]"]), float(relayout["xaxis.range[1]"])
    x0, x1 = max(t0_full, x0), min(t1_full, x1)
    if crop:                                       # restrict view to the analysis region
        x0, x1 = max(x0, rs), min(x1, re_)
        if x1 <= x0:
            x0, x1 = rs, re_

    time_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                             row_heights=[0.625, 0.375])     # frame-sync row enlarged ×1.25
    fft_fig = go.Figure()
    per_file_rates, stim_freqs, readbits, isi_all = [], [], [], []

    # vertical stagger step for frame syncs: fraction × (first file's TTL peak-to-peak)
    ttl_step = 0.0
    if stagger and ttl_name and ttl_name != chan:
        try:
            ttl_step = stagger_frac * 1.3 * float(np.ptp(get_channel(files[0], ttl_name)))
        except Exception:
            ttl_step = 0.0

    row1_ylab = f"{chan} ({rec0.units(chan)})"
    for idx, path in enumerate(files):
        rec = get_recording(path)
        fs = rec.fs
        y = get_channel(path, chan)
        t = np.arange(len(y)) / fs
        color = PALETTE[idx % len(PALETTE)]
        name = os.path.basename(path)
        fl = get_flicker(path, ttl_name)
        stim_freqs.append(fl.freq if fl else None)

        i0, i1 = max(0, int(x0 * fs)), min(len(y), int(x1 * fs))
        if i1 <= i0:
            i0, i1 = 0, len(y)

        eff_abs = amap.get(path)                     # this trace's own value, if set
        if eff_abs is None:
            eff_abs = float(absth) if absth is not None else None
        det_i = dict(det, abs_threshold=(float(eff_abs) if eff_abs is not None else None))
        st = detect_spikes(y, fs, **det_i)
        in_reg = st.times[(st.times >= rs) & (st.times <= re_)]
        if len(in_reg) > 1:
            isi_all.append(np.diff(in_reg) * 1000.0)     # ms, for the ISI histogram
        y_disp = y
        disp_spikes = in_reg if crop else st.times        # hide excluded spikes when cropped

        # ---- ROW 1: either the analog signal, or the spike train (0/1 or binned) ----
        if spike_train:
            if tbin > 0:                                  # binned counts
                bw = tbin / 1000.0
                edges = np.arange(0.0, t1_full + bw, bw)
                counts, _ = np.histogram(disp_spikes, bins=edges)
                centers = 0.5 * (edges[:-1] + edges[1:])
                m = (centers >= x0) & (centers <= x1)
                time_fig.add_trace(go.Scatter(x=centers[m], y=counts[m], mode="lines",
                                              line_shape="hv", legendgroup=name,
                                              line=dict(width=1, color=color),
                                              opacity=(0.7 if multi else 0.9), name=name), row=1, col=1)
                row1_ylab = f"spikes / {tbin:g} ms"
            else:                                         # 0/1 impulses at spike times
                vis = disp_spikes[(disp_spikes >= x0) & (disp_spikes <= x1)]
                xx = np.repeat(vis, 3)
                yy = np.tile([0.0, 1.0, np.nan], len(vis))
                time_fig.add_trace(go.Scattergl(x=xx, y=yy, mode="lines", legendgroup=name,
                                                line=dict(width=1, color=color),
                                                opacity=(0.7 if multi else 0.9), name=name), row=1, col=1)
                row1_ylab = "spike (0/1)"
        else:
            dt, dy = minmax_decimate(t[i0:i1], y_disp[i0:i1])
            time_fig.add_trace(go.Scattergl(x=dt, y=dy, mode="lines", legendgroup=name,
                                            line=dict(width=0.6, color=color),
                                            opacity=(0.6 if multi else 0.9), name=name), row=1, col=1)
            if show_spikes:
                sp = in_reg[(in_reg >= x0) & (in_reg <= x1)]
                sp_y = y[np.clip((sp * fs).astype(int), 0, len(y) - 1)]
                time_fig.add_trace(go.Scattergl(x=sp, y=sp_y, mode="markers",
                                                marker=dict(color=(color if multi else "red"),
                                                            size=5, symbol="circle-open"),
                                                legendgroup=name, showlegend=False,
                                                name=f"{name} spikes"), row=1, col=1)
            row1_ylab = f"{chan} ({rec0.units(chan)})"

        # ---- ROW 2: frame sync per file (zeroed in excluded regions if 'zero'; optional stagger) ----
        if ttl_name and ttl_name != chan:
            try:
                ttl = get_channel(path, ttl_name)
                tt, ty = minmax_decimate((np.arange(len(ttl)) / fs)[i0:i1], ttl[i0:i1])
                time_fig.add_trace(go.Scattergl(x=tt, y=ty + idx * ttl_step, mode="lines",
                                                legendgroup=name, showlegend=False,
                                                line=dict(width=0.6, color=color),
                                                opacity=(0.6 if multi else 0.9),
                                                name=f"{name} TTL"), row=2, col=1)
            except Exception:
                pass

        rate = binned_rate(in_reg - rs, 0.0, re_ - rs)
        if rate is not None:
            per_file_rates.append(rate)
            f, pw = power_w(rate)
            fft_fig.add_trace(go.Scatter(x=f, y=power_db(pw), mode="lines", legendgroup=name,
                                         line=dict(width=(1 if multi else 2), color=color),
                                         opacity=(0.45 if multi else 1.0), name=name))
        thr_txt = (f", thr {det_i['abs_threshold']:.1f}"
                   if (method in ("abs", "mad_floor") and det_i["abs_threshold"] is not None)
                   else "")
        readbits.append(f"{name}: {len(in_reg)} spk in region" + thr_txt
                        + (f", stim {fl.freq:.2f}Hz" if fl else ", no flicker"))

    if not crop:                                   # shade excluded blocks (skip when cropped out)
        for (a, b, lbl, pos) in [(t0_full, rs, "excluded (adapting)", "bottom left"),
                                 (re_, t1_full, "excluded", "bottom right")]:
            if b > a + 1e-6:
                time_fig.add_vrect(x0=a, x1=b, fillcolor="gray", opacity=0.4, line_width=0,
                                   annotation_text=lbl, annotation_position=pos,
                                   annotation=dict(font_size=10), row="all", col=1)

    ttl_ylab = (f"{ttl_name} (staggered)" if (stagger and multi) else (ttl_name or "TTL"))
    time_fig.update_yaxes(title_text=row1_ylab, row=1, col=1)
    time_fig.update_yaxes(title_text=ttl_ylab, row=2, col=1)
    time_fig.update_xaxes(title_text="time (s)", row=2, col=1, range=[x0, x1])
    time_fig.update_layout(margin=dict(l=55, r=20, t=30, b=40), uirevision="keep",
                           showlegend=False)   # file colors are evident from the Files list

    # power spectrum: group average + stim marker
    if multi and len(per_file_rates) >= 2:
        n = min(len(r) for r in per_file_rates)
        f, pw = power_w(np.mean([r[:n] for r in per_file_rates], axis=0))
        fft_fig.add_trace(go.Scatter(x=f, y=power_db(pw), mode="lines",
                                     line=dict(width=3, color="black"), name="GROUP AVG"))
    sfreqs = [s for s in stim_freqs if s]
    if sfreqs:
        sf = float(np.mean(sfreqs))
        fft_fig.add_vline(x=sf, line_dash="dash", line_color="orange",
                          annotation_text=f"stim {sf:.2f} Hz",
                          annotation_position="bottom right",
                          annotation=dict(font=dict(size=10, color="#c60")))
    fft_fig.update_layout(
        title=dict(text="spike-train power  10·log₁₀(2|X[k]|²/N²)  (inside region)", x=0.5,
                   xanchor="center", y=0.97, yanchor="top", font=dict(size=12)),
        xaxis_title="frequency (Hz)", yaxis_title="power (dB, R=1Ω)",
        xaxis_range=[0, FMAX], margin=dict(l=55, r=15, t=34, b=40),
        legend=dict(x=0.99, y=0.97, xanchor="right", yanchor="top", font=dict(size=9),
                    bgcolor="rgba(255,255,255,0.65)", bordercolor="#ccc", borderwidth=1),
        showlegend=True)

    # ISI histogram: pooled in-region inter-spike intervals (companion to the FFT)
    if isi_all:
        isis = np.concatenate(isi_all)
        hi = np.percentile(isis, 99) if len(isis) else 0.0
        shown = isis[isis <= hi] if hi > 0 else isis
        isi_fig = go.Figure(go.Histogram(x=shown, nbinsx=60, marker_color="#3367d6"))
        isi_fig.update_layout(
            title=dict(text=f"ISI histogram ({len(isis)} intervals, ≤99th pct)", x=0.5,
                       xanchor="center", y=0.97, yanchor="top", font=dict(size=12)),
            xaxis_title="inter-spike interval (ms)", yaxis_title="count",
            margin=dict(l=50, r=12, t=34, b=40), bargap=0.03, showlegend=False)
    else:
        isi_fig = blank_fig("no spikes in region for ISI")

    view = ("spike-train" if spike_train else "analog")
    mode = (f"GROUP of {len(files)} (avg→power; {view} view"
            + ("; spikes shown" if (show_spikes and not spike_train) else "")
            + "; frame syncs overlaid)" if multi else f"single-file inspect ({view})")
    readout = f"[{mode}]  region {rs:.2f}-{re_:.2f}s  |  " + "  |  ".join(readbits)
    return time_fig, fft_fig, isi_fig, readout


# ---- auto absolute threshold: seed one value PER TRACE (k·MAD of each file) ----
@app.callback(Output("absth-seed", "data"), Output("absth-sync", "value"),
              Output("method", "value", allow_duplicate=True), Output("absth-msg", "children"),
              Input("auto-absth", "n_clicks"),
              State("file", "value"), State("chan", "value"), State("k", "value"),
              State("method", "value"), prevent_initial_call=True)
def auto_absth(_n, files, chan, k, method):
    files = [f for f in (files or []) if f and loadable(f)]
    if not files or not chan:
        return no_update, no_update, no_update, "select file(s) + a signal channel first"
    seed, bits = {}, []
    for path in files:
        try:
            y = get_channel(path, chan)
            sigma = float(np.median(np.abs(y - np.median(y))) * 1.4826)   # robust σ
            seed[path] = round(float(k) * sigma, 2)
            bits.append(f"{os.path.basename(path)}={seed[path]:g}")
        except Exception:
            pass
    if not seed:
        return no_update, no_update, no_update, "could not compute thresholds"
    # works for both "absolute" and "k·MAD ≥ floor"; only nudge plain k·MAD over to absolute
    new_method = method if method in ("abs", "mad_floor") else "abs"
    return seed, [], new_method, "auto abs (k·MAD per trace) → " + ", ".join(bits)


# ---- build the per-trace abs-threshold editor (shown when NOT synced) ----------
@app.callback(Output("absth-editor", "children"), Output("absth-editor", "style"),
              Input("absth-sync", "value"), Input("file", "value"), Input("absth-seed", "data"),
              State("absth", "value"), prevent_initial_call=False)
def build_absth_editor(sync, files, seed, single):
    if "sync" in (sync or []):
        return [], {"display": "none"}
    files = [f for f in (files or []) if f and loadable(f)]
    seed = seed or {}
    default = single if single is not None else 20
    rows = [html.Div("per-trace abs threshold:", style={"fontSize": "10px", "color": "#555",
                                                        "marginBottom": "2px"})]
    for p in files:
        rows.append(html.Div([
            html.Span(os.path.basename(p), style={"fontSize": "10px", "flex": "1",
                                                  "overflow": "hidden", "textOverflow": "ellipsis",
                                                  "whiteSpace": "nowrap"}),
            dcc.Input(id={"type": "absth-trace", "path": p}, type="number",
                      value=seed.get(p, default), debounce=True,
                      style={"width": "70px", "marginLeft": "4px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "2px"}))
    if not files:
        rows.append(html.Span("select file(s) to set per-trace thresholds",
                              style={"fontSize": "10px", "color": "#999"}))
    return rows, {"display": "block", "marginTop": "4px",
                  "borderLeft": "2px solid #3367d6", "paddingLeft": "6px"}


# ---- collect the per-trace inputs into absth-map (render reads this) ------------
@app.callback(Output("absth-map", "data"),
              Input({"type": "absth-trace", "path": ALL}, "value"),
              Input("absth-sync", "value"), Input("file", "value"),
              prevent_initial_call=True)
def collect_absth(_vals, sync, _files):
    if "sync" in (sync or []):
        return {}                                    # synced → render falls back to the single value
    amap = {}
    for item in (ctx.inputs_list[0] or []):          # each: {"id": {...,"path":p}, "value": v}
        v = item.get("value")
        if v is not None:
            try:
                amap[item["id"]["path"]] = float(v)
            except (TypeError, ValueError):
                pass
    return amap


# ---- data store: pick a cell -> load its recordings + prefill stimulus -------
@app.callback(Output("file", "options", allow_duplicate=True),
              Output("file", "value", allow_duplicate=True),
              Output("sel-cell", "data"), Output("stim-type", "value"),
              Output("stim-params", "value"), Output("recent-cells", "data"),
              Input("cell-select", "value"), State("recent-cells", "data"),
              prevent_initial_call=True)
def pick_cell(vals, recent):
    vals = vals if isinstance(vals, list) else ([vals] if vals else [])
    if not vals:
        return no_update, no_update, [], None, None, no_update
    ds = DataStore()
    opts, checked, sel_list, seen = [], [], [], set()
    stype, sparams = None, None
    for v in vals:
        date, cell = v.split("|")
        sel_list.append({"date": date, "cell": cell})
        cm = ds.cell(date, cell)
        files, default = cell_default_files(cm)      # default = openable / spike files only
        for p in files:
            if p not in seen:
                seen.add(p)
                opts.append({"label": " " + os.path.basename(p), "value": p})
        checked += default
        if stype is None:                            # prefill from the first stimulus found
            for r in cm.data.get("recordings", []):
                if r.get("stimulus"):
                    stype = r["stimulus"].get("type")
                    sparams = ", ".join(f"{k}={v}" for k, v in
                                        (r["stimulus"].get("params") or {}).items() if v is not None)
                    break
    recent = [x for x in (recent or []) if x not in vals]        # most-recent-first, de-duped
    recent = list(vals) + recent
    return opts, checked, sel_list, stype, sparams, recent[:50]


# ---- re-sort the cell(s) dropdown (incl. "opened recently") ------------------
@app.callback(Output("cell-select", "options", allow_duplicate=True),
              Input("cell-sort", "value"), Input("recent-cells", "data"),
              prevent_initial_call="initial_duplicate")
def sort_cells(sort, recent):
    return store_cell_options(sort or "recent", recent=recent)


# ---- save stimulus metadata to the cell's data recordings --------------------
@app.callback(Output("store-msg", "children"), Input("save-meta", "n_clicks"),
              State("sel-cell", "data"), State("stim-type", "value"),
              State("stim-params", "value"), prevent_initial_call=True)
def save_meta(_n, sel, stype, sparams):
    sels = sel if isinstance(sel, list) else ([sel] if sel else [])
    if not sels or not stype:
        return "pick a cell and a stimulus type first"
    ds = DataStore()
    params = parse_params(sparams)
    total, cells = 0, []
    for s in sels:
        cm = ds.cell(s["date"], s["cell"])
        n = 0
        for r in cm.data.get("recordings", []):
            if r.get("kind", "recording") == "recording" and str(r.get("file", "")).endswith((".abf", ".csv")):
                cm.set_stimulus(r["id"], stype, params, source="user"); n += 1
        cm.save(); total += n; cells.append(f"{s['date']}/{s['cell']}")
    ds.update_index()
    return f"saved stimulus '{stype}' {params} to {total} recordings in " + ", ".join(cells)


# ---- run the analysis for the selected cell(s), dispatched by their stimulus type --
@app.callback(Output("store-msg", "children", allow_duplicate=True),
              Output("gallery-trigger", "data", allow_duplicate=True),
              Input("run-cell", "n_clicks"), State("sel-cell", "data"),
              State("file", "value"), State("run-name", "value"),
              prevent_initial_call=True)
def run_cell(_n, sel, checked, run_name):
    import re
    sels = sel if isinstance(sel, list) else ([sel] if sel else [])
    if not sels:
        return "pick a cell first", no_update
    user_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", (run_name or "").strip())
    checked = [c for c in (checked or []) if c]
    ds, msgs = DataStore(), []
    for s in sels:
        tag = f"{s['date']}/{s['cell']}"
        cm = ds.cell(s["date"], s["cell"])
        stype = next((r["stimulus"].get("type") for r in cm.data.get("recordings", [])
                      if r.get("stimulus")), None)       # the cell's stimulus type (explicit metadata)
        try:
            if stype == "gaussian_noise":                # spikes×stimulus reverse correlation → STA
                nm = user_name or "sta"
                r = run_cell_noise(ds, s["date"], s["cell"], name=nm)
                sm = r.summary[0]
                msgs.append(f"{tag} [{nm}]: STA {sm['n_epochs']} epochs, "
                            f"peak {sm['peak_ms']:.1f} ms {sm['peak_sign']}")
            else:                                        # default: square-wave (flicker) ON/OFF
                nm = user_name or "flicker"
                cmdir = str(cm.dir)
                sub = [f for f in checked if f.startswith(cmdir) and f.endswith(".abf")]
                r = run_cell_flicker(ds, s["date"], s["cell"], n_shuffle=500,
                                     name=nm, include=(sub or None))
                p = r.tables["pooled_onoff"][0]
                msgs.append(f"{tag} [{nm}]: {len(r.summary)} file(s), "
                            f"{p['flicker_hz']:.1f} Hz, '{p['verdict']}'")
        except SystemExit as e:
            msgs.append(f"{tag}: {e}")
        except Exception as e:
            msgs.append(f"{tag}: error {e}")
    return "ran → " + "  |  ".join(msgs) + "  (outputs saved; see gallery)", (_n or 1)


# ---- output-image gallery for the selected cell + full-screen pop-out --------
@app.callback(Output("outputs-gallery", "children"),
              Input("cell-select", "value"), Input("gallery-trigger", "data"),
              prevent_initial_call=False)
def build_gallery(cell_val, _trig):
    vals = cell_val if isinstance(cell_val, list) else ([cell_val] if cell_val else [])
    if not vals:
        return [html.Span("pick a cell to see its output images",
                          style={"color": "#888", "fontSize": "12px"})]
    thumbs = []
    for v in vals:
        try:
            date, cell = v.split("|")
            imgs = output_gallery(date, cell)
            if len(vals) > 1:                        # label each cell's group when several picked
                thumbs.append(html.Div(f"{date}/{cell}", style={"width": "100%", "fontSize": "11px",
                                                                "fontWeight": "bold", "color": "#555"}))
            thumbs += imgs
        except Exception as e:
            thumbs.append(html.Span(f"(no outputs: {e})", style={"color": "#888", "fontSize": "12px"}))
    return thumbs


@app.callback(Output("output-modal", "style"), Output("modal-img", "src"),
              Input({"type": "out-thumb", "src": ALL}, "n_clicks"),
              Input({"type": "raw-thumb", "src": ALL}, "n_clicks"),
              Input("modal-close", "n_clicks"), Input("modal-backdrop", "n_clicks"),
              prevent_initial_call=True)
def toggle_modal(_thumbs, _raw, _close, _backdrop):
    trig = ctx.triggered_id
    if trig in ("modal-close", "modal-backdrop"):         # ✕, or click off the image
        return {"display": "none"}, no_update
    if isinstance(trig, dict) and ctx.triggered and ctx.triggered[0].get("value"):  # real click
        if trig.get("type") == "out-thumb":                       # processed figure (PNG on disk)
            return _MODAL_SHOWN, _img_datauri(trig["src"])
        if trig.get("type") == "raw-thumb":                       # raw trace -> render full waveform
            uri = big_waveform_datauri(trig["src"])
            if uri:
                return _MODAL_SHOWN, uri
    return no_update, no_update


# ---- clientside: Escape closes pop-outs; hovering any graph fills the preview pane ----
app.clientside_callback(
    """
    function(n) {
        if (!window._neitzEsc) {
            window._neitzEsc = true;
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape' || e.keyCode === 27) {
                    var im = document.getElementById('output-modal');
                    if (im && im.style.display !== 'none') {
                        var b = document.getElementById('modal-close'); if (b) { b.click(); }
                        return;
                    }
                    var ex = document.getElementById('explorer-modal');
                    if (ex && ex.style.display !== 'none') {
                        var c = document.getElementById('exp-close'); if (c) { c.click(); }
                    }
                }
            });
            // hover any graph thumbnail (.gprev) -> ask the server to render the real
            // (labelled) plot into the preview pane, by writing its "kind|path" to a hidden input
            document.addEventListener('mouseover', function(e) {
                var t = e.target;
                if (t && t.tagName === 'IMG' && t.classList && t.classList.contains('gprev')) {
                    var ps = t.getAttribute('data-ps');
                    var inp = document.getElementById('hover-sink');
                    if (ps && inp && inp.value !== ps) {
                        var setter = Object.getOwnPropertyDescriptor(
                            window.HTMLInputElement.prototype, 'value').set;
                        setter.call(inp, ps);
                        inp.dispatchEvent(new Event('input', {bubbles: true}));
                    }
                }
            });
            // leaving the open cell(s) dropdown closes it (blur the react-select input)
            document.addEventListener('mouseout', function(e) {
                var wrap = document.getElementById('cell-select-wrap');
                if (!wrap) return;
                var to = e.relatedTarget;
                if (wrap.contains(e.target) && (!to || !wrap.contains(to))) {
                    var inp = wrap.querySelector('input');
                    if (inp) { inp.blur(); }
                }
            });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("kb-dummy", "data"),
    Input("once", "n_intervals"),
)


# ============================================================
# Data Explorer callbacks
# ============================================================
@app.callback(Output("explorer-modal", "style"),
              Output("exp-date", "data", allow_duplicate=True),
              Output("exp-cell", "data", allow_duplicate=True),
              Input("open-explorer", "n_clicks"), Input("exp-close", "n_clicks"),
              prevent_initial_call=True)
def toggle_explorer(_open, _close):
    if ctx.triggered_id == "exp-close":
        return {"display": "none"}, no_update, no_update
    dates = sorted({c["date"] for c in DataStore().index()}, reverse=True)
    return _EXPLORER_SHOWN, (dates[0] if dates else None), None


@app.callback(Output("exp-date", "data"), Output("exp-cell", "data", allow_duplicate=True),
              Input({"type": "exp-date", "date": ALL}, "n_clicks"), prevent_initial_call=True)
def exp_pick_date(_clicks):
    t = ctx.triggered_id
    if isinstance(t, dict) and ctx.triggered and ctx.triggered[0].get("value"):
        return t["date"], None
    return no_update, no_update


@app.callback(Output("exp-cell", "data"),
              Input({"type": "exp-cell", "cell": ALL}, "n_clicks"), prevent_initial_call=True)
def exp_pick_cell(_clicks):
    t = ctx.triggered_id
    if isinstance(t, dict) and ctx.triggered and ctx.triggered[0].get("value"):
        return t["cell"]
    return no_update


@app.callback(Output("exp-cell", "data", allow_duplicate=True),
              Input("exp-back", "n_clicks"), prevent_initial_call=True)
def exp_back(_n):
    return None


@app.callback(Output("exp-cards", "children"),
              Output("exp-files", "options"), Output("exp-files", "value"),
              Output("exp-detail", "children"), Output("exp-breadcrumb", "children"),
              Output("exp-back", "style"), Output("exp-prev", "children"),
              Input("exp-date", "data"), Input("exp-cell", "data"), prevent_initial_call=True)
def exp_render(date, cell):
    # reset the hover-preview on every navigation (no stale graph showing)
    prev = [html.Img(id="exp-prev-img", style={"width": "100%", "height": "100%",
                                               "objectFit": "contain", "display": "none"}),
            html.Span("hover any graph to preview it here", id="exp-prev-hint",
                      style={"color": "#999", "fontSize": "12px"})]
    if not date:
        return ([no_update] * 6) + [prev]
    bc = explorer_breadcrumb(date, cell)
    if not cell:                                       # DAY view: cell thumbnails
        hint = [html.Div("select a cell to see its files + manifest",
                         style={"color": "#999", "fontSize": "12px"})]
        return explorer_day_cards(date), [], [], hint, bc, {"display": "none"}, prev
    # CELL view: file checklist + manifest JSON
    return ([], explorer_file_options(date, cell), [],
            explorer_detail(date, cell), bc, {"display": "inline-block", "fontSize": "12px"}, prev)


# ---- explorer rail: sort headers + per-column search rebuild the date rows -----
@app.callback(Output("rail-sort", "data"),
              Input({"type": "rail-sort", "col": ALL}, "n_clicks"),
              State("rail-sort", "data"), prevent_initial_call=True)
def set_rail_sort(_clicks, cur):
    t = ctx.triggered_id
    if not (isinstance(t, dict) and ctx.triggered and ctx.triggered[0].get("value")):
        return no_update
    cur = cur or {"col": "date", "dir": "desc"}
    if cur.get("col") == t["col"]:
        return {"col": t["col"], "dir": ("asc" if cur.get("dir") == "desc" else "desc")}
    return {"col": t["col"], "dir": "asc"}


@app.callback(Output("exp-dates", "children"),
              Input("rail-sort", "data"), Input("rail-q-label", "value"),
              Input("rail-q-date", "value"), Input("rail-q-cells", "value"),
              Input("exp-date", "data"), Input("store-rev", "data"),
              prevent_initial_call=False)
def rebuild_rail(sort, ql, qd, qc, active, _rev):
    return explorer_dates_body(active=active, sort=sort,
                               search={"label": ql, "date": qd, "cells": qc})


# ---- hover preview: render the REAL (labelled) plot for the hovered thumbnail -----
@app.callback(Output("exp-prev", "children", allow_duplicate=True),
              Input("hover-sink", "value"), prevent_initial_call=True)
def render_hover(ps):
    if not ps or "|" not in ps:
        return no_update
    kind, path = ps.split("|", 1)
    if kind == "wave":
        uri = big_waveform_datauri(path) if loadable(path) else None
    else:
        uri = _img_datauri(path) if os.path.exists(path) else None
    if not uri:
        return no_update
    return [html.Img(src=uri, style={"width": "100%", "height": "100%", "objectFit": "contain"}),
            html.Div(os.path.basename(path),
                     style={"position": "absolute", "bottom": "3px", "left": "8px",
                            "fontSize": "12px", "color": "#333", "fontWeight": "bold",
                            "background": "rgba(255,255,255,0.75)", "padding": "0 4px",
                            "borderRadius": "3px"})]


@app.callback(Output("file", "options", allow_duplicate=True),
              Output("file", "value", allow_duplicate=True),
              Output("explorer-modal", "style", allow_duplicate=True),
              Input("exp-open-viewer", "n_clicks"), State("exp-files", "value"),
              prevent_initial_call=True)
def exp_open_viewer(_n, sel):
    sel = [s for s in (sel or []) if s]
    if not sel:
        return no_update, no_update, no_update
    return file_options(sel), sel, {"display": "none"}


# ---- the 🗑 for the checked file(s) (appears at the bottom of the selection) -------
@app.callback(Output("del-files", "style"), Output("del-files", "children"),
              Input("exp-files", "value"), prevent_initial_call=False)
def del_files_button(sel):
    sel = [s for s in (sel or []) if s]
    base = {"color": "white", "background": "#b00", "border": "none", "borderRadius": "4px",
            "padding": "4px 10px", "fontWeight": "bold", "fontSize": "12px", "cursor": "pointer"}
    if not sel:
        return {"display": "none"}, ""
    return dict(base, display="inline-block"), f"🗑 delete {len(sel)} selected"


# ---- delete: open the (warning-only) confirmation modal -------------------------
#   targets: a whole DAY {kind:"date", date} · a cell's checked files / a figure {date,cell,paths}
@app.callback(Output("del-modal", "style"), Output("del-targets", "data"),
              Output("del-list", "children"),
              Input({"type": "del-date", "date": ALL}, "n_clicks"),
              Input({"type": "del-output", "src": ALL}, "n_clicks"),
              Input("del-files", "n_clicks"),
              State("exp-files", "value"), State("exp-date", "data"), State("exp-cell", "data"),
              prevent_initial_call=True)
def open_delete(_ddates, _douts, _dfiles, sel, date, cell):
    trig = ctx.triggered_id
    fired = bool(ctx.triggered and ctx.triggered[0].get("value"))
    if isinstance(trig, dict) and trig.get("type") == "del-date":       # a whole day
        if not fired:
            return (no_update,) * 3
        d = trig["date"]
        cells = [c for c in DataStore().index() if c["date"] == d]
        lst = [html.Div(f"ALL of {d} — {len(cells)} cell(s):",
                        style={"fontWeight": "bold", "color": "#b00"})] \
            + [html.Div(f"{c['cell']} — {c.get('label') or ''}") for c in cells]
        return _DEL_SHOWN, {"kind": "date", "date": d}, lst
    if isinstance(trig, dict) and trig.get("type") == "del-output":     # one figure (+ siblings)
        if not fired:
            return (no_update,) * 3
        p = Path(trig["src"])
        paths = sorted(str(x) for x in p.parent.glob(p.stem + ".*"))
        lst = [html.Div("output figure (+ same-name pdf/svg):", style={"fontWeight": "bold"})] \
            + [html.Div(os.path.basename(x)) for x in paths]
        return _DEL_SHOWN, {"date": date, "cell": cell, "paths": paths}, lst
    if trig == "del-files":                                             # the checked recordings
        s = [x for x in (sel or []) if x]
        if not fired or not s or not date or not cell:
            return (no_update,) * 3
        lst = [html.Div("recordings:", style={"fontWeight": "bold"})] \
            + [html.Div(os.path.basename(p)) for p in s]
        return _DEL_SHOWN, {"date": date, "cell": cell, "paths": s}, lst
    return (no_update,) * 3


@app.callback(Output("del-modal", "style", allow_duplicate=True),
              Input("del-cancel", "n_clicks"), prevent_initial_call=True)
def cancel_delete(_n):
    return {"display": "none"}


@app.callback(Output("del-msg", "children", allow_duplicate=True),
              Output("del-modal", "style", allow_duplicate=True),
              Output("store-rev", "data", allow_duplicate=True),
              Output("exp-cards", "children", allow_duplicate=True),
              Output("exp-files", "options", allow_duplicate=True),
              Output("exp-files", "value", allow_duplicate=True),
              Output("exp-detail", "children", allow_duplicate=True),
              Output("cell-select", "options", allow_duplicate=True),
              Input("del-confirm", "n_clicks"), State("del-targets", "data"),
              State("store-rev", "data"), prevent_initial_call=True)
def confirm_delete(_n, targets, rev):
    nu = (no_update,) * 6              # store-rev, cards, files-opts, files-val, detail, cells
    if not targets:
        return "nothing to delete", no_update, *nu
    rev = (rev or 0) + 1               # bump -> rail rebuilds (date removed)
    try:
        if targets.get("kind") == "date":
            delete_date(targets["date"])
            msg = html.Span(f"✓ deleted all of {targets['date']}", style={"color": "#070"})
            return (msg, {"display": "none"}, rev,
                    [html.Div("pick a date", style={"color": "#999", "fontSize": "12px"})],
                    [], [], [], store_cell_options())
        date, cell = targets["date"], targets["cell"]
        removed, _ = delete_files(date, cell, targets["paths"])
        msg = html.Span(f"✓ deleted {len(removed)} item(s)", style={"color": "#070"})
        return (msg, {"display": "none"}, rev, no_update,
                explorer_file_options(date, cell), [],
                explorer_detail(date, cell), store_cell_options())
    except Exception as e:
        return f"delete error: {e}", no_update, *nu


# ---- import new experiment data into the store (copies + auto-groups by date) --
# (the Import button now lives in the Data Explorer window)
@app.callback(Output("cell-select", "options", allow_duplicate=True),
              Output("store-rev", "data", allow_duplicate=True),
              Output("exp-msg", "children"),
              Output("store-msg", "children", allow_duplicate=True),
              Input("import-data", "n_clicks"),
              State("stim-type", "value"), State("stim-params", "value"),
              State("store-rev", "data"), prevent_initial_call=True)
def import_data(_n, stype, sparams, rev):
    import re
    folder = native_choose_folder()
    if not folder:
        return no_update, no_update, no_update, no_update
    abfs = sorted(glob.glob(os.path.join(folder, "**", "*.abf"), recursive=True))
    if not abfs:
        return no_update, no_update, f"no .abf files found in {folder}", no_update
    by_date = {}
    for f in abfs:
        m = re.match(r"(\d{4})_(\d{2})_(\d{2})", os.path.basename(f))
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "undated"
        by_date.setdefault(date, []).append(f)

    stim = None
    if stype and stype != "(none)":
        stim = {"type": stype, "params": parse_params(sparams), "source": "user"}

    ds = DataStore()
    made = []
    for date, fs in sorted(by_date.items()):
        cm = ds.new_cell(date, label=os.path.basename(folder.rstrip("/")))
        for f in fs:
            cm.add_recording(f, label=os.path.basename(f), stimulus=stim)
        cm.save()
        made.append(f"{date}/{cm.data['cell']} ({len(fs)})")
    ds.update_index()
    msg = f"imported {len(abfs)} recordings → " + ", ".join(made)
    return (store_cell_options(), (rev or 0) + 1, msg, msg)   # bump store-rev -> rail rebuilds


# ---- back up the whole data store to this computer's mirror -------------------
@app.callback(Output("store-msg", "children", allow_duplicate=True),
              Input("backup-mirror", "n_clicks"), prevent_initial_call=True)
def backup_mirror(_n):
    from neitz.dataio import mirror_dir, mirror_store
    if mirror_dir() is None:
        return "no mirror configured — set one with `neitz mirror --set PATH`"
    try:
        info = mirror_store()
        return f"backed up store → {info['dst']}  (via {info['method']})"
    except Exception as e:
        return f"backup error: {e}"


if __name__ == "__main__":
    app.run(debug=False, port=8050)
