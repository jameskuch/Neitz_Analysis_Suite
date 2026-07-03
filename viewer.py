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
    # drop the f=0 (DC) bin: the mean was subtracted so X[0]≈0 → a spurious ~−80 dB spike
    # that drags the y-axis. Keep only 0 < f ≤ FMAX.
    keep = (f > 0) & (f <= FMAX)
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


_IMG_CACHE: dict = {}                                  # (path, mtime) -> data-URI (poll re-renders cheaply)


def _img_datauri(path):
    try:
        mt = os.path.getmtime(str(path))
    except OSError:
        mt = 0
    key = (str(path), mt)
    uri = _IMG_CACHE.get(key)
    if uri is None:
        with open(path, "rb") as f:
            uri = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        if len(_IMG_CACHE) > 300:
            _IMG_CACHE.clear()
        _IMG_CACHE[key] = uri
    return uri


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
_EXPLBL = {"fontSize": "12px", "fontWeight": "bold", "color": "#666", "display": "block",
           "marginBottom": "2px"}     # field label inside the Data Explorer detail pane


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
# region start/end overlays: start flush with the plot's left (l margin), end flush with the
# plot's right (r margin). Shared so a callback can hide them when cropping.
_OV_H = "16px"   # shared overlay-box height: start / end+crop / show-spikes / stagger all equal
_START_OV = ov(top="34px", left="78px", height=_OV_H)
_END_OV = ov(top="34px", right="20px", height=_OV_H, justifyContent="flex-end")
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
        dx, dy = minmax_decimate(np.arange(len(y)), y, n_target=1000)  # min/max keeps spikes
        fig = Figure(figsize=(width_in, height_in), dpi=64)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.plot(dx, dy, color="#27408b", linewidth=0.4)
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
        dt, dy = minmax_decimate(t, y, n_target=4000)   # min/max keeps spikes (matches main view)
        fig = Figure(figsize=(11, 4.2), dpi=110)
        ax = fig.add_subplot(111)
        ax.plot(dt, dy, color="#27408b", linewidth=0.6)
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


# Rail columns are sized by CSS vars (--rc1/2/3 on #exp-rail) so the header, search row and every
# data row stay in lock-step; drag handles (assets/rail-resize.js) update the vars + persist them.
_RAIL_C1 = {"width": "var(--rc1)", "minWidth": 0, "boxSizing": "border-box", "position": "relative"}
_RAIL_DATE = {"width": "var(--rc2)", "minWidth": 0, "boxSizing": "border-box", "position": "relative"}
_RAIL_N = {"width": "var(--rc3)", "minWidth": 0, "textAlign": "center", "boxSizing": "border-box",
           "position": "relative"}
_RAIL_DEAD = {"flex": "1 1 0", "minWidth": "8px"}          # dead spacer column (not resizable)
_RAIL_TRASH = {"flex": "0 0 22px"}                          # trash column (not resizable)
_RZ_HANDLE = {"position": "absolute", "top": 0, "right": "-3px", "width": "7px", "height": "100%",
              "cursor": "col-resize", "zIndex": 5}          # drag handle at a column's right edge
_RAILHDR = {"background": "#222", "color": "#bcd", "border": "none", "cursor": "pointer",
            "fontSize": "12px", "padding": "1px 4px", "borderRadius": "3px", "fontWeight": "bold"}
_RAILQ = {"fontSize": "12px", "padding": "2px 4px", "boxSizing": "border-box", "minWidth": 0,
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
                         style=dict(_RAIL_C1, overflow="hidden", textOverflow="ellipsis",
                                    whiteSpace="nowrap")),
                html.Div(r["date"], style=_RAIL_DATE),
                html.Div(str(r["n"]), style=_RAIL_N),
                html.Div(style=_RAIL_DEAD),
            ], id={"type": "exp-date", "date": r["date"]}, n_clicks=0,
               style={"flex": "1", "minWidth": 0, "display": "flex", "gap": "4px",
                      "alignItems": "center", "cursor": "pointer"}),
            html.Button("🗑", id={"type": "del-date", "date": r["date"]}, n_clicks=0,
                        title=f"delete all of {r['date']}",
                        style=dict(_RAIL_TRASH, border="none", background="none", color="#e66",
                                   cursor="pointer", fontSize="12px", padding="0")),
        ], style={"padding": "6px 8px", "fontSize": "13px", "color": "white",
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
        stim = {"sq_wave": "sq wave", "flicker": "sq wave"}.get(stim, stim)   # friendly display
        thumb = html.Div([
            html.Img(src=spark, className="gprev", style=dict(_THUMB_IMG, width="220px"),
                     **{"data-ps": "wave|" + p}) if spark
            else html.Div("—", style={"height": "62px", "color": "#999"}),
            html.Div(os.path.basename(p), style={"fontSize": "11px", "wordBreak": "break-all",
                                                 "color": "#e3e9ff", "fontWeight": "bold"}),
            html.Div(f"stim: {stim}" if stim else "stim: —",
                     style={"fontSize": "10px", "color": "#9aa7c0"}),
        ], style={"display": "inline-block", "verticalAlign": "top",
                  "background": "#2a2a36", "borderRadius": "4px", "padding": "2px 4px"})
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


def _fact_row(label, value):
    return html.Tr([
        html.Td(label, style={"color": "#777", "padding": "3px 12px 3px 0", "verticalAlign": "top",
                              "whiteSpace": "nowrap"}),
        html.Td(value, style={"padding": "3px 0"})])


def explorer_detail(date, cell):
    """Lower part of the detail pane: read-only recording facts (from the files) + output
    figures (click to enlarge, 🗑 to delete) + the raw manifest tucked in a disclosure.
    The editable cell metadata above this is static layout, filled by `fill_cell_meta`."""
    cm = DataStore().cell(date, cell)

    # recording facts read straight from the files (sample rate / duration / channels)
    files = [p for p in cell_data_files(cm) if loadable(p)]
    rates, durs, chans = set(), [], None
    for p in files:
        try:
            rec = get_recording(p)
            rates.add(round(rec.fs))
            durs.append(rec.duration)
            if chans is None:
                chans = list(getattr(rec, "channel_names", []) or [])
        except Exception:
            pass
    rec_names = ", ".join(os.path.basename(p) for p in files) or "none"
    rate_txt = " · ".join(f"{r/1000:g} kHz" for r in sorted(rates)) if rates else "—"
    if durs and round(min(durs)) != round(max(durs)):
        dur_txt = f"{min(durs):.0f}–{max(durs):.0f} s per sweep"
    elif durs:
        dur_txt = f"{durs[0]:.0f} s"
    else:
        dur_txt = "—"
    facts = html.Table([
        _fact_row("Recordings", f"{len(files)} · {rec_names}"),
        _fact_row("Sample rate", rate_txt),
        _fact_row("Duration", dur_txt),
        _fact_row("Channels", " · ".join(chans) if chans else "—"),
    ], style={"fontSize": "14px", "borderCollapse": "collapse", "marginBottom": "10px"})

    # processed output figures — DISK is the source of truth (glob the PNGs); group them by the
    # analysis (run name from the Analysis View), each group under its title + a horizontal rule.
    outdir = cm.dir / "outputs"
    pngs = sorted(outdir.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if outdir.exists() else []
    label_by_analysis = {o["analysis"]: o.get("label")     # analysis key -> friendly run name
                         for o in cm.data.get("outputs", []) if o.get("label")}

    def _out_thumb(p):                                     # one figure tile (checkbox + image + 🗑)
        cap = html.Span(p.name, title=str(p.relative_to(outdir)),
                        style={"fontSize": "9px", "flex": "1", "overflow": "hidden",
                               "textOverflow": "ellipsis", "whiteSpace": "nowrap"})
        return html.Div([
            dcc.Checklist(id={"type": "out-check", "src": str(p)},      # select for multi-delete
                          options=[{"label": "", "value": str(p)}], value=[], className="out-check",
                          style={"position": "absolute", "top": "3px", "left": "3px", "zIndex": 3}),
            html.Img(src=_img_datauri(str(p)), className="gprev",
                     id={"type": "out-thumb", "src": str(p)}, n_clicks=0,
                     **{"data-ps": "img|" + str(p)},
                     style={"height": "64px", "border": "1px solid #ccc", "cursor": "pointer",
                            "background": "white", "display": "block"}),
            html.Div([
                html.Div(cap, style={"flex": "1", "minWidth": "0"}),
                html.Button("🗑", id={"type": "del-output", "src": str(p)}, n_clicks=0,
                            title="delete this figure",
                            style={"fontSize": "13px", "padding": "0 3px", "color": "#ff7a7a",
                                   "border": "none", "background": "none", "cursor": "pointer"}),
            ], style={"display": "flex", "alignItems": "center", "maxWidth": "130px"}),
        ], style={"margin": "3px", "position": "relative"})

    groups = {}                                            # analysis folder -> [png, …] (newest first)
    for p in pngs:
        groups.setdefault(p.relative_to(outdir).parts[0], []).append(p)
    out_groups = []
    for i, (analysis, ps) in enumerate(groups.items()):
        title = label_by_analysis.get(analysis) or analysis
        out_groups.append(html.Div([
            (html.Hr(style={"border": "none", "borderTop": "1px solid #3a3a46", "margin": "8px 0 4px"})
             if i > 0 else None),
            html.Div(f"{title}  ·  {len(ps)} file(s)", title=title,
                     style={"fontSize": "11px", "fontWeight": "bold", "color": "#9fb0ff",
                            "margin": "2px 0"}),
            html.Div([_out_thumb(p) for p in ps], style={"display": "flex", "flexWrap": "wrap"}),
        ]))

    return [
        html.Div("Recording facts — read from the files",
                 style={"fontWeight": "bold", "fontSize": "13px", "color": "#555",
                        "marginTop": "4px"}),
        facts,
        html.Div([
            html.Span("Outputs — grouped by analysis (drag to select · ☑ + button or 🗑 to delete)",
                      style={"fontWeight": "bold", "fontSize": "13px", "color": "#555"}),
            html.Button("📂 Open in Finder", id="open-outputs-finder", n_clicks=0,
                        title="reveal the selected figures' location(s) in Finder",
                        style={"marginLeft": "10px", "fontSize": "11px", "padding": "1px 8px",
                               "color": "#cfe3ff", "background": "#2f3142",
                               "border": "1px solid #555", "borderRadius": "4px", "cursor": "pointer"})
            if out_groups else None,
            html.Button("🗑 Delete selected", id="del-outputs", n_clicks=0,
                        style={"marginLeft": "6px", "fontSize": "11px", "padding": "1px 8px",
                               "color": "#ff7a7a", "background": "#2f3142",
                               "border": "1px solid #555", "borderRadius": "4px", "cursor": "pointer"})
            if out_groups else None,
            html.Span(id="finder-msg", style={"marginLeft": "8px", "fontSize": "11px", "color": "#7fdc7f"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "4px", "flexWrap": "wrap"}),
        html.Div(out_groups or [html.Span("none yet", style={"color": "#999", "fontSize": "13px"})],
                 id="exp-outputs-grid", style={"marginBottom": "8px"}),
        html.Details([
            html.Summary("raw manifest (JSON)",
                         style={"fontSize": "12px", "color": "#888", "cursor": "pointer"}),
            html.Div(_json_tree(cm.data, top=True),
                     style={"maxHeight": "30vh", "overflowY": "auto", "border": "1px solid #eee",
                            "padding": "6px", "background": "#fbfbfb", "marginTop": "4px"}),
        ]),
    ]


def explorer_breadcrumb(date, cell):
    # 📂 date is itself the "back to day" control (the old button is gone): click it to
    # drop back to the all-cells view. A left arrow separates it from the current cell.
    parts = [html.Span(f"📂 {date}", id="exp-back", n_clicks=0,
                       title="back to the day (all cells)",
                       className=("glow-target" if cell else ""),
                       style={"fontWeight": "bold",
                              "cursor": "pointer" if cell else "default",
                              "color": "#cfe0ff" if cell else "inherit"})]
    if cell:
        parts += [html.Span("  ←  ", style={"color": "#9fc0ff", "fontWeight": "bold",
                                            "fontSize": "15px"}),
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
                   "width": "100%", "height": "100%", "background": "#15151d",
                   "zIndex": 2500, "flexDirection": "column", "padding": "10px",
                   "boxSizing": "border-box"}

_DEL_SHOWN = {"display": "flex", "position": "fixed", "top": 0, "left": 0,
              "width": "100%", "height": "100%", "background": "rgba(0,0,0,0.6)",
              "zIndex": 2800, "alignItems": "center", "justifyContent": "center"}


# ============================================================
# App
# ============================================================
# background-callback manager: Run Analysis (analysis + 4K kaleido exports, ~1-2 min) runs in a
# worker process so it doesn't block the UI, and its outputs auto-update the front end when done.
import tempfile as _tempfile
import diskcache as _diskcache
from dash import DiskcacheManager as _DiskcacheManager
_bg_manager = _DiskcacheManager(_diskcache.Cache(os.path.join(_tempfile.gettempdir(), "neitz_dash_bg")))

app = Dash(__name__, suppress_callback_exceptions=True,   # detail-pane buttons (e.g. del-outputs)
           background_callback_manager=_bg_manager)       # are created dynamically by explorer_detail
app.title = "Neitz ABF Viewer"
_files = discover_abf()


def nav_toggle(active, dark=False):
    """Upper view switcher: 'Analysis View' (left) … 'Data Explorer' (right), always both. Each sits
    in a big rounded-rect pill whose gradient (a blurred rounded-rectangle behind the text — see the
    .nav-pill CSS) is small at rest and blooms a little on hover, feathering into the background with
    no clipped edges. The SELECTED view gets an inner text glow (yellow / blue) and isn't clickable;
    the OTHER is the clickable target (ids open-explorer / exp-close, unchanged)."""
    def pill(label, which, selected, click_id):
        cls = f"nav-pill nav-{which}" + (" nav-sel" if selected else " nav-clickable")
        if selected:                                    # the current view: no arrow, not clickable
            return html.Span(label, className=cls)
        arrow_l = html.Span("➤", style={"display": "inline-block", "transform": "scaleX(-1)",
                                         "marginRight": "7px"})
        arrow_r = html.Span("➤", style={"marginLeft": "7px"})
        kids = [arrow_l, label] if which == "av" else [label, arrow_r]   # arrow points its direction
        return html.Span(kids, id=click_id, n_clicks=0, className=cls,
                         title=("open the Data Explorer" if which == "de" else "go to the Analysis View"))
    av = pill("Analysis View", "av", active == "analysis", "exp-close")
    de = pill("Data Explorer", "de", active == "explorer", "open-explorer")
    return html.Div([av, de], style={"display": "flex", "justifyContent": "flex-start",
                                     "alignItems": "center",
                                     "gap": "12px", "flexWrap": "wrap",
                                     "marginTop": "16px"})   # lower the row (room above for the gradient)

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "display": "flex", "gap": "10px",
           "height": "100vh", "padding": "8px", "boxSizing": "border-box"},
    children=[

    # ================= LEFT SIDEBAR: controls (drag the splitter to resize) =====
    html.Div(id="sidebar", style={"flex": "0 0 20%", "maxWidth": "50%", "minWidth": "160px",
                    "height": "100%", "overflowY": "auto", "paddingRight": "6px",
                    "boxSizing": "border-box"}, children=[

        # upper-left view switcher (consistent layout in both windows)
        html.Div([
            nav_toggle("analysis"),
        ], style={"marginBottom": "8px"}),

        # ---- compartment: data store (cell select + files + stimulus) ----
        card("Data store", [

            # cell(s) label + inline sort control on the same row (sort = dropdown order)
            html.Div([
                html.Label("cell(s)", style=dict(_LBL, marginBottom=0)),
                dcc.RadioItems(id="cell-sort", value="recent", inline=True,
                               options=[{"label": "recent", "value": "recent"},
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

            # stimulus + cell metadata editing now lives in the Data Explorer; the analysis
            # view only RUNS the analysis (dispatched by the cell's saved stimulus type).
            html.Div(html.Button("▶ Run analysis", id="run-cell", n_clicks=0,
                                 style={"width": "100%", "marginTop": "2px"})),
            html.Div([html.Label("run by the cell's stimulus type · run name keeps a variant "
                                 "(blank = auto)", style=dict(_LBL, fontWeight="normal")),
                      dcc.Input(id="run-name", type="text", value="", debounce=True,
                                placeholder="auto (sq wave / sta)",
                                style={"width": "100%", "boxSizing": "border-box"})],
                     style={"marginTop": "6px"}),
            html.Div(id="store-msg", style={"marginTop": "6px", "minHeight": "14px"}),
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
            # horizontal rule under the channel pickers, above polarity / spike-detect algorithm
            html.Div(style={"borderTop": "1px solid #ccc", "margin": "2px 0 8px"}),
            # polarity | spike detect algorithm — side by side, separated by a vertical divider
            html.Div([
                html.Div([html.Label("polarity", style=_LBL),
                          dcc.RadioItems(id="polarity",
                                         options=[{"label": p, "value": p} for p in ("neg", "pos", "abs")],
                                         value="neg", inline=True, **PERSIST)],
                         style={"flex": "0 0 auto"}),
                html.Div(style={"borderLeft": "1px solid #ccc", "alignSelf": "stretch",
                                "margin": "0 10px"}),
                html.Div([html.Label("spike detect algorithm", style=_LBL),
                          dcc.RadioItems(id="method",
                                         options=[{"label": "k·MAD", "value": "mad"},
                                                  {"label": "absolute", "value": "abs"},
                                                  {"label": "k·MAD ≥ floor", "value": "mad_floor"},
                                                  {"label": "MATLAB (Sara)", "value": "matlab"}],
                                         value="mad", inline=True, **PERSIST)],
                         style={"flex": "1", "minWidth": 0}),
            ], style=dict(_FIELD, display="flex", alignItems="flex-start")),
            html.Div(id="method-note", style={"fontSize": "10px", "color": "#3367d6",
                                              "marginTop": "-2px", "marginBottom": "4px"}),
            # k·MAD slider (narrowed) with refractory to its right
            html.Div([
                html.Div([html.Label("k (MAD)", style=_LBL),
                          dcc.Slider(id="k", min=2, max=15, step=0.5, value=6,
                                     marks={2: "2", 6: "6", 10: "10", 15: "15"},
                                     tooltip={"placement": "bottom"}, **PERSIST)],
                         id="k-wrap", style={"flex": "1", "minWidth": 0}),
                html.Div([html.Label("refractory (ms)", style=_LBL),
                          dcc.Input(id="refr", type="number", value=2, debounce=True,
                                    style={"width": "58px"}, **PERSIST)],
                         style={"flex": "0 0 auto", "marginLeft": "12px"}),
            ], style=dict(_FIELD, display="flex", alignItems="flex-end")),
            # "abs thresh (all)" box removed — the per-trace grid replaces it. #absth is kept
            # HIDDEN as the carrier of the shared (synced) value that mirrors into the boxes.
            dcc.Input(id="absth", type="number", value=20, debounce=True,
                      style={"display": "none"}, **PERSIST),
            # sync toggle (per-box "auto" buttons in the grid below replace the old global one)
            dcc.Checklist(id="absth-sync",
                          options=[{"label": " sync abs — use one threshold for all traces",
                                    "value": "sync"}],
                          value=["sync"], labelStyle={"fontSize": "11px"},
                          style={"marginTop": "6px"}, **PERSIST),
            html.Div(id="absth-editor", style={"display": "none", "marginTop": "4px"}),
            html.Div(id="absth-msg", style={"fontSize": "10px", "color": "#666",
                                            "marginTop": "3px", "wordBreak": "break-all"}),
            # ---- trial-alignment nudge: shift each file's frame-sync + spikes together so trial
            #      starts line up. "⇄ auto" computes each file's offset from its TTL first-onset. ----
            html.Hr(style={"border": "none", "borderTop": "1px solid #ddd", "margin": "8px 0 6px"}),
            html.Div([
                html.Span("trial align — frame-sync nudge (ms)",
                          style={"fontSize": "11px", "fontWeight": "bold", "color": "#333"}),
                html.Button("⇄ auto", id="align-auto", n_clicks=0,
                            title="align every file's first flicker onset to the first file's",
                            style={"fontSize": "10px", "padding": "0 6px", "marginLeft": "8px",
                                   "cursor": "pointer"}),
                html.Button("reset", id="align-reset", n_clicks=0,
                            style={"fontSize": "10px", "padding": "0 6px", "marginLeft": "4px",
                                   "cursor": "pointer"}),
            ], style={"display": "flex", "alignItems": "center", "marginTop": "2px",
                      "flexWrap": "wrap"}),
            html.Div(id="align-editor", style={"marginTop": "4px"}),
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

    # draggable divider between the sidebar and the graphs (JS in assets/splitter.js;
    # remembers the width as a fraction of the window so it adapts across screens)
    html.Div(id="splitter", title="drag to resize",
             style={"flex": "0 0 6px", "cursor": "col-resize", "background": "#dcdce4",
                    "borderRadius": "3px", "alignSelf": "stretch"}),

    # ================= RIGHT PANEL: graphs (80% width, full height) =============
    html.Div(style={"flex": "1 1 0", "minWidth": 0, "height": "100%",
                    "display": "flex", "flexDirection": "column"}, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px",
                        "flex": "0 0 auto"}, children=[
            html.Div(id="readout", style={"fontWeight": "bold", "fontSize": "12px",
                                          "padding": "2px 0", "flex": "1 1 auto"}),
            # true full-screen toggle (browser Fullscreen API; wired in assets/fullscreen.js)
            html.Button("⛶ Full screen", id="fs-toggle", n_clicks=0,
                        title="enter / exit full screen (or press F)",
                        style={"fontSize": "11px", "padding": "2px 8px", "cursor": "pointer",
                               "flex": "0 0 auto", "whiteSpace": "nowrap"}),
        ]),
        # signal + frame-sync (frame-sync row enlarged) — gets the lion's share of height.
        # Region & display controls float in the corners, hugging the graph.
        html.Div(style={"flex": "3 1 0", "minHeight": 0, "position": "relative"}, children=[
            dcc.Graph(id="time", style={"height": "100%"},
                      config={"responsive": True, "scrollZoom": True, "doubleClick": "reset"},
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
                      dcc.Checklist(id="region-mode", className="cb-right",  # right-handed: label left, box right
                                    options=[{"label": "crop", "value": "crop"}], value=[],
                                    inline=True, labelStyle={"fontSize": "10px"}, **PERSIST)],
                     id="end-box", style=_END_OV),
            # bottom-LEFT, same level as "stagger frame sync %". "show detected spikes" (default
            # ON) HIDES when "show binned spikes" is checked, and keeps its value for when it
            # reappears.
            html.Div([
                dcc.Checklist(id="disp-show", className="cb-right",   # right-handed: label left, box right
                              options=[{"label": " show detected spikes", "value": "show_spikes"}],
                              value=["show_spikes"], inline=True, labelStyle={"fontSize": "10px"}),
                dcc.Checklist(id="disp-binned", className="cb-right",
                              options=[{"label": " show binned spikes", "value": "spike_train"}],
                              value=[], inline=True, labelStyle={"fontSize": "10px"}),
            ], style=ov(bottom="calc(38% + 25px)", right="6px", height=_OV_H,
                        justifyContent="flex-end")),   # up 1.25×height, under Im_prime; checkboxes flush right
            # just above the frame-sync x-axis, right-aligned with "bin (ms)": stagger %
            html.Div([html.Span("stagger frame sync %", style=_OVL),
                      dcc.Input(id="stagger-pct", type="number", value=0, min=0, max=100, step=5,
                                debounce=True, style=dict(_OVI, width="48px"), **PERSIST)],
                     style=ov(bottom="26px", right="8px")),
        ]),
        # bottom strip (less tall): FFT at half width + ISI histogram at the other half
        html.Div(style={"flex": "1 1 0", "minHeight": 0, "display": "flex", "gap": "6px"},
                 children=[
            html.Div([dcc.Graph(id="fft", style={"height": "100%"}, config={"responsive": True},
                                figure=blank_fig("")),
                      # FFT bin width (ms): sets the spike-train sampling before the transform.
                      # Coarser bins low-pass the impulse train (fewer harmonics); default 5 ms.
                      html.Div([html.Span("fft bin (ms)", style=_OVL),
                                dcc.Input(id="fft-bin", type="number", value=5, min=1, step=1,
                                          debounce=True, style=dict(_OVI, width="42px"), **PERSIST)],
                               style=ov(top="34px", left="62px"))],
                     style={"flex": "1 1 0", "minWidth": 0, "position": "relative"}),
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
    dcc.Store(id="align-map"),                            # {file path: trial-align offset (SECONDS)}
    dcc.Store(id="align-seed"),                           # {file path: offset (ms) shown in the editor}
    dcc.Store(id="recent-cells", storage_type="local"),   # most-recently-opened date|cell list
    dcc.Store(id="last-session", storage_type="local"),   # last cell + checked files (auto-loaded on startup)
    dcc.Store(id="rail-sort", data={"col": "date", "dir": "desc"}),   # explorer rail sort
    dcc.Store(id="store-rev", data=0),                    # bumped when the store changes (rail refresh)
    dcc.Store(id="store-fp"),                             # last-seen fingerprint of the selected cell(s)
    dcc.Interval(id="poll", interval=3000),               # watches disk -> auto-refresh Analysis View
    dcc.Store(id="exp-date"),                             # explorer: selected date
    dcc.Store(id="exp-cell"),                             # explorer: selected cell (within date)
    dcc.Store(id="exp-autosel"),                          # explorer: file path to auto-check on open
                                                          # (one file selected in Analysis View)
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
        # header bar — view switcher + status message. (Import/Backup → rail bottom panel;
        # Open-selected → thumbnails action bar; breadcrumb → rail top row.)
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px",
                        "color": "white", "marginBottom": "8px", "flex": "0 0 auto"}, children=[
            nav_toggle("explorer", dark=True),
            html.Span(id="exp-msg", style={"fontSize": "12px", "color": "#7fdc7f"}),
        ]),
        # body: left rail (full height) · middle (browser over a hover-preview) · right detail (full height)
        html.Div(style={"flex": "1 1 0", "minHeight": 0, "display": "flex", "gap": "8px"},
                 children=[
            # LEFT COLUMN: rail (fills) + a small bottom panel (Import / Backup mirror)
            html.Div(style={"flex": "0 0 380px", "minHeight": 0, "display": "flex",
                            "flexDirection": "column", "gap": "6px"}, children=[
              html.Div(id="exp-rail",
                     style={"flex": "1 1 0", "minHeight": 0, "display": "flex",
                            "flexDirection": "column", "background": "#15151d",
                            "border": "1px solid #2a2a35", "borderRadius": "6px",
                            "overflow": "hidden",
                            "--rc1": "150px", "--rc2": "76px", "--rc3": "30px"}, children=[
                # top row: the date ← cell back control, centered over the experiments list
                html.Div(id="exp-breadcrumb",
                         style={"textAlign": "center", "fontSize": "15px", "color": "white",
                                "padding": "4px", "borderBottom": "1px solid #2a2a35",
                                "flex": "0 0 auto"}),
                # sortable column headers — each (resizable) column carries a drag handle
                html.Div([
                    html.Div([html.Button("experiment ⇅", id={"type": "rail-sort", "col": "label"},
                                          n_clicks=0, style=dict(_RAILHDR, flex="1", textAlign="left")),
                              html.Div(className="rail-rz", **{"data-col": "1"}, style=_RZ_HANDLE)],
                             style=dict(_RAIL_C1, display="flex")),
                    html.Div([html.Button("date ⇅", id={"type": "rail-sort", "col": "date"},
                                          n_clicks=0, style=dict(_RAILHDR, flex="1")),
                              html.Div(className="rail-rz", **{"data-col": "2"}, style=_RZ_HANDLE)],
                             style=dict(_RAIL_DATE, display="flex")),
                    html.Div([html.Button("# ⇅", id={"type": "rail-sort", "col": "cells"},
                                          n_clicks=0, style=dict(_RAILHDR, flex="1")),
                              html.Div(className="rail-rz", **{"data-col": "3"}, style=_RZ_HANDLE)],
                             style=dict(_RAIL_N, display="flex")),
                    html.Div(style=_RAIL_DEAD),
                    html.Span(style=_RAIL_TRASH),
                ], style={"display": "flex", "gap": "4px", "padding": "4px 8px",
                          "alignItems": "stretch"}),
                # per-column search
                html.Div([
                    dcc.Input(id="rail-q-label", type="text", placeholder="search…", debounce=True,
                              style=dict(_RAILQ, **_RAIL_C1)),
                    dcc.Input(id="rail-q-date", type="text", placeholder="date", debounce=True,
                              style=dict(_RAILQ, **_RAIL_DATE)),
                    dcc.Input(id="rail-q-cells", type="text", placeholder="#", debounce=True,
                              style=dict(_RAILQ, **_RAIL_N)),
                    html.Div(style=_RAIL_DEAD),
                    html.Span(style=_RAIL_TRASH),
                ], style={"display": "flex", "gap": "4px", "padding": "0 8px 4px"}),
                # the rows (rebuilt by a callback)
                html.Div(id="exp-dates", style={"flex": "1 1 0", "overflowY": "auto"}),
              ]),
              # bottom panel (~10% height): Import + Backup mirror
              html.Div([
                  html.Button("📥 Import data…", id="import-data", n_clicks=0,
                              style={"flex": "1", "fontWeight": "bold"}),
                  html.Button("⤓ Backup mirror", id="backup-mirror", n_clicks=0,
                              style={"flex": "1", "fontWeight": "bold"}),
              ], style={"flex": "0 0 10%", "minHeight": "44px", "background": "#15151d",
                        "border": "1px solid #2a2a35", "borderRadius": "6px", "padding": "8px",
                        "display": "flex", "alignItems": "center", "gap": "8px"}),
            ]),
            # middle column: browser (top 2/3) over the hover preview (bottom 1/3)
            html.Div(style={"flex": "1 1 0", "minWidth": 0, "display": "flex",
                            "flexDirection": "column", "gap": "6px"}, children=[
                html.Div(style={"flex": "2 1 0", "minHeight": 0, "display": "flex",
                                "flexDirection": "column", "background": "#23232c",
                                "borderRadius": "6px"}, children=[
                    html.Div([
                        html.Div(id="exp-cards",
                                 style={"display": "flex", "flexWrap": "wrap", "gap": "18px",
                                        "alignItems": "flex-start"}),
                        # files as a desktop-icon grid (wrap into rows/columns)
                        dcc.Checklist(id="exp-files", options=[], value=[], inline=True,
                                      labelStyle={"display": "inline-flex", "alignItems": "flex-start",
                                                  "verticalAlign": "top", "background": "#23232c",
                                                  "borderRadius": "5px", "padding": "5px", "margin": "5px"},
                                      inputStyle={"marginRight": "5px", "marginTop": "2px"}),
                    ], style={"flex": "1 1 0", "minHeight": 0, "overflowY": "auto", "padding": "10px"}),
                    # bottom action bar: delete (left) + Open selected (right) — shown when files chosen
                    html.Div([
                        html.Button("", id="del-files", n_clicks=0, style={"display": "none"}),
                        html.Button("📈 Open selected in viewer", id="exp-open-viewer", n_clicks=0,
                                    style={"display": "none", "fontWeight": "bold"}),
                    ], style={"display": "flex", "justifyContent": "flex-end", "gap": "8px",
                              "alignItems": "center", "padding": "6px 10px",
                              "borderTop": "1px solid #333"}),
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
            # right: detail pane — editable cell metadata (Save / Backup live here now) above
            # the callback-filled recording facts + outputs + raw JSON (exp-detail). Bigger font.
            html.Div(id="exp-rightpane",
                     style={"flex": "0 0 32%", "minHeight": 0, "overflowY": "auto",
                            "background": "white", "borderRadius": "6px", "padding": "12px",
                            "fontSize": "14px", "display": "flex", "flexDirection": "column",
                            "gap": "8px"}, children=[
                # header: cell label + actions
                html.Div([
                    html.Div(id="exp-cell-label",
                             style={"fontWeight": "bold", "fontSize": "16px", "flex": "1",
                                    "minWidth": 0}),
                    html.Button("🔧 Fix abf (Neitz)", id="fix-abf", n_clicks=0,
                                title="Apply the Neitz Lab config: stimulus = sq wave, "
                                      "channels Im_prime/Vm_sec/TTL; re-read the files fresh",
                                style={"fontSize": "12px"}),
                    html.Button("💾 Save metadata", id="save-meta", n_clicks=0,
                                style={"fontSize": "12px"}),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px",
                          "borderBottom": "1px solid #eee", "paddingBottom": "8px"}),
                # editable cell metadata (type / stimulus / params / notes)
                html.Div("Cell metadata — edit, then Save",
                         style={"fontWeight": "bold", "fontSize": "13px", "color": "#555"}),
                html.Div([html.Label("type", style=_EXPLBL),
                          dcc.Input(id="cell-type", type="text", debounce=True,
                                    placeholder="e.g. ipRGC",
                                    style={"width": "100%", "boxSizing": "border-box"})]),
                html.Div([html.Label("stimulus", style=_EXPLBL),
                          dcc.Dropdown(id="stim-type", style={"width": "100%"},
                                       options=[{"label": "sq wave", "value": "sq_wave"},
                                                {"label": "gaussian_noise", "value": "gaussian_noise"},
                                                {"label": "checkerboard", "value": "checkerboard"},
                                                {"label": "(none)", "value": "(none)"}])]),
                html.Div([html.Label("params (k=v, …)", style=_EXPLBL),
                          dcc.Input(id="stim-params", type="text", debounce=True,
                                    placeholder="flicker_hz=2, frame_rate=60",
                                    style={"width": "100%", "boxSizing": "border-box"})]),
                html.Div([html.Label("notes", style=_EXPLBL),
                          dcc.Textarea(id="cell-notes",
                                       style={"width": "100%", "boxSizing": "border-box",
                                              "minHeight": "44px", "fontSize": "13px"})]),
                html.Div(id="exp-save-msg", style={"fontSize": "12px", "color": "#070"}),
                # callback-filled: recording facts + outputs + collapsible raw JSON
                html.Div(id="exp-detail", style={"minHeight": 0}),
            ]),
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
              Input("absth-map", "data"), Input("fft-bin", "value"), Input("align-map", "data"),
              Input("time", "relayoutData"), prevent_initial_call=True)
def render(files, chan, ttl_name, polarity, method, k, absth, refr, rstart, rend,
           disp_show, disp_binned, stagger_pct, region_mode, train_bin, absth_map, fft_bin,
           align_map, relayout):
    return build_figures(files, chan, ttl_name, polarity, method, k, absth, refr, rstart, rend,
                         disp_show, disp_binned, stagger_pct, region_mode, train_bin, absth_map,
                         fft_bin=fft_bin, align_map=align_map, relayout=relayout, trig=ctx.triggered_id)


def build_figures(files, chan, ttl_name, polarity, method, k, absth, refr, rstart, rend,
                  disp_show, disp_binned, stagger_pct, region_mode, train_bin, absth_map,
                  fft_bin=5.0, align_map=None, relayout=None, trig=None):
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
    # FFT bin width (ms) → spike-train sampling rate (Hz) for binned_rate + power_w. Both MUST use
    # the same rate or the frequency axis is wrong. Guard blank/≤0 → the 5 ms (200 Hz) default.
    fft_ms = float(fft_bin) if fft_bin else 5.0
    fft_rate = 1000.0 / max(1.0, fft_ms)

    rec0 = get_recording(files[0])
    fs0 = rec0.fs
    y0 = get_channel(files[0], chan)
    t0_full, t1_full = 0.0, len(y0) / fs0

    rs = float(rstart) if rstart is not None else t0_full
    re_ = float(rend) if rend is not None else t1_full

    x0, x1 = t0_full, t1_full
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
    per_file_rates, stim_freqs, isi_all = [], [], []

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

        # per-file trial-alignment nudge (s): shifts this file's analog, TTL AND spikes together,
        # so trial starts line up across files. i0/i1 sample the ORIGINAL data for the window that,
        # once shifted by +off, lands in the visible [x0,x1]. Spikes/traces are then plotted at +off.
        off = 0.0
        if align_map:
            try:
                off = float(align_map.get(path, 0.0) or 0.0)
            except (TypeError, ValueError):
                off = 0.0

        i0, i1 = max(0, int((x0 - off) * fs)), min(len(y), int((x1 - off) * fs))
        if i1 <= i0:
            i0, i1 = 0, len(y)

        eff_abs = amap.get(path)                     # this trace's own value, if set
        if eff_abs is None:
            eff_abs = float(absth) if absth is not None else None
        det_i = dict(det, abs_threshold=(float(eff_abs) if eff_abs is not None else None))
        st = detect_spikes(y, fs, **det_i)
        at = st.times + off                          # spike times in aligned (display) coordinates
        in_reg = at[(at >= rs) & (at <= re_)]
        if len(in_reg) > 1:
            isi_all.append(np.diff(in_reg) * 1000.0)     # ms, for the ISI histogram (offset cancels)
        y_disp = y
        disp_spikes = in_reg if crop else at              # hide excluded spikes when cropped

        # ---- ROW 1: either the analog signal, or the spike train (0/1 or binned) ----
        if spike_train:
            if tbin > 0:                                  # binned counts
                bw = tbin / 1000.0
                lo = min(0.0, off)                         # cover the offset-shifted span
                edges = np.arange(lo, t1_full + max(0.0, off) + bw, bw)
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
            time_fig.add_trace(go.Scattergl(x=dt + off, y=dy, mode="lines", legendgroup=name,
                                            line=dict(width=0.6, color=color),
                                            opacity=(0.6 if multi else 0.9), name=name), row=1, col=1)
            if show_spikes:
                sp = in_reg[(in_reg >= x0) & (in_reg <= x1)]         # display (aligned) coords
                sp_y = y[np.clip(((sp - off) * fs).astype(int), 0, len(y) - 1)]   # original index for y
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
                time_fig.add_trace(go.Scattergl(x=tt + off, y=ty + idx * ttl_step, mode="lines",
                                                legendgroup=name, showlegend=False,
                                                line=dict(width=0.6, color=color),
                                                opacity=(0.6 if multi else 0.9),
                                                name=f"{name} TTL"), row=2, col=1)
            except Exception:
                pass

        rate = binned_rate(in_reg - rs, 0.0, re_ - rs, bin_rate=fft_rate)
        if rate is not None:
            per_file_rates.append(rate)
            f, pw = power_w(rate, bin_rate=fft_rate)
            fft_fig.add_trace(go.Scatter(x=f, y=power_db(pw), mode="lines", legendgroup=name,
                                         line=dict(width=(1 if multi else 2), color=color),
                                         opacity=(0.45 if multi else 1.0), name=name))

    if not crop:                                   # shade excluded blocks (skip when cropped out)
        for (a, b, lbl, pos) in [(t0_full, rs, "excluded (adapting)", "bottom left"),
                                 (re_, t1_full, "excluded", "bottom right")]:
            if b > a + 1e-6:
                time_fig.add_vrect(x0=a, x1=b, fillcolor="gray", opacity=0.4, line_width=0,
                                   annotation_text=lbl, annotation_position=pos,
                                   annotation=dict(font_size=10), row="all", col=1)

    ttl_ylab = (f"{ttl_name} (staggered)" if (stagger and multi) else (ttl_name or "TTL"))
    # fixedrange on both y-axes makes the drag-box zoom X-ONLY: Plotly can't axis-lock a thin box
    # into a y-only zoom, so a narrow left-right drag zooms the time axis precisely (no minimum
    # width / no broad-horizontal-lines artifact). Y still autoranges to the data; wheel-scroll
    # zooms the x-axis for fine control. fixedrange blocks USER y-zoom only, not the code's autorange.
    time_fig.update_yaxes(title_text=row1_ylab, row=1, col=1, fixedrange=True)
    # the frame-sync row's y gets its OWN uirevision tied to the staggered extent, so it
    # re-autoranges (all traces fit) when the stagger % or the number of traces changes —
    # the global uirevision="keep" would otherwise pin the old y-range and clip the spread.
    time_fig.update_yaxes(title_text=ttl_ylab, row=2, col=1, autorange=True, fixedrange=True,
                          uirevision=f"ttl-{stagger_pct}-{len(files)}")
    time_fig.update_xaxes(title_text="time (s)", row=2, col=1, range=[x0, x1])
    time_fig.update_layout(margin=dict(l=55, r=20, t=30, b=40), uirevision="keep",
                           showlegend=False)   # file colors are evident from the Files list

    # power spectrum: group average + stim marker
    if multi and len(per_file_rates) >= 2:
        n = min(len(r) for r in per_file_rates)
        f, pw = power_w(np.mean([r[:n] for r in per_file_rates], axis=0), bin_rate=fft_rate)
        fft_fig.add_trace(go.Scatter(x=f, y=power_db(pw), mode="lines",
                                     line=dict(width=3, color="black"), name="GROUP AVG"))
    sfreqs = [s for s in stim_freqs if s]
    if sfreqs:
        sf = float(np.mean(sfreqs))
        fft_fig.add_vline(x=sf, line_dash="dash", line_color="orange",
                          annotation_text=f"stim {sf:.2f} Hz",
                          annotation_position="bottom right",
                          annotation=dict(font=dict(size=10, color="#c60")))
    # legend OUTSIDE the plot on the right; the right margin grows with the longest label so the
    # plot area shrinks to make room instead of the legend overlapping the data.
    legend_names = [os.path.basename(p) for p in files] + (["GROUP AVG"] if multi else [])
    maxlen = max((len(s) for s in legend_names), default=8)
    r_margin = int(min(240, max(80, maxlen * 6.5 + 26)))
    fft_fig.update_layout(
        title=dict(text=f"spike-train power  10·log₁₀(2|X[k]|²/N²)  (inside region · {fft_ms:g} ms bins)",
                   x=0.5, xanchor="center", y=0.97, yanchor="top", font=dict(size=12)),
        xaxis_title="frequency (Hz)", yaxis_title="power (dB, R=1Ω)",
        xaxis_range=[0, FMAX], margin=dict(l=55, r=r_margin, t=34, b=40),
        legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top", font=dict(size=9),
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
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

    # Compact one-line readout (the old per-file dump blew up to many lines with 40+ files).
    view = ("spike-train" if spike_train else "analog")
    freqs = sorted({round(s, 2) for s in stim_freqs if s})
    freq_txt = (f" · stim {freqs[0]:.2f} Hz" if len(freqs) == 1
                else f" · stim {min(freqs):.2f}–{max(freqs):.2f} Hz" if freqs else "")
    label = (f"{len(files)} files · avg" if multi else "1 file")
    readout = f"{label} · region {rs:.2f}–{re_:.2f}s · {view} view{freq_txt}"
    return time_fig, fft_fig, isi_fig, readout


# ---- 4K Plotly exports saved on "Run Analysis" (kaleido → PDF/SVG, 3840×2160 = 4K full-screen).
#      Built from the SAME build_figures() the GUI uses, so they're pixel-faithful regardless of the
#      actual (possibly small) browser window. ----------------------------------------------------
def export_window_figures(outdir, stem, *, files, chan, ttl_name, polarity, method, k, absth, refr,
                          rstart, rend, region_mode, stagger_pct, disp_show, disp_binned,
                          train_bin, absth_map, fft_bin=5.0, align_map=None):
    """Save the four requested 4K figures into outdir. Returns {name: Path} of what was written."""
    import copy
    import plotly.io as pio
    W, H = 3840, 2160                                  # 4K (27" full-screen)
    files = [f for f in (files or []) if f and loadable(f)]
    if not files or not chan:
        return {}
    tf, ff, isf, _ = build_figures(files, chan, ttl_name, polarity, method, k, absth, refr,
                                   rstart, rend, disp_show, disp_binned, stagger_pct,
                                   region_mode, train_bin, absth_map, fft_bin=fft_bin,
                                   align_map=align_map)
    cp = copy.deepcopy                                  # traces can't live in two figures at once
    saved = {}

    def _write(fig, name, fmts, h=None):
        H_ = h or H                                        # optional shorter height (e.g. tables)
        fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white")
        for ext in fmts:
            p = outdir / f"{name}.{ext}"
            w, hh = (1920, int(H_ * 1920 / W)) if ext == "png" else (W, H_)   # PNG = lighter preview
            pio.write_image(fig, str(p), format=ext, width=w, height=hh)
            saved[f"{name}_{ext}"] = p

    sig = [t for t in tf.data if getattr(t, "yaxis", "y") != "y2"]      # signal + spike markers
    ttl = [t for t in tf.data if getattr(t, "yaxis", "y") == "y2"]      # frame syncs
    is_spike = lambda t: ("markers" in (getattr(t, "mode", "") or "")
                          or str(getattr(t, "name", "")).endswith("spikes"))
    try:
        dur = len(get_channel(files[0], chan)) / get_recording(files[0]).fs
    except Exception:
        dur = None
    rs = float(rstart) if rstart not in (None, "") else 0.0
    re_ = float(rend) if rend not in (None, "") else (dur or 0.0)

    # (1) full window: the top two rows (signal + frame-sync) are shrunk to 80% (left-justified) and
    #     TIME-ALIGNED; the right 20% is a 250 ms zoom of the stimulus (same rows, raw full-res).
    #     Bottom row: spike-train power | ISI. -------------------------------------------------------
    z0 = rs                                            # 250 ms stimulus zoom, from the region start
    z1 = z0 + 0.25
    combo = make_subplots(rows=3, cols=2, column_widths=[0.8, 0.2],
                          row_heights=[0.42, 0.26, 0.32], vertical_spacing=0.09,
                          horizontal_spacing=0.045,
                          subplot_titles=("", "stimulus · 250 ms", "", "", "spike-train power", "ISI"))
    for t in sig:
        combo.add_trace(cp(t), row=1, col=1)
    for t in ttl:
        combo.add_trace(cp(t), row=2, col=1)
    # right-column 250 ms zoom: raw (un-decimated) signal + frame-sync over [z0, z1]
    for idx, path in enumerate(files):
        recz = get_recording(path); fsz = recz.fs; colz = PALETTE[idx % len(PALETTE)]
        yz = get_channel(path, chan)
        a, b = max(0, int(z0 * fsz)), min(len(yz), int(z1 * fsz))
        tz = np.arange(a, b) / fsz
        combo.add_trace(go.Scattergl(x=tz, y=yz[a:b], mode="lines", line=dict(width=0.8, color=colz),
                                     opacity=0.7, showlegend=False), row=1, col=2)
        if ttl_name and ttl_name != chan:
            try:
                tt = get_channel(path, ttl_name)
                combo.add_trace(go.Scattergl(x=tz, y=tt[a:b], mode="lines", line=dict(width=0.8, color=colz),
                                             opacity=0.7, showlegend=False), row=2, col=2)
            except Exception:
                pass
    for t in ff.data:
        combo.add_trace(cp(t), row=3, col=1)
    for t in isf.data:
        combo.add_trace(cp(t), row=3, col=2)
    combo.update_yaxes(title_text=(tf.layout.yaxis.title.text or chan), row=1, col=1)
    combo.update_yaxes(title_text=(tf.layout.yaxis2.title.text or (ttl_name or "frame sync")), row=2, col=1)
    combo.update_xaxes(title_text="time (s)", row=2, col=1)
    combo.update_xaxes(title_text="time (s)", row=2, col=2)
    combo.update_xaxes(title_text="frequency (Hz)", range=[0, FMAX], row=3, col=1)
    combo.update_yaxes(title_text="power (dB)", row=3, col=1)
    combo.update_xaxes(title_text="ISI (ms)", row=3, col=2)
    combo.update_yaxes(title_text="count", row=3, col=2)
    # alignment: frame-sync time-axis matches the signal time-axis (x3↔x); zoom rows share x (x4↔x2)
    if dur:
        combo.update_xaxes(range=[0, dur], row=1, col=1)
    combo.update_xaxes(matches="x", row=2, col=1)      # #1a: signal & frame-sync aligned in time
    combo.update_xaxes(range=[z0, z1], row=1, col=2)
    combo.update_xaxes(range=[z0, z1], matches="x2", row=2, col=2)
    if dur and re_ > rs:                                # shade excluded blocks on the full time rows
        for r in (1, 2):
            if rs > 0:
                combo.add_vrect(x0=0, x1=rs, fillcolor="gray", opacity=0.22, line_width=0, row=r, col=1)
            if re_ < dur:
                combo.add_vrect(x0=re_, x1=dur, fillcolor="gray", opacity=0.22, line_width=0, row=r, col=1)
    combo.update_layout(showlegend=False, margin=dict(l=70, r=40, t=80, b=60),
                        title=dict(text=stem, x=0.5, xanchor="center", font=dict(size=22)))
    _write(combo, "window_4k", ("pdf", "svg", "png"))

    # (2) spike-train power graph ------------------------------------------------------------------
    _write(go.Figure(ff), "power_4k", ("pdf", "png"))

    # (3) analog: top = signal+spikes (color), middle = same B&W no spikes, bottom = frame syncs ----
    a3 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                       row_heights=[0.4, 0.3, 0.3])
    for t in sig:
        a3.add_trace(cp(t), row=1, col=1)
    for t in sig:
        if is_spike(t):
            continue
        bw = cp(t)
        bw.update(line=dict(color="#222", width=getattr(t.line, "width", 0.6)), opacity=0.9)
        a3.add_trace(bw, row=2, col=1)
    for t in ttl:
        a3.add_trace(cp(t), row=3, col=1)
    a3.update_yaxes(title_text=f"{chan} (color)", row=1, col=1)
    a3.update_yaxes(title_text=f"{chan} (B&W, no spikes)", row=2, col=1)
    a3.update_yaxes(title_text=(ttl_name or "frame sync"), row=3, col=1)
    a3.update_xaxes(title_text="time (s)", row=3, col=1)
    a3.update_layout(showlegend=False, margin=dict(l=70, r=40, t=80, b=60),
                     title=dict(text=f"{stem} — analog + frame sync", x=0.5, xanchor="center",
                                font=dict(size=22)))
    _write(a3, "analog_framesync_4k", ("pdf", "png"))

    # (4) frame syncs, each file in its OWN panel (un-staggered so every carrier is visible) -------
    tf0, _, _, _ = build_figures(files, chan, ttl_name, polarity, method, k, absth, refr,
                                 rstart, rend, disp_show, disp_binned, 0, region_mode,
                                 train_bin, absth_map, fft_bin=fft_bin, align_map=align_map)
    sep_traces = [t for t in tf0.data if getattr(t, "yaxis", "y") == "y2"]
    if sep_traces:
        n = len(sep_traces)
        sep = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            subplot_titles=[str(getattr(t, "name", "") or "").replace(" TTL", "")
                                            for t in sep_traces])
        for i, t in enumerate(sep_traces):
            sep.add_trace(cp(t), row=i + 1, col=1)
        sep.update_xaxes(title_text="time (s)", row=n, col=1)
        sep.update_layout(showlegend=False, margin=dict(l=70, r=40, t=90, b=60),
                          title=dict(text=f"{stem} — frame syncs (separated)", x=0.5,
                                     xanchor="center", font=dict(size=22)))
        _write(sep, "framesync_separated_4k", ("pdf", "png"))

    # (5) ASSUMPTIONS page — exactly which detection settings were used + per-file spike counts,
    #     so it's verifiable that Run Analysis honored the live GUI settings. ----------------------
    from datetime import datetime as _dt
    _meth = {"mad": "k·MAD", "abs": "absolute", "mad_floor": "k·MAD ≥ floor",
             "matlab": "MATLAB (Sara)"}.get(method, method)
    amap = absth_map or {}
    det0 = dict(polarity=polarity, method=method, k=(float(k) if k not in (None, "") else 6.0),
                abs_threshold=(float(absth) if absth not in (None, "") else None),
                refractory_s=((float(refr) / 1000.0) if refr else 0.002))
    per = []                                            # (file, n_total, n_region, threshold, flicker_hz)
    for path in files:
        try:
            rec = get_recording(path); fs = rec.fs
            y = get_channel(path, chan)
            ea = amap.get(path)
            if ea is None:
                ea = det0["abs_threshold"]
            st = detect_spikes(y, fs, **dict(det0, abs_threshold=(float(ea) if ea is not None else None)))
            inreg = st.times[(st.times >= rs) & (st.times <= re_)]
            fl = get_flicker(path, ttl_name)
            per.append((os.path.basename(path), len(st), int(len(inreg)),
                        (f"{st.threshold:.2f}" if st.threshold is not None else "—"),
                        (f"{fl.freq:.2f}" if fl else "—")))
        except Exception as e:
            per.append((os.path.basename(path), "err", str(e)[:20], "—", "—"))
    uses_abs = method in ("abs", "mad_floor", "matlab")
    settings = [
        ("cell / run", stem),
        ("signal channel", str(chan)),
        ("frame-sync (TTL) channel", str(ttl_name)),
        ("polarity", {"neg": "neg (downward)", "pos": "pos (upward)", "abs": "abs (either)"}.get(polarity, str(polarity))),
        ("spike-detect method", _meth),
        ("k (·MAD)", (f"{det0['k']:g}" if method in ("mad", "mad_floor") else "— (not used)")),
        ("absolute threshold", ("per-trace (see table)" if amap else
                                (f"{det0['abs_threshold']:g}" if (uses_abs and det0['abs_threshold'] is not None)
                                 else ("max/3 auto" if method == "matlab" else "— (not used)")))),
        ("refractory (ms)", (f"{(refr if refr else 2)}" if method != "matlab" else "— (not used)")),
        ("analysis region (s)", f"{rs:.2f} – {re_:.2f}" + ("  (crop ON)" if "crop" in (region_mode or []) else "")),
        ("stagger frame-sync %", str(stagger_pct or 0)),
        ("files analyzed", str(len(files))),
        ("total spikes (all files)", str(sum(p[1] for p in per if isinstance(p[1], int)))),
        ("total in-region spikes", str(sum(p[2] for p in per if isinstance(p[2], int)))),
        ("generated", _dt.now().isoformat(timespec="seconds")),
    ]
    _ns, _np = len(settings), len(per)                 # size each table's domain to its row count
    info = make_subplots(rows=2, cols=1, vertical_spacing=0.06,
                         row_heights=[_ns / (_ns + _np), _np / (_ns + _np)],
                         specs=[[{"type": "table"}], [{"type": "table"}]],
                         subplot_titles=("detection settings used", "per-file spike counts"))
    info.add_trace(go.Table(
        columnwidth=[34, 66],
        header=dict(values=["<b>parameter</b>", "<b>value</b>"], fill_color="#2f3142",
                    font=dict(color="white", size=20), align="left", height=40),
        cells=dict(values=[[s[0] for s in settings], [s[1] for s in settings]],
                   fill_color=[["#f3f4fb", "#ffffff"] * 8], font=dict(size=19), align="left", height=34)),
        row=1, col=1)
    info.add_trace(go.Table(
        columnwidth=[40, 16, 16, 16, 14],
        header=dict(values=["<b>file</b>", "<b>spikes (total)</b>", "<b>spikes (region)</b>",
                            "<b>threshold</b>", "<b>flicker Hz</b>"], fill_color="#2f3142",
                    font=dict(color="white", size=20), align="left", height=40),
        cells=dict(values=[[p[0] for p in per], [p[1] for p in per], [p[2] for p in per],
                           [p[3] for p in per], [p[4] for p in per]],
                   fill_color=[["#f3f4fb", "#ffffff"] * 16], font=dict(size=19), align="left", height=34)),
        row=2, col=1)
    info.update_layout(margin=dict(l=40, r=40, t=90, b=40),
                       title=dict(text=f"{stem} — analysis assumptions", x=0.5, xanchor="center",
                                  font=dict(size=24)))
    fit_h = min(H, 360 + 50 * (len(settings) + len(per)))     # fit height to the row count
    _write(info, "assumptions_4k", ("pdf", "png"), h=fit_h)
    return saved


# ---- auto-seed the per-trace abs boxes with the MATLAB max/3 default the moment MATLAB is
#      selected (or files/polarity change) — no need to click the auto-abs button -------------
@app.callback(Output("absth-seed", "data", allow_duplicate=True),
              Output("absth-sync", "value", allow_duplicate=True),
              Output("absth-msg", "children", allow_duplicate=True),
              Input("method", "value"), Input("file", "value"), Input("polarity", "value"),
              State("chan", "value"), prevent_initial_call=True)
def seed_matlab_threshold(method, files, polarity, chan):
    if method != "matlab":
        return no_update, no_update, no_update
    from neitz.spikes import _highpass_fft, HIGHPASS_SPIKES_HZ
    files = [f for f in (files or []) if f and loadable(f)]
    if not files or not chan:
        return no_update, no_update, no_update
    seed, bits = {}, []
    for path in files:
        try:
            tr = _highpass_fft(get_channel(path, chan), get_recording(path).fs, HIGHPASS_SPIKES_HZ)
            tr = tr - np.median(tr)
            if polarity == "neg":
                tr = -tr
            elif polarity == "abs":
                tr = np.abs(tr)
            seed[path] = round(float(np.max(tr)) / 3.0, 2)
            bits.append(f"{os.path.basename(path)}={seed[path]:g}")
        except Exception:
            pass
    if not seed:
        return no_update, no_update, no_update
    return seed, [], "MATLAB max/3 auto-seeded → " + ", ".join(bits)


# ---- build the per-trace abs-threshold grid (ALWAYS shown; one box per on-graph trace,
#      wrapping left→right. When synced, all boxes show but only the first is editable and
#      its value is mirrored into the rest) ----------------------------------------------
# ---- enable/disable controls that aren't relevant to the chosen detection method --------
@app.callback(Output("k", "disabled"), Output("k-wrap", "style"), Output("refr", "disabled"),
              Output("polarity", "options"),
              Output("absth-sync", "options"), Output("method-note", "children"),
              Input("method", "value"))
def toggle_controls(method):
    k_off = method in ("abs", "matlab")                # k·MAD not used
    abs_on = method in ("abs", "mad_floor", "matlab")  # absolute thresholds are used
    # dcc.Slider's `disabled` has no visible effect in this Dash build, so grey the wrapper
    k_style = {"flex": "1", "minWidth": 0, "opacity": (0.4 if k_off else 1),
               "pointerEvents": ("none" if k_off else "auto")}
    pol = [{"label": p, "value": p} for p in ("neg", "pos", "abs")]   # polarity used by all methods
    sync = [{"label": " sync abs — use one threshold for all traces", "value": "sync",
             "disabled": not abs_on}]
    note = ("MATLAB (Sara): polarity sets orientation; the per-box ‘auto’ fills the max/3 "
            "default; k & refractory are not used."
            if method == "matlab" else "")
    return k_off, k_style, (method == "matlab"), pol, sync, note


# ---- build the per-trace abs-threshold grid (ALWAYS shown; one box per on-graph trace,
#      wrapping left→right. When synced, all boxes show but only the first is editable and
#      its value is mirrored into the rest). Only relevant for the absolute-threshold methods.
@app.callback(Output("absth-editor", "children"), Output("absth-editor", "style"),
              Input("absth-sync", "value"), Input("file", "value"), Input("absth-seed", "data"),
              Input("absth", "value"), Input("method", "value"), prevent_initial_call=False)
def build_absth_editor(sync, files, seed, single, method):
    if method not in ("abs", "mad_floor", "matlab"):  # abs thresholds unused → hide the grid
        return [], {"display": "none"}
    files = [f for f in (files or []) if f and loadable(f)]
    if not files:
        return ([html.Span("select file(s) to set per-trace thresholds",
                           style={"fontSize": "10px", "color": "#999"})], {"display": "none"})
    synced = "sync" in (sync or [])
    seed = seed or {}
    default = single if single is not None else 20
    cells = []
    for i, p in enumerate(files):
        locked = synced and i > 0                    # synced: only the first box is editable
        val = default if synced else seed.get(p, default)
        cells.append(html.Div([
            html.Span(os.path.basename(p), title=os.path.basename(p),
                      style={"fontSize": "10px", "color": "#555", "display": "block",
                             "maxWidth": "100px", "overflow": "hidden",
                             "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
            dcc.Input(id={"type": "absth-trace", "path": p}, type="number", value=val,
                      debounce=True, disabled=locked,
                      style={"width": "72px",
                             "background": "#eee" if locked else "white"}),
            # per-box "auto" — BELOW the box; sets this trace's threshold (max/3 for MATLAB,
            # k·MAD otherwise). When synced, only the first is active and it fills every box.
            html.Button("auto", id={"type": "absth-auto", "path": p}, n_clicks=0, disabled=locked,
                        style={"fontSize": "9px", "padding": "0 4px", "marginTop": "2px",
                               "width": "50px", "lineHeight": "13px",
                               "cursor": "default" if locked else "pointer",
                               "opacity": 0.4 if locked else 1}),
        ], style={"margin": "0 8px 6px 0", "display": "flex", "flexDirection": "column",
                  "alignItems": "flex-start"}))
    head = ("one threshold (sync on): edit the first box — all match"
            if synced else "per-trace abs thresholds")
    return ([html.Div(head, style={"fontSize": "10px", "color": "#555",
                                    "marginBottom": "3px", "width": "100%"}),
             html.Div(cells, style={"display": "flex", "flexWrap": "wrap"})],
            {"display": "block", "marginTop": "4px",
             "borderLeft": "2px solid #3367d6", "paddingLeft": "6px"})


# ---- per-box "auto": compute one trace's threshold and drop it in (all boxes when synced) ----
@app.callback(Output({"type": "absth-trace", "path": ALL}, "value", allow_duplicate=True),
              Input({"type": "absth-auto", "path": ALL}, "n_clicks"),
              State("absth-sync", "value"), State("chan", "value"), State("method", "value"),
              State("k", "value"), State("polarity", "value"), prevent_initial_call=True)
def per_trace_auto(_clicks, sync, chan, method, k, polarity):
    from neitz.spikes import _highpass_fft, HIGHPASS_SPIKES_HZ
    outs = ctx.outputs_list[0] if ctx.outputs_list else []
    trig = ctx.triggered_id
    if not (isinstance(trig, dict) and ctx.triggered and ctx.triggered[0].get("value")):
        return [no_update] * len(outs)
    path = trig["path"]
    try:
        y = get_channel(path, chan)
        if method == "matlab":
            tr = _highpass_fft(y, get_recording(path).fs, HIGHPASS_SPIKES_HZ)
            tr = tr - np.median(tr)
            if polarity == "neg":
                tr = -tr
            elif polarity == "abs":
                tr = np.abs(tr)
            val = round(float(np.max(tr)) / 3.0, 2)
        else:
            val = round(float(k) * float(np.median(np.abs(y - np.median(y))) * 1.4826), 2)
    except Exception:
        return [no_update] * len(outs)
    if "sync" in (sync or []):                          # synced → fill every box
        return [val] * len(outs)
    return [val if o["id"]["path"] == path else no_update for o in outs]   # else just this trace


# ---- collect the per-trace inputs into absth-map (render reads this). When synced, the
#      first (editable) box becomes the single shared threshold so the mirrors follow it. --
@app.callback(Output("absth-map", "data"), Output("absth", "value", allow_duplicate=True),
              Input({"type": "absth-trace", "path": ALL}, "value"),
              Input("absth-sync", "value"), Input("file", "value"),
              prevent_initial_call=True)
def collect_absth(_vals, sync, _files):
    items = ctx.inputs_list[0] or []                 # each: {"id": {...,"path":p}, "value": v}
    if "sync" in (sync or []):                        # synced → render uses the single value
        master = no_update
        for item in items:                            # first non-empty box drives the shared value
            v = item.get("value")
            if v is not None:
                try:
                    master = float(v)
                except (TypeError, ValueError):
                    master = no_update
                break
        return {}, master
    amap = {}
    for item in items:
        v = item.get("value")
        if v is not None:
            try:
                amap[item["id"]["path"]] = float(v)
            except (TypeError, ValueError):
                pass
    return amap, no_update


# ================= trial-alignment nudge (frame-sync + spikes shifted together) =================
# The align-editor shows one ms box per file; the first file is the reference (offset 0, disabled).
# "⇄ auto" seeds each other file with (ref.t0 − file.t0) from the TTL first-onset; "reset" clears.
# The box values (ms) are collected into align-map (SECONDS), which build_figures applies.
@app.callback(Output("align-editor", "children"),
              Input("file", "value"), Input("align-seed", "data"),
              prevent_initial_call=False)
def build_align_editor(files, seed):
    files = [f for f in (files or []) if f and loadable(f)]
    if len(files) < 2:
        return [html.Span("select 2+ files to nudge trials into alignment",
                          style={"fontSize": "10px", "color": "#999"})]
    seed = seed or {}
    cells = []
    for i, p in enumerate(files):
        ref = (i == 0)                                   # first file = reference (offset 0)
        cells.append(html.Div([
            html.Span(("① " if ref else "") + os.path.basename(p), title=os.path.basename(p),
                      style={"fontSize": "10px", "color": "#555", "display": "block",
                             "maxWidth": "100px", "overflow": "hidden",
                             "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
            dcc.Input(id={"type": "align-trace", "path": p}, type="number",
                      value=(0 if ref else seed.get(p, 0)), debounce=True, disabled=ref, step=1,
                      style={"width": "72px", "background": "#eee" if ref else "white"}),
        ], style={"margin": "0 8px 6px 0", "display": "flex", "flexDirection": "column",
                  "alignItems": "flex-start"}))
    return [html.Div("reference = ① (offset 0); others shift to match.  + = later, − = earlier.",
                     style={"fontSize": "10px", "color": "#555", "marginBottom": "3px",
                            "width": "100%"}),
            html.Div(cells, style={"display": "flex", "flexWrap": "wrap"})]


@app.callback(Output("align-seed", "data"),
              Input("align-auto", "n_clicks"), Input("align-reset", "n_clicks"),
              State("file", "value"), State("ttl", "value"), prevent_initial_call=True)
def set_align_seed(_a, _r, files, ttl_name):
    if ctx.triggered_id == "align-reset":
        return {}
    files = [f for f in (files or []) if f and loadable(f)]
    if len(files) < 2:
        return {}
    try:                                                 # reference = first file's flicker onset
        ref = get_flicker(files[0], ttl_name)
        ref_t0 = ref.t0 if ref else None
    except Exception:
        ref_t0 = None
    if ref_t0 is None:
        return {}
    seed = {}
    for p in files[1:]:                                  # each other file: shift its t0 onto ref's
        try:
            fl = get_flicker(p, ttl_name)
            if fl:
                seed[p] = round((ref_t0 - fl.t0) * 1000.0, 1)     # ms
        except Exception:
            pass
    return seed


@app.callback(Output("align-map", "data"),
              Input({"type": "align-trace", "path": ALL}, "value"),
              prevent_initial_call=True)
def collect_align(_vals):
    items = ctx.inputs_list[0] or []                     # each: {"id": {...,"path":p}, "value": ms}
    amap = {}
    for item in items:
        v = item.get("value")
        if v:                                            # skip 0/None (no shift)
            try:
                amap[item["id"]["path"]] = float(v) / 1000.0      # ms → s (build_figures uses s)
            except (TypeError, ValueError):
                pass
    return amap


# ---- data store: pick a cell -> load its recordings + prefill stimulus -------
def _cell_files(vals):
    """(file options, default-checked, sel-cell list, set of existing files) for date|cell value(s).
    Only files that still exist on disk are listed — so it also drops anything deleted in the
    Explorer."""
    vals = vals if isinstance(vals, list) else ([vals] if vals else [])
    ds = DataStore()
    opts, checked, sel_list, seen = [], [], [], set()
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
    return opts, checked, sel_list, seen


@app.callback(Output("file", "options", allow_duplicate=True),
              Output("file", "value", allow_duplicate=True),
              Output("sel-cell", "data"), Output("recent-cells", "data"),
              Input("cell-select", "value"), State("recent-cells", "data"),
              State("last-session", "data"), prevent_initial_call=True)
def pick_cell(vals, recent, sess):
    vals = vals if isinstance(vals, list) else ([vals] if vals else [])
    if not vals:
        return no_update, no_update, [], no_update
    opts, checked, sel_list, seen = _cell_files(vals)
    # restore the exact files last worked on, if they belong to this selection (resume on reload)
    sess_files = [f for f in ((sess or {}).get("files") or []) if f in seen]
    if sess_files:
        checked = sess_files
    recent = [x for x in (recent or []) if x not in vals]        # most-recent-first, de-duped
    recent = list(vals) + recent
    return opts, checked, sel_list, recent[:50]


# ---- returning to the Analysis View: re-sync its files + output gallery with the store, so
#      deletions made in the Explorer take effect WITHOUT a browser refresh (and a run won't
#      reference a now-deleted file). -----------------------------------------------------------
@app.callback(Output("file", "options", allow_duplicate=True),
              Output("file", "value", allow_duplicate=True),
              Output("sel-cell", "data", allow_duplicate=True),
              Output("gallery-trigger", "data", allow_duplicate=True),
              Input("exp-close", "n_clicks"),
              State("cell-select", "value"), State("file", "value"),
              prevent_initial_call=True)
def resync_on_close(_n, vals, checked):
    if not _n or not vals:
        return no_update, no_update, no_update, no_update
    opts, default, sel_list, seen = _cell_files(vals)
    keep = [f for f in (checked or []) if f in seen]    # drop files deleted in the Explorer
    return opts, (keep or default), sel_list, (_n or 0) + 1


def _store_fingerprint(vals):
    """Cheap signature of the selected cell(s) on disk: (#output PNGs + newest mtime) and the raw-file
    set. Lets the poll detect run-completion / Explorer deletions without a callback round-trip."""
    vals = vals if isinstance(vals, list) else ([vals] if vals else [])
    ds = DataStore()
    out, fil = [], []
    for v in vals:
        try:
            date, cell = v.split("|")
            cm = ds.cell(date, cell)
            od = cm.dir / "outputs"
            n, mt = 0, 0.0
            if od.exists():
                for p in od.rglob("*.png"):
                    n += 1
                    m = p.stat().st_mtime
                    if m > mt:
                        mt = m
            out.append(f"{v}#{n}#{mt:.1f}")
            files, _ = cell_default_files(cm)
            fil.append(v + "#" + ":".join(sorted(os.path.basename(f) for f in files)))
        except Exception:
            pass
    return {"out": "|".join(out), "files": "|".join(fil)}


# ---- POLL: every 3 s, diff the selected cell(s)' on-disk state. New/changed outputs (a finished
#      Run Analysis, or a figure deleted in the Explorer) refresh the gallery; a changed raw-file set
#      (files deleted in the Explorer) refreshes the file checklist. This force-refreshes the Analysis
#      View automatically, independent of background-callback delivery or navigation. ----------------
@app.callback(Output("gallery-trigger", "data", allow_duplicate=True),
              Output("file", "options", allow_duplicate=True),
              Output("file", "value", allow_duplicate=True),
              Output("store-fp", "data"),
              Input("poll", "n_intervals"),
              State("cell-select", "value"), State("file", "value"), State("store-fp", "data"),
              prevent_initial_call=True)
def poll_refresh(_n, vals, checked, prev):
    if not vals:
        return no_update, no_update, no_update, no_update
    fp = _store_fingerprint(vals)
    if not prev:                                        # first tick: establish baseline, no refresh
        return no_update, no_update, no_update, fp
    if prev.get("out") == fp["out"] and prev.get("files") == fp["files"]:
        return no_update, no_update, no_update, no_update      # nothing changed
    gal = ((_n or 0) + 1) if prev.get("out") != fp["out"] else no_update   # outputs changed → gallery
    if prev.get("files") != fp["files"]:                # raw files changed → refresh the checklist
        opts, default, _sel, seen = _cell_files(vals)
        keep = [f for f in (checked or []) if f in seen]
        return gal, opts, (keep or default), fp
    return gal, no_update, no_update, fp


# ---- persist the working session (cell + checked files) and auto-load it on startup --
@app.callback(Output("last-session", "data"),
              Input("cell-select", "value"), Input("file", "value"),
              prevent_initial_call=True)
def save_session(cellv, filev):
    return {"cell": cellv, "files": [f for f in (filev or []) if f]}


@app.callback(Output("cell-select", "value"),
              Input("once", "n_intervals"), State("last-session", "data"),
              prevent_initial_call=True)
def restore_session(_n, sess):
    cell = (sess or {}).get("cell")
    return cell if cell else no_update


# ---- re-sort the cell(s) dropdown (incl. "opened recently") ------------------
@app.callback(Output("cell-select", "options", allow_duplicate=True),
              Input("cell-sort", "value"), Input("recent-cells", "data"),
              prevent_initial_call="initial_duplicate")
def sort_cells(sort, recent):
    return store_cell_options(sort or "recent", recent=recent)


# ---- Data Explorer: fill the editable cell-metadata fields for the selected cell ----
@app.callback(Output("exp-cell-label", "children"),
              Output("stim-type", "value"), Output("stim-params", "value"),
              Output("cell-type", "value"), Output("cell-notes", "value"),
              Input("exp-cell", "data"), State("exp-date", "data"),
              prevent_initial_call=True)
def fill_cell_meta(cell, date):
    if not (date and cell):
        return "", None, None, None, ""
    cm = DataStore().cell(date, cell)
    label = cm.data.get("label") or ""
    title = f"{date} / {cell}" + (f" — {label}" if label else "")
    stype, sparams = None, None
    for r in cm.data.get("recordings", []):
        if r.get("stimulus"):
            stype = r["stimulus"].get("type")
            if stype == "flicker":              # legacy value -> the current "sq wave" option
                stype = "sq_wave"
            sparams = ", ".join(f"{k}={v}" for k, v in
                                (r["stimulus"].get("params") or {}).items() if v is not None)
            break
    return title, stype, sparams, cm.data.get("cell_type"), (cm.data.get("notes") or "")


# ---- Data Explorer: Save metadata (stimulus + cell type + notes) for the selected cell --
@app.callback(Output("exp-save-msg", "children"), Input("save-meta", "n_clicks"),
              State("exp-date", "data"), State("exp-cell", "data"),
              State("stim-type", "value"), State("stim-params", "value"),
              State("cell-type", "value"), State("cell-notes", "value"),
              prevent_initial_call=True)
def save_meta(_n, date, cell, stype, sparams, ctype, notes):
    if not (date and cell):
        return "select a cell in the Explorer first"
    ds = DataStore()
    cm = ds.cell(date, cell)
    params = parse_params(sparams)
    n = 0
    if stype:
        for r in cm.data.get("recordings", []):
            if r.get("kind", "recording") == "recording" and str(r.get("file", "")).endswith((".abf", ".csv")):
                cm.set_stimulus(r["id"], stype, params, source="user"); n += 1
    cm.data["cell_type"] = (ctype or None)
    cm.data["notes"] = notes or ""
    cm.save()
    ds.update_index()
    bits = [f"stimulus '{stype}' {params} → {n} recordings"] if stype else []
    bits.append(f"type={ctype or '—'}, notes saved")
    return "✓ " + "; ".join(bits)


# ---- Data Explorer: "Fix abf (Neitz)" — apply the standard lab config to a cell, non-destructive:
#      stimulus = sq wave (preserving any existing params, default frame_rate=60), verify the
#      Im_prime/Vm_sec/TTL channels, and clear the in-memory cache so the files are re-read fresh.
NEITZ_CHANNELS = ["Im_prime", "Vm_sec", "TTL"]


@app.callback(Output("exp-save-msg", "children", allow_duplicate=True),
              Output("stim-type", "value", allow_duplicate=True),
              Output("store-rev", "data", allow_duplicate=True),
              Output("exp-detail", "children", allow_duplicate=True),
              Input("fix-abf", "n_clicks"),
              State("exp-date", "data"), State("exp-cell", "data"), State("store-rev", "data"),
              prevent_initial_call=True)
def fix_abf(_n, date, cell, rev):
    if not (date and cell):
        return "select a cell in the Explorer first", no_update, no_update, no_update
    ds = DataStore()
    cm = ds.cell(date, cell)
    n, warns = 0, []
    for r in cm.data.get("recordings", []):
        f = str(r.get("file", ""))
        if not f.endswith(".abf"):
            continue
        n += 1
        p = dict((r.get("stimulus") or {}).get("params") or {})   # preserve existing params
        p.setdefault("frame_rate", 60)                            # lab-standard display rate
        cm.set_stimulus(r["id"], "sq_wave", p, source="neitz-fix")
        _CACHE.pop(str(cm.dir / f), None)                         # force a fresh re-read
    # verify the channel config on the first recording (non-destructive — just warn on mismatch)
    try:
        first = next((str(cm.dir / r["file"]) for r in cm.data.get("recordings", [])
                      if str(r.get("file", "")).endswith(".abf")), None)
        if first:
            chans = list(load_recording(first).channel_names)
            missing = [c for c in NEITZ_CHANNELS if c not in chans]
            if missing:
                warns.append(f"⚠ channels {chans} missing {missing} — not the Neitz config!")
    except Exception as e:
        warns.append(f"⚠ channel check failed: {e}")
    cm.save()
    ds.update_index()
    msg = (f"✓ Neitz config applied to {n} recording(s): stimulus = sq wave, "
           f"channels Im_prime/Vm_sec/TTL, files re-read fresh")
    if warns:
        msg += "   " + "; ".join(warns)
    return msg, "sq_wave", (rev or 0) + 1, explorer_detail(date, cell)


def _attach_output_files(ds, date, cell, analysis, saved):
    """Record extra files (e.g. the 4K exports) on the just-written output for <analysis>."""
    cm = ds.cell(date, cell)
    rec = next((o for o in reversed(cm.data.get("outputs", [])) if o.get("analysis") == analysis), None)
    if rec is not None:
        rec.setdefault("files", {}).update(
            {k: os.path.relpath(str(p), cm.dir) for k, p in saved.items()})
        cm.save()
        ds.update_index()


# ---- run the analysis for the selected cell(s), dispatched by their stimulus type --
@app.callback(Output("store-msg", "children", allow_duplicate=True),
              Output("gallery-trigger", "data", allow_duplicate=True),
              Input("run-cell", "n_clicks"), State("sel-cell", "data"),
              State("file", "value"), State("run-name", "value"),
              State("polarity", "value"), State("method", "value"), State("k", "value"),
              State("absth", "value"), State("refr", "value"), State("absth-map", "data"),
              State("chan", "value"), State("ttl", "value"),
              State("region-start", "value"), State("region-end", "value"),
              State("region-mode", "value"), State("stagger-pct", "value"),
              State("disp-show", "value"), State("disp-binned", "value"), State("train-bin", "value"),
              State("fft-bin", "value"), State("align-map", "data"),
              background=True,                              # run off the UI thread (analysis + 4K
              running=[(Output("run-cell", "disabled"), True, False),   # exports take ~1-2 min);
                       (Output("run-cell", "children"), "⏳ Running… (~1-2 min)", "▶ Run analysis")],
              prevent_initial_call=True)                   # outputs auto-refresh the UI when done
def run_cell(_n, sel, checked, run_name, polarity, method, k, absth, refr, absth_map,
             chan, ttl, rstart, rend, region_mode, stagger_pct, disp_show, disp_binned, train_bin,
             fft_bin, align_map):
    import re
    sels = sel if isinstance(sel, list) else ([sel] if sel else [])
    if not sels:
        return "pick a cell first", no_update
    raw_name = (run_name or "").strip()                  # the user's friendly run name (preserved)
    user_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name)
    detect = dict(polarity=polarity, method=method,      # the live GUI spike-detection settings
                  k=float(k) if k is not None else None,
                  abs_threshold=float(absth) if absth is not None else None,
                  refractory_s=(float(refr) / 1000.0) if refr else 0.002)
    abs_map = absth_map or None                          # per-trace abs thresholds {path: value}
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
                r = run_cell_noise(ds, s["date"], s["cell"], name=nm, run_label=raw_name or None)
                sm = r.summary[0]
                msgs.append(f"{tag} [{nm}]: STA {sm['n_epochs']} epochs, "
                            f"peak {sm['peak_ms']:.1f} ms {sm['peak_sign']}")
            else:                                        # default: square-wave (flicker) ON/OFF
                nm = user_name or "flicker"
                cmdir = str(cm.dir)
                sub = [f for f in checked if f.startswith(cmdir) and f.endswith(".abf")]
                r = run_cell_flicker(ds, s["date"], s["cell"], n_shuffle=500,
                                     name=nm, include=(sub or None),
                                     detect=detect, abs_map=abs_map, run_label=raw_name or None)
                p = r.tables["pooled_onoff"][0]
                msgs.append(f"{tag} [{nm}]: {len(r.summary)} file(s), "
                            f"{p['flicker_hz']:.1f} Hz, '{p['verdict']}'")
                try:                                     # also save the 4K window exports (kaleido)
                    exp_files = sub or [str(cm.dir / rr["file"]) for rr in cm.data.get("recordings", [])
                                        if str(rr.get("file", "")).endswith(".abf")]
                    saved = export_window_figures(
                        cm.output_dir(nm), f"{tag} [{raw_name or nm}]",
                        files=exp_files, chan=chan, ttl_name=ttl, polarity=polarity, method=method,
                        k=k, absth=absth, refr=refr, rstart=rstart, rend=rend, region_mode=region_mode,
                        stagger_pct=stagger_pct, disp_show=disp_show, disp_binned=disp_binned,
                        train_bin=train_bin, absth_map=abs_map, fft_bin=fft_bin, align_map=align_map)
                    if saved:
                        _attach_output_files(ds, s["date"], s["cell"], nm, saved)
                        msgs[-1] += f" +{len(saved)} 4K file(s)"
                except Exception as e:
                    msgs[-1] += f" (4K export skipped: {e})"
        except SystemExit as e:
            msgs.append(f"{tag}: {e}")
        except Exception as e:
            msgs.append(f"{tag}: error {e}")
    banner = html.Div([
        html.Span("✓ Analysis complete", style={"fontWeight": "bold", "fontSize": "16px"}),
        html.Div("  ·  ".join(msgs), style={"fontSize": "12px", "marginTop": "3px"}),
        html.Div("outputs saved — see the gallery below ↓",
                 style={"fontSize": "11px", "marginTop": "2px", "opacity": 0.8}),
    ], style={"background": "#e7f6e7", "border": "1.5px solid #4fae4f", "borderRadius": "6px",
              "padding": "9px 11px", "color": "#0a5a0a"})
    return banner, (_n or 1)


# instant feedback the moment "Run analysis" is clicked (the server run_cell — analysis + 4K
# kaleido exports — can take ~30-60s; the dcc.Loading spinner around #store-msg also spins).
app.clientside_callback(
    """function(n){
        if(!n) return window.dash_clientside.no_update;
        var H = 'dash_html_components';
        return {namespace:H, type:'Div', props:{
            children:[
                {namespace:H, type:'Span', props:{children:'⏳ Running analysis…',
                    style:{fontWeight:'bold', fontSize:'16px'}}},
                {namespace:H, type:'Div', props:{children:'computing + writing the 4K figures (~2–4 min) — please wait; outputs appear below as they finish',
                    style:{fontSize:'11px', marginTop:'2px', opacity:0.85}}}
            ],
            style:{background:'#fff6e0', border:'1.5px solid #e0a93a', borderRadius:'6px',
                   padding:'9px 11px', color:'#8a5a00'}}};
    }""",
    Output("store-msg", "children", allow_duplicate=True),
    Input("run-cell", "n_clicks"), prevent_initial_call=True)


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
            // close the cell(s) dropdown only when the pointer is over NEITHER the control NOR
            // its open menu (the menu — with Search / Select-all — is portalled outside the wrap)
            document.addEventListener('mouseout', function(e) {
                var wrap = document.getElementById('cell-select-wrap');
                if (!wrap) return;
                var inp = wrap.querySelector('input');
                var lbId = inp && inp.getAttribute('aria-controls');
                var lb = lbId ? document.getElementById(lbId) : null;
                var menu = lb ? (lb.parentElement || lb)
                              : document.querySelector('.Select__menu, .Select-menu-outer');
                if (!menu) return;                          // menu not open -> nothing to close
                var inRegion = function(n) {
                    return !!(n && (wrap.contains(n) || menu.contains(n)));
                };
                if (inRegion(e.target) && !inRegion(e.relatedTarget)) {
                    if (inp) {
                        inp.dispatchEvent(new KeyboardEvent('keydown',
                            {key: 'Escape', code: 'Escape', keyCode: 27, which: 27, bubbles: true}));
                        inp.blur();
                    }
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
def _date_cell_of(path):
    """(date, cell) parsed from a store file path <root>/<date>/<cell>/…, else (None, None)."""
    try:
        rel = os.path.relpath(path, str(DataStore().root))
        parts = rel.split(os.sep)
        if len(parts) >= 2 and not parts[0].startswith(".."):
            return parts[0], parts[1]
    except Exception:
        pass
    return None, None


@app.callback(Output("explorer-modal", "style"),
              Output("exp-date", "data", allow_duplicate=True),
              Output("exp-cell", "data", allow_duplicate=True),
              Output("exp-autosel", "data"),
              Input("open-explorer", "n_clicks"), Input("exp-close", "n_clicks"),
              State("file", "value"), prevent_initial_call=True)
def toggle_explorer(_open, _close, files):
    if ctx.triggered_id == "exp-close":
        return {"display": "none"}, no_update, no_update, None
    sel = [f for f in (files or []) if f]
    if len(sel) == 1:                          # exactly one file checked in Analysis View ->
        d, c = _date_cell_of(sel[0])           # jump straight to it and pre-check it
        if d and c:
            return _EXPLORER_SHOWN, d, c, sel[0]
    dates = sorted({c["date"] for c in DataStore().index()}, reverse=True)
    return _EXPLORER_SHOWN, (dates[0] if dates else None), None, None


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
    if not _n:                 # ignore the n_clicks reset when the breadcrumb rebuilds
        return no_update
    return None


@app.callback(Output("exp-cards", "children"),
              Output("exp-files", "options"), Output("exp-files", "value"),
              Output("exp-detail", "children"), Output("exp-breadcrumb", "children"),
              Output("exp-prev", "children"),
              Output("exp-autosel", "data", allow_duplicate=True),
              Input("exp-date", "data"), Input("exp-cell", "data"),
              State("exp-autosel", "data"), prevent_initial_call=True)
def exp_render(date, cell, autosel):
    # reset the hover-preview on every navigation (no stale graph showing)
    prev = [html.Img(id="exp-prev-img", style={"width": "100%", "height": "100%",
                                               "objectFit": "contain", "display": "none"}),
            html.Span("hover any graph to preview it here", id="exp-prev-hint",
                      style={"color": "#999", "fontSize": "12px"})]
    if not date:
        return ([no_update] * 5) + [prev, no_update]
    bc = explorer_breadcrumb(date, cell)
    if not cell:                                       # DAY view: cell thumbnails
        hint = [html.Div("select a cell to see its files + manifest",
                         style={"color": "#999", "fontSize": "12px"})]
        return explorer_day_cards(date), [], [], hint, bc, prev, no_update
    # CELL view: file checklist + manifest JSON
    opts = explorer_file_options(date, cell)
    checked, consume = [], no_update
    if autosel:                                        # pre-check the file we jumped in from
        if autosel in {o["value"] for o in opts}:
            checked = [autosel]
        consume = None                                 # one-shot: clear after this render
    return ([], opts, checked,
            explorer_detail(date, cell), bc, prev, consume)


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
              Output("exp-open-viewer", "style"),
              Input("exp-files", "value"), prevent_initial_call=False)
def del_files_button(sel):
    sel = [s for s in (sel or []) if s]
    base = {"color": "white", "background": "#b00", "border": "none", "borderRadius": "4px",
            "padding": "4px 10px", "fontWeight": "bold", "fontSize": "12px", "cursor": "pointer"}
    if not sel:                                          # nothing chosen → hide both action buttons
        return {"display": "none"}, "", {"display": "none"}
    return (dict(base, display="inline-block"), f"🗑 delete {len(sel)} selected",
            {"display": "inline-block", "fontWeight": "bold"})


# ---- Data Explorer: reveal the SELECTED output figures' location(s) in the OS file browser ------
@app.callback(Output("finder-msg", "children"),
              Input("open-outputs-finder", "n_clicks"),
              State({"type": "out-check", "src": ALL}, "value"), prevent_initial_call=True)
def open_outputs_in_finder(_n, out_checks):
    checked = [v[0] for v in (out_checks or []) if v]
    if not checked:
        return "select figure(s) first"
    folders = sorted({os.path.dirname(p) for p in checked})
    try:
        sysname = platform.system()
        if sysname == "Darwin":
            subprocess.run(["open", "-R", *checked])          # reveal + select all in Finder
            return f"✓ revealed {len(checked)} figure(s) in Finder"
        opener = ["explorer"] if sysname == "Windows" else ["xdg-open"]
        for d in folders:
            subprocess.run(opener + [d])
        return f"✓ opened {len(folders)} location(s)"
    except Exception as e:
        return f"open failed: {e}"


# ---- delete: open the (warning-only) confirmation modal -------------------------
#   targets: a whole DAY {kind:"date", date} · a cell's checked files / a figure {date,cell,paths}
@app.callback(Output("del-modal", "style"), Output("del-targets", "data"),
              Output("del-list", "children"),
              Input({"type": "del-date", "date": ALL}, "n_clicks"),
              Input({"type": "del-output", "src": ALL}, "n_clicks"),
              Input("del-files", "n_clicks"), Input("del-outputs", "n_clicks"),
              State("exp-files", "value"), State("exp-date", "data"), State("exp-cell", "data"),
              State({"type": "out-check", "src": ALL}, "value"),
              prevent_initial_call=True)
def open_delete(_ddates, _douts, _dfiles, _dmulti, sel, date, cell, out_checks):
    trig = ctx.triggered_id
    fired = bool(ctx.triggered and ctx.triggered[0].get("value"))
    if trig == "del-outputs":                                           # multiple checked figures
        checked = [v[0] for v in (out_checks or []) if v]               # each box is [] or [path]
        if not fired or not checked:
            return (no_update,) * 3
        paths = sorted({str(x) for src in checked for x in Path(src).parent.glob(Path(src).stem + ".*")})
        lst = [html.Div(f"{len(checked)} selected figure(s) (+ same-name pdf/svg/png):",
                        style={"fontWeight": "bold", "color": "#b00"})] \
            + [html.Div(os.path.basename(x)) for x in paths]
        return _DEL_SHOWN, {"date": date, "cell": cell, "paths": paths}, lst
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
@app.callback(Output("exp-save-msg", "children", allow_duplicate=True),
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
