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
import glob
import base64
import platform
import subprocess
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update, ALL

from neitz.io import load_recording
from neitz.spikes import detect_spikes
from neitz.analysis import flicker as flk
from neitz.dataio import DataStore
from neitz.run import run_cell_flicker

# default browse location is the managed data store (~/Documents/ephysdataio)
EPHYS_ROOT = os.path.expanduser(os.environ.get("EPHYSDATAIO_ROOT", "~/Documents/ephysdataio"))
DEFAULT_GLOBS = [os.path.join(EPHYS_ROOT, "**", "*.abf"),
                 os.path.join(EPHYS_ROOT, "**", "*.csv")]
DEFAULT_DIR = EPHYS_ROOT if os.path.isdir(EPHYS_ROOT) else os.getcwd()
_last_dir = DEFAULT_DIR          # remembered folder; native dialogs open here
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


def spectrum(rate, bin_rate=BIN_RATE):
    r = rate - rate.mean()
    amp = np.abs(np.fft.rfft(r))
    f = np.fft.rfftfreq(len(r), d=1.0 / bin_rate)
    keep = f <= FMAX
    return f[keep], amp[keep]


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
def store_cell_options():
    try:
        idx = DataStore().index()
    except Exception:
        idx = []
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
                "zIndex": 2000, "alignItems": "center", "justifyContent": "center"}


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

        html.Div("Neitz ABF Viewer", style={"fontWeight": "bold", "fontSize": "15px",
                                            "marginBottom": "8px"}),

        # ---- compartment: data store ----
        card("Data store", [
            html.Button("📥 Import data…", id="import-data", n_clicks=0,
                        style={"fontWeight": "bold", "width": "100%", "marginBottom": "6px"}),
            html.Div([html.Label("cell", style=_LBL),
                      dcc.Dropdown(id="cell-select", options=store_cell_options(),
                                   placeholder="pick a date / cell…", style={"width": "100%"})],
                     style=_FIELD),
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
                html.Button("▶ Run sq wave", id="run-cell", n_clicks=0,
                            style={"marginLeft": "4px"}),
                html.Button("⤓ Backup mirror", id="backup-mirror", n_clicks=0,
                            style={"marginLeft": "4px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
            html.Div(id="store-msg", style={"fontSize": "11px", "color": "#070",
                                            "marginTop": "6px"}),
        ]),

        # ---- compartment: files ----
        card("Files", [
            html.Div([html.Button("📁 File…", id="browse-file", n_clicks=0),
                      html.Button("📂 Folder…", id="browse-folder", n_clicks=0,
                                  style={"marginLeft": "6px"})], style={"marginBottom": "6px"}),
            html.Div(dcc.Checklist(id="file", options=file_options(_files),
                                   value=[_files[0]] if _files else [],
                                   labelStyle={"display": "block", "fontSize": "11px",
                                               "whiteSpace": "nowrap", "overflow": "hidden",
                                               "textOverflow": "ellipsis"},
                                   inputStyle={"marginRight": "4px"}),
                     style={"maxHeight": "150px", "overflowY": "auto",
                            "border": "1px solid #ccc", "padding": "4px", "background": "white"}),
            html.Div(id="meta", style={"fontSize": "10px", "color": "#333", "lineHeight": "1.45",
                                       "background": "#f6f6f6", "padding": "6px",
                                       "borderRadius": "4px", "marginTop": "6px"}),
        ]),

        # ---- compartment: channels & spike detection ----
        card("Channels & spike detection", [
            html.Div([html.Label("signal channel", style=_LBL),
                      dcc.Dropdown(id="chan", style={"width": "100%"})], style=_FIELD),
            html.Div([html.Label("TTL channel", style=_LBL),
                      dcc.Dropdown(id="ttl", style={"width": "100%"})], style=_FIELD),
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
                html.Div([html.Label("abs thresh", style=_LBL),
                          dcc.Input(id="absth", type="number", value=20, debounce=True,
                                    style={"width": "85px"}, **PERSIST)]),
                html.Div([html.Label("refractory (ms)", style=_LBL),
                          dcc.Input(id="refr", type="number", value=2, debounce=True,
                                    style={"width": "85px"}, **PERSIST)],
                         style={"marginLeft": "10px"}),
            ], style={"display": "flex"}),
            html.Button("🎯 auto abs (per trace)", id="auto-absth", n_clicks=0,
                        style={"marginTop": "6px", "fontSize": "11px", "width": "100%"}),
            html.Div(id="absth-msg", style={"fontSize": "10px", "color": "#666",
                                            "marginTop": "3px", "wordBreak": "break-all"}),
        ]),

        # ---- compartment: region & display ----
        card("Region & display", [
            html.Div([
                html.Div([html.Label("region start (s)", style=_LBL),
                          dcc.Input(id="region-start", type="number", debounce=True,
                                    style={"width": "95px"})]),
                html.Div([html.Label("region end (s)", style=_LBL),
                          dcc.Input(id="region-end", type="number", debounce=True,
                                    style={"width": "95px"})], style={"marginLeft": "10px"}),
            ], style={"display": "flex", "marginBottom": "6px"}),
            html.Div([html.Label("excluded regions", style=_LBL),
                      dcc.RadioItems(id="region-mode",
                                     options=[{"label": " show", "value": "show"},
                                              {"label": " crop", "value": "crop"},
                                              {"label": " baseline", "value": "baseline"}],
                                     value="show", inline=True,
                                     labelStyle={"fontSize": "12px", "marginRight": "8px"},
                                     **PERSIST)], style=_FIELD),
            html.Div([html.Label("display", style=_LBL),
                      dcc.Checklist(id="dispopts",
                                    options=[{"label": " stagger frame syncs", "value": "stagger"},
                                             {"label": " hide detected spikes", "value": "hide_spikes"},
                                             {"label": " spike-train view (0/1)", "value": "spike_train"}],
                                    value=[], labelStyle={"display": "block", "fontSize": "12px"},
                                    **PERSIST)], style=_FIELD),
            html.Div([html.Label("spike-train bin (ms, 0=impulses)", style=_LBL),
                      dcc.Input(id="train-bin", type="number", value=0, min=0, debounce=True,
                                style={"width": "110px"}, **PERSIST)], style=_FIELD),
        ]),

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
        # signal + frame-sync (frame-sync row enlarged) — gets the lion's share of height
        html.Div(dcc.Graph(id="time", style={"height": "100%"}, config={"responsive": True}),
                 style={"flex": "3 1 0", "minHeight": 0}),
        # bottom strip (less tall): FFT at half width + ISI histogram at the other half
        html.Div(style={"flex": "1 1 0", "minHeight": 0, "display": "flex", "gap": "6px"},
                 children=[
            html.Div(dcc.Graph(id="fft", style={"height": "100%"}, config={"responsive": True}),
                     style={"flex": "1 1 0", "minWidth": 0}),
            html.Div(dcc.Graph(id="isi", style={"height": "100%"}, config={"responsive": True}),
                     style={"flex": "1 1 0", "minWidth": 0}),
        ]),
    ]),

    # ---- invisible state + overlays ----
    dcc.Store(id="sel-cell"),
    dcc.Store(id="gallery-trigger"),
    dcc.Store(id="absth-map"),                            # {file path: per-trace abs threshold}
    dcc.Store(id="last-folder", storage_type="local"),   # remembers data folder across sessions
    dcc.Interval(id="once", interval=300, max_intervals=1),
    # ---- full-screen pop-out for an output image ----
    html.Div(id="output-modal", style={"display": "none"}, children=[
        html.Button("✕ close", id="modal-close", n_clicks=0,
                    style={"position": "absolute", "top": "12px", "right": "16px",
                           "fontSize": "15px", "padding": "4px 10px"}),
        html.Img(id="modal-img", style={"maxWidth": "94vw", "maxHeight": "92vh",
                                        "boxShadow": "0 0 24px #000", "background": "white"}),
    ]),
])


# ---- native dialogs -> file checklist --------------------------------------
@app.callback(Output("file", "options"), Output("file", "value"), Output("last-folder", "data"),
              Input("browse-file", "n_clicks"), Input("browse-folder", "n_clicks"),
              State("file", "options"), prevent_initial_call=True)
def browse(_bf, _bfo, cur_opts):
    global _last_dir
    trig = ctx.triggered_id
    if trig == "browse-file":
        path = native_choose_file()
        if not path:
            return no_update, no_update, no_update
        _last_dir = os.path.dirname(path) or _last_dir
        opts = list(cur_opts or [])
        if path not in [o["value"] for o in opts]:
            opts = file_options([path]) + opts
        return opts, [path], no_update
    if trig == "browse-folder":
        folder = native_choose_folder()
        if not folder:
            return no_update, no_update, no_update
        files = discover_abf(folder)
        if not files:
            return no_update, no_update, no_update
        _last_dir = folder
        return file_options(files), [files[0]], folder
    return no_update, no_update, no_update


# ---- restore the remembered data folder on page load (once) ----------------
@app.callback(Output("file", "options", allow_duplicate=True),
              Output("file", "value", allow_duplicate=True),
              Input("once", "n_intervals"), State("last-folder", "data"),
              prevent_initial_call=True)
def restore_folder(_n, last_folder):
    global _last_dir
    if last_folder:
        files = discover_abf(last_folder)
        if files:
            _last_dir = last_folder
            return file_options(files), []
    return no_update, no_update


# ---- load first file -> channels, metadata, auto-fill region ---------------
@app.callback(Output("chan", "options"), Output("chan", "value"),
              Output("ttl", "options"), Output("ttl", "value"), Output("meta", "children"),
              Output("region-start", "value"), Output("region-end", "value"),
              Input("file", "value"), prevent_initial_call=False)
def load_meta(files):
    files = files or []
    if not files:
        return [], None, [], None, "No file selected.", None, None
    try:
        rec = get_recording(files[0])
    except Exception as e:
        return [], None, [], None, f"Failed to load: {e}", None, None
    opts = [{"label": n, "value": n} for n in rec.channel_names]
    ttl_default = next((n for n in rec.channel_names if "ttl" in n.lower()), rec.channel_names[-1])
    fl = get_flicker(files[0], ttl_default)
    rstart = round(fl.t0, 2) if fl else 0.0
    rend = round(fl.t1, 2) if fl else round(rec.duration, 2)
    md = rec.metadata()
    fields = "  ·  ".join(str(md[k]) for k in
                          ["file", "protocol", "sample rate", "duration", "channels",
                           "recorded", "creator"] if k in md)
    n = len(files)
    txt = f"{n} file{'s' if n != 1 else ''} selected (region auto-filled from file 1)   ·   {fields}"
    return opts, rec.channel_names[0], opts, ttl_default, txt, rstart, rend


# ---- render time + fft -----------------------------------------------------
@app.callback(Output("time", "figure"), Output("fft", "figure"), Output("isi", "figure"),
              Output("readout", "children"),
              Input("file", "value"), Input("chan", "value"), Input("ttl", "value"),
              Input("polarity", "value"), Input("method", "value"), Input("k", "value"),
              Input("absth", "value"), Input("refr", "value"),
              Input("region-start", "value"), Input("region-end", "value"),
              Input("dispopts", "value"), Input("region-mode", "value"), Input("train-bin", "value"),
              Input("absth-map", "data"), Input("time", "relayoutData"), prevent_initial_call=True)
def render(files, chan, ttl_name, polarity, method, k, absth, refr, rstart, rend,
           dispopts, region_mode, train_bin, absth_map, relayout):
    files = [f for f in (files or []) if f]
    if not files:
        return blank_fig("No file selected"), blank_fig(""), blank_fig(""), "No file selected."
    if not chan:
        return no_update, no_update, no_update, no_update

    det = dict(polarity=polarity, method=method, k=float(k),
               abs_threshold=float(absth) if absth is not None else None,
               refractory_s=(float(refr) / 1000.0) if refr else 0.002)
    amap = absth_map or {}                          # per-trace absolute thresholds
    multi = len(files) > 1
    opts = dispopts or []
    stagger = "stagger" in opts
    hide_spikes = "hide_spikes" in opts
    spike_train = "spike_train" in opts
    region_mode = region_mode or "show"
    crop = region_mode == "crop"           # show only the analysis region (drop excluded blocks)
    baseline_mode = region_mode == "baseline"   # full axis, but flatten excluded blocks to each channel's baseline
    show_spikes = not hide_spikes          # spikes shown by default (single AND multi)
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

    # vertical stagger step for frame syncs (from first file's TTL peak-to-peak)
    ttl_step = 0.0
    if stagger and ttl_name and ttl_name != chan:
        try:
            ttl_step = 1.3 * float(np.ptp(get_channel(files[0], ttl_name)))
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
        excl = (t < rs) | (t > re_)                       # excluded-region mask
        inmask = ~excl
        if baseline_mode:                                 # flatten excluded region to in-region baseline (median)
            base_y = float(np.median(y[inmask])) if inmask.any() else float(np.median(y))
            y_disp = np.where(excl, base_y, y)
        else:
            y_disp = y
        disp_spikes = in_reg if (baseline_mode or crop) else st.times   # hide excluded spikes when baseline/crop

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
                if baseline_mode:                          # flatten excluded frame-sync to its baseline (low level)
                    base_ttl = float(np.median(ttl[inmask])) if inmask.any() else float(np.median(ttl))
                    ttl_disp = np.where(excl, base_ttl, ttl)
                else:
                    ttl_disp = ttl
                tt, ty = minmax_decimate((np.arange(len(ttl)) / fs)[i0:i1], ttl_disp[i0:i1])
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
            f, amp = spectrum(rate)
            fft_fig.add_trace(go.Scatter(x=f, y=amp, mode="lines", legendgroup=name,
                                         line=dict(width=(1 if multi else 2), color=color),
                                         opacity=(0.45 if multi else 1.0), name=name))
        thr_txt = (f", thr {det_i['abs_threshold']:.1f}"
                   if (method in ("abs", "mad_floor") and det_i["abs_threshold"] is not None)
                   else "")
        readbits.append(f"{name}: {len(in_reg)} spk in region" + thr_txt
                        + (f", stim {fl.freq:.2f}Hz" if fl else ", no flicker"))

    if not crop:                                   # shade excluded blocks (skip when cropped out)
        for (a, b, lbl) in [(t0_full, rs, "excluded (adapting)"), (re_, t1_full, "excluded")]:
            if b > a + 1e-6:
                time_fig.add_vrect(x0=a, x1=b, fillcolor="gray", opacity=0.13, line_width=0,
                                   annotation_text=lbl, annotation_position="top left",
                                   annotation=dict(font_size=10), row="all", col=1)

    ttl_ylab = (f"{ttl_name} (staggered)" if (stagger and multi) else (ttl_name or "TTL"))
    time_fig.update_yaxes(title_text=row1_ylab, row=1, col=1)
    time_fig.update_yaxes(title_text=ttl_ylab, row=2, col=1)
    time_fig.update_xaxes(title_text="time (s)", row=2, col=1, range=[x0, x1])
    time_fig.update_layout(margin=dict(l=55, r=20, t=30, b=40), uirevision="keep",
                           legend=dict(orientation="h", y=1.12), showlegend=multi)

    # FFT: group average + stim marker
    if multi and len(per_file_rates) >= 2:
        n = min(len(r) for r in per_file_rates)
        f, amp = spectrum(np.mean([r[:n] for r in per_file_rates], axis=0))
        fft_fig.add_trace(go.Scatter(x=f, y=amp, mode="lines",
                                     line=dict(width=3, color="black"), name="GROUP AVG"))
    sfreqs = [s for s in stim_freqs if s]
    if sfreqs:
        sf = float(np.mean(sfreqs))
        fft_fig.add_vline(x=sf, line_dash="dash", line_color="orange",
                          annotation_text=f"stim {sf:.2f} Hz",
                          annotation_position="bottom right",
                          annotation=dict(font=dict(size=10, color="#c60")))
    fft_fig.update_layout(
        title=dict(text="spike-train spectrum (inside region)", x=0.5, xanchor="center",
                   y=0.97, yanchor="top", font=dict(size=12)),
        xaxis_title="frequency (Hz)", yaxis_title="amplitude",
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
    mode = (f"GROUP of {len(files)} (avg→FFT; {view} view"
            + ("; spikes shown" if (show_spikes and not spike_train) else "")
            + "; frame syncs overlaid)" if multi else f"single-file inspect ({view})")
    readout = f"[{mode}]  region {rs:.2f}-{re_:.2f}s  |  " + "  |  ".join(readbits)
    return time_fig, fft_fig, isi_fig, readout


# ---- auto absolute threshold: one value PER TRACE (k·MAD of each file) --------
@app.callback(Output("absth-map", "data"), Output("absth-msg", "children"),
              Output("method", "value", allow_duplicate=True),
              Input("auto-absth", "n_clicks"),
              State("file", "value"), State("chan", "value"), State("k", "value"),
              prevent_initial_call=True)
def auto_absth(_n, files, chan, k):
    files = [f for f in (files or []) if f]
    if not files or not chan:
        return no_update, "select file(s) + a signal channel first", no_update
    amap, bits = {}, []
    for path in files:
        try:
            y = get_channel(path, chan)
            sigma = float(np.median(np.abs(y - np.median(y))) * 1.4826)   # robust σ
            thr = round(float(k) * sigma, 2)
            amap[path] = thr
            bits.append(f"{os.path.basename(path)}={thr:g}")
        except Exception:
            pass
    if not amap:
        return no_update, "could not compute thresholds", no_update
    return amap, "auto abs (k·MAD per trace) → " + ", ".join(bits), "abs"


# ---- typing a single abs value reverts to ONE uniform threshold for all traces -
@app.callback(Output("absth-map", "data", allow_duplicate=True),
              Output("absth-msg", "children", allow_duplicate=True),
              Input("absth", "value"), prevent_initial_call=True)
def clear_absmap(_v):
    return {}, "abs thresh: uniform across all traces"


# ---- data store: pick a cell -> load its recordings + prefill stimulus -------
@app.callback(Output("file", "options", allow_duplicate=True),
              Output("file", "value", allow_duplicate=True),
              Output("sel-cell", "data"), Output("stim-type", "value"),
              Output("stim-params", "value"),
              Input("cell-select", "value"), prevent_initial_call=True)
def pick_cell(val):
    if not val:
        return no_update, no_update, None, None, None
    date, cell = val.split("|")
    cm = DataStore().cell(date, cell)
    files = cell_data_files(cm)
    opts = [{"label": " " + os.path.basename(p), "value": p} for p in files]
    stype, sparams = None, None
    for r in cm.data.get("recordings", []):          # prefill from the first stimulus found
        if r.get("stimulus"):
            stype = r["stimulus"].get("type")
            sparams = ", ".join(f"{k}={v}" for k, v in (r["stimulus"].get("params") or {}).items()
                                if v is not None)
            break
    return opts, files, {"date": date, "cell": cell}, stype, sparams


# ---- save stimulus metadata to the cell's data recordings --------------------
@app.callback(Output("store-msg", "children"), Input("save-meta", "n_clicks"),
              State("sel-cell", "data"), State("stim-type", "value"),
              State("stim-params", "value"), prevent_initial_call=True)
def save_meta(_n, sel, stype, sparams):
    if not sel or not stype:
        return "pick a cell and a stimulus type first"
    cm = DataStore().cell(sel["date"], sel["cell"])
    params = parse_params(sparams)
    n = 0
    for r in cm.data.get("recordings", []):
        if r.get("kind", "recording") == "recording" and str(r.get("file", "")).endswith((".abf", ".csv")):
            cm.set_stimulus(r["id"], stype, params, source="user"); n += 1
    cm.save(); DataStore().update_index()
    return f"saved stimulus '{stype}' {params} to {n} recordings in {sel['date']}/{sel['cell']}"


# ---- run the flicker analysis on the selected cell ---------------------------
@app.callback(Output("store-msg", "children", allow_duplicate=True),
              Output("gallery-trigger", "data", allow_duplicate=True),
              Input("run-cell", "n_clicks"), State("sel-cell", "data"),
              prevent_initial_call=True)
def run_cell(_n, sel):
    if not sel:
        return "pick a cell first", no_update
    try:
        res = run_cell_flicker(DataStore(), sel["date"], sel["cell"], n_shuffle=500)
        p = res.tables["pooled_onoff"][0]
        return (f"ran sq wave on {sel['date']}/{sel['cell']}: {p['n_trials']} trials, "
                f"{p['flicker_hz']:.1f} Hz, verdict '{p['verdict']}' — "
                f"outputs saved (png/pdf/svg + csv + json); see the gallery above"), _n
    except SystemExit as e:
        return str(e), no_update
    except Exception as e:
        return f"error: {e}", no_update


# ---- output-image gallery for the selected cell + full-screen pop-out --------
@app.callback(Output("outputs-gallery", "children"),
              Input("cell-select", "value"), Input("gallery-trigger", "data"),
              prevent_initial_call=False)
def build_gallery(cell_val, _trig):
    if not cell_val:
        return [html.Span("pick a cell to see its output images",
                          style={"color": "#888", "fontSize": "12px"})]
    try:
        date, cell = cell_val.split("|")
        return output_gallery(date, cell)
    except Exception as e:
        return [html.Span(f"(no outputs: {e})", style={"color": "#888", "fontSize": "12px"})]


@app.callback(Output("output-modal", "style"), Output("modal-img", "src"),
              Input({"type": "out-thumb", "src": ALL}, "n_clicks"),
              Input("modal-close", "n_clicks"), prevent_initial_call=True)
def toggle_modal(_thumbs, _close):
    trig = ctx.triggered_id
    if trig == "modal-close":
        return {"display": "none"}, no_update
    if isinstance(trig, dict) and trig.get("type") == "out-thumb":
        if ctx.triggered and ctx.triggered[0].get("value"):       # a real click
            return _MODAL_SHOWN, _img_datauri(trig["src"])
    return no_update, no_update


# ---- import new experiment data into the store (copies + auto-groups by date) --
@app.callback(Output("cell-select", "options", allow_duplicate=True),
              Output("store-msg", "children", allow_duplicate=True),
              Input("import-data", "n_clicks"),
              State("stim-type", "value"), State("stim-params", "value"),
              prevent_initial_call=True)
def import_data(_n, stype, sparams):
    import re
    folder = native_choose_folder()
    if not folder:
        return no_update, no_update
    abfs = sorted(glob.glob(os.path.join(folder, "**", "*.abf"), recursive=True))
    if not abfs:
        return no_update, f"no .abf files found in {folder}"
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
    return store_cell_options(), f"imported {len(abfs)} recordings → " + ", ".join(made)


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
