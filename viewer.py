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
import platform
import subprocess
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

from neitz.io import load_recording
from neitz.spikes import detect_spikes
from neitz.analysis import flicker as flk

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


# ============================================================
# App
# ============================================================
app = Dash(__name__)
app.title = "Neitz ABF Viewer"
_files = discover_abf()

app.layout = html.Div(style={"font-family": "sans-serif", "margin": "12px"}, children=[
    # file "columns" viewer: browse buttons + left-to-right (Finder-columns) checklist
    html.Div(style={"display": "flex", "gap": "12px", "alignItems": "flex-start",
                    "marginBottom": "4px"}, children=[
        html.Div([
            html.Span("Neitz ABF Viewer", style={"fontWeight": "bold", "fontSize": "14px"}),
            html.Div([html.Button("📁 File…", id="browse-file", n_clicks=0, style={"height": "30px"}),
                      html.Button("📂 Folder…", id="browse-folder", n_clicks=0,
                                  style={"height": "30px", "marginLeft": "6px"})], style={"marginTop": "4px"}),
        ], style={"flex": "0 0 auto"}),
        html.Div(dcc.Checklist(id="file",
                               options=file_options(_files),
                               value=[_files[0]] if _files else [],
                               inline=True,
                               labelStyle={"display": "inline-block", "width": "175px",
                                           "fontSize": "11px", "whiteSpace": "nowrap",
                                           "overflow": "hidden", "textOverflow": "ellipsis",
                                           "verticalAlign": "top", "marginRight": "6px"},
                               inputStyle={"marginRight": "4px"}),
                 style={"flex": "1", "maxHeight": "84px", "overflowY": "auto",
                        "border": "1px solid #ccc", "padding": "4px", "background": "white"}),
    ]),
    # details: full-width row beneath the file viewer (same compact font)
    html.Div(id="meta", style={"fontSize": "11px", "color": "#333", "lineHeight": "1.5",
                               "background": "#f6f6f6", "padding": "6px", "borderRadius": "4px",
                               "marginBottom": "6px"}),
    html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "8px",
                    "alignItems": "flex-end"}, children=[
        html.Div([html.Label("signal channel"), dcc.Dropdown(id="chan", style={"width": "170px"})]),
        html.Div([html.Label("TTL channel"), dcc.Dropdown(id="ttl", style={"width": "140px"})]),
        html.Div([html.Label("polarity"),
                  dcc.RadioItems(id="polarity", options=[{"label": p, "value": p} for p in ("neg", "pos", "abs")],
                                 value="neg", inline=True, **PERSIST)]),
        html.Div([html.Label("threshold"),
                  dcc.RadioItems(id="method", options=[{"label": "k·MAD", "value": "mad"},
                                                       {"label": "absolute", "value": "abs"},
                                                       {"label": "k·MAD ≥ floor", "value": "mad_floor"}],
                                 value="mad", inline=True, **PERSIST)]),
        html.Div([html.Label("k (MAD)"),
                  dcc.Slider(id="k", min=2, max=15, step=0.5, value=6,
                             marks={2: "2", 6: "6", 10: "10", 15: "15"},
                             tooltip={"placement": "bottom"}, **PERSIST)], style={"width": "220px"}),
        html.Div([html.Label("abs thresh"),
                  dcc.Input(id="absth", type="number", value=20, debounce=True,
                            style={"width": "85px"}, **PERSIST)]),
        html.Div([html.Label("refractory (ms)"),
                  dcc.Input(id="refr", type="number", value=2, debounce=True,
                            style={"width": "75px"}, **PERSIST)]),
        html.Div([html.Label("region start (s)"),
                  dcc.Input(id="region-start", type="number", debounce=True, style={"width": "95px"})]),
        html.Div([html.Label("region end (s)"),
                  dcc.Input(id="region-end", type="number", debounce=True, style={"width": "95px"})]),
        html.Div([html.Label("display"),
                  dcc.Checklist(id="dispopts",
                                options=[{"label": " stagger frame syncs", "value": "stagger"},
                                         {"label": " hide detected spikes", "value": "hide_spikes"},
                                         {"label": " spike-train view (0/1)", "value": "spike_train"}],
                                value=[], labelStyle={"display": "block", "fontSize": "12px"}, **PERSIST)]),
        html.Div([html.Label("excluded regions"),
                  dcc.RadioItems(id="region-mode",
                                 options=[{"label": " show", "value": "show"},
                                          {"label": " crop", "value": "crop"},
                                          {"label": " baseline", "value": "baseline"}],
                                 value="show", labelStyle={"display": "block", "fontSize": "12px"}, **PERSIST)]),
        html.Div([html.Label("spike-train bin (ms, 0=impulses)"),
                  dcc.Input(id="train-bin", type="number", value=0, min=0, debounce=True,
                            style={"width": "110px"}, **PERSIST)]),
    ]),
    html.Div(id="readout", style={"margin": "4px 0", "fontWeight": "bold", "fontSize": "12px"}),
    dcc.Graph(id="time", style={"height": "560px"}),
    dcc.Graph(id="fft", style={"height": "340px"}),
    dcc.Store(id="last-folder", storage_type="local"),   # remembers data folder across sessions
    dcc.Interval(id="once", interval=300, max_intervals=1),
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
@app.callback(Output("time", "figure"), Output("fft", "figure"), Output("readout", "children"),
              Input("file", "value"), Input("chan", "value"), Input("ttl", "value"),
              Input("polarity", "value"), Input("method", "value"), Input("k", "value"),
              Input("absth", "value"), Input("refr", "value"),
              Input("region-start", "value"), Input("region-end", "value"),
              Input("dispopts", "value"), Input("region-mode", "value"), Input("train-bin", "value"),
              Input("time", "relayoutData"), prevent_initial_call=True)
def render(files, chan, ttl_name, polarity, method, k, absth, refr, rstart, rend,
           dispopts, region_mode, train_bin, relayout):
    files = [f for f in (files or []) if f]
    if not files:
        return blank_fig("No file selected"), blank_fig(""), "No file selected."
    if not chan:
        return no_update, no_update, no_update

    det = dict(polarity=polarity, method=method, k=float(k),
               abs_threshold=float(absth) if absth is not None else None,
               refractory_s=(float(refr) / 1000.0) if refr else 0.002)
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
                             row_heights=[0.7, 0.3])
    fft_fig = go.Figure()
    per_file_rates, stim_freqs, readbits = [], [], []

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

        st = detect_spikes(y, fs, **det)
        in_reg = st.times[(st.times >= rs) & (st.times <= re_)]
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
        readbits.append(f"{name}: {len(in_reg)} spk in region"
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
                          annotation_text=f"stim {sf:.2f} Hz", annotation_position="top")
    fft_fig.update_layout(title="spike-train spectrum (inside region)",
                          xaxis_title="frequency (Hz)", yaxis_title="amplitude",
                          xaxis_range=[0, FMAX], margin=dict(l=55, r=20, t=40, b=40),
                          legend=dict(orientation="h", y=1.2), showlegend=True)

    view = ("spike-train" if spike_train else "analog")
    mode = (f"GROUP of {len(files)} (avg→FFT; {view} view"
            + ("; spikes shown" if (show_spikes and not spike_train) else "")
            + "; frame syncs overlaid)" if multi else f"single-file inspect ({view})")
    readout = f"[{mode}]  region {rs:.2f}-{re_:.2f}s  |  " + "  |  ".join(readbits)
    return time_fig, fft_fig, readout


if __name__ == "__main__":
    app.run(debug=False, port=8050)
