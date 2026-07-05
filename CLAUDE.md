# CLAUDE.md — working agreement for AI agents on this repo

Electrophysiology analysis for the Neitz lab (James Kuchenbecker). One tested core
package (`neitz/`) with three front-ends: a **Dash GUI** (`viewer.py`), a **CLI**
(`neitz` / `python -m neitz`), and a **Python API** (`neitz.run`). This file is the
portable hand-off so any Claude instance (Claude Code, Desktop, a fresh clone) can be
productive immediately. Keep it current when the architecture changes.

## Environment / how to run

- **Python**: `/Users/j/miniconda3/bin/python` (conda base — has `pyabf`, `dash`, `plotly`,
  `scipy`, `pandas`, `matplotlib`). Plain `python` may not be it.
- **GUI**: `/Users/j/miniconda3/bin/python viewer.py` → http://127.0.0.1:8050
- **Desktop app** (double-click → native window, NOT a browser tab): `NeitzAnalysisSuite.app`
  (macOS) / a Desktop shortcut (Windows 11) launches `neitz_app.py`. See **Desktop app** below.
- **Tests**: `/Users/j/miniconda3/bin/python -m pytest -q` (≈43 tests; synthetic tests run
  anywhere, real-`.abf` tests skip if store data is absent).
- **CLI**: `neitz cell <date> <cell>` | `neitz flicker|spikes|noise|mirror …`
  (after `pip install -e ".[gui]"`).
- **Restart the GUI backend** (do this after editing `viewer.py`, and whenever the user asks):
  ```bash
  pkill -9 -f viewer.py; sleep 1; lsof -ti tcp:8050 | xargs -r kill -9; sleep 1
  nohup /Users/j/miniconda3/bin/python viewer.py > /tmp/viewer.log 2>&1 & disown
  curl -s --retry 15 --retry-delay 1 --retry-connrefused -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8050/
  lsof -ti tcp:8050   # confirm ONE pid
  ```
  (With the desktop app open, a restart auto-reloads its window — see below.)

## Desktop app — native window (macOS + Windows 11)

A double-clickable launcher that opens the viewer as its OWN app window — its own dock/taskbar icon
and macOS Space, NOT a Chrome tab (like Slack). `neitz_app.py` is the cross-platform core; the OS
wrappers just point at it. Deps: `pip install -e ".[gui,app]"` (`pywebview`; `pythonnet` on Windows;
`pyobjc` is pulled on macOS; the Edge **WebView2** runtime ships with Win11).

- **`neitz_app.py`** (cross-platform, `sys.platform`-branched): reuses a healthy back end on
  127.0.0.1:8050 or starts `python viewer.py` itself (freeing the port first if a half-dead process
  holds it), then opens a **pywebview** window — WKWebView (macOS) / WebView2 (Windows). **Single
  instance** via a lock socket (port **8051**): a second launch exits instead of stacking a window
  (macOS also won't relaunch a running .app). On window close it stops ONLY a back end it *started*
  (a reused / manually-run server is left alone). "Reuse the same instance" was James's explicit ask.
- **Front-end auto-reload** (James's ask, mirrored from `benaqTools_py`): `viewer.py` serves
  `/neitz-health` → `{boot, code_mtime}` (`boot` = a per-process uuid). `assets/autoreload.js` polls
  it every 2 s and `location.reload()`s when `boot` changes — so relaunching after a code edit
  refreshes the open window with no Cmd-R. Works in a plain browser tab too (restart viewer.py → the
  tab refreshes itself). `code_mtime` (newest source mtime at boot) is exposed for launcher use.
- **Full screen in the app**: the browser Fullscreen API is a **no-op** in WKWebView / WebView2, so
  `neitz_app.py` exposes `toggle_fullscreen` via pywebview `js_api` and `assets/fullscreen.js` calls
  `window.pywebview.api.toggle_fullscreen()` when it detects `window.pywebview` (falling back to the
  Fullscreen API in a plain browser tab). The `F` key + the ⛶ button share that one `toggle()`.
- **Icon — one source, both platforms**: `scripts/build_icon.py` renders `assets/app_icon.svg`
  (needs `cairosvg`) → `assets/app_icon.icns` (macOS, via `iconutil`) + `assets/app_icon.ico`
  (Windows, 16→256). The SVG is the trichromatic **cone mosaic + single-unit spike-train** icon.
  Both rendered icons are **committed**, so `build_windows_app.py` needs no `cairosvg`/Cairo on
  Windows (only regenerating from the SVG does); `build_macos_app.py` falls back to the committed
  `.icns` if `cairosvg` is absent.
- **Build** (idempotent — re-run after editing the SVG / viewer.py / neitz_app.py):
  - macOS: `python scripts/build_macos_app.py` → `NeitzAnalysisSuite.app`. Its launcher `exec`s
    `neitz_app.py` via conda base in the **foreground**, so the .app is the responsible, dock-resident
    process (Cmd-Q stops it cleanly); ad-hoc **codesigned** + quarantine cleared for a stable identity.
    Drag to `/Applications` + the Dock.
  - Windows 11: `python scripts\build_windows_app.py` → `NeitzAnalysisSuite.vbs` (silent, no console)
    + a Desktop shortcut carrying the `.ico`; right-click its taskbar icon → **Pin to taskbar**. Sets
    an AppUserModelID so Windows gives it its own taskbar identity.
- **Tracked vs not**: the icon **source + `.icns`/`.ico`**, `neitz_app.py`, and the build scripts are
  tracked; the built **`.app` / `.vbs` bundles bake machine-specific absolute paths → gitignored**
  (regenerate with the build scripts). `build/` (icon intermediates) is ignored.

## Architecture

```
neitz/                         tested core (no GUI deps)
  io/      abf.py (Recording), csv.py (CsvSpikeRecording + siso loaders), figures.py (save PNG/PDF/SVG)
           io/__init__.py: load_recording(path) dispatches .abf vs .csv
  spikes.py                    detect_spikes(signal, fs, polarity, method='mad'|'abs'|'mad_floor'|'matlab', k, abs_threshold, refractory_s)
                               'matlab' = detect_spikes_matlab = Sara's spikeDetectorOnline.m (500 Hz HP, max/3 threshold, 4σ noise gate; respects polarity)
  analysis/  revcorr.py (reverse_correlation, average_filter, normalize_filter 'max'|'std'),
             flicker.py (detect_flicker, cycle/transition PSTH, vector_strength, shift_test),
             strf.py (spatiotemporal reverse correlation)
  stimulus/  base.py + NoiseParadigm / FlickerParadigm / CheckerboardParadigm
  dataio/    DataStore, CellManifest, config (mirror), mirror_store  (the managed store)
  run.py     headless orchestration -> Result; run_cell_flicker(...name=,include=), run_cell_noise(...) (STA)
  plots.py   flicker_onoff_figure, flicker_cycle_grid
  cli.py / __main__.py
viewer.py                      the Dash GUI (single file, ~1500 lines) — built ON the core
examples/                      thin client scripts (reference implementations)
tests/                         pytest regression net
```

**Unifying idea**: noise and flicker are the same operation (reverse correlation between
stimulus and spikes) at different dimensionality; flicker is the periodic special case
(cycle/transition PSTH). Cone isolation (S/L/M/LM-iso) is metadata, not different math.

## Data-store integrity — NON-NEGOTIABLE RULES (James: "let's not let that happen again")

File↔manifest tracking must never silently break. When you touch ANY output-writing, recording,
deletion, or store code, preserve these invariants — and run `pytest -k integrity`:

1. **Two sources of truth, kept consistent.** The `manifest.json` is the truth for METADATA
   (recordings, stimulus, the `outputs[]` records). The DISK is the truth for FILES. The GUI shows
   output figures by **globbing `outputs/**/*.png`** (NOT by trusting the manifest's file list) and
   groups them by the `<analysis>` folder — so a stale/incomplete manifest can never *hide* real
   files on disk. Keep it that way: never make the display depend solely on the manifest file-list.
2. **One record per analysis, with ALL its files.** A run records every file it produced in ONE
   `record_output(analysis, files=…)` call; extra files (e.g. the 4K exports) are attached to the
   SAME record via `_attach_output_files`. `record_output` **REPLACES** an existing record of the
   same `analysis` name (never appends) so the manifest can't accrue duplicate/zombie records.
3. **Files always live under their parent.** Every output goes in `<date>/<cell>/outputs/<analysis>/`
   — never elsewhere, never detached from the cell. That folder path IS the association to the
   parent cell + the `analysis` key.
4. **Deletes update BOTH.** Deleting moves files to `.trash/` AND updates the manifest; never leave
   orphans (file on disk, in no record) or dangling refs (record points at a missing file) on
   purpose. (Dangling refs are tolerated by the display because of rule 1, but prune them when you
   touch the record — see the dedup/reconcile in the store-cleanup path.)
5. **There is a regression test:** `tests/test_store_integrity.py` runs an analysis and asserts the
   invariants (record exists, no duplicates, every recorded file is on disk, no orphan PNGs in the
   folder). If you change the output pipeline and it fails, you broke tracking — fix it, don't
   delete the test.

## Managed data store (source of truth)

- Location: `~/Documents/ephysdataio/` (override `$EPHYSDATAIO_ROOT`). Layout:
  `<date>/<cell>/{raw/, outputs/<analysis>/}` + per-cell `manifest.json` + store-wide `index.json`.
- `manifest.json`: cell label/type/notes + `recordings[]` (id, label, kind, file, stimulus,
  channels, fs, duration) + `outputs[]` (analysis runs). The GUI reads/writes this.
- Import COPIES files into `<cell>/raw/` with proper names. Analysis writes figures
  (PNG+PDF+SVG) + metrics.csv + result.json into `<cell>/outputs/<name>/` and records them.
- Optional per-computer mirror (Google Drive etc.): `neitz mirror --set PATH`; true rsync
  mirror with deletes; auto after analysis. The config file `~/.config/neitz/config.json` must
  be a JSON **object** `{"mirror": "…"}` — a bare quoted string is invalid JSON and is silently
  ignored (`mirror_dir()` → None). Always set it via `neitz mirror --set`, never by hand. Note
  `.trash/` lives inside the store, so deletes also propagate to the mirror on the next sync.

## Data in the store & validated findings (durable science, not just code)

- **2017-01-18/c01** (`20170118Bc4`, S-iso **gaussian noise**): the Python temporal-STA
  reproduces Sara Patterson's MATLAB ground truth **exactly** — biphasic filter, peak
  **22.2 ms OFF**, 15 epochs, low-pass tuning peaking ~18–20 Hz. Pinned as the
  `test_run_cell_noise_sta_peak` regression. Normalization gotcha (made explicit in
  `normalize_filter`): MATLAB divides the filter by `max(abs())` (trough → −1.0); dividing by
  `std` instead yields the *same* filter at ~5× scale. Sara's reference math lives in
  `MTFanalysis` inside `analyzeOnline.m` (case IsoSTA/GaussianNoise), not in the `iprgc4jim.m`
  wrapper. (Her `SaraipRGC/` source tree was removed from the repo in the restructure.)
- **2026-06-02/c01 (ipRGC) + c02 (notipRGC)** (Barak's recordings): **voltage-clamp**, ONE
  continuous sweep ~79–92 s @ 20 kHz, 3 channels — ch0 `Im_prime` (current, pA), ch1 `Vm_sec`
  (mV), ch2 `TTL` (frame clock). The "spikes" are **biphasic escaped action currents** on ch0
  (~60–200 pA over ~2 pA noise) — detect on the inward (negative) peak at ~6×MAD. Stimulus is
  **2 Hz black/white full-field flicker** (a light-sensitivity screen), NOT gaussian noise, and
  its timing is **fully recoverable from the TTL** — no stimulus file needed. Finding: a pooled
  circular-shift test shows **neither cell has a significant transient ON/OFF response** (ipRGC
  ON 1.29× p=0.56 / OFF 1.23× p=0.66, 2931 spk; notipRGC ON 1.35× p=0.42 / OFF 1.26× p=0.67,
  722 spk). Earlier per-trial ON/OFF labels were noise. The ipRGC recording is somewhat unstable
  (action-current amplitude drifted ±30→±200 pA across trials).
- **Rig caveat**: across otherwise-identical flicker trials the pre-flicker adapting block is
  driven inconsistently — e.g. 0040's frame sync is flat (0 Hz carrier) where 0041 runs the
  ~240 Hz carrier. Flag this if frame-sync timing matters for an analysis.
- **Method caveat**: `flicker.shift_test`'s circular-shift null only *rotates* a periodic PSTH,
  so its p-value has a floor (~latency-window/period) and low power. The default null is
  `'jitter'` (per-spike), which actually detects locked responses; `'shift'` is kept for
  reference only.

## Conventions & decisions (don't relearn these)

- **Undo / redo — `UNDO_TRACK` IS THE CONTRACT (NON-NEGOTIABLE):** ctrl-Z / shift-ctrl-Z (also
  ctrl-Y) walk a 50-entry history of "program changes" (`viewer.py`, bottom: `undo_record` /
  `undo_apply`, `assets/undoredo.js` → hidden `#undo-key`). The list `UNDO_TRACK` (id, prop pairs)
  is the SINGLE source of truth for what's captured — **whenever you add a control that changes the
  analysis/display, add its `(id, prop)` to `UNDO_TRACK`; whenever you remove one, delete its row.**
  Both callbacks are built from that list, so keeping it current keeps undo correct. Design notes:
  an equality guard (`_snap_key`) makes a restore a no-op in `undo_record` (no loop); per-file/-trace
  editors are captured via their SEED stores (`align-seed`, `absth-seed`) so a restore rebuilds the
  boxes → re-derives the maps → renders; `undoredo.js` leaves native text-undo alone while a text
  field is focused. This rule was James's explicit ask ("always remember to update the undo/redo
  queue"). A debounced input only records on blur/Enter (that's how dcc.Input commits).
- **Settable FFT bin** (power panel overlay `#fft-bin`, default 5 ms = 200 Hz): threaded through
  `binned_rate` + `power_w` (both MUST share the rate) via `build_figures(fft_bin=)`. Coarser bins
  low-pass the impulse train → fewer harmonics; the title shows the bin and the spectrum caps at its
  Nyquist. The **f=0 (DC) bin is dropped** in `power_w` (mean is subtracted → X[0]≈0 was a spurious
  ~−80 dB spike). Mean subtraction stays.
- **FFT "view input" toggle** (`#fft-input`, in the power panel under `#fft-bin`): swaps the power
  spectrum for the *exact array `power_w` transforms* — spikes binned at the fft bin, mean-subtracted,
  over the region (x = time, y = rate−mean). `build_figures(fft_input=)` branches the per-file +
  group-avg traces and the panel title/axes; the stim-frequency marker shows only on the spectrum.
  Distinct from "show binned spikes" (that uses the separate `#train-bin`). In `UNDO_TRACK`.
- **Trial-alignment nudge (D1)** (sidebar "trial align — frame-sync nudge (ms)"): a per-file time
  offset shifts that file's analog + TTL + spikes TOGETHER so trial starts line up. "⇄ auto" seeds
  each file's offset from its TTL first-onset, THEN refines to sample precision by FFT
  cross-correlating the two frame-sync waveforms (`fine_align_offset` — the onset t0 is only ~10 ms
  accurate); a per-file ms grid fine-tunes; "reset" clears. Boxes use `step="any"` (offsets are
  decimal ms). Stored in `align-seed` (ms, editor) → `align-map` (seconds, what `build_figures`
  applies). Coupled by construction — the region mask, FFT, ISI all run in the aligned frame.
  **Focusing an align box highlights that file's traces in both graphs** (clientside Plotly.restyle,
  `assets/alignfocus.js` → `#align-focus`; match → opacity 1, others → 0.12, restored on blur) — a
  full Python re-render on focus would be too slow. NOTE: this aligns the Analysis View + 4K exports;
  making the pooled ON/OFF stats in `run_cell_flicker` consume the offsets is a known follow-up.
- **Zoom policy** (`build_figures`): a user zoom OR an alignment nudge (trig `time` / `align-map`)
  KEEPS the current window; ANY other parameter change reverts to the default full view. Enforced by
  the layout `uirevision` = a hash of every NON-align parameter (constant across zoom+align → Plotly
  holds the zoom; changes on any other edit → Plotly resets) AND honoring the relayout range only
  when keeping. (Replaces the old blanket `uirevision="keep"`.)
- **Graph interaction toolbar** (vertical icon strip left of the signal graph — `graph_tools()` /
  `#graph-toolbar`, a "current tool" palette; the first of a growing set): the active tool lives in
  the `graph-mode` store (in `UNDO_TRACK`). `set_graph_mode` (buttons → store), a clientside callback
  highlights the active button, and `build_figures(graph_mode=)` sets each axis' `fixedrange` so a
  drag-box zooms **X only** (`zoomx`, default = the historical behavior), **Y only** (`zoomy`), or
  **both** (`zoombox`). Reset (`#gm-reset`) is a clientside `Plotly.relayout` back to the default
  window — stashed as `time_fig.layout.meta.xr = [x0, x1]`. `graph_mode` is NOT in the `uirevision`
  hash, so switching tools keeps the current zoom. Planned tools (edit-region drag-handles, highlight,
  measure, drag-threshold, edit-spikes) plug into this same framework.
- **ISI histogram** honors the "bin (ms)" box (`train-bin`): `xbins` size = the ms value; 0 → auto
  (60 bins). (That box also drives the spike-train row-1 view when "show binned spikes" is on.)
- **Group → one average** (files panel checkbox `#group-avg`, needs 3+ checked): `build_figures`
  collapses the selected files to a single mean trace — the aligned (post-nudge) analog + TTL are
  averaged incrementally onto a common grid (running sum+count, memory-safe), the FFT shows only the
  group-avg line, ISI stays pooled. Uncheck to ungroup. `do_group` gates it.
- **Files panel** (Analysis View): `all` / `none` buttons + a live `(k/n)` count next to the label;
  **drag a box over the rows to TOGGLE** them (`assets/fileselect.js`, on `#file-box`) — selected↔
  deselected; a plain click still toggles one row.
- **Compact readout** (top of graphs): a ONE-LINER (count · region · view · stim-freq range). The
  old per-file spike-count dump blew up to many lines with 40+ files — don't reintroduce it there.
- **Outputs are NOT shown in the Analysis View** (the old "Cell outputs" gallery was removed) — view
  processed figures in the **Data Explorer** (click to enlarge) or via **Finder** only. The
  `out-thumb` enlarge-modal still serves the Explorer; `gallery-trigger` is now a dead store.
- **True full screen**: `⛶ Full screen` button (next to the readout) + the `F` key toggle the
  browser Fullscreen API (`assets/fullscreen.js`); relabels on `fullscreenchange`.
- **Flicker auto-window includes the final cycle**: `detect_flicker` extends `t1` by one `period`
  (clamped to the envelope) so the last complete frame isn't clipped into the excluded block
  (regression: `test_detect_flicker_freq_and_region` asserts `t1 − last_on_edge > 0.4`).
- **"sq wave" stimulus type** (changed 2026-06): the stimulus metadata value is now stored as
  `sq_wave` (GUI label "sq wave"). Existing manifests were migrated `flicker` → `sq_wave`, and the
  GUI maps any legacy `flicker` value back to the "sq wave" option / display. BUT the *analysis*
  internals stay `flicker` for data compatibility — the output folder (`outputs/flicker/`),
  `run_cell_flicker`, and the default run name are all still `flicker`. Run dispatch keys off
  `stimulus.type` (only `gaussian_noise` → STA; everything else incl. `sq_wave` → the flicker
  analysis), so storing `sq_wave` doesn't change dispatch. Don't rename the analysis internals.
- **Spike power graph** uses the explicit FFT formula `W = 2·|X[k]|² / N²` (single-sided,
  R=1Ω), NOT a Welch PSD — the user specifically wanted this. See `power_w()`.
- **Per-trace abs thresholds** (methods `abs`/`mad_floor`/`matlab`): a "sync" checkbox (one value
  for all, mirrored through the hidden `#absth` carrier) over an always-shown wrapping grid of
  per-trace boxes, EACH with its own **"auto"** button (max/3 for `matlab`, k·MAD otherwise;
  synced → only the first auto is active and it fills every box). The old global "auto abs"
  button was removed as redundant. For `matlab`, selecting it auto-seeds the boxes with max/3.
- **MATLAB (Sara) detection** respects polarity (neg flips / pos as-is / abs rectifies) and uses
  the abs boxes as its threshold (max/3 default); k & refractory are not used (greyed).
- **Deletes** (Data Explorer): a 🗑 on each date (whole day), a 🗑 under the checked files,
  a 🗑 on each output figure, AND a per-figure **select checkbox** (`out-check`, pattern id) + a
  **🗑 Delete selected** button (`del-outputs`) for multi-deleting figures at once — each pops a
  **confirmation-warning** modal (no password; `open_delete` builds the target list, all share
  `confirm_delete`). Items are MOVED to a reversible `.trash/` inside the store (`<date>_<cell>/`
  for files, `<date>_ALL/` for a whole day), not unlinked. `delete_files` / `delete_date` helpers.
  (`del-outputs` is created dynamically by `explorer_detail`, so the app sets
  `suppress_callback_exceptions=True`.)
- **"🔧 Fix abf (Neitz)"** button (Explorer detail header, next to Save metadata; callback `fix_abf`):
  one click applies the lab standard to a cell — sets every `.abf` recording's `stimulus.type` to
  `sq_wave` (preserving existing params, defaulting `frame_rate=60`, `source="neitz-fix"`), verifies
  the `Im_prime/Vm_sec/TTL` channels (warns, non-destructively, if a file isn't that config), and
  clears the in-memory `_CACHE` for the cell's files so they're re-read fresh. Non-destructive — it
  never modifies the `.abf` binary. Handy for the blank-stimulus session cells and as a recovery
  button. (Channel ROLES are already auto-standard: `chan` defaults to `Im_prime`, `ttl` to the
  first `*ttl*` name.)
- **Returning to the Analysis View re-syncs it** (`resync_on_close` on `exp-close`): rebuilds the
  `#file` options/value (existing files only, so anything deleted in the Explorer drops out),
  refreshes `sel-cell`, and bumps `gallery-trigger` — so Explorer deletions take effect WITHOUT a
  browser refresh and a subsequent Run no longer silently no-ops on a now-deleted file. `pick_cell`
  and `resync_on_close` share the `_cell_files(vals)` helper.
- **Run Analysis is a BACKGROUND callback** (`background=True`, `DiskcacheManager` — needs
  `diskcache`/`multiprocess`/`psutil`, in `[gui]` extras): the analysis + 4K kaleido exports take
  **~2–4 min** (the worker re-imports the module on spawn and kaleido is slow per call), so it runs
  off the UI thread — the UI stays responsive and the outputs **auto-refresh when done** (no browser
  refresh). `running=[…]` disables the run button + shows "⏳ Running… (~1-2 min)" during the run; a
  clientside callback shows an instant message; `#store-msg` is wrapped in `dcc.Loading`. The
  background worker spawns from `viewer.py` import, so module-level code must stay import-safe.
  Feedback is a prominent **banner** in `#store-msg`: an amber "⏳ Running analysis…" the instant
  the button is clicked (clientside callback returning a styled component spec) → a green
  "✓ Analysis complete" banner with the summary when `run_cell` returns. (The old `dcc.Loading`
  spinner was removed — the banners + the `running` button label are the feedback.)
- **Output thumbnails**: 64px in the Explorer detail pane (`explorer_detail`); the Analysis-View
  "Cell outputs" gallery (`output_gallery`) is a separate 150px grid.
- **Auto-refresh via polling** (`poll_refresh`, a 3 s `dcc.Interval`): diffs a cheap on-disk
  fingerprint of the selected cell(s) (`_store_fingerprint` — #output PNGs + newest mtime, and the
  raw-file set). When outputs change (a finished Run Analysis, or a figure deleted in the Explorer)
  it bumps `gallery-trigger`; when the raw-file set changes (files deleted in the Explorer) it
  refreshes the `#file` checklist. This force-refreshes the Analysis View **without** relying on the
  background callback's result reaching the browser or on navigation — and makes outputs appear
  progressively *during* a run. `_img_datauri` is mtime-cached so the repeated gallery rebuilds are
  cheap. (Complements `resync_on_close`, which still syncs on the explicit Explorer→Analysis click.)
- **Some Sara CSVs aren't time-series** (e.g. `siso-DLP.csv` has a phase label in col 0).
  `CsvSpikeRecording` raises on those; the viewer's `loadable()` filters them and prefers
  spike-train CSVs. Don't "fix" by forcing them open.
- **Run variants**: `run_cell_flicker(..., name=, include=)` writes to `outputs/<name>/` so a
  run that excludes a recording coexists with the original instead of overwriting it. The GUI
  "run name" box + checked-files drive this.
- **Run Analysis USES the live GUI detection settings** (fixed 2026-06): `run_cell` passes
  `detect=` (polarity/method/k/abs_threshold/refractory_s), `abs_map=` (per-trace abs
  `{path: value}`), and `run_label=` (the un-sanitized friendly name) into
  `run_cell_flicker`/`run_cell_noise`. Before this, the run silently used `FlickerParadigm`
  DEFAULTS (`polarity='neg'`, k=6) and recorded those regardless of the GUI — so e.g. a `pos`
  selection was stored as `neg`. The settings now land in the output's `params` (+ `abs_per_trace`
  when per-trace) and the friendly name in the output record's `label` (the folder/`analysis`
  key stays sanitized, e.g. "4 epochs" → `4_epochs`). Per-file detection feeds
  `FlickerParadigm.group_from_trials` so per-trace thresholds also drive the pooled ON/OFF.
  NOTE: signal/TTL **channel** is still NOT threaded (paradigm default `Im_prime`/`TTL`) — a
  known follow-up.
- **Run Analysis also saves 4K Plotly exports** (added 2026-06, needs **kaleido** — in the `[gui]`
  extras): the flicker branch calls `export_window_figures(...)` which rebuilds the GUI graphs via
  the shared `build_figures(...)` (the `render` callback is now a thin wrapper over it) and writes,
  at 3840×2160 (4K full-screen, regardless of the actual window), into `outputs/<name>/`:
  `window_4k.{pdf,svg}` (signal + frame-sync TIME-ALIGNED at 80% width, a 250 ms stimulus zoom in
  the right 20%, power | ISI below), `power_4k.pdf`, `analog_framesync_4k.pdf` (color signal+spikes
  / B&W no-spikes / color frame-syncs, x-shared), `framesync_separated_4k.pdf` (each file's
  frame-sync in its own un-staggered panel), and **`assumptions_4k.pdf`** — a table page of the
  exact detection settings used (polarity/method/k/threshold/refractory/region/channels) + per-file
  spike counts + thresholds (so it's verifiable Run Analysis honored the live settings; height
  fits the row count). Each figure also gets a lighter **`.png`** (1920×1080) so it surfaces in the
  Explorer/gallery (which glob `outputs/**/*.png`); PDF/SVG stay 4K. Files are attached to the
  output record via `_attach_output_files`. Wrapped in try/except so a kaleido failure never breaks
  the run. (`plots.flicker_cycle_grid` sizes its grid to the epoch count: ≤5 → 1×n no empty slot,
  10 → 2×5.)
- **Run dispatch by stimulus type** (NOT guessed from file contents — driven by the manifest's
  explicit `stimulus.type`): `gaussian_noise` → `run_cell_noise` (reverse-correlation STA from
  the spike + stimulus CSVs, `outputs/sta/`); else → `run_cell_flicker`. Data format (abf-analog
  vs spike-CSV) is ORTHOGONAL to stimulus type — a spike-CSV is pre-detected spike times (no
  detection). The S-iso STA peak (22.2 ms OFF on 2017-01-18/c01) is pinned as a regression test
  and matches Sara's MATLAB ground truth.

## Stimulus source & reproduction (seed-based; June2026 Stage rig)

June2026-rig stimuli (repo `June2026StageMATLAB`) are **gamma-corrected** for the LightCrafter and
their noise is **regenerated from a seed here** — no per-frame stimulus values are shipped.

- **Session manifest.** The rig writes `YYYY_MM_DD_stim_manifest.jsonl` (JSON-Lines, one trial per
  line, in acquisition order): `{stimulus, stim_type, cone_isolation, seed, mu, sigma, checks_x/y,
  n_updates, update_every_n_frames, refresh_rate_hz, stim_frames, gamma, noise_method, fill_order,
  timestamp}`. `stim_type` ∈ `gaussian_noise | checkerboard | sq_wave | jitter`. Trials pair to
  Clampex `.abf` files **by order** (day-level, sorted by NNNN); `timestamp` is the cross-check.
- **Reproduce, don't store.** `stimulus/reproduce.py::reproduce_noise(seed, n_y, n_x, n_updates,
  mu, sigma)` → the exact `(n_y, n_x, n_updates)` **linear** noise, MATLAB-bit-identical: `mt19937ar`
  `rand()` == numpy `RandomState`, MATLAB `norminv` == `scipy.special.ndtri` (**inverse-CDF, NOT
  `randn`** — MATLAB's ziggurat is unreproducible in Python), `order='F'`, clip [0,1]. Pinned to a
  MATLAB reference by `tests/test_reproduce.py` (~1e-12). STA **correlates against the linear `v`**
  (the projector is gamma-corrected, so emitted light is linear in `v`); `gamma_adjust(v) =
  v**(1/2.2056)` is the DAC code that was sent (display verification only).
- **Reader.** `io/stim.py`: `load_session_manifest` / `find_session_manifest` (strict by date),
  `stimulus_metadata(record) -> (stim_type, params)` for `CellManifest.set_stimulus(...,
  source="stim-manifest")`, `pair_by_order`, `noise_from_record(record, per_frame=False)` (the STA
  tensor; raises for sq_wave/jitter), `sent_codes_from_record`. **Complements — does not replace —**
  the historical stimulus-CSV path (`io/csv.load_stimulus_epochs_csv`, used by 2017-era cells).
- **Analysis-side wiring (status).** The reproduction API is stable — wire to it, don't fork it.
  - ✅ **Import auto-fill** — `io.stim.apply_session_manifest(cm, source_dir, date=)` finds the day's
    manifest next to the `.abf`s, pairs rows→recordings by order, `set_stimulus(source="stim-manifest")`,
    and copies the manifest into the cell dir. `viewer.import_data` calls it (replaces hand-entry;
    no-ops when absent).
  - ✅ **Data Explorer** surfaces `stim_type · cone · seed · grid` (file tiles `_stim_brief`, detail
    "Stimulus" row `_cell_stim_summary`, "Epochs" row `_cell_epoch_summary`). The **file browser
    groups recordings into "N epochs" blocks** by `neitz.io.epoch_groups` (runs of the same
    `stim_signature`): `explorer_file_options` inserts a full-width **disabled** header option before
    each group (`_epoch_group_header`; CSS `#exp-files :has(input:disabled)` → `display:block`), so
    headers can't be selected and the selection/delete/open callbacks are untouched. Flat fallback
    when there's no useful grouping. (Rail-level *sorting* by stim_type is still a follow-up.)
  - ✅ **Seed-based STA/STRF core** (`run.py`, tested against injected filters): `analysis_for_stim_type`
    (sq_wave→flicker, gaussian_noise→sta, checkerboard→strf), `record_from_stimulus`,
    `sta_from_records` / `strf_from_records` (per-epoch reverse correlation vs the **contrast** v−mean,
    averaged/pooled — the Neitz model: each recording is one epoch synced to its own spikes),
    `epoch_response` (bin spikes to the update grid from `t0`).
  - ✅ **Store-driven run + dispatch.** `run.load_seed_epochs` (per abf: detect spikes → `t0` from
    `analysis.flicker.frame_clock_onset` → regenerate stimulus from seed → `epoch_response`);
    `run_cell_noise` DISPATCHES seed-abf vs legacy stim-CSV; `run_cell_checkerboard` (STRF);
    `plots.strf_figure`; `_save_seed_run` (one integrity-safe manifest record). `viewer.run_cell`
    routes `stim_type`: gaussian_noise→STA, checkerboard→STRF, else flicker, threading the live
    detection settings. Validated by an integration test with a mocked `Recording`.
  - 🟢 **Ready; awaits a real seeded-noise cell for a smoke test.** June2026 noise trials run **no
    adapting carrier** (per James), so `frame_clock_onset`'s "first sustained frame-clock run = the
    stimulus `t0`" is correct by design (not a guess). Cadence is metadata (`refresh_rate_hz`,
    `update_every_n_frames`); `t0` falls back to 0.0 if no clock is found. The only thing a first real
    cell confirms is that the TTL is the per-frame clock we expect. Noise needs NO cross-trial nudge —
    each epoch self-syncs; the frame-sync nudge is a square-wave (cycle-averaging) tool only.

## GUI structure (viewer.py)

- Two-column flex shell with a **draggable splitter** (`#splitter`, `assets/splitter.js`,
  width persisted as a window fraction): **left sidebar** (collapsible `<details>` cards: Data
  store, Channels & spike detection, Cell outputs) + **right graphs** (signal+frame-sync on top,
  FFT-power | ISI-histogram on the bottom). An upper-left `nav_toggle()` switcher flips between
  **Analysis View** and the **Data Explorer** (same widget in both; the inactive side is the
  clickable target, with a hover glow).
- Channels card: signal/TTL dropdowns, then **polarity | spike-detect algorithm** side by side
  (a `<hr>` above them), the k·MAD slider with **refractory** beside it, and the per-trace
  abs-threshold grid (see conventions). The region/display controls are **overlays floated onto
  the graphs** via `ov()` (start/end + `crop`, hide-spikes/spike-train, stagger%, ISI bin).
- The time graph re-renders detail on zoom (relayout); **scrollZoom** enables fine zooming, and
  the frame-sync y-axis re-autoranges on stagger via a per-axis `uirevision`. Both y-axes are
  **`fixedrange=True`** so the drag-box zoom is **X-ONLY** — Plotly can't axis-lock a thin (narrow
  left-right) box into a y-only zoom, so a precise narrow x-zoom works in one drag (the old behavior
  collapsed a too-narrow box into broad horizontal min/max-envelope lines). `fixedrange` blocks USER
  y-zoom only, NOT the code's `autorange` (stagger still fits). Trade-off: no drag-box Y-zoom (Y just
  autoranges). NOTE: a `dragmode='select'`→relayout approach for this does NOT work — the global
  `uirevision="keep"` (needed to preserve a real user zoom while the data re-decimates) reverts a
  programmatic `Plotly.relayout` range, so the fix must ride the native user-zoom path.
- **Data Explorer** (`#explorer-modal`, full-screen, all-dark): a **resizable** sortable/
  searchable **rail** of dates (`#exp-rail`; column widths are CSS vars `--rc1/2/3` driven by
  drag handles in `assets/rail-resize.js`, persisted; a dead spacer + fixed trash) with a
  centered `date ← cell` **back** row on top and an **Import / Backup mirror** bottom panel ·
  **middle** = cell thumbnails / file tiles over a hover-preview, with a bottom action bar
  (**delete + Open selected**, shown only when files are checked) · **right** (`#exp-rightpane`)
  = **editable cell metadata** (type / stimulus type+params / notes) + **Save metadata**, over
  read-only recording facts + output figures (🗑 to delete) + a collapsible raw-manifest JSON.
  Metadata editing lives ONLY here now (not the sidebar). Escape closes the image pop-out first,
  then the explorer. Output figures' 🗑 is light-red (`#ff7a7a`) so it's visible on the dark theme.
  Opening the Explorer from the Analysis View with **exactly one file checked** auto-jumps to that
  file's cell and pre-checks it (`exp-autosel` store + `_date_cell_of`; `toggle_explorer` sets it,
  `exp_render` consumes it one-shot).
- Waveform sparklines / big waveforms are matplotlib-Agg PNG data-URIs, cached by mtime.
- Responsive: CSS `zoom` media-queries scale the whole UI on smaller windows (root height
  counter-scaled). Dash 4.2 renders checklist/radio/dropdown options as `.dash-options-list-option`
  — alignment/colors are set on those classes (labelStyle is ignored); see `assets/viewer.css`.

## How to work here

1. **Verify before declaring done**: parse syntax → confirm every callback-referenced id still
   exists in `app.layout` → `pytest -q` → server smoke (HTTP 200) → restart backend.
2. For `viewer.py`, when adding/removing components, an output written by >1 callback needs
   `allow_duplicate=True` on all but one; pattern-matching ids use `ALL` + guard
   `ctx.triggered[0]["value"]` to ignore creation.
3. **Git**: remote `origin` = the `jameskuch` fork; work on branch `kuch`. Commit when the user
   asks (they usually do, per change). End commit messages with the Co-Authored-By trailer.
4. Keep changes in the idiom of the surrounding code (inline Dash styles, compact helpers).
5. Update this file + `README.md` when the architecture or run story changes.

## Known stale / TODO

- 6 session cells (2025-12 → 2026-02) have blank stimulus metadata pending entry (auto-fillable
  once the stim-manifest import wiring lands — see *Stimulus source & reproduction*).
- Checkerboard/Gaussian noise now HAVE a seed-based on-disk format (the stim manifest +
  `reproduce.py` / `io/stim.py`); the import / Data-Explorer / `run_cell_noise` wiring is still TODO.
- No CI yet.
- `RESTRUCTURE_DESIGN.md` is a historical design record (the restructure is done) — keep
  for provenance, don't treat as current.
