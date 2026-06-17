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
  the frame-sync y-axis re-autoranges on stagger via a per-axis `uirevision`.
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

- 6 session cells (2025-12 → 2026-02) have blank stimulus metadata pending entry.
- No CI yet; checkerboard has no on-disk stimulus format / CLI.
- `RESTRUCTURE_DESIGN.md` is a historical design record (the restructure is done) — keep
  for provenance, don't treat as current.
