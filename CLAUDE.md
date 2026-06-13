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
  spikes.py                    detect_spikes(signal, fs, polarity, method='mad'|'abs'|'mad_floor', k, abs_threshold, refractory_s) -> SpikeTrain
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
  mirror with deletes; auto after analysis.

## Conventions & decisions (don't relearn these)

- **"sq wave" is the user-facing name for `flicker`** — the GUI label says "sq wave" but the
  stored stimulus value, folder (`outputs/flicker/`), and `run_cell_flicker` all stay `flicker`
  for data compatibility. Don't rename the internals.
- **Spike power graph** uses the explicit FFT formula `W = 2·|X[k]|² / N²` (single-sided,
  R=1Ω), NOT a Welch PSD — the user specifically wanted this. See `power_w()`.
- **Per-trace abs thresholds**: a "sync" checkbox (one value for all) vs a per-trace editor;
  "auto abs" seeds each trace's `k·MAD` and keeps `mad_floor` if already selected.
- **Deletes** (Data Explorer): a 🗑 on each date (whole day), a 🗑 under the checked files,
  and a 🗑 on each output figure — each pops a **confirmation-warning** modal (no password).
  Items are MOVED to a reversible `.trash/` inside the store (`<date>_<cell>/` for files,
  `<date>_ALL/` for a whole day), not unlinked. `delete_files` / `delete_date` helpers.
- **Some Sara CSVs aren't time-series** (e.g. `siso-DLP.csv` has a phase label in col 0).
  `CsvSpikeRecording` raises on those; the viewer's `loadable()` filters them and prefers
  spike-train CSVs. Don't "fix" by forcing them open.
- **Run variants**: `run_cell_flicker(..., name=, include=)` writes to `outputs/<name>/` so a
  run that excludes a recording coexists with the original instead of overwriting it. The GUI
  "run name" box + checked-files drive this.
- **Run dispatch by stimulus type** (NOT guessed from file contents — driven by the manifest's
  explicit `stimulus.type`): `gaussian_noise` → `run_cell_noise` (reverse-correlation STA from
  the spike + stimulus CSVs, `outputs/sta/`); else → `run_cell_flicker`. Data format (abf-analog
  vs spike-CSV) is ORTHOGONAL to stimulus type — a spike-CSV is pre-detected spike times (no
  detection). The S-iso STA peak (22.2 ms OFF on 2017-01-18/c01) is pinned as a regression test
  and matches Sara's MATLAB ground truth.

## GUI structure (viewer.py)

- Two-column flex shell: **left sidebar 20%** (collapsible `<details>` cards: Data store,
  Channels & spike detection, Cell outputs) + **right 80%** graphs (signal+frame-sync on top,
  FFT-power | ISI-histogram on the bottom).
- The region/display controls are **overlays floated onto the graphs** (start top-left, end +
  a `crop` checkbox top-right, hide-spikes/spike-train bottom-right, stagger% just above the
  frame-sync x-axis, bin inside the ISI), via the `ov()` helper, positioned below the Plotly
  toolbar. Compact `_OVI` textboxes.
- **Data Explorer** pop-out (`📂`): dates rail → cell thumbnail cards (waveform + latest output)
  → file checklist (middle) + detail/JSON/preview (right, split 75/25). Import lives here.
  Escape closes the image pop-out first, then the explorer (clientside keydown handler).
- Waveform sparklines / big waveforms are matplotlib-Agg PNG data-URIs, cached by mtime.

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
