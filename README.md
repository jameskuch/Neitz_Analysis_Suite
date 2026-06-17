# Neitz Analysis Suite

Electrophysiology analysis for the Neitz lab: ABF/CSV I/O, spike detection,
reverse-correlation (STA / linear filter), flicker light-response, and spatiotemporal
STRF — with a managed data store and three front-ends (**GUI · CLI · Python API**).

## Quick start

```bash
cd /Users/j/Neitz_Analysis_Suite
python viewer.py            # then open http://127.0.0.1:8050  (Ctrl-C to stop)
```

That's the program. A draggable splitter divides the **left sidebar** (controls) from the
**right graphs**, and an upper-left switcher flips between the **Analysis View** and the
**Data Explorer**:
- **cell(s)** — multi-select one or more `date / cell — label`s (with an inline
  date / label / type sort); the checked **files** load into the graphs.
- **▶ Run analysis** — dispatches on the cell's **stimulus type** (set in the Explorer):
  `gaussian_noise` → reverse-correlation **STA** (linear filter, from the spike + stimulus
  CSVs); otherwise the square-wave ON/OFF analysis on the *checked* files. It uses the **live
  spike-detection settings** (polarity / algorithm / k / per-trace abs / refractory) and records
  them. A **run name** keeps variants side by side (blank = auto: `sta` / `flicker`) and is
  preserved as the output's friendly label; outputs → `<cell>/outputs/<name>/`. The flicker run
  also writes **4K PDF/SVG exports** of the current graphs (rendered at 3840×2160 regardless of
  window size): the full window (signal + frame-sync time-aligned at 80% width with a 250 ms
  stimulus zoom on the right, power | ISI below), the spike-train power graph, an analog panel
  (color signal + spikes / B&W without spikes / color frame-syncs), the frame-syncs separated per
  file, and an **assumptions** page (a table of the exact detection settings used + per-file spike
  counts/thresholds). A
  lighter **PNG preview** of each is saved too, so they appear (click-to-enlarge) in the Explorer's
  Outputs grid alongside the analysis figures. Run Analysis runs **in the background** (it takes
  ~2–4 min with the 4K exports) so the UI stays responsive, the run button shows "⏳ Running…", and
  the outputs **auto-refresh when done** — no browser refresh needed.
- **Channels & spike detection** — signal / TTL channel, **polarity** (neg/pos/abs), and
  the **spike-detect algorithm**: `k·MAD`, `absolute`, `k·MAD ≥ floor`, or
  **MATLAB (Sara)** (a faithful port of `spikeDetectorOnline.m`: 500 Hz high-pass, max/3
  threshold, 4σ noise gate). The absolute-threshold methods show a wrapping grid of
  per-trace boxes, each with its own **auto** button, plus a global **sync**.
- **Data Explorer** (dark, full-screen) — a resizable, sortable/searchable **rail** of
  dates → cell thumbnails → files; the right pane **edits cell metadata** (type, stimulus
  type+params, notes) and **Saves** it, above read-only recording facts + output figures
  (click to enlarge, 🗑 to delete) + the raw manifest. **Import data** and **Backup mirror**
  live in the rail's bottom panel; **Open selected in viewer** (with a delete) appears when
  files are checked. Output figures can be deleted one at a time (🗑) or **multi-selected** (a
  checkbox per figure + **Delete selected**). Deletes are confirmation-warned and trashed
  reversibly. Switching here from the Analysis View with exactly **one file checked** jumps
  straight to that file's cell and pre-selects it; switching **back** re-syncs the Analysis View
  (files + output gallery) so Explorer deletions take effect without a browser refresh. (The view
  switcher uses a thick ➤ arrow in an oval that grows on hover.)
- **Graphs**: signal + per-file frame-sync (adjustable **stagger %**, auto-fits), a
  spike-train **power** spectrum (`W = 2·|X[k]|²/N²`), and an **ISI histogram**; live spike
  detection, an editable analysis **region** (start / end, optional **crop**), and a
  **spike-train** view — controls floated on the graphs; mouse-wheel for fine zoom.

First time only (installs the package, the `neitz` command, Dash, and **kaleido** — the latter
drives the 4K PDF/SVG exports; it bundles a headless Chromium on first use):
```bash
pip install -e ".[gui]"
```

## The data store

All data + outputs live **outside the repo** at `~/Documents/ephysdataio/`
(override with `EPHYSDATAIO_ROOT`), laid out by **date → cell**:

```
~/Documents/ephysdataio/
  index.json                     # all cells
  2026-06-02/c01/                # <date>/<cell>
    manifest.json                # recordings (stimulus type+params, labels) + outputs
    raw/   2026_06_02_0040.abf …
    outputs/flicker/  flicker_onoff.{png,pdf,svg}  metrics.csv  result.json
                      window_4k.{pdf,svg,png}  power_4k.{pdf,png}  analog_framesync_4k.{pdf,png}
                      framesync_separated_4k.{pdf,png}  assumptions_4k.{pdf,png}
```

The `manifest.json` is the source of truth: stimulus metadata, friendly labels, and
every generated output (paths + params + timestamp).

## CLI (`neitz`)

```bash
neitz cell 2026-06-02 c01                 # run the sq-wave (flicker) analysis on a cell -> outputs into it
neitz flicker "~/Documents/ephysdataio/2026-06-02"   # per-file VS + pooled ON/OFF
neitz spikes  "~/Documents/ephysdataio/2026-06-02/c01" --out spikes.csv
neitz noise   ~/Documents/ephysdataio/2017-01-18/c01/raw/2017_01_18_siso-spikes.csv ~/Documents/ephysdataio/2017-01-18/c01/raw/2017_01_18_siso-stdev.csv --out sta   # reverse correlation -> sta.json/.npz
neitz --help                              # all commands     (or: python -m neitz …)
```

## Backup / mirror (per computer)

The mirror path is machine-local (kept in `~/.config/neitz/config.json`, not in the repo):

```bash
neitz mirror --set "/Users/j/Library/CloudStorage/GoogleDrive-…/My Drive/ephysdataio"
neitz mirror                              # sync now (true mirror via rsync)
neitz mirror --show
```

It also auto-backs-up after each `neitz cell …` run. On another computer, set its own
path once. (`neitz mirror --no-delete` for additive-only.)

## Python API

```python
from neitz.io import load_recording          # .abf -> Recording, spike .csv -> CsvSpikeRecording
from neitz.spikes import detect_spikes
from neitz.analysis import revcorr, flicker, strf
from neitz.dataio import DataStore
from neitz.run import run_cell_flicker, run_flicker, from_folder, Result

rec = load_recording("…/2026-06-02/c01/raw/2026_06_02_0040.abf")
st  = detect_spikes(rec.channel("Im_prime"), rec.fs, polarity="neg", k=6)  # or method="mad_floor"
fl  = flicker.detect_flicker(rec.channel("TTL"), rec.fs)

ds  = DataStore()                            # ~/Documents/ephysdataio
res = run_cell_flicker(ds, "2026-06-02", "c01")     # writes outputs into the cell
```

- `io/` — `Recording` (abf), `CsvSpikeRecording`, `load_recording`, `csv` loaders, `save_figure` (PNG/PDF/SVG).
- `spikes.py` — `detect_spikes` (`mad`, `abs`, `mad_floor = max(k·MAD, floor)`, `matlab` = Sara's `spikeDetectorOnline.m`).
- `analysis/` — `revcorr` (STA/linear filter), `flicker` (PSTH, vector strength, shift test), `strf` (checkerboard).
- `stimulus/` — `NoiseParadigm`, `FlickerParadigm`, `CheckerboardParadigm`.
- `dataio/` — `DataStore`, `CellManifest`, mirror.    `run.py` — orchestration + `Result`.

See **`README_TEST.md`** for a full feature/how-to guide and **`RESTRUCTURE_DESIGN.md`**
for the data-store design. Reference scripts are in **`examples/`**.

## Tests

```bash
pytest                     # algorithm tests run on synthetic data;
                           # real-.abf integration tests use the store, skip if absent
```
