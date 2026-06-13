# Neitz Analysis Suite

Electrophysiology analysis for the Neitz lab: ABF/CSV I/O, spike detection,
reverse-correlation (STA / linear filter), flicker light-response, and spatiotemporal
STRF — with a managed data store and three front-ends (**GUI · CLI · Python API**).

## Quick start

```bash
cd /Users/j/Neitz_Analysis_Suite
python viewer.py            # then open http://127.0.0.1:8050  (Ctrl-C to stop)
```

That's the program. The GUI is a left sidebar (controls) + right graphs:
- **cell(s)** — multi-select one or more `date / cell — label`s (with an inline
  date / label / type sort); the checked **files** load into the graphs. Set
  **stimulus type / params + Save metadata** to record what was shown.
- **▶ Run sq wave** — runs the square-wave (flicker) analysis on the *checked* files; a
  **run name** keeps variants side by side (e.g. excluding a recording); writes PNG/PDF/SVG
  + CSV + JSON into `<cell>/outputs/<name>/`.
- **📂 Data explorer** — browse the store by date → cell thumbnails (waveform + latest
  output) → files; click to enlarge raw traces and figures; **Import data** new recordings;
  delete a whole day (🗑 by each date), selected files, or a figure (a confirmation
  warning; trashed reversibly). **⤓ Backup mirror**
  copies the store to Google Drive.
- **Graphs**: signal + per-file frame-sync (adjustable **stagger %**), a spike-train
  **power** spectrum (`W = 2·|X[k]|²/N²`), and an **ISI histogram**; live spike detection
  (polarity, k·MAD / absolute / floor, with per-trace thresholds), an editable analysis
  **region** (start / end, optional **crop**), and a **spike-train** view — all floated as
  controls on the graphs.

First time only (installs the package, the `neitz` command, and Dash):
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
- `spikes.py` — `detect_spikes` (`mad`, `abs`, `mad_floor = max(k·MAD, floor)`).
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
