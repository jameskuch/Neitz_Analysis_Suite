# Neitz Analysis Suite

Electrophysiology analysis for the Neitz lab: ABF/CSV I/O, spike detection,
reverse-correlation (STA / linear filter), flicker light-response, and spatiotemporal
STRF — with a managed data store and three front-ends (**GUI · CLI · Python API**).

## Quick start

```bash
cd /Users/j/Neitz_Analysis_Suite
python viewer.py            # then open http://127.0.0.1:8050  (Ctrl-C to stop)
```

That's the program. In the GUI (top to bottom):
- **cell (data store)** — pick a `date / cell — label`; it loads that cell's recordings.
- **stimulus type / params + Save metadata** — record what was shown (esp. for cells
  with blank metadata).
- **▶ Run flicker → cell** — runs the analysis; writes PNG/PDF/SVG + CSV + JSON into the cell.
- **⤓ Backup to mirror** — copy the store to Google Drive (after you set the path, below).
- Or **📁 Browse file…** to open any `.abf` or spike `.csv` directly; live spike detection,
  spike-train FFT, region show/crop/baseline, multi-file group averaging.

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
neitz cell 2026-06-02 c01                 # run flicker on a stored cell -> outputs into it
neitz flicker "~/Documents/ephysdataio/2026-06-02"   # per-file VS + pooled ON/OFF
neitz spikes  "~/Documents/ephysdataio/2026-06-02/c01" --out spikes.csv
neitz noise   siso-spikes.csv siso-stdev.csv --out sta   # reverse correlation -> sta.json/.npz
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
