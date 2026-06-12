# Restructure Design — `ephysdataio` data store + manifest

Design doc for the major restructure. **Nothing is executed until this is signed off.**
Open questions for you are marked **[CONFIRM]**.

---

## 1. Goals

1. A managed data store outside the repo, under **`~/Documents/ephysdataio/`**.
2. A **JSON manifest** is the source of truth for every managed file (raw data *and*
   analysis output): stimulus type + parameters, channels, and every generated output.
3. Missing metadata → the **GUI queries the user** and writes it back to the manifest.
4. The **GUI reads the manifest** to: browse data (date → cell), route to the right
   pipeline, and label/title graphs from the hierarchy.
5. Outputs (CSV / PNG / **PDF / SVG**) are written into a **`date → cell`** folder
   hierarchy, and their paths are recorded back into the manifest as they're produced.
6. Repo is tidied: redundant files deleted, third-party deps removed, big data moved out.

---

## 2. On-disk layout

```
~/Documents/ephysdataio/                 # data root (EPHYSDATAIO_ROOT env var overrides)
  index.json                             # lightweight index of all cells/tissues
  2026-06-02/                            # <date> (acquisition date, ISO yyyy-mm-dd)
    ipRGC_c1/                            # <cell or tissue id> (filesystem-safe label)
      manifest.json                      # this cell's full record (see §3)
      raw/                               # raw recordings managed here (optional)
        2026_06_02_0040.abf
        ...
      outputs/                           # generated outputs, grouped by analysis
        flicker_onoff/
          flicker_onoff.png
          flicker_onoff.pdf
          flicker_onoff.svg
          metrics.csv
          result.json
    notipRGC_c1/
      ...
  2017-01-18/
    20170118Bc4/                         # the S-iso cell (migrated from SaraipRGC/)
      raw/  siso-spikes.csv  siso-stdev.csv  ...
      outputs/ ...
```

- **Raw data**: the GUI's **"Import data"** action **copies** the source files into
  `<cell>/raw/`, renaming to a **proper title format** if the source isn't already
  properly named. The manifest records both the stored path and the original source path.
  - *Proper title format*: `YYYY_MM_DD_<id>[_<tag>].<ext>`. A file already matching the
    `YYYY_MM_DD_NNNN` pattern (e.g. `2026_06_02_0040.abf`) is kept as-is; otherwise it is
    renamed using the cell's date + a recording id (e.g. `siso-spikes.csv` →
    `2017_01_18_siso-spikes.csv`).
  - De-dup: a source whose content already exists in the store (same name + size) is
    **not copied again**.
- **Outputs** always go under `<cell>/outputs/<analysis>/`.

---

## 3. Manifest schema

`index.json`
```json
{
  "version": 1,
  "cells": [
    {"date": "2026-06-02", "cell": "ipRGC_c1", "path": "2026-06-02/ipRGC_c1",
     "cell_type": "ipRGC", "n_recordings": 5, "n_outputs": 3}
  ]
}
```

`<date>/<cell>/manifest.json`
```json
{
  "version": 1,
  "date": "2026-06-02",
  "cell": "ipRGC_c1",
  "tissue": null,
  "cell_type": "ipRGC",
  "notes": "",
  "recordings": [
    {
      "id": "0040",
      "file": "raw/2026_06_02_0040.abf",
      "stimulus": {
        "type": "flicker",                 // flicker | gaussian_noise | checkerboard | ...
        "params": {"flicker_hz": 2.0, "carrier_hz": 120,
                   "cone_isolation": null, "frame_rate": 60},
        "source": "ttl"                    // ttl (auto) | user | file
      },
      "channels": {"signal": "Im_prime", "ttl": "TTL"},
      "fs": 20000, "duration_s": 79.2
    }
  ],
  "outputs": [
    {
      "analysis": "flicker_onoff_pooled",
      "created": "2026-06-12T14:30:00",
      "params": {"polarity": "neg", "method": "mad", "k": 6, "n_shuffle": 1000},
      "inputs": ["0040","0041","0042","0043","0044"],
      "files": {"figure_png": "outputs/flicker_onoff/flicker_onoff.png",
                "figure_pdf": "outputs/flicker_onoff/flicker_onoff.pdf",
                "figure_svg": "outputs/flicker_onoff/flicker_onoff.svg",
                "metrics_csv": "outputs/flicker_onoff/metrics.csv",
                "result_json": "outputs/flicker_onoff/result.json"},
      "summary": {"verdict": "no sig.", "on_p": 0.45}
    }
  ]
}
```

- **Stimulus param sets** per type (the GUI form fields):
  - `flicker`: flicker_hz, carrier_hz, cone_isolation, frame_rate (most auto-recovered from TTL)
  - `gaussian_noise`: cone_isolation (S/L/M/LM/achromatic), stdev, frame_rate, bins_per_frame, seeds
  - `checkerboard`: n_y, n_x, stixel_size, stdev, frame_rate, seeds
- **Cell/tissue id**: a **forced number** plus an optional **free-text label**. The
  folder is `c<NN>` (auto-incremented per date, zero-padded), and the manifest stores a
  free-text `label` (e.g. "ipRGC", "20170118Bc4"). The GUI shows "c01 — ipRGC". (Folder
  may be rendered `c01_ipRGC` for readability; the forced `c<NN>` is canonical.)

---

## 4. New code

```
neitz/dataio/
  store.py     DataStore  -> resolve root; read/write index; list/open cells
  manifest.py  CellManifest -> recordings, stimulus get/set, output_dir(), record_output(), save()
  config.py    root resolution: EPHYSDATAIO_ROOT > ~/Documents/ephysdataio
neitz/io/figures.py
  save_figure(fig, out_dir, name, formats=("png","pdf","svg")) -> {fmt: path}
```

API sketch:
```python
from neitz.dataio import DataStore
ds   = DataStore()                                   # ~/Documents/ephysdataio
cell = ds.cell("2026-06-02", "ipRGC_c1")             # CellManifest (creates if new)
cell.add_recording("….abf", stimulus={"type":"flicker","params":{…}})
cell.get_stimulus("0040")                            # -> params or None (None -> GUI asks)
out = cell.output_dir("flicker_onoff")               # …/outputs/flicker_onoff/
cell.record_output("flicker_onoff", files={…}, params={…}, summary={…}); cell.save()
```

The existing `neitz.run.Result` gains a `.save_into(cell, analysis, params)` that writes
JSON/CSV/figures under the cell and records them in the manifest.

---

## 5. GUI integration

- **Browse by manifest**: the viewer reads `index.json` and shows a **date → cell** tree
  (the raw file/folder browse stays as a fallback for ad-hoc files).
- **Metadata query**: selecting a recording whose `stimulus` is missing opens a **modal
  form** (stimulus-type dropdown + the param fields for that type) → written to the
  manifest. Auto-filled where possible (flicker freq from TTL, fs/channels from the abf).
- **Pipeline routing**: stimulus `type` selects the analysis (flicker / noise / checkerboard).
- **Labeling**: figure titles include `date · cell · cell_type · stimulus(params)`.
- **Outputs**: analyses run from the GUI write into `<cell>/outputs/` (PNG+PDF+SVG+CSV)
  and append to the manifest.

---

## 6. Output formats

Every figure is saved as **PNG + PDF + SVG** (PDF & SVG are Illustrator-editable vectors).
Centralised in `save_figure()`; default formats configurable. CSVs and a `result.json`
accompany each analysis output.

---

## 7. Repo cleanup / migration

| Item | Action |
|---|---|
| `jk.py`, `jk2.py`, `jk3.py` | **delete** (GUI/CLI cover this) |
| `extract_siso2.py`, `extract_siso4.py` | **delete** (→ `neitz.run.run_noise` / CLI `neitz noise`) |
| `analyze_flicker.py`, `flicker_onoff_pooled.py`, `extract_abf_spikes.py`, `flicker_onoff_psth.py` | **keep as examples** → move to `examples/` |
| `legacy/` | **delete** |
| `abfUtilities-main/` | **delete** (unused; we use `pyabf`) |
| `neitz.egg-info/` | leave gitignored (auto-generated, not a dependency) |
| `SaraipRGC/` (759 MB) | **migrate** into `ephysdataio` (data → a 2017 cell; redundant `c4_bigger`/`c4_smaller`/`process_*` dropped), then remove from repo |
| `data/` (barak abfs) | **migrate** into `ephysdataio` (2026-06-02 → ipRGC + notipRGC cells), then remove from repo |
| `ipRGC barak/` | **delete** — its abfs are already duplicated in `data/`; migrate from `data/` only (no redundant copy) |
| generated `*.csv` / `*.png` at root | regenerate into the data store; remove strays |

---

## 8. Build phases (after sign-off)

- **A — Cleanup**: deletions, move SaraipRGC out, move example scripts to `examples/`.
- **B — `neitz.dataio`**: DataStore + CellManifest + `save_figure` (PNG/PDF/SVG) + tests.
- **C — Wire `run`/CLI** to read/write the manifest and emit outputs into the store.
- **D — GUI**: manifest-driven date→cell browse, metadata-query modal, manifest labeling.

Each phase is tested and committed separately; destructive steps (Phase A) confirmed first.

---

## 9. Resolved decisions

1. **Import = copy** the source into `<cell>/raw/`, renaming to the proper title format
   if needed; record original + stored paths; de-dup identical files.
2. **Cell id = forced `c<NN>` + free-text label.**
3. **Migrate** `data/` and `SaraipRGC/` contents into `ephysdataio` now.
4. **`ipRGC barak/` is deleted** (redundant with `data/`); migrate from `data/` only.
5. Output formats: **PNG + PDF + SVG**. Flicker scripts kept as `examples/`.
   `abfUtilities-main` deleted. `neitz.egg-info` left gitignored.
