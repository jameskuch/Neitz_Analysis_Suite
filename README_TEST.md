# Neitz Analysis Suite — Feature Guide & How-To

Everything added during the `neitz` package refactor, with copy-pasteable examples.
The same analysis core is reachable three ways: a **GUI**, a **CLI**, and a
**Python API**. All numbers are locked in by a regression test suite (`pytest`).

> Use the conda-base Python that has the deps: `/Users/j/miniconda3/bin/python`.

---

## 0. Setup

```bash
pip install -e .            # core library (numpy, scipy, pyabf, pandas, ...)
pip install -e ".[gui]"     # + Dash/Plotly for the viewer
pip install -e ".[dev]"     # + pytest
pytest                      # run the test suite (35 tests; .abf tests skip if data absent)
```

After `pip install -e .` the **`neitz`** command is on your PATH. (Re-run the install
after editing `pyproject.toml`.) You can always use `python -m neitz ...` instead.

---

## 1. Loading recordings

`load_recording` dispatches by extension — `.abf` → `Recording`, spike `.csv` →
`CsvSpikeRecording` (see §7). Channels are addressed **by name** or index.

```python
from neitz.io import load_recording, Recording

rec = load_recording("~/Documents/ephysdataio/2026-06-02/c01/raw/2026_06_02_0044.abf")
rec.fs               # 20000.0
rec.duration         # 92.0
rec.channel_names    # ['Im_prime', 'Vm_sec', 'TTL']
im  = rec.channel("Im_prime")     # by name (case-insensitive, partial match ok)
ttl = rec.channel("TTL")
rec.metadata()       # sample rate, protocol, recording date, Clampex version, ...
```

---

## 2. Spike detection

`detect_spikes` works on any 1-D signal. Three threshold methods, selectable polarity,
and a refractory period.

```python
from neitz.spikes import detect_spikes

# k * robust-noise-sigma (MAD) — noise-adaptive
st = detect_spikes(im, rec.fs, polarity="neg", method="mad", k=6)

# absolute pA threshold
st = detect_spikes(im, rec.fs, polarity="neg", method="abs", abs_threshold=50)

# k*MAD with an absolute FLOOR -> threshold = max(k*MAD, floor)
#   keeps noise-adaptivity but rejects small proximal-cell spikes
st = detect_spikes(im, rec.fs, polarity="neg", method="mad_floor", k=6, abs_threshold=50)

len(st)                 # spike count
st.times                # spike times (s)
st.rate(rec.duration)   # Hz
st.threshold            # effective threshold actually used
st.binary(10_000, duration=rec.duration)   # 0/1 train at 10 kHz (siso-spikes.csv style)
```

---

## 3. Analysis engines

```python
from neitz.analysis import revcorr, flicker, strf
```

**Reverse correlation (linear filter / STA)** — `neitz.analysis.revcorr`
```python
f = revcorr.reverse_correlation(stimulus, response, filter_len=360, zero_pad=60)
out = revcorr.average_filter(stimuli, responses, 360, normalize="max")  # 'max'|'std'|None
freqs, tuning = revcorr.temporal_tuning(out["average"], bin_rate=360)
```

**Flicker** — `neitz.analysis.flicker`
```python
fl = flicker.detect_flicker(ttl, rec.fs)        # square-wave timing from the TTL
fl.freq, fl.t0, fl.t1                           # e.g. 2.0 Hz, 15.4 s, 74.0 s
vs, p = flicker.vector_strength(spike_times, fl.t0, fl.freq)
# pooled transition-triggered ON/OFF significance (jitter null by default)
res = flicker.shift_test(trials, "on", pre_s=0.1, post_s=0.4, bin_s=0.01)
```

**Spatiotemporal STRF (checkerboard)** — `neitz.analysis.strf`
```python
S = strf.spatiotemporal_revcorr(stim, response, filter_len=30)   # stim (n_checks, n_time)
cube = strf.reshape_strf(S, n_y, n_x)                            # (n_y, n_x, filter_len)
strf.peak_check(S)                                               # strongest check
```

---

## 4. Paradigms (the high-level interface)

A `Paradigm` bundles "detect + analyze" for one stimulus type. Cone isolation
(S/L/M/LM-iso) is metadata — the math is identical.

```python
from neitz.stimulus import FlickerParadigm, NoiseParadigm, CheckerboardParadigm

# Flicker — per-recording and per-cell (pooled) light response
fp = FlickerParadigm(current_channel="Im_prime", ttl_channel="TTL",
                     polarity="neg", method="mad", k=6)
r = fp.analyze_recording(rec, name="0044")
r.freq, r.vector_strength, r.rayleigh_p, r.n_in_region
grp = fp.analyze_group([rec1, rec2, rec3])      # pools trials -> ON/OFF + p-values

# Gaussian noise -> reverse correlation
out = NoiseParadigm(normalize="max").analyze(stimuli, responses)

# Checkerboard -> STRF
res = CheckerboardParadigm(n_y=6, n_x=8, filter_len=30).analyze(stimulus, response)
res.peak_yx, res.peak_time_ms, res.strf.shape   # ((y,x), ms, (6,8,30))
```

---

## 5. CLI (`neitz`)

```bash
# flicker: per-file vector strength + per-cell pooled ON/OFF
neitz flicker "~/Documents/ephysdataio/2026-06-02" --out flick
#   -> prints tables, writes flick.json + flick_summary.csv
#   options: --polarity --method {mad,abs,mad_floor} --k --abs-threshold --refractory-ms --n-shuffle

# spikes: detect -> binary spike-train CSV (siso-spikes.csv layout)
neitz spikes "~/Documents/ephysdataio/2026-06-02/c01" --out spikes.csv --k 6

# noise: S-iso reverse correlation (STA / linear filter)
neitz noise ~/Documents/ephysdataio/2017-01-18/c01/raw/2017_01_18_siso-spikes.csv ~/Documents/ephysdataio/2017-01-18/c01/raw/2017_01_18_siso-stdev.csv --out sta
#   -> prints peak latency/sign, writes sta.json + sta.npz (filter arrays)

python -m neitz flicker "~/Documents/ephysdataio/2026-06-02"     # equivalent without the console script
```

---

## 6. Headless API + saveable results

`neitz.run` orchestrates folder → results and saves them for later/elsewhere.

```python
from neitz.run import run_flicker, from_folder, run_spike_export, run_noise, run_strf, Result

res = run_flicker(from_folder("~/Documents/ephysdataio/2026-06-02"))   # per-file + pooled-per-cell
res.summary            # list of per-file dict rows
res.tables["pooled_onoff"]    # per-cell ON/OFF table
res.save("flick")      # -> flick.json (+ flick.npz if arrays)
res.save_csv("flick_summary.csv")

later = Result.load("flick")          # reload anywhere
later.summary, later.arrays

run_spike_export("~/Documents/ephysdataio/2026-06-02/c01", "spikes.csv", k=6)
run_noise("~/Documents/ephysdataio/2017-01-18/c01/raw/2017_01_18_siso-spikes.csv", "~/Documents/ephysdataio/2017-01-18/c01/raw/2017_01_18_siso-stdev.csv").save("sta")
run_strf(stimulus, response, n_y=6, n_x=8).save("strf")   # STRF cube -> strf.npz
```

---

## 7. Opening spike CSVs (NEW)

A spike CSV (`col 0 = time`, `cols 1.. = spike channels`, values binary **or**
amplitude) can be loaded like a recording, and **opened in the viewer**.

**Programmatic:**
```python
from neitz.io import load_recording           # dispatches .csv -> CsvSpikeRecording
rec = load_recording("~/Documents/ephysdataio/2017-01-18/c01/raw/2017_01_18_siso-spikes.csv")
rec.channel_names         # ['ch1', 'ch2', ... 'ch15']
rec.fs                    # 10000 (from the time column)
rec.channel("ch1")        # that spike column

# or the raw arrays:
from neitz.io import csv as ncsv
time, spikes = ncsv.load_spikes_csv("~/Documents/ephysdataio/2017-01-18/c01/raw/2017_01_18_siso-spikes.csv")   # (N,), (N, M)
spike_times  = ncsv.spike_times_from_matrix(spikes, time)          # per-channel times
```

**In the GUI:** click **Browse file…** and pick the `.csv` (the dialog now accepts
`.abf` *and* `.csv`). Each spike column appears as a channel `ch1, ch2, …`. To mark
the spikes, set **polarity = pos**, **threshold = absolute**, **abs thresh ≈ 0.5**
(or use **spike-train view**). The TTL/flicker/region features don't apply to spike
CSVs (no frame sync), so the analysis region is the full trace.

---

## 8. The GUI viewer

```bash
python viewer.py            # http://127.0.0.1:8050
```

Features:
- **Native macOS dialogs** — 📁 Browse file… (`.abf` or spike `.csv`), 📂 Browse folder…
  (repopulates the file list). The data folder + detection settings **persist across
  sessions**.
- **Files as a checkbox grid** — check **1** to inspect, **2+** to group-average.
- **Channels by name**, with an acquisition **metadata** strip (rate, protocol, date…).
- **Live spike detection** — polarity (neg/pos/abs), threshold (**k·MAD / absolute /
  k·MAD ≥ floor**), `k` slider, refractory. Number boxes apply on **Enter / click-away**;
  radios/checkboxes apply instantly.
- **Excluded regions** (radio): **show** (shade adapting/post blocks) · **crop** (clip
  to the analysis region) · **baseline** (flatten excluded regions to each channel's
  baseline). The **region start/end** boxes auto-fill from the detected flicker and are
  editable.
- **Display options**: stagger overlaid frame syncs · hide detected spikes ·
  **spike-train view** (0/1 impulses, or binned counts via the bin box).
- **Multi-file group mode**: signals + frame syncs overlaid in matching colors; FFT
  shows each file's spike-train spectrum + a bold **GROUP-AVG**.
- **Spike-train FFT** with the detected stimulus frequency marked.
- **Linked x-axes** (signal + frame sync) with zoom that re-renders at full detail.

---

## 9. Tests

```bash
pytest                      # 35 tests
pytest tests/test_spikes.py -v
```

- Algorithm tests run on **synthetic data** (anywhere): spike detection (incl.
  `mad_floor`), reverse correlation, flicker, STRF receptive-field recovery, CSV loaders.
- Real-`.abf` integration tests pin the validated numbers (1157 spikes / 2.0 Hz on 0044)
  and **skip** when the gitignored data is absent.

---

## Where things live

```
neitz/
  io/abf.py        Recording (abf)        io/csv.py   CSV loaders + CsvSpikeRecording
  io/__init__.py   load_recording() factory
  spikes.py        detect_spikes / SpikeTrain
  analysis/        revcorr · flicker · strf
  stimulus/        NoiseParadigm · FlickerParadigm · CheckerboardParadigm
  run.py           run_flicker/spikes/noise/strf + Result (save/load)
  cli.py           the `neitz` command
viewer.py          Dash GUI
tests/             pytest suite
legacy/            retired Neitz.py monolith + 4 Hz-flicker STA tutorial
```
