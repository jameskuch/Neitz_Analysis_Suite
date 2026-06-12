# Neitz Analysis Suite

Electrophysiology analysis for the Neitz lab: ABF I/O, spike / action-current
detection, reverse-correlation (spike-triggered average / linear filter), and
flicker light-response analysis, with a Dash GUI viewer.

## Install

```bash
pip install -e .            # core library (numpy, scipy, pyabf, ...)
pip install -e ".[gui]"     # + Dash/Plotly for the viewer
pip install -e ".[dev]"     # + pytest for the test suite
```

## Library (`neitz/`)

```python
from neitz.io.abf import Recording
from neitz.spikes import detect_spikes
from neitz.analysis import revcorr, flicker

rec = Recording.load("path/to.abf")
im  = rec.channel("Im_prime")           # channels by name
st  = detect_spikes(im, rec.fs, polarity="neg", k=6)        # or method="mad_floor"
fl  = flicker.detect_flicker(rec.channel("TTL"), rec.fs)    # square-wave timing from TTL
```

- `io/abf.py` — `Recording`: path-explicit ABF loader, channels by name/index.
- `spikes.py` — `detect_spikes` (`mad`, `abs`, or `mad_floor = max(k·MAD, floor)`).
- `analysis/revcorr.py` — reverse correlation / linear filter, `normalize_filter('max'|'std')`.
- `analysis/flicker.py` — `detect_flicker`, cycle/transition PSTH, vector strength, shift test.

## GUI

```bash
python viewer.py            # http://127.0.0.1:8050
```

Interactive ABF viewer: native file dialogs, live spike detection, spike-train
FFT, editable analysis region (show / crop / baseline), multi-file group
averaging, persisted settings.

## Tests

```bash
pytest                     # algorithm tests run on synthetic data;
                           # real-.abf integration tests skip if data absent
```
