# examples/

Reference scripts showing how to use the `neitz` package directly. They produce the
multi-panel figures used during development.

- `analyze_flicker.py` — per-file flicker vector strength + cycle-PSTH grid
- `flicker_onoff_pooled.py` — pooled, significance-tested ON/OFF per cell
- `flicker_onoff_psth.py` — per-file transition-triggered ON/OFF PSTHs
- `extract_abf_spikes.py` — detect spikes → binary spike-train CSV (siso layout)

> **Note:** their hardcoded `ABF_GLOBS` point at the old in-repo `data/ipRGC barak/…`
> layout, which has moved into the data store (`~/Documents/ephysdataio/<date>/<cell>/`).
> Update the glob (or use `neitz.run` / the `neitz` CLI / the GUI) to run them against
> the store. They will be migrated to the manifest-driven flow in a later phase.

For routine work prefer the CLI (`neitz flicker …`) or `neitz.run` — see `README_TEST.md`.
