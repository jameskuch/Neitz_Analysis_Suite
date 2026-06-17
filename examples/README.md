# examples/

Reference scripts showing how to use the `neitz` package directly. They produce the
multi-panel figures used during development.

- `analyze_flicker.py` — per-file flicker vector strength + cycle-PSTH grid
- `flicker_onoff_pooled.py` — pooled, significance-tested ON/OFF per cell
- `flicker_onoff_psth.py` — per-file transition-triggered ON/OFF PSTHs
- `extract_abf_spikes.py` — detect spikes → binary spike-train CSV (siso layout)

> **Note:** their `ABF_GLOBS` read from the managed data store
> (`~/Documents/ephysdataio/<date>/<cell>/`, override with `$EPHYSDATAIO_ROOT`),
> defaulting to the `2026-06-02` cells. Edit the glob to target other cells, or use the
> manifest-driven flow (`neitz.run` / the `neitz` CLI / the GUI) for routine work.

For routine work prefer the CLI (`neitz flicker …`) or `neitz.run` — see `README_TEST.md`.
