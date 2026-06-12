# legacy/

Quarantined code from before the `neitz` package refactor. Kept for reference,
**not** part of the active codebase.

- **Neitz.py** — the original 1247-line monolith. Its reusable parts were ported
  into the `neitz` package (ABF I/O → `neitz.io.abf.Recording`; spike detection →
  `neitz.spikes`; CSV loaders → `neitz.io.csv`; reverse correlation/flicker →
  `neitz.analysis`). What remains here is the **4 Hz-flicker + contrast-CSV STA
  tutorial** subsystem, which has no in-repo data and was not ported.
- **sta_tutorial.py**, **abf_ret.py** — the STA tutorial examples. They depend on
  `Neitz.py` and on `2026_02_04_*.abf` recordings that are no longer in the repo,
  so they cannot run as-is. Run from inside this folder if reviving.
- **extract_siso.py** — the first siso reverse-correlation iteration, superseded by
  the package-based `extract_siso2.py` / `extract_siso4.py` in the project root.

To bring any of this back to life, port the needed pieces onto the `neitz`
package (guarded by `tests/`) rather than reusing the monolith.
