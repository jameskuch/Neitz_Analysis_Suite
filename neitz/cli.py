"""
neitz.cli — command-line interface to the analysis pipeline.

After `pip install -e .`:   neitz flicker data/ipRGC\\ barak --out flicker
Without reinstalling:       python -m neitz flicker "data/ipRGC barak" --out flicker

Subcommands:
  flicker FOLDER     per-file vector strength + per-cell pooled ON/OFF
  spikes  FOLDER     detect spikes -> binary spike-train CSV (siso-spikes layout)
  noise   SPIKE_CSV STIM_CSV    S-iso reverse correlation (STA / linear filter)
"""
from __future__ import annotations
import argparse
import sys

from . import run as R
from .stimulus import FlickerParadigm, NoiseParadigm


def _print_table(rows, cols=None):
    if not rows:
        print("  (no rows)"); return
    cols = cols or list(rows[0].keys())
    print("  " + "  ".join(f"{c}" for c in cols))
    for r in rows:
        print("  " + "  ".join(_fmt(r.get(c)) for c in cols))


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _add_detect_args(p):
    p.add_argument("--channel", default="Im_prime")
    p.add_argument("--polarity", default="neg", choices=["neg", "pos", "abs"])
    p.add_argument("--method", default="mad", choices=["mad", "abs", "mad_floor"])
    p.add_argument("--k", type=float, default=6.0)
    p.add_argument("--abs-threshold", type=float, default=None)
    p.add_argument("--refractory-ms", type=float, default=2.0)


def _flicker_paradigm(a):
    return FlickerParadigm(current_channel=a.channel, polarity=a.polarity, method=a.method,
                           k=a.k, abs_threshold=a.abs_threshold,
                           refractory_s=a.refractory_ms / 1000.0)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="neitz", description="Neitz electrophysiology analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("flicker", help="flicker light-response (VS + pooled ON/OFF)")
    f.add_argument("folder")
    f.add_argument("--out", help="write <out>.json + <out>_summary.csv")
    f.add_argument("--no-pooled", action="store_true")
    f.add_argument("--n-shuffle", type=int, default=1000)
    _add_detect_args(f)

    s = sub.add_parser("spikes", help="detect spikes -> binary spike-train CSV")
    s.add_argument("folder")
    s.add_argument("--out", default="spikes.csv")
    s.add_argument("--out-rate", type=int, default=10000)
    _add_detect_args(s)

    n = sub.add_parser("noise", help="S-iso reverse correlation (STA / linear filter)")
    n.add_argument("spike_csv")
    n.add_argument("stim_csv")
    n.add_argument("--out", help="write <out>.json + <out>.npz")
    n.add_argument("--normalize", default="max", choices=["max", "std", "none"])
    n.add_argument("--zero-pad", type=int, default=None)

    a = ap.parse_args(argv)

    if a.cmd == "flicker":
        res = R.run_flicker_folder(a.folder, paradigm=_flicker_paradigm(a),
                                   pooled=not a.no_pooled, n_shuffle=a.n_shuffle)
        print(f"flicker: {len(res.summary)} file(s)")
        _print_table(res.summary, ["file", "group", "flicker_hz", "n_in_region",
                                   "vector_strength", "rayleigh_p"])
        if res.tables.get("pooled_onoff"):
            print("\npooled ON/OFF per cell:")
            _print_table(res.tables["pooled_onoff"],
                         ["cell", "n_trials", "flicker_hz", "on_ratio", "on_p",
                          "off_ratio", "off_p", "verdict"])
        if a.out:
            res.save(a.out); res.save_csv(a.out + "_summary.csv")
            print(f"\nwrote {a.out}.json, {a.out}_summary.csv")

    elif a.cmd == "spikes":
        res = R.run_spike_export(a.folder, a.out, channel=a.channel, out_rate=a.out_rate,
                                 polarity=a.polarity, method=a.method, k=a.k,
                                 abs_threshold=a.abs_threshold,
                                 refractory_s=a.refractory_ms / 1000.0)
        _print_table(res.summary, ["file", "duration_s", "n_spikes", "rate_hz"])
        print(f"\nwrote {a.out}")

    elif a.cmd == "noise":
        norm = None if a.normalize == "none" else a.normalize
        res = R.run_noise(a.spike_csv, a.stim_csv,
                          paradigm=NoiseParadigm(normalize=norm,
                                                 zero_pad=(a.zero_pad if a.zero_pad is not None else 60)),
                          zero_pad=a.zero_pad)
        _print_table(res.summary)
        if a.out:
            res.save(a.out)
            print(f"\nwrote {a.out}.json, {a.out}.npz")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
