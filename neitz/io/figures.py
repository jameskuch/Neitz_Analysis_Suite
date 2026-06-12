"""
save_figure — write a matplotlib figure in multiple formats at once.

Defaults to PNG + PDF + SVG (PDF & SVG are Adobe Illustrator-editable vectors).
Returns {format: path} so callers can record the paths in a manifest.
"""
from __future__ import annotations
from pathlib import Path

DEFAULT_FORMATS = ("png", "pdf", "svg")


def save_figure(fig, out_dir, name, formats=DEFAULT_FORMATS, dpi=150) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fmt in formats:
        p = out_dir / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths[fmt] = str(p)
    return paths
