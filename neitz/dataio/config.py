"""Data-store location + filename conventions for ephysdataio."""
from __future__ import annotations
import os
import re
from pathlib import Path

DEFAULT_ROOT = Path.home() / "Documents" / "ephysdataio"

# "proper" raw filename: YYYY_MM_DD_<id...>  (e.g. 2026_06_02_0040.abf)
_PROPER_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\S+")


def data_root() -> Path:
    """Root of the managed data store (EPHYSDATAIO_ROOT overrides the default)."""
    return Path(os.environ.get("EPHYSDATAIO_ROOT", str(DEFAULT_ROOT)))


def is_proper_name(filename) -> bool:
    return bool(_PROPER_RE.match(Path(filename).stem))


def proper_name(filename, date) -> str:
    """Keep an already date-stamped name; otherwise prefix with the cell's date."""
    p = Path(filename)
    if is_proper_name(p.name):
        return p.name
    d = str(date).replace("-", "_")
    return f"{d}_{p.name}"
