"""Data-store location, machine-local config, and filename conventions."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

DEFAULT_ROOT = Path.home() / "Documents" / "ephysdataio"

# "proper" raw filename: YYYY_MM_DD_<id...>  (e.g. 2026_06_02_0040.abf)
_PROPER_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\S+")


# ---- machine-local config (~/.config/neitz/config.json) --------------------
# Per-computer settings that must NOT live in the repo or the data store — most
# importantly the mirror/backup path (e.g. this machine's Google Drive folder,
# whose absolute path differs across computers).
def config_path() -> Path:
    return Path.home() / ".config" / "neitz" / "config.json"


def load_config() -> dict:
    p = config_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_config(cfg: dict) -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2))
    return p


def data_root() -> Path:
    """Root of the managed data store. EPHYSDATAIO_ROOT > config > default."""
    env = os.environ.get("EPHYSDATAIO_ROOT")
    if env:
        return Path(env)
    cfg = load_config().get("data_root")
    return Path(cfg) if cfg else DEFAULT_ROOT


def mirror_dir():
    """This computer's mirror/backup path, or None. EPHYSDATAIO_MIRROR > config."""
    env = os.environ.get("EPHYSDATAIO_MIRROR")
    if env:
        return Path(env)
    m = load_config().get("mirror")
    return Path(m).expanduser() if m else None


def set_mirror(path) -> Path:
    cfg = load_config()
    cfg["mirror"] = str(Path(path).expanduser())
    return save_config(cfg)


def auto_mirror() -> bool:
    """Whether to auto-back-up to the mirror after an analysis run (default True)."""
    return bool(load_config().get("auto_mirror", True))


def is_proper_name(filename) -> bool:
    return bool(_PROPER_RE.match(Path(filename).stem))


def proper_name(filename, date) -> str:
    """Keep an already date-stamped name; otherwise prefix with the cell's date."""
    p = Path(filename)
    if is_proper_name(p.name):
        return p.name
    d = str(date).replace("-", "_")
    return f"{d}_{p.name}"
