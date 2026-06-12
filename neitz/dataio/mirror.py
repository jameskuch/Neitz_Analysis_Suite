"""
mirror_store — duplicate the ephysdataio data store to a configured mirror
(e.g. this computer's Google Drive folder). True mirror: the destination is made
an exact copy of the source (extra files in the mirror are deleted).

Uses rsync when available (fast, incremental), with a pure-Python fallback.
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

from .config import data_root, mirror_dir


def mirror_store(src=None, dst=None, *, delete=True) -> dict:
    """
    Copy `src` (default: the data store) to `dst` (default: the configured mirror).
    With delete=True the mirror is made an exact copy (removes extras).
    Returns {'method', 'src', 'dst', 'n_copied' (python only)}.
    """
    src = Path(src) if src is not None else Path(data_root())
    dst = Path(dst) if dst is not None else mirror_dir()
    if dst is None:
        raise RuntimeError("no mirror configured — set one with "
                           "`neitz mirror --set PATH` or neitz.dataio.set_mirror(path)")
    if not src.exists():
        raise RuntimeError(f"data store not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    if shutil.which("rsync"):
        cmd = ["rsync", "-a"] + (["--delete"] if delete else []) + [f"{src}/", f"{dst}/"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"rsync failed: {r.stderr.strip()[-300:]}")
        return {"method": "rsync", "src": str(src), "dst": str(dst)}

    n = _py_mirror(src, dst, delete=delete)
    return {"method": "python", "src": str(src), "dst": str(dst), "n_copied": n}


def _py_mirror(src: Path, dst: Path, *, delete: bool) -> int:
    src_rel = set()
    n_copied = 0
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        src_rel.add(rel)
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            need = (not target.exists()
                    or target.stat().st_size != p.stat().st_size
                    or target.stat().st_mtime < p.stat().st_mtime - 1e-6)
            if need:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
                n_copied += 1
    if delete:
        for p in sorted(dst.rglob("*"), key=lambda q: len(q.parts), reverse=True):
            if p.relative_to(dst) not in src_rel:
                if p.is_dir():
                    try:
                        p.rmdir()
                    except OSError:
                        pass
                else:
                    p.unlink()
    return n_copied
