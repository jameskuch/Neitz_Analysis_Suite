#!/usr/bin/env python
"""
rebuild_backend.py — force-close, rebuild, and restart the viewer.py back end.

A dev/ops convenience (sibling of run.py) that guarantees a clean back end running the LATEST code,
with no stale process, port, or cache lingering. It does three things:

  1. FORCE-CLOSE — kill any running `viewer.py` and free port 8050 (the Dash server) + 8051
     (neitz_app.py's single-instance lock), so nothing stale is left holding on.
  2. REBUILD    — clear stale Python bytecode (`__pycache__` / `*.pyc`) under the repo, wipe the
     background-callback diskcache (`<tempdir>/neitz_dash_bg`, which can hold stale run results),
     and refresh the editable install (`pip install -e ".[gui]"`).
  3. RESTART    — start a fresh `viewer.py` (detached, survives this script), wait for
     `/neitz-health`, and report the new boot id + PID (confirming exactly ONE server on 8050).

Run it with the conda-base interpreter that has the GUI deps:

    /Users/j/miniconda3/bin/python neitz/rebuild_backend.py
    # or, since it ships in the package:
    /Users/j/miniconda3/bin/python -m neitz.rebuild_backend

Flags:
    --no-reinstall   skip the `pip install -e` step (faster; just clears caches)
    --no-restart     force-close + rebuild only, don't start a new server
    --quiet          less chatter
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # neitz/ -> repo root (holds viewer.py)
VIEWER = REPO / "viewer.py"
HOST, PORT = "127.0.0.1", 8050
LOCK_PORT = 8051                                        # neitz_app.py single-instance lock
HEALTH = f"http://{HOST}:{PORT}/neitz-health"
BG_CACHE = Path(tempfile.gettempdir()) / "neitz_dash_bg"   # DiskcacheManager dir (see viewer.py)
LOG_DIR = Path.home() / ".neitz"
VIEWER_LOG = LOG_DIR / "viewer.log"
PY = sys.executable                                    # relaunch with the SAME interpreter
IS_WIN = sys.platform.startswith("win")

_QUIET = False


def say(msg):
    if not _QUIET:
        print(msg, flush=True)


# ---------------------------------------------------------------- process / port control
def _pids_on_port(port):
    """PIDs LISTENing on `port` (own user), via the OS's own tool (no elevated perms needed)."""
    pids = set()
    try:
        if IS_WIN:
            out = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                                 capture_output=True, text=True).stdout
            for ln in out.splitlines():
                p = ln.split()
                if len(p) >= 5 and p[1].endswith(f":{port}") and p[3].upper() == "LISTENING":
                    pids.add(int(p[4]))
        else:
            out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                                 capture_output=True, text=True).stdout
            pids = {int(x) for x in out.split()}
    except Exception as e:
        say(f"   (could not list pids on {port}: {e})")
    return pids


def _kill(pids):
    for pid in pids:
        try:
            if IS_WIN:
                subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True)
            else:
                os.kill(pid, signal.SIGKILL)
        except Exception as e:
            say(f"   (could not kill pid {pid}: {e})")


def force_close():
    say("1. force-closing the back end …")
    if not IS_WIN:                                      # kill by name first (catches a server not on 8050)
        subprocess.run(["pkill", "-9", "-f", "viewer.py"], capture_output=True)
    killed = set()
    for port in (PORT, LOCK_PORT):
        pids = _pids_on_port(port)
        if pids:
            _kill(pids)
            killed |= pids
    time.sleep(1.0)
    left = _pids_on_port(PORT)
    if left:
        say(f"   ⚠ port {PORT} STILL held by {left} — trying once more")
        _kill(left)
        time.sleep(1.0)
        left = _pids_on_port(PORT)
    say(f"   killed {sorted(killed) or 'nothing'} · port {PORT} now {'FREE' if not left else f'HELD by {left}'}")


# ---------------------------------------------------------------- rebuild (clear caches + reinstall)
def _rmtree(p):
    try:
        shutil.rmtree(p)
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        say(f"   (could not remove {p}: {e})")
        return False


def rebuild(reinstall=True):
    say("2. rebuilding …")
    # a) stale Python bytecode across the repo (skip .git / virtualenvs / node_modules)
    skip = {".git", ".venv", "venv", "node_modules", "build", "dist"}
    n_pyc = 0
    for cache in REPO.rglob("__pycache__"):
        if skip & set(cache.parts):
            continue
        if _rmtree(cache):
            n_pyc += 1
    for pyc in REPO.rglob("*.pyc"):
        if skip & set(pyc.parts):
            continue
        try:
            pyc.unlink()
            n_pyc += 1
        except Exception:
            pass
    say(f"   cleared {n_pyc} bytecode item(s) (__pycache__ / *.pyc)")
    # b) the background-callback diskcache (stale Run-Analysis results)
    say(f"   diskcache {BG_CACHE}: {'wiped' if _rmtree(BG_CACHE) else 'absent'}")
    # c) refresh the editable install so pyproject / entry-point / dep changes take effect
    if reinstall:
        say(f"   pip install -e \".[gui]\"  ({PY}) …")
        r = subprocess.run([PY, "-m", "pip", "install", "-e", ".[gui]"],
                           cwd=str(REPO), capture_output=True, text=True)
        if r.returncode == 0:
            tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-1:] or ["done"]
            say(f"   ✓ reinstall OK — {tail[0]}")
        else:
            say("   ⚠ reinstall FAILED (continuing anyway):")
            for ln in (r.stderr or r.stdout).splitlines()[-6:]:
                say(f"       {ln}")
    else:
        say("   (skipping pip reinstall — --no-reinstall)")


# ---------------------------------------------------------------- restart + health
def _health(timeout=1.0):
    try:
        with urllib.request.urlopen(HEALTH, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def restart(wait=45.0):
    say("3. restarting the back end …")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if VIEWER_LOG.exists():
        try:
            VIEWER_LOG.replace(LOG_DIR / "viewer.log.1")
        except Exception:
            pass
    logf = open(VIEWER_LOG, "w")
    kw = {}
    if IS_WIN:
        kw["creationflags"] = 0x00000008 | 0x00000200   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True                  # detach so the server survives this script
    proc = subprocess.Popen([PY, str(VIEWER)], cwd=str(REPO), stdout=logf,
                            stderr=subprocess.STDOUT, **kw)
    say(f"   launched: {PY} {VIEWER}  (pid {proc.pid}) · log {VIEWER_LOG}")
    end = time.time() + wait
    while time.time() < end:
        h = _health()
        if h is not None:
            pids = sorted(_pids_on_port(PORT))
            say(f"   ✓ up — boot {h.get('boot', '?')[:8]} · root {h.get('root')}")
            say(f"   server pid(s) on {PORT}: {pids}"
                + ("   ⚠ MORE THAN ONE — expected exactly one!" if len(pids) > 1 else ""))
            say(f"\n✅ back end rebuilt & restarted → http://{HOST}:{PORT}/")
            return True
        time.sleep(0.4)
    say(f"   ⚠ no /neitz-health within {wait:.0f}s — check {VIEWER_LOG}")
    return False


def main(argv=None):
    global _QUIET
    ap = argparse.ArgumentParser(description="Force-close, rebuild, and restart the viewer.py back end.")
    ap.add_argument("--no-reinstall", action="store_true", help="skip `pip install -e` (just clear caches)")
    ap.add_argument("--no-restart", action="store_true", help="force-close + rebuild only, don't start a server")
    ap.add_argument("--quiet", action="store_true", help="less output")
    args = ap.parse_args(argv)
    _QUIET = args.quiet

    if not VIEWER.exists():
        print(f"ERROR: {VIEWER} not found (REPO={REPO})", file=sys.stderr)
        return 2
    say(f"# rebuild_backend · repo {REPO} · python {PY}\n")
    force_close()
    rebuild(reinstall=not args.no_reinstall)
    if args.no_restart:
        say("\n✅ force-closed & rebuilt (--no-restart: server NOT started)")
        return 0
    return 0 if restart() else 1


if __name__ == "__main__":
    sys.exit(main())
