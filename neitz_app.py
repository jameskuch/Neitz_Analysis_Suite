#!/usr/bin/env python
"""
neitz_app.py — cross-platform native-window launcher for the Neitz Analysis Suite.

Double-clicked from the macOS .app (WKWebView) or the Windows launcher (WebView2 / Edge), this:
  1. reuses a healthy back end already serving on 127.0.0.1:8050, or starts `python viewer.py`
     itself (freeing the port first if something half-dead is holding it) and waits for it;
  2. opens the viewer in a real native window — its own dock / taskbar icon and macOS Space,
     NOT a browser tab;
  3. enforces a single app instance (a lock socket), so a second double-click doesn't stack a
     duplicate window on the same back end;
  4. on window close, stops the back end it started (a reused / manually-run server is left alone).

The front end auto-reloads when the back end restarts (see assets/autoreload.js + the /neitz-health
route in viewer.py), so editing viewer.py and relaunching shows current code with no manual refresh.

Runs the same on macOS and Windows 11 — the OS-specific bits (kill-by-port, no-console spawn,
taskbar identity) branch on sys.platform. Run directly for a native window without the bundle:

    python neitz_app.py
"""
from __future__ import annotations
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
VIEWER = REPO / "viewer.py"
HOST, PORT = "127.0.0.1", 8050
URL = f"http://{HOST}:{PORT}/"
HEALTH = URL + "neitz-health"
LOCK_PORT = 8051                       # a second instance fails to bind this -> it exits quietly
TITLE = "Neitz Analysis Suite"
IS_WIN = sys.platform.startswith("win")

LOG_DIR = Path.home() / ".neitz"
LOG = LOG_DIR / "viewer.log"


def health(timeout=1.0):
    """The back end's /neitz-health dict, or None if nothing is answering on the port."""
    try:
        with urllib.request.urlopen(HEALTH, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _pids_on_port(port):
    """PIDs LISTENing on `port` (own-user processes) — used to free a half-dead port before a
    fresh start. Uses the OS's own tool so it needs no elevated permissions."""
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
    except Exception:
        pass
    return pids


def _kill(pids):
    for pid in pids:
        try:
            if IS_WIN:
                subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True)
            else:
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def _spawn_backend():
    """Start `python viewer.py` (same interpreter as this launcher) with output to a rotated log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        try:
            LOG.replace(LOG.with_name("viewer.log.1"))
        except Exception:
            pass
    logf = open(LOG, "w")
    kw = {}
    if IS_WIN:
        kw["creationflags"] = 0x08000000       # CREATE_NO_WINDOW: no console pop-up for the server
    return subprocess.Popen([sys.executable, str(VIEWER)], cwd=str(REPO),
                            stdout=logf, stderr=subprocess.STDOUT, **kw)


def _wait_health(timeout=45.0):
    end = time.time() + timeout
    while time.time() < end:
        if health() is not None:
            return True
        time.sleep(0.4)
    return False


def ensure_backend():
    """Reuse a healthy server if one is already serving, else (re)start one. Returns the Popen we
    started, or None if we reused an existing server (which we then leave running on exit)."""
    if health() is not None:
        return None                            # single-instance reuse of a running back end
    pids = _pids_on_port(PORT)                 # port held but not healthy -> free it, start fresh
    if pids:
        _kill(pids)
        time.sleep(0.5)
    proc = _spawn_backend()
    _wait_health()                             # open the window regardless; it surfaces any errors
    return proc


def acquire_single_instance():
    """Bind a lock port so a second launch exits instead of stacking a duplicate window. Returns
    the bound socket (keep it referenced for the process lifetime) or None if already held."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((HOST, LOCK_PORT))
        s.listen(1)
        return s
    except OSError:
        return None


def main():
    lock = acquire_single_instance()
    if lock is None:
        return                                 # another window is already open — reuse it, exit

    if IS_WIN:
        try:                                   # stable identity so Windows gives it its own taskbar icon
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("com.neitz.analysissuite")
        except Exception:
            pass

    proc = ensure_backend()

    try:
        import webview
    except Exception:                          # pywebview missing -> degrade to a browser tab, leave
        import webbrowser                      # the back end running so the tab keeps working
        webbrowser.open(URL)
        return

    webview.create_window(TITLE, URL, width=1440, height=900, min_size=(940, 620))
    try:
        webview.start()                        # blocks on the main thread until the window closes
    finally:
        if proc is not None:                   # stop only the back end WE started (leave a reused one)
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
            except Exception:
                pass
    _ = lock                                    # keep the lock socket alive until here


if __name__ == "__main__":
    main()
