#!/usr/bin/env python
"""
neitz_app.py — cross-platform native-window launcher for the Neitz Analysis Suite.

Double-clicked from the macOS .app or the Windows launcher, this:
  1. KILLS then REBUILDS the back end on every launch — frees port 8050 (killing any stale/leftover
     server) and, on **macOS**, runs a full REBUILD (clear stale bytecode + the background-callback
     diskcache + `pip install -e ".[gui]"`, via `neitz.rebuild_backend.rebuild`) before starting a
     fresh `python viewer.py`, so the icon always runs freshly-rebuilt latest code. (Mirrors
     benaqTools' benaq_app.py calling `restart_full.sh`; James's ask. Windows stays a plain fresh
     restart for now.) There is no separate compiled front end — Dash fingerprints `assets/` by
     mtime, so a fresh back end + fresh window load already picks up changed CSS/JS.
  2. opens the viewer in its OWN window (its own dock / taskbar icon and Space), NOT a browser tab:
       - preferred: a real native window via pywebview (WKWebView on macOS, WebView2 on Windows);
         a spinner loading-page shows INSTANTLY while the back end boots in the background, so the
         window + Dock icon appear at once (a slow, windowless launch makes macOS drop the .app
         identity → generic python icon), then swaps to the real UI once the server is healthy;
       - fallback: a chromeless Edge/Chrome `--app` window — used when pywebview's GUI backend is
         unavailable (e.g. Windows-on-ARM without the .NET Desktop Runtime that WinForms needs).
  3. enforces a single app instance (a lock socket), so a second launch doesn't stack a window;
  4. on close (red button / Cmd-Q / signal / exit) tears the back end DOWN — "quit = everything
     down" — sweeping port 8050 so nothing lingers to be reused, keeping every next start fresh.

Everything is logged to ~/.neitz/app.log; on Windows a fatal error also pops a message box (there's
no console when launched from the .vbs / shortcut). To debug a launch, run it with a console:

    python neitz_app.py
"""
from __future__ import annotations
import atexit
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# .absolute() NOT .resolve(): on a mapped network drive (e.g. this repo shared from a Mac VM host as
# Z:\) .resolve() rewrites paths to their UNC form (\\Mac\Home\…), and a UNC path used as a
# subprocess working directory fails on Windows. .absolute() keeps the drive-letter form.
REPO = Path(__file__).absolute().parent
VIEWER = REPO / "viewer.py"
HOST, PORT = "127.0.0.1", 8050
URL = f"http://{HOST}:{PORT}/"
HEALTH = URL + "neitz-health"
LOCK_PORT = 8051                       # a second instance fails to bind this -> it exits quietly
TITLE = "Neitz Analysis Suite"
IS_WIN = sys.platform.startswith("win")

LOG_DIR = Path.home() / ".neitz"       # always LOCAL (C:\Users\<you>\.neitz on Windows)
APP_LOG = LOG_DIR / "app.log"
VIEWER_LOG = LOG_DIR / "viewer.log"

log = logging.getLogger("neitz_app")

# Shown in the native window INSTANTLY while the back end restarts in the background — so the window
# + Dock icon appear at once (a slow windowless launch makes macOS drop the .app identity and the
# Dock tile falls back to the generic python icon). Swapped for the real UI (window.load_url) once
# the back end is healthy.
_LOADING_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
 html,body{height:100%;margin:0;background:#0f1220;color:#e8ebf3;
   font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif;}
 .w{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;}
 .s{width:34px;height:34px;border-radius:50%;border:3px solid #29304a;border-top-color:#5a86ff;
   animation:r .9s linear infinite;}
 @keyframes r{to{transform:rotate(360deg)}}
 .t{font-size:22px;font-weight:600;letter-spacing:.4px;}
 .u{font-size:13px;color:#8b93a7;}
</style></head><body><div class="w"><div class="s"></div>
 <div class="t">Neitz Analysis Suite</div>
 <div class="u">Starting — rebuilding the analysis server…</div>
</div></body></html>"""


def _setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(APP_LOG, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    log.addHandler(logging.StreamHandler())            # visible when run from a console
    log.info("platform=%s executable=%s", sys.platform, sys.executable)
    log.info("REPO=%s", REPO)
    log.info("VIEWER=%s exists=%s", VIEWER, VIEWER.exists())
    log.info("URL=%s  log=%s", URL, APP_LOG)


def _fatal(msg):
    log.exception(msg)
    if IS_WIN:                                          # no console under pythonw/.vbs — show a dialog
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, f"{msg}\n\nDetails in:\n{APP_LOG}\n{VIEWER_LOG}", "Neitz Analysis Suite", 0x10)
        except Exception:
            pass


def health(timeout=1.0):
    """The back end's /neitz-health dict, or None if nothing is answering on the port."""
    try:
        with urllib.request.urlopen(HEALTH, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _pids_on_port(port):
    """PIDs LISTENing on `port` (own-user processes) — used to free the port before a fresh start.
    Uses the OS's own tool so it needs no elevated permissions."""
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
        log.warning("could not list pids on port %s: %s", port, e)
    return pids


def _kill(pids):
    for pid in pids:
        try:
            if IS_WIN:
                subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True)
            else:
                os.kill(pid, signal.SIGKILL)
        except Exception as e:
            log.warning("could not kill pid %s: %s", pid, e)


def _spawn_backend():
    """Start `python viewer.py` (same interpreter as this launcher). Its imports resolve via the
    script's own directory (sys.path[0]), so the working directory does NOT need to be the repo —
    we use a LOCAL cwd so a UNC/network repo path can't break the spawn on Windows."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if VIEWER_LOG.exists():
        try:
            VIEWER_LOG.replace(LOG_DIR / "viewer.log.1")
        except Exception:
            pass
    logf = open(VIEWER_LOG, "w")
    kw = {}
    if IS_WIN:
        kw["creationflags"] = 0x08000000       # CREATE_NO_WINDOW: no console pop-up for the server
    # Stamp OUR pid so the back end can watchdog us and self-exit the instant the GUI quits (Cmd-Q /
    # red-button / crash) — a hard guarantee on top of _teardown that no server outlives the window.
    env = dict(os.environ, NEITZ_APP_PARENT_PID=str(os.getpid()))
    log.info("starting back end: %s %s (cwd=%s, parent_pid=%s)", sys.executable, VIEWER, LOG_DIR,
             os.getpid())
    return subprocess.Popen([sys.executable, str(VIEWER)], cwd=str(LOG_DIR),
                            stdout=logf, stderr=subprocess.STDOUT, env=env, **kw)


def _wait_health(timeout=45.0):
    end = time.time() + timeout
    while time.time() < end:
        if health() is not None:
            return True
        time.sleep(0.4)
    return False


def _rebuild():
    """macOS ONLY (James's ask): a double-click of the dock/desktop icon KILLS then REBUILDS the back
    end — not just restarts it. Mirrors benaqTools' benaq_app.py calling `restart_full.sh`; for our
    Dash app the "rebuild" is pure Python, so we reuse `neitz.rebuild_backend.rebuild()` — the SAME
    steps as the standalone `python -m neitz.rebuild_backend`: clear stale bytecode + wipe the
    background-callback diskcache + `pip install -e ".[gui]"`. We call ONLY that (not its
    `force_close`), because this launcher already freed port 8050 above and must NOT kill its own
    single-instance lock on 8051. Best-effort — a failure (e.g. offline pip) never blocks the launch.
    Windows is intentionally left as a plain fresh-restart for now."""
    if IS_WIN:
        return
    try:
        from neitz import rebuild_backend
        rebuild_backend._QUIET = True                  # its progress -> our log, not double-printed
        log.info("rebuilding the back end (clear caches + pip install -e '.[gui]') …")
        rebuild_backend.rebuild(reinstall=True)
        log.info("rebuild complete")
    except Exception as e:
        log.warning("rebuild step skipped (%s) — starting the existing back end", e)


def ensure_backend():
    """FRESH-START + REBUILD on every launch (James's ask, mirrored from benaq_app.py): free port 8050
    (kill any healthy OR half-dead server), REBUILD the back end (macOS — see `_rebuild`), then spawn
    a new `viewer.py` and wait for it. So a double-click of the icon always runs freshly-rebuilt,
    latest code. Returns the Popen we started (non-None on success) so the caller tears it down on
    close."""
    pids = _pids_on_port(PORT)
    if pids:
        log.info("fresh start: freeing port %s held by %s", PORT, pids)
        _kill(pids)
        time.sleep(0.5)
    _rebuild()                                          # macOS: kill (above) -> REBUILD -> start
    proc = _spawn_backend()
    if _wait_health():
        log.info("back end is up (fresh)")
    else:
        log.warning("back end did not answer within timeout; opening the window anyway "
                    "(check %s)", VIEWER_LOG)
    return proc


def _teardown(proc):
    """Quit = everything down (benaq_app.py's rule): stop the back end we started AND sweep port
    8050, so no half-dead server lingers to be reused — that would defeat the fresh-start policy and
    leave an orphan process. Idempotent (terminating a dead proc / sweeping a free port are no-ops)."""
    if proc is not None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        except Exception as e:
            log.warning("teardown terminate failed: %s", e)
    leftover = _pids_on_port(PORT)
    if leftover:
        log.info("teardown: sweeping port %s (%s)", PORT, leftover)
        _kill(leftover)


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


def _set_win_icon(win):
    """Give the native (WinForms) window OUR taskbar icon, not pythonw's default Python icon."""
    ico = str(REPO / "assets" / "app_icon.ico")

    def _apply():
        try:
            import clr
            clr.AddReference("System.Drawing")
            from System.Drawing import Icon as _Icon
            win.native.Icon = _Icon(ico)
            log.info("set native window icon from %s", ico)
        except Exception as e:
            log.warning("could not set window icon: %s", e)
    try:
        win.events.shown += _apply
    except Exception:
        _apply()


def _set_macos_dock_identity():
    """Force OUR Dock icon (+ menu/Dock name) onto the running NSApplication — the "icon displays
    properly" fix James mirrored from benaq_app.py. The .app launcher `exec`s python, so
    `[NSBundle mainBundle]` resolves to the interpreter and macOS would otherwise show the Dock tile
    as a generic 'Python' icon. Call this BEFORE `webview.create_window` so pywebview reuses the same
    shared NSApplication. Needs pyobjc (pulled by the `[app]` extra on macOS); all best-effort."""
    if IS_WIN:
        return
    try:
        from AppKit import NSApplication, NSImage
        icns = REPO / "assets" / "app_icon.icns"
        img = NSImage.alloc().initWithContentsOfFile_(str(icns)) if icns.exists() else None
        if img is not None:
            NSApplication.sharedApplication().setApplicationIconImage_(img)
            log.info("set macOS Dock icon from %s", icns)
    except Exception as e:
        log.warning("dock icon set skipped: %s", e)
    try:
        from Foundation import NSBundle
        info = NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["CFBundleName"] = TITLE               # menu-bar / Dock label
    except Exception:
        pass


def _try_native_window():
    """Open the pywebview native window (spinner loading-page first) and boot a FRESH back end in the
    background, swapping to the real UI when healthy. Blocks until the window closes; tears the back
    end down on every exit path. Returns True if it actually ran, False if pywebview / its GUI backend
    is unavailable (so the caller can fall back to a chromeless browser window)."""
    try:
        import webview
    except Exception as e:
        log.warning("pywebview not importable (%s)", e)
        return False

    class _Api:
        # exposed as window.pywebview.api.* — lets assets/fullscreen.js toggle NATIVE fullscreen
        # (the browser Fullscreen API is a no-op in WKWebView / WebView2).
        def toggle_fullscreen(self):
            try:
                webview.windows[0].toggle_fullscreen()
            except Exception:
                pass
            return True

    # state shared with the background boot thread + the close/exit hooks
    state = {"proc": None, "closing": False}

    def _cleanup(*_a):
        state["closing"] = True
        _teardown(state.get("proc"))

    atexit.register(_cleanup)                          # interpreter exit (covers Cmd-Q teardown)

    def _on_signal(_sig, _frm):                        # kill / Ctrl-C -> clean up then exit now
        _cleanup()
        os._exit(0)
    for _s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_s, _on_signal)
        except Exception:
            pass

    try:
        _set_macos_dock_identity()                    # our Dock icon BEFORE the window (macOS no-op elsewhere)
        win = webview.create_window(TITLE, html=_LOADING_HTML, width=1440, height=900,
                                    min_size=(940, 620), js_api=_Api())
        try:
            win.events.closed += _cleanup             # red-button close OR Cmd-Q teardown
        except Exception:
            pass
        if IS_WIN:
            _set_win_icon(win)

        def _boot():
            # background thread (once the GUI is up): the kill + REBUILD + fresh start happens HERE, so
            # the slow rebuild never delays the window/icon appearing. Swap to the real UI when up.
            state["proc"] = ensure_backend()
            if state["closing"]:                      # window closed while we were starting
                _teardown(state.get("proc"))
                return
            try:
                win.load_url(URL)
            except Exception as e:
                log.warning("load_url failed: %s", e)

        log.info("webview.start() — native window open (loading page); back end booting in background")
        webview.start(_boot)                          # blocks on the main thread until the window closes
        log.info("webview.start() returned — window closed")
        _cleanup()                                    # belt-and-suspenders when start() returns
        return True
    except Exception as e:
        log.warning("pywebview backend failed (%s) — falling back to a browser app window", e)
        _teardown(state.get("proc"))
        return False


def _browser_exes():
    """Edge/Chrome executables that can host a chromeless --app window (no .NET/pywebview needed)."""
    if IS_WIN:
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pfx = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        cands = [pf + r"\Microsoft\Edge\Application\msedge.exe",
                 pfx + r"\Microsoft\Edge\Application\msedge.exe",
                 pf + r"\Google\Chrome\Application\chrome.exe",
                 pfx + r"\Google\Chrome\Application\chrome.exe"]
    else:
        cands = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                 "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    return [c for c in cands if os.path.exists(c)]


def _app_mode_window():
    """A chromeless standalone window via Edge/Chrome --app (own window + Space; its favicon is our
    icon). Returns the browser Popen, or None if no browser was found."""
    profile = str(LOG_DIR / "app-profile")
    for exe in _browser_exes():
        try:
            p = subprocess.Popen([exe, f"--app={URL}", f"--user-data-dir={profile}",
                                  "--no-first-run", "--no-default-browser-check"])
            log.info("opened app-mode window via %s", exe)
            return p
        except Exception as e:
            log.warning("could not launch %s: %s", exe, e)
    return None


def _run():
    lock = acquire_single_instance()
    if lock is None:
        log.info("another instance already running (lock %s held) — exiting", LOCK_PORT)
        return

    if IS_WIN:
        try:                                   # stable identity so Windows gives it its own taskbar icon
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("com.neitz.analysissuite")
        except Exception as e:
            log.warning("could not set AppUserModelID: %s", e)

    if _try_native_window():                   # best path: native window (loading page + fresh boot)
        _ = lock
        return

    # pywebview's GUI backend is unavailable (e.g. Windows-on-ARM without the .NET Desktop Runtime)
    # -> bring a FRESH back end up synchronously (the browser window needs it now), then open a
    # chromeless Edge/Chrome --app window and block on it. Tear the back end down when it closes.
    proc = ensure_backend()
    browser = _app_mode_window()
    if browser is None:
        log.warning("no Edge/Chrome found for an --app window; opening a normal browser tab")
        import webbrowser
        webbrowser.open(URL)
    else:
        try:
            browser.wait()                     # returns when the --app window is closed
        except Exception:
            pass
        _teardown(proc)                        # quit = everything down (the window owned this server)
    _ = lock


def main():
    _setup_logging()
    try:
        _run()
    except Exception:
        _fatal("Neitz Analysis Suite failed to start")
        raise


if __name__ == "__main__":
    main()
