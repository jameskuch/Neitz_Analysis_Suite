#!/usr/bin/env python
"""
neitz_app.py — cross-platform native-window launcher for the Neitz Analysis Suite.

Double-clicked from the macOS .app or the Windows launcher, this:
  1. reuses a healthy back end already serving on 127.0.0.1:8050, or starts `python viewer.py`
     itself (freeing the port first if something half-dead is holding it) and waits for it;
  2. opens the viewer in its OWN window (its own dock / taskbar icon and Space), NOT a browser tab:
       - preferred: a real native window via pywebview (WKWebView on macOS, WebView2 on Windows);
       - fallback: a chromeless Edge/Chrome `--app` window — used when pywebview's GUI backend is
         unavailable (e.g. Windows-on-ARM without the .NET Desktop Runtime that WinForms needs).
         Our icon still shows there because it's the Dash favicon (assets/favicon.ico).
  3. enforces a single app instance (a lock socket), so a second launch doesn't stack a window;
  4. on window close, stops the back end it started (a reused / manually-run server is left alone).

The front end auto-reloads when the back end restarts (assets/autoreload.js + /neitz-health).

Everything is logged to ~/.neitz/app.log; on Windows a fatal error also pops a message box (there's
no console when launched from the .vbs / shortcut). To debug a launch, run it with a console:

    python neitz_app.py
"""
from __future__ import annotations
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
    log.info("starting back end: %s %s (cwd=%s)", sys.executable, VIEWER, LOG_DIR)
    return subprocess.Popen([sys.executable, str(VIEWER)], cwd=str(LOG_DIR),
                            stdout=logf, stderr=subprocess.STDOUT, **kw)


def _wait_health(timeout=45.0):
    end = time.time() + timeout
    while time.time() < end:
        if health() is not None:
            return True
        time.sleep(0.4)
    return False


def _norm(p):
    return os.path.normcase(os.path.normpath(p)) if p else ""


def ensure_backend():
    """Reuse a healthy server if it's serving the SAME data root we want, else (re)start one so a
    changed EPHYSDATAIO_ROOT actually takes effect. Returns the Popen we started, or None if we
    reused an existing server (which we then leave running on exit)."""
    h = health()
    if h is not None:
        want = os.environ.get("EPHYSDATAIO_ROOT")           # what THIS launch was told to use
        have = h.get("root")                                # what the running server is using
        if want and have and _norm(want) != _norm(have):
            log.info("running back end uses data root %r but %r requested — restarting it",
                     have, want)
            _kill(_pids_on_port(PORT))
            time.sleep(0.5)
        else:
            log.info("reusing back end already serving on %s (root=%s)", URL, have)
            return None                        # single-instance reuse of a matching back end
    else:
        pids = _pids_on_port(PORT)             # port held but not healthy -> free it, start fresh
        if pids:
            log.info("freeing stale port %s held by %s", PORT, pids)
            _kill(pids)
            time.sleep(0.5)
    proc = _spawn_backend()
    if _wait_health():
        log.info("back end is up")
    else:
        log.warning("back end did not answer within timeout; opening the window anyway "
                    "(check %s)", VIEWER_LOG)
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


def _try_native_window():
    """Open the pywebview native window and block until it closes. Returns True if it actually ran,
    False if pywebview or its GUI backend is unavailable (so the caller can fall back). On Windows,
    pywebview needs the .NET Desktop Runtime (System.Windows.Forms); without it start() raises and
    we fall back to a chromeless browser window instead of dying."""
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

    try:
        win = webview.create_window(TITLE, URL, width=1440, height=900, min_size=(940, 620),
                                    js_api=_Api())
        if IS_WIN:                             # give the native (WinForms) window OUR taskbar icon,
            ico = str(REPO / "assets" / "app_icon.ico")   # not pythonw's default Python icon

            def _set_icon():
                try:
                    import clr
                    clr.AddReference("System.Drawing")
                    from System.Drawing import Icon as _Icon
                    win.native.Icon = _Icon(ico)
                    log.info("set native window icon from %s", ico)
                except Exception as e:
                    log.warning("could not set window icon: %s", e)
            try:
                win.events.shown += _set_icon
            except Exception:
                _set_icon()
        log.info("webview.start() — native window open")
        webview.start()                        # blocks on the main thread until the window closes
        log.info("webview.start() returned — window closed")
        return True
    except Exception as e:
        log.warning("pywebview backend failed (%s) — falling back to a browser app window", e)
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

    proc = ensure_backend()

    if _try_native_window():                   # best path: real native window (own dock/taskbar icon)
        if proc is not None:                   # window closed -> stop the back end WE started
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        _ = lock
        return

    # pywebview's GUI backend is unavailable (e.g. Windows-on-ARM without the .NET Desktop Runtime)
    # -> chromeless Edge/Chrome --app window. Block on it so we hold the single-instance lock while
    # it's open; leave the back end running (the window owns it now).
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
