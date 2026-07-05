#!/usr/bin/env python
r"""
Build the Windows 11 launcher for the Neitz Analysis Suite — a native window (NOT a browser tab).

Run this ON Windows, from the repo root, with the same Python you use for the viewer:

    python scripts\build_windows_app.py

It (1) renders assets\app_icon.ico from the SVG, (2) writes a silent double-clickable launcher
`NeitzAnalysisSuite.vbs` in the repo (runs `pythonw neitz_app.py` with no console window), and
(3) creates a Desktop shortcut carrying our icon. neitz_app.py opens the viewer in a WebView2
(Edge/Chromium) window with its own taskbar icon + a stable AppUserModelID, and reuses a single
instance. Pin the Desktop shortcut (or the running window) to the taskbar to keep it there.

Prereqs on Windows: `pip install -e ".[gui,app]"` (pulls pywebview + pythonnet); the Edge WebView2
runtime ships with Windows 11. PowerShell (used to create the shortcut) is always present.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APP_LABEL = "Neitz Analysis Suite"


def main() -> None:
    if not sys.platform.startswith("win"):
        print(f"NOTE: this builds the *Windows* launcher; you're on {sys.platform}. It will still "
              f"render assets/app_icon.ico, but run it on Windows 11 to create the shortcut.")

    # 1) icon — prefer the committed assets/app_icon.ico so Windows needs NO cairosvg/Cairo (painful
    #    to install there); only render from the SVG if the .ico is somehow missing.
    ico = REPO / "assets" / "app_icon.ico"
    if not ico.exists():
        print("  app_icon.ico missing — rendering it from the SVG (needs cairosvg)…")
        subprocess.run([sys.executable, str(REPO / "scripts" / "build_icon.py")], check=True)
    if not ico.exists():
        raise SystemExit("assets/app_icon.ico not found and could not be generated.")
    try:                                                    # guard: a git line-ending mangle corrupts it
        from PIL import Image
        Image.open(ico).verify()
        print(f"  icon OK: {ico}")
    except Exception as e:
        print(f"  WARNING: {ico} is not a readable icon ({e}).\n"
              f"  If you pulled it from git on Windows, autocrlf may have corrupted it. With the\n"
              f"  committed .gitattributes this won't recur; restore a clean copy with:\n"
              f"    git add --renormalize . && git checkout -- assets/app_icon.ico\n"
              f"  then re-run this script.")

    # the windowless interpreter next to this one (python.exe -> pythonw.exe)
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    pyw = str(pythonw if pythonw.exists() else sys.executable)

    # 2) a silent double-clickable launcher in the repo (no console flash)
    vbs = REPO / "NeitzAnalysisSuite.vbs"
    vbs.write_text(
        'Set W = CreateObject("WScript.Shell")\r\n'
        f'W.CurrentDirectory = "{REPO}"\r\n'
        f'W.Run """{pyw}"" ""{REPO / "neitz_app.py"}""", 0, False\r\n',
        encoding="utf-8",
    )
    print(f"  wrote {vbs}")

    if not sys.platform.startswith("win"):
        print("  (skipped Desktop shortcut — not on Windows)")
        return

    # 3) Desktop shortcut with our icon, via PowerShell + WScript.Shell (no extra pip deps).
    #    IconLocation MUST carry the icon index (",0") — WScript.Shell shows a blank page icon
    #    without it. Also print whether .NET can load the .ico, so a bad icon is obvious here.
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    lnk = desktop / f"{APP_LABEL}.lnk"
    ps = (
        "Add-Type -AssemblyName System.Drawing; "
        f"try {{ $i=[System.Drawing.Icon]::new('{ico}'); Write-Output \"icon loads: $($i.Width)x$($i.Height)\" }} "
        "catch { Write-Output \"ICON INVALID: $($_.Exception.Message)\" }; "
        "$W = New-Object -ComObject WScript.Shell; "
        f"$s = $W.CreateShortcut('{lnk}'); "
        f"$s.TargetPath = '{pyw}'; "
        f"$s.Arguments = '\"{REPO / 'neitz_app.py'}\"'; "
        f"$s.WorkingDirectory = '{REPO}'; "
        f"$s.IconLocation = '{ico},0'; "
        f"$s.Description = '{APP_LABEL}'; "
        "$s.Save(); "
        "Write-Output ('IconLocation: ' + $s.IconLocation)"
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=True)
    subprocess.run(["ie4uinit.exe", "-show"], capture_output=True)   # nudge Explorer's icon cache
    print(f"  wrote {lnk}")
    print("done — double-click the Desktop shortcut (or NeitzAnalysisSuite.vbs), then right-click "
          "its taskbar icon > Pin to taskbar.")


if __name__ == "__main__":
    main()
