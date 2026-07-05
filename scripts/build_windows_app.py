#!/usr/bin/env python
r"""
Build the Windows 11 launcher for the Neitz Analysis Suite — a native window (NOT a browser tab).

Run this ON Windows, from the repo, with the same Python you use for the viewer:

    python scripts\build_windows_app.py

It (1) ensures assets\app_icon.ico exists, (2) copies the icon into a LOCAL folder
(%LOCALAPPDATA%\NeitzAnalysisSuite) and uses that folder as the launcher's working directory —
Windows won't render a shortcut icon that lives on a network share, and a UNC/network working
directory breaks process launch — and (3) creates a Desktop shortcut that runs `pythonw
neitz_app.py` (no console) with our icon. neitz_app.py opens the viewer in a WebView2 window,
single-instance, with its own taskbar icon (AppUserModelID).

Prereqs on Windows: `pip install -e ".[gui,app]"` (dash, plotly, pywebview, pythonnet, …); the Edge
WebView2 runtime ships with Windows 11. PowerShell (used for the shortcut) is always present.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

# .absolute() NOT .resolve(): on a mapped network drive (e.g. this repo shared from a Mac VM host as
# Z:\) .resolve() rewrites paths to their UNC form (\\Mac\Home\…), and UNC breaks as a working
# directory on Windows. .absolute() keeps the drive-letter form the user actually launched with.
REPO = Path(__file__).absolute().parent.parent
APP_LABEL = "Neitz Analysis Suite"


def main() -> None:
    win = sys.platform.startswith("win")
    if not win:
        print(f"NOTE: this builds the *Windows* launcher; you're on {sys.platform}. It still renders "
              f"assets/app_icon.ico, but run it on Windows 11 to create the shortcut.")

    # 1) icon — prefer the committed assets/app_icon.ico (Windows then needs no cairosvg/Cairo).
    ico = REPO / "assets" / "app_icon.ico"
    if not ico.exists():
        print("  app_icon.ico missing — rendering it from the SVG (needs cairosvg)…")
        subprocess.run([sys.executable, str(REPO / "scripts" / "build_icon.py")], check=True)
    if not ico.exists():
        raise SystemExit("assets/app_icon.ico not found and could not be generated.")
    try:                                                    # PIL is optional; PowerShell verifies too
        from PIL import Image
        Image.open(ico).verify()
        print(f"  icon OK: {ico}")
    except ImportError:
        pass
    except Exception as e:
        print(f"  WARNING: {ico} may be corrupt ({e}); restore a clean copy:\n"
              f"    git add --renormalize . && git checkout -- assets/app_icon.ico")

    # 2) a LOCAL working folder (never a network path): holds the icon copy and is the launcher's
    #    working directory, so Explorer can render the icon and Windows can launch with a valid cwd.
    local = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "NeitzAnalysisSuite"
    local.mkdir(parents=True, exist_ok=True)
    local_ico = local / "app_icon.ico"
    try:
        shutil.copy2(ico, local_ico)
    except Exception as e:
        print(f"  (could not copy icon to {local_ico}: {e}; using {ico})")
        local_ico = ico

    # the windowless interpreter next to this one (python.exe -> pythonw.exe)
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    pyw = str(pythonw if pythonw.exists() else sys.executable)
    app = REPO / "neitz_app.py"

    # 3) a silent double-clickable launcher (no console flash); cwd LOCAL, script path absolute
    vbs = REPO / "NeitzAnalysisSuite.vbs"
    try:
        vbs.write_text(
            'Set W = CreateObject("WScript.Shell")\r\n'
            f'W.CurrentDirectory = "{local}"\r\n'
            f'W.Run """{pyw}"" ""{app}""", 0, False\r\n',
            encoding="utf-8",
        )
        print(f"  wrote {vbs}")
    except Exception as e:
        print(f"  (could not write {vbs}: {e})")

    if not win:
        print("  (skipped Desktop shortcut — not on Windows)")
        return

    # 4) Desktop shortcut via PowerShell + WScript.Shell. IconLocation needs the ",0" index and a
    #    LOCAL icon path; WorkingDirectory must be LOCAL too (a UNC cwd fails to launch).
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    lnk = desktop / f"{APP_LABEL}.lnk"
    ps = (
        "Add-Type -AssemblyName System.Drawing; "
        f"try {{ $i=[System.Drawing.Icon]::new('{local_ico}'); Write-Output \"icon loads: $($i.Width)x$($i.Height)\" }} "
        "catch { Write-Output \"ICON INVALID: $($_.Exception.Message)\" }; "
        "$W = New-Object -ComObject WScript.Shell; "
        f"$s = $W.CreateShortcut('{lnk}'); "
        f"$s.TargetPath = '{pyw}'; "
        f"$s.Arguments = '\"{app}\"'; "
        f"$s.WorkingDirectory = '{local}'; "
        f"$s.IconLocation = '{local_ico},0'; "
        f"$s.Description = '{APP_LABEL}'; "
        "$s.Save(); "
        "Write-Output ('IconLocation: ' + $s.IconLocation)"
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=True)
    subprocess.run(["ie4uinit.exe", "-show"], capture_output=True)   # nudge Explorer's icon cache
    print(f"  wrote {lnk}")
    print("done — double-click the Desktop shortcut. If the icon doesn't refresh, log out/in or "
          "restart Explorer. To debug a failed launch, run visibly:  python neitz_app.py")


if __name__ == "__main__":
    main()
