#!/usr/bin/env python
"""
Build the macOS app for the Neitz Analysis Suite — a native window (NOT a browser tab).

    /Users/j/miniconda3/bin/python scripts/build_macos_app.py

The bundle's launcher runs `neitz_app.py` in the FOREGROUND (via conda base), so the .app is the
responsible, dock-resident process: it opens the viewer in a pywebview/WKWebView window with its
own dock icon + macOS Space, and Cmd-Q stops it cleanly. Icon comes from assets/app_icon.svg via
scripts/build_icon.py. Idempotent — re-run after editing the SVG, viewer.py, or neitz_app.py.
"""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = "/Users/j/miniconda3/bin/python"          # conda interpreter (fallback if conda.sh absent)
APP_NAME = "NeitzAnalysisSuite"
DISPLAY = "Neitz Analysis Suite"

INFO_PLIST = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>{DISPLAY}</string>
  <key>CFBundleDisplayName</key><string>{DISPLAY}</string>
  <key>CFBundleIdentifier</key><string>com.neitz.analysissuite</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleSignature</key><string>????</string>
  <key>CFBundleExecutable</key><string>{APP_NAME}</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
"""

# Foreground launcher: the .app process becomes the pywebview window (its own dock tile / Space).
LAUNCHER = f"""#!/bin/bash
# Neitz Analysis Suite — open the native window (pywebview / WKWebView). Foreground so THIS .app is
# the responsible process: it holds the dock tile and Cmd-Q stops it cleanly.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/miniconda3/bin:$PATH"
APP_DIR="{REPO}"
cd "$APP_DIR" || {{ /usr/bin/osascript -e 'display alert "Neitz Analysis Suite" message "Repo not found: {REPO}"'; exit 1; }}
LOG_DIR="$HOME/.neitz"; mkdir -p "$LOG_DIR"
[ -f "$LOG_DIR/app.log" ] && mv -f "$LOG_DIR/app.log" "$LOG_DIR/app.log.1" 2>/dev/null
# Activate conda base so viewer.py's background-callback workers resolve the same deps.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate base; PY="python"
else
  PY="{PYTHON}"
fi
exec "$PY" neitz_app.py >"$LOG_DIR/app.log" 2>&1
"""


def main() -> None:
    # 1) refresh the icon from the SVG; fall back to the committed .icns if cairosvg is unavailable
    try:
        subprocess.run([sys.executable, str(REPO / "scripts/build_icon.py")], check=True)
    except Exception as e:
        print(f"  (icon regen skipped: {e}; using committed assets/app_icon.icns)")
    icns = REPO / "assets/app_icon.icns"
    if not icns.exists():
        raise SystemExit("assets/app_icon.icns not found (regen failed and none committed).")

    # 2) (re)assemble the bundle
    app = REPO / f"{APP_NAME}.app"
    if app.exists():
        shutil.rmtree(app)
    (app / "Contents/MacOS").mkdir(parents=True)
    (app / "Contents/Resources").mkdir(parents=True)
    shutil.copy2(icns, app / "Contents/Resources/AppIcon.icns")
    (app / "Contents/Info.plist").write_text(INFO_PLIST)
    (app / "Contents/PkgInfo").write_text("APPL????")
    launcher = app / "Contents/MacOS" / APP_NAME
    launcher.write_text(LAUNCHER)
    launcher.chmod(0o755)

    # 3) stable code identity: clear any quarantine, then ad-hoc sign (so macOS remembers the app
    #    across launches and consistently shows our icon). Non-fatal if the tools are unavailable.
    subprocess.run(["xattr", "-cr", str(app)], check=False)
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=False)
    subprocess.run(["touch", str(app)], check=False)

    print(f"app  : {app}")
    print("done — double-click it, or drag it to /Applications and the Dock.")


if __name__ == "__main__":
    main()
