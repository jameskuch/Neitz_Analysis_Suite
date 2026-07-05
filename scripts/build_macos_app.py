#!/usr/bin/env python
"""
Build the double-clickable macOS app for the Neitz Analysis Suite viewer.

    /Users/j/miniconda3/bin/python scripts/build_macos_app.py

Pipeline: assets/app_icon.svg  --cairosvg-->  1024² master  -->  .iconset  --iconutil-->
assets/app_icon.icns  -->  NeitzAnalysisSuite.app (a Finder/Dock launcher that starts
viewer.py if it isn't already running, then opens the browser to the Dash GUI).

Everything here is idempotent — re-run it after editing the SVG to refresh the icon + app.
"""
from __future__ import annotations
import io
import shutil
import subprocess
from pathlib import Path

import cairosvg
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
PYTHON = "/Users/j/miniconda3/bin/python"          # the conda interpreter that runs the GUI
APP_NAME = "NeitzAnalysisSuite"                     # bundle executable / .app basename
DISPLAY = "Neitz Analysis Suite"
PORT = 8050

# icon layout: render the squircle at 1672 px, inset into a 2048 canvas (≈9% margin, macOS-style)
ICON_PX, CANVAS_PX = 1672, 2048
MARGIN = (CANVAS_PX - ICON_PX) // 2

INFO_PLIST = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>{DISPLAY}</string>
  <key>CFBundleDisplayName</key><string>{DISPLAY}</string>
  <key>CFBundleIdentifier</key><string>edu.uw.neitz.analysissuite</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleSignature</key><string>????</string>
  <key>CFBundleExecutable</key><string>{APP_NAME}</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>LSMinimumSystemVersion</key><string>10.13</string>
  <key>LSUIElement</key><false/>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
"""

LAUNCHER = f"""#!/bin/bash
# Neitz Analysis Suite launcher: ensure the Dash backend is up, then open the browser.
PY="{PYTHON}"
APP_DIR="{REPO}"
LOG="/tmp/neitz_viewer.log"
URL="http://127.0.0.1:{PORT}/"

cd "$APP_DIR" || {{ /usr/bin/osascript -e 'display alert "Neitz Analysis Suite" message "Repo not found: {REPO}"'; exit 1; }}

# start the backend only if nothing is already listening on the port
if ! /usr/sbin/lsof -ti "tcp:{PORT}" >/dev/null 2>&1; then
  /usr/bin/nohup "$PY" viewer.py > "$LOG" 2>&1 &
fi

# wait (up to ~20 s) for the server to answer, then open it
up=""
for i in $(seq 1 40); do
  if /usr/bin/curl -s -o /dev/null "$URL"; then up=1; break; fi
  sleep 0.5
done
if [ -z "$up" ]; then
  /usr/bin/osascript -e 'display alert "Neitz Analysis Suite" message "The viewer backend did not start within 20 s. See '"$LOG"' for details."'
  exit 1
fi
/usr/bin/open "$URL"
"""


def render_master() -> Image.Image:
    png = cairosvg.svg2png(url=str(REPO / "assets/app_icon.svg"),
                           output_width=ICON_PX, output_height=ICON_PX)
    icon = Image.open(io.BytesIO(png)).convert("RGBA")
    master = Image.new("RGBA", (CANVAS_PX, CANVAS_PX), (0, 0, 0, 0))
    master.paste(icon, (MARGIN, MARGIN), icon)
    return master


def build_iconset(master: Image.Image, iconset: Path) -> None:
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)
    # (px, iconutil name) — the @2x entries are the retina doubles
    for px, name in [(16, "16x16"), (32, "16x16@2x"), (32, "32x32"), (64, "32x32@2x"),
                     (128, "128x128"), (256, "128x128@2x"), (256, "256x256"),
                     (512, "256x256@2x"), (512, "512x512"), (1024, "512x512@2x")]:
        master.resize((px, px), Image.LANCZOS).save(iconset / f"icon_{name}.png")


def main() -> None:
    build = REPO / "build"
    build.mkdir(exist_ok=True)
    master = render_master()
    master.resize((1024, 1024), Image.LANCZOS).save(build / "icon_master_1024.png")

    iconset = build / "icon.iconset"
    build_iconset(master, iconset)
    icns = REPO / "assets/app_icon.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)

    app = REPO / f"{APP_NAME}.app"
    if app.exists():
        shutil.rmtree(app)
    (app / "Contents/MacOS").mkdir(parents=True)
    (app / "Contents/Resources").mkdir(parents=True)
    shutil.copy2(icns, app / "Contents/Resources/icon.icns")
    (app / "Contents/Info.plist").write_text(INFO_PLIST)
    (app / "Contents/PkgInfo").write_text("APPL????")
    launcher = app / "Contents/MacOS" / APP_NAME
    launcher.write_text(LAUNCHER)
    launcher.chmod(0o755)
    subprocess.run(["touch", str(app)])                       # nudge Finder to pick up the icon

    print(f"icns : {icns}")
    print(f"app  : {app}")
    print("done — double-click the .app or drag it to /Applications and the Dock.")


if __name__ == "__main__":
    main()
