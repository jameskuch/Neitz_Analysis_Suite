#!/usr/bin/env python
"""
Render assets/app_icon.svg into the platform icon files:

    assets/app_icon.icns   (macOS, 16->1024 incl. @2x; needs `iconutil`, so built only on macOS)
    assets/app_icon.ico    (Windows, multi-resolution 16->256)

Both come from one SVG source, so the macOS app and the Windows app show the same icon. Re-run
after editing the SVG.  Usage:  python scripts/build_icon.py
"""
from __future__ import annotations
import io
import shutil
import subprocess
import sys
from pathlib import Path

import cairosvg
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SVG = REPO / "assets/app_icon.svg"
ASSETS = REPO / "assets"
BUILD = REPO / "build"
# render the squircle at 1672 px and inset it in a 2048 canvas (~9% margin, macOS-style float)
ICON_PX, CANVAS_PX = 1672, 2048
MARGIN = (CANVAS_PX - ICON_PX) // 2


def render_master() -> Image.Image:
    png = cairosvg.svg2png(url=str(SVG), output_width=ICON_PX, output_height=ICON_PX)
    icon = Image.open(io.BytesIO(png)).convert("RGBA")
    master = Image.new("RGBA", (CANVAS_PX, CANVAS_PX), (0, 0, 0, 0))
    master.paste(icon, (MARGIN, MARGIN), icon)
    return master


def build_icns(master: Image.Image) -> Path:
    iconset = BUILD / "icon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)
    for px, name in [(16, "16x16"), (32, "16x16@2x"), (32, "32x32"), (64, "32x32@2x"),
                     (128, "128x128"), (256, "128x128@2x"), (256, "256x256"),
                     (512, "256x256@2x"), (512, "512x512"), (1024, "512x512@2x")]:
        master.resize((px, px), Image.LANCZOS).save(iconset / f"icon_{name}.png")
    icns = ASSETS / "app_icon.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)
    return icns


def build_ico(master: Image.Image) -> Path:
    ico = ASSETS / "app_icon.ico"
    base = master.resize((256, 256), Image.LANCZOS)          # crisp source; Pillow embeds each size
    base.save(ico, format="ICO",
              sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return ico


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    master = render_master()
    master.resize((1024, 1024), Image.LANCZOS).save(BUILD / "icon_master_1024.png")
    written = []
    if sys.platform == "darwin":
        try:
            written.append(build_icns(master))
        except Exception as e:                               # iconutil missing -> still emit the .ico
            print(f"  (skipped .icns: {e})")
    written.append(build_ico(master))
    for p in written:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
