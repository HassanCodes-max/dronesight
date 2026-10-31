"""Build the free static site. Everything runs in the visitor's browser, so there is no server.

    python deploy/fetch_ort.py
    python deploy/build_static.py dist

The models and their runtime are published with the page. Static hosts cannot send headers, so the
page carries its Content Security Policy in a meta tag.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STATIC = ROOT / "dronesight/app/static"
PAGE_FILES = ["rescue.js", "local.js", "tracker.js", "hazards.js", "risk.js", "style.css", "rescue.css", "favicon.svg"]
BROWSER_MODELS = [ROOT / "models" / name for name in ("person.onnx", "water.onnx", "fire.onnx")]
# Hugging Face redirects large files to its CDN, so they need that one extra source. Blobs are a
# visitor's own video, and the runtime needs to compile WebAssembly. Nothing here can send it anywhere else.
CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self' data: https://*.hf.co; "
       "media-src 'self' blob: https://*.hf.co; "
       "connect-src 'self' https://*.hf.co; object-src 'none'; base-uri 'none'; form-action 'none'")
CARD = """---
title: DroneSight
emoji: 🛰️
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
short_description: See who needs help first in a drone video
---

Add a drone video and see who needs help first, on your own device.
"""


def main(out):
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for name in PAGE_FILES:
        shutil.copy(STATIC / name, out / name)
    for model in BROWSER_MODELS:
        shutil.copy(model, out / model.name)
    shutil.copytree(ROOT / "vendor/ort", out / "ort")
    meta = f'  <meta http-equiv="Content-Security-Policy" content="{CSP}">\n'
    html = (STATIC / "rescue.html").read_text(encoding="utf-8")
    html = html.replace(' data-edition="server"', "")
    html = html.replace('  <meta name="viewport"', meta + '  <meta name="viewport"', 1)
    (out / "index.html").write_text(html, encoding="utf-8", newline="\n")
    (out / "README.md").write_text(CARD, encoding="utf-8", newline="\n")
    print(f"{sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(*sys.argv[1:])
