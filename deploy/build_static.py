"""Build the free static edition: the sample flight and its detections, with no server.

    python -m dronesight.app.seed demo/flight.mp4 "Sample flight"
    python deploy/fetch_ort.py
    python deploy/build_static.py dist [assistant-url]

It reads the seeded database, so the detections are the worker's real output. Visitors' own videos
are analysed in their browser, so the model and its runtime are published too. Static hosts cannot
send headers, so the page carries its Content Security Policy in a meta tag.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deploy.build_worker import topics  # noqa: E402
from dronesight.app import db  # noqa: E402

STATIC = ROOT / "dronesight/app/static"
BROWSER_MODEL = ROOT / "models/person_960.onnx"
# Hugging Face redirects large files to its CDN, so they need that one extra source. Blobs are a
# visitor's own video, and the runtime needs to compile WebAssembly.
CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self' data: https://*.hf.co; "
       "media-src 'self' blob: https://*.hf.co; "
       "connect-src 'self' https://*.hf.co {assistant}; object-src 'none'; base-uri 'none'; form-action 'none'")
CARD = """---
title: DroneSight
emoji: 🛰️
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
short_description: Find people in drone footage without watching all of it
---

Ask a question about drone footage and jump straight to the moment.
"""


def main(out, assistant=""):
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(STATIC, out)
    shutil.copy(ROOT / "demo/flight.mp4", out / "flight.mp4")
    shutil.copy(BROWSER_MODEL, out / BROWSER_MODEL.name)
    shutil.copytree(ROOT / "vendor/ort", out / "ort")
    with db.transaction() as conn:
        video = conn.execute("SELECT * FROM videos WHERE id = 'Sample flight' AND status = 'ready'").fetchone()
        if video is None:
            sys.exit("Seed the sample flight first.")
        (out / "index.json").write_text(json.dumps(db.video_index(conn, video), separators=(",", ":")))
    (out / "facts.json").write_text(json.dumps(topics(), separators=(",", ":")))
    meta = f'  <meta http-equiv="Content-Security-Policy" content="{CSP.format(assistant=assistant)}">\n'
    # The mission is the front page. The recorded flight moves to its own page.
    for source, target in [("index.html", "footage.html"), ("mission.html", "index.html")]:
        html = (out / source).read_text(encoding="utf-8")
        html = html.replace('data-edition="server"', f'data-edition="static" data-ask-url="{assistant}"')
        html = html.replace('  <meta name="viewport"', meta + '  <meta name="viewport"', 1)
        (out / target).write_text(html, encoding="utf-8", newline="\n")
    (out / "mission.html").unlink()
    (out / "README.md").write_text(CARD, encoding="utf-8", newline="\n")
    print(f"{sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
