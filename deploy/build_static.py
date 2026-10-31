"""Build the free static edition: the sample flight and its detections, with no server.

    python -m dronesight.app.seed demo/flight.mp4 "Sample flight"
    python deploy/build_static.py dist

It reads the seeded database, so the detections are the worker's real output. Static hosts cannot
send headers, so the page carries its Content Security Policy in a meta tag.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dronesight.app import db  # noqa: E402

STATIC = ROOT / "dronesight/app/static"
# Hugging Face redirects large files to its CDN, so the video needs that one extra source.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "media-src 'self' https://*.hf.co; "
       "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'")
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


def main(out):
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(STATIC, out)
    shutil.copy(ROOT / "demo/flight.mp4", out / "flight.mp4")
    with db.transaction() as conn:
        video = conn.execute("SELECT * FROM videos WHERE id = 'Sample flight' AND status = 'ready'").fetchone()
        if video is None:
            sys.exit("Seed the sample flight first.")
        (out / "index.json").write_text(json.dumps(db.video_index(conn, video), separators=(",", ":")))
    page = out / "index.html"
    html = page.read_text(encoding="utf-8")
    meta = f'  <meta http-equiv="Content-Security-Policy" content="{CSP}">\n'
    html = html.replace('data-edition="server"', 'data-edition="static"')
    html = html.replace('  <meta name="viewport"', meta + '  <meta name="viewport"', 1)
    page.write_text(html, encoding="utf-8", newline="\n")
    (out / "README.md").write_text(CARD, encoding="utf-8", newline="\n")
    print(f"{sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
