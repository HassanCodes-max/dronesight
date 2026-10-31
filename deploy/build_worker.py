"""Build the Cloudflare Worker that answers free questions about the sample flight.

    python deploy/build_worker.py dist-worker/worker.js

Combines the worker's detections with demo/scene.json, so the assistant's facts are real output.
Paste the result into the Cloudflare dashboard and add a Workers AI binding named AI.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dronesight.app import db  # noqa: E402

ORIGIN = "https://hassanali0987654-dronesight.static.hf.space"
ROW_SECONDS = 2
GROUPS = {"cars and vans": ("car", "van"), "trucks and buses": ("truck", "bus"), "motorbikes": ("motor",),
          "bicycles": ("bicycle",), "tricycles": ("tricycle", "awning-tricycle")}
ABOUT = """DroneSight finds people in recorded drone video and lets you jump to the moment.
People are found by a YOLO11s model trained on the VisDrone and HERIDAL datasets, four times a second.
On validation data it found about 65% of people in VisDrone city footage and 34% in HERIDAL search-and-rescue footage.
This sample is 42 seconds from the VisDrone dataset. It has no GPS, so locations on a map are unknown."""


def clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def facts(index, scene):
    people = {s["t"]: len(s["boxes"]) for s in index["samples"]}
    busiest = max(people, key=people.get)
    lines = [ABOUT, "", f"Length: {clock(index['duration'])}. Busiest moment: about {people[busiest]} people at {clock(busiest)}.",
             "", f"Peak counts in each {ROW_SECONDS} second stretch:"]
    for start in range(0, int(index["duration"]) + 1, ROW_SECONDS):
        rows = [s for s in scene["samples"] if start <= s["t"] < start + ROW_SECONDS]
        peak = max((n for t, n in people.items() if start <= t < start + ROW_SECONDS), default=0)
        parts = [f"{peak} people"]
        for name, classes in GROUPS.items():
            most = max((sum(s["counts"].get(c, 0) for c in classes) for s in rows), default=0)
            if most:
                parts.append(f"{most} {name}")
        lines.append(f"{clock(start)}: " + ", ".join(parts))
    lines += ["", "Rough scene descriptions:"]
    lines += [f"{clock(c['t'])}: {c['text']}" for c in scene["captions"]]
    return "\n".join(lines)


def main(out):
    scene = json.loads((ROOT / "demo/scene.json").read_text(encoding="utf-8"))
    with db.transaction() as conn:
        video = conn.execute("SELECT * FROM videos WHERE id = 'Sample flight' AND status = 'ready'").fetchone()
        if video is None:
            sys.exit("Seed the sample flight first.")
        index = db.video_index(conn, video)
    template = (ROOT / "worker/ask.template.js").read_text(encoding="utf-8")
    code = template.replace("__ORIGIN__", ORIGIN).replace("__FACTS__", json.dumps(facts(index, scene)))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(code, encoding="utf-8", newline="\n")
    print(f"{len(code) / 1000:.1f} KB in {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
