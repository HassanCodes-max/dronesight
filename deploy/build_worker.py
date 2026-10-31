"""Build the Cloudflare Worker that answers questions and sends alerts to headquarters.

    python deploy/build_worker.py dist-worker/worker.js

Combines the worker's detections, demo/scene.json and the mission data, so the assistant's facts are
real output. Paste the result into the Cloudflare dashboard and add a Workers AI binding named AI.
Telegram alerts stay off until TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set as Cloudflare secrets.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dronesight.app import db  # noqa: E402

ORIGIN = "https://hassanali0987654-dronesight.static.hf.space"
MISSIONS = ROOT / "dronesight/app/static/mission/mission.json"
ROW_SECONDS = 2
TOP_PEOPLE = 15
GROUPS = {"cars and vans": ("car", "van"), "trucks and buses": ("truck", "bus"), "motorbikes": ("motor",),
          "bicycles": ("bicycle",), "tricycles": ("tricycle", "awning-tricycle")}
ABOUT = """DroneSight finds people in recorded drone video and lets you jump to the moment.
People are found by a YOLO11s model trained on the VisDrone and HERIDAL datasets, four times a second.
On its test images it found about 44% of people in VisDrone city footage and 67% in HERIDAL search-and-rescue footage.
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


def ranked(mission):
    return sorted(((person, sector["id"]) for sector in mission["sectors"] for person in sector["people"]), key=lambda item: item[0]["rank"])


def mission_facts(mission):
    people = ranked(mission)
    levels = Counter(person["level"] for person, _ in people)
    lines = [f"Simulated {mission['title'].lower()} mission. A drone at {mission['altitude_m']} m searches {len(mission['sectors'])} sectors, "
             f"each about {mission['sector_width_m']} m wide. Locations are simulated, accurate to about {mission['error_m']} m.",
             "The priority order uses fixed rules, not a trained model: being in the hazard counts most, then how far it is to safety, "
             "then being near the hazard, then being alone. Emergency responders have not validated these rules.",
             "People and hazards are real model output on photos with people added by the C2A dataset, and the model misses people.",
             "", f"Found about {len(people)} people: {levels['CRITICAL']} critical, {levels['HIGH']} high, {levels['WATCH']} watch.", "", "Sectors:"]
    for sector in mission["sectors"]:
        critical = sum(person["level"] == "CRITICAL" for person in sector["people"])
        lines.append(f"{sector['id']}: {len(sector['people'])} people, {critical} critical")
    lines += ["", "Most urgent people, in priority order:"]
    for person, sector in people[:TOP_PEOPLE]:
        lines.append(f"{person['rank']}. {person['level']}, sector {sector}, {', '.join(person['reasons'])}, at {person['latitude']}, {person['longitude']}")
    return "\n".join(lines)


def alerts(mission):
    return {person["id"]: {"level": person["level"], "sector": sector, "reasons": ", ".join(person["reasons"]),
                           "latitude": person["latitude"], "longitude": person["longitude"]}
            for person, sector in ranked(mission) if person["level"] == "CRITICAL"}


def main(out):
    scene = json.loads((ROOT / "demo/scene.json").read_text(encoding="utf-8"))
    with db.transaction() as conn:
        video = conn.execute("SELECT * FROM videos WHERE id = 'Sample flight' AND status = 'ready'").fetchone()
        if video is None:
            sys.exit("Seed the sample flight first.")
        index = db.video_index(conn, video)
    missions = json.loads(MISSIONS.read_text(encoding="utf-8"))
    topics = {"flight": facts(index, scene), **{name: mission_facts(mission) for name, mission in missions.items()}}
    template = (ROOT / "worker/ask.template.js").read_text(encoding="utf-8")
    code = (template.replace("__ORIGIN__", ORIGIN).replace("__FACTS__", json.dumps(topics))
            .replace("__ALERTS__", json.dumps({name: alerts(mission) for name, mission in missions.items()})))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(code, encoding="utf-8", newline="\n")
    print(f"{len(code) / 1000:.1f} KB in {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
