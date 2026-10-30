"""Turn a plain-English question into a filter over stored moments, then rank the clips.

This is deterministic on purpose: it needs no LLM key, costs nothing per query, and text from
users or footage can never become instructions. It only understands the listed intents.
"""
import re
from dataclasses import dataclass

LABEL_WORDS = {
    "person": ["person", "people", "human", "humans", "pedestrian", "pedestrians", "someone",
               "anybody", "anyone", "survivor", "survivors", "victim", "victims"],
}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}
CLIP_GAP_SECONDS = 2.0


@dataclass
class Query:
    label: str | None
    min_count: int = 1
    start: float | None = None
    end: float | None = None
    busiest: bool = False


def _seconds(value, unit):
    return float(value) * (60 if unit.startswith("m") else 1)


def parse(text):
    """Return a Query, or None when the question is outside what the index can answer."""
    q = text.lower().strip()
    tokens = re.findall(r"[a-z]+|\d+", q)
    label = next((name for name, words in LABEL_WORDS.items() if set(words) & set(tokens)), None)
    if label is None:
        return None
    query = Query(label=label)

    if m := re.search(r"(?:at least|more than|over|min(?:imum)?(?: of)?)\s+(\d+|[a-z]+)", q):
        n = NUMBER_WORDS.get(m[1], int(m[1]) if m[1].isdigit() else None)
        if n is not None:
            query.min_count = n + 1 if m.group(0).startswith(("more", "over")) else n
    elif m := re.search(r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten)\+?\s+(?:or more\s+)?(?:people|persons|humans)", q):
        query.min_count = NUMBER_WORDS.get(m[1], int(m[1]) if m[1].isdigit() else 1)
    elif re.search(r"\b(group|groups|crowd|crowds|several|many)\b", q):
        query.min_count = 3

    if m := re.search(r"(?:after|from)\s+(\d+(?:\.\d+)?)\s*(s|sec|second|seconds|m|min|minute|minutes)\b", q):
        query.start = _seconds(m[1], m[2])
    if m := re.search(r"(?:before|until|to)\s+(\d+(?:\.\d+)?)\s*(s|sec|second|seconds|m|min|minute|minutes)\b", q):
        query.end = _seconds(m[1], m[2])

    query.busiest = bool(re.search(r"\b(most|busiest|largest|biggest)\b", q))
    return query


def group_moments(rows):
    """Merge per-frame detections into clips of nearby time, keeping the strongest evidence."""
    frames = {}
    for row in rows:
        frames.setdefault(row["t"], []).append(row)
    clips = []
    for t in sorted(frames):
        boxes = frames[t]
        if clips and t - clips[-1]["end"] <= CLIP_GAP_SECONDS:
            clip = clips[-1]
            clip["end"] = t
        else:
            clip = {"start": t, "end": t, "peak_count": 0, "peak_t": t, "tracks": set(), "boxes": []}
            clips.append(clip)
        clip["tracks"].update(b["track_id"] for b in boxes)
        if len(boxes) > clip["peak_count"]:
            clip.update(peak_count=len(boxes), peak_t=t, boxes=boxes)
    return clips


def run(conn, video_id, text):
    query = parse(text)
    if query is None:
        return {"understood": False, "clips": [],
                "answer": "I can search for people right now. Try “groups of people” or "
                          "“at least 2 people after 30 seconds”."}
    sql = "SELECT t, track_id, conf, x1, y1, x2, y2, lat, lon FROM moments WHERE video_id = ? AND label = ?"
    args = [video_id, query.label]
    if query.start is not None:
        sql += " AND t >= ?"
        args.append(query.start)
    if query.end is not None:
        sql += " AND t <= ?"
        args.append(query.end)
    rows = conn.execute(sql + " ORDER BY t", args).fetchall()

    by_time = {}
    for row in rows:
        by_time.setdefault(row["t"], []).append(row)
    kept = [r for group in by_time.values() if len(group) >= query.min_count for r in group]
    clips = group_moments(kept)
    clips.sort(key=lambda c: (-c["peak_count"], -len(c["tracks"]), c["start"]))
    if query.busiest:
        clips = clips[:1]
    clips = clips[:20]

    out = []
    for clip in clips:
        located = [b for b in clip["boxes"] if b["lat"] is not None]
        out.append({
            "start": clip["start"], "end": clip["end"], "peak_t": clip["peak_t"],
            "count": clip["peak_count"], "tracks": len(clip["tracks"]),
            "confidence": round(max(b["conf"] for b in clip["boxes"]), 3),
            "boxes": [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in clip["boxes"]],
            "location": ({"lat": sum(b["lat"] for b in located) / len(located),
                          "lon": sum(b["lon"] for b in located) / len(located)} if located else None),
        })
    if out:
        answer = f"Found {len(out)} matching clip{'s' if len(out) != 1 else ''}. " \
                 f"The strongest shows {out[0]['count']} {'person' if out[0]['count'] == 1 else 'people'} " \
                 f"at {_clock(out[0]['peak_t'])}."
    else:
        answer = "Nothing in this footage matched. The detector can miss very small people, so this is not proof that nobody is there."
    return {"understood": True, "clips": out, "answer": answer}


def _clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"
