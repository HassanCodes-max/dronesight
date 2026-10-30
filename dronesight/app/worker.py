"""Background worker: a separate process with no route to the network-facing API.

Run with `python -m dronesight.app.worker`. It claims queued videos from the database, indexes
them, and deletes anything past its retention time.
"""
import time
from pathlib import Path

from . import db
from .config import settings
from .vision import IouTracker, OnnxDetector

POLL_SECONDS = 2


def looks_like_video(path):
    """Check magic bytes, because a filename or Content-Type proves nothing."""
    with open(path, "rb") as f:
        head = f.read(12)
    return head[4:8] == b"ftyp" or head[:4] == b"\x1a\x45\xdf\xa3"


def probe(path):
    """Return (fps, frames, width, height), or None if the file cannot be decoded."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
    finally:
        cap.release()
    if not ok or fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        return None
    return fps, frames, width, height


def set_status(video_id, **fields):
    names = ", ".join(f"{k} = ?" for k in fields)  # keys are fixed names from this module
    with db.transaction() as conn:
        conn.execute(f"UPDATE videos SET {names} WHERE id = ?", (*fields.values(), video_id))


def index_video(video_id, path, detector):
    import cv2

    info = None if not looks_like_video(path) else probe(path)
    if info is None:
        set_status(video_id, status="failed", message="This file could not be read as a video.")
        return
    fps, frames, width, height = info
    duration = frames / fps
    if duration > settings.max_video_seconds:
        set_status(video_id, status="failed",
                   message=f"Videos can be up to {settings.max_video_seconds // 60} minutes long.")
        return
    set_status(video_id, duration=duration, width=width, height=height, message="Finding people")

    stride = max(1, round(fps * settings.frame_stride_seconds))
    tracker = IouTracker()
    cap = cv2.VideoCapture(str(path))
    rows = []
    try:
        for frame_no in range(frames):
            if not cap.grab():
                break
            if frame_no % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            boxes, scores = detector(frame)
            t = frame_no / fps
            for box, score, track in zip(boxes.tolist(), scores.tolist(), tracker.update(boxes)):
                rows.append((video_id, t, track, "person", score, *box))
            set_status(video_id, progress=(frame_no + 1) / frames)
    finally:
        cap.release()
    with db.transaction() as conn:
        conn.executemany(
            "INSERT INTO moments (video_id, t, track_id, label, conf, x1, y1, x2, y2) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        conn.execute("UPDATE videos SET status = 'ready', progress = 1, message = '' WHERE id = ?",
                     (video_id,))


def claim_next():
    with db.transaction() as conn:
        row = conn.execute("SELECT id, path FROM videos WHERE status = 'queued' "
                           "ORDER BY created_at LIMIT 1").fetchone()
        if row:
            conn.execute("UPDATE videos SET status = 'processing', message = 'Starting' "
                         "WHERE id = ? AND status = 'queued'", (row["id"],))
    return row


def recover_interrupted():
    """A crash mid-video must not leave it stuck: requeue it and clear partial results."""
    with db.transaction() as conn:
        ids = [r["id"] for r in conn.execute("SELECT id FROM videos WHERE status = 'processing'")]
        for video_id in ids:
            conn.execute("DELETE FROM moments WHERE video_id = ?", (video_id,))
            conn.execute("UPDATE videos SET status = 'queued', progress = 0 WHERE id = ?", (video_id,))


def purge_expired():
    with db.transaction() as conn:
        rows = conn.execute("SELECT id, path FROM videos WHERE expires_at IS NOT NULL AND expires_at < ?",
                            (time.time(),)).fetchall()
        for row in rows:
            Path(row["path"]).unlink(missing_ok=True)
            conn.execute("DELETE FROM videos WHERE id = ?", (row["id"],))
    return len(rows)


def main():
    db.init()
    recover_interrupted()
    detector = OnnxDetector(settings.model_path, settings.conf_threshold)
    while True:
        purge_expired()
        job = claim_next()
        if job is None:
            time.sleep(POLL_SECONDS)
            continue
        try:
            index_video(job["id"], job["path"], detector)
        except Exception as exc:  # one bad file must not stop the queue
            print(f"job {job['id']} failed: {type(exc).__name__}: {exc}", flush=True)
            set_status(job["id"], status="failed", message="Something went wrong processing this video.")


if __name__ == "__main__":
    main()
