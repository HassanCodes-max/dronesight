"""Add a public sample flight: `python -m dronesight.app.seed demo/flight.mp4 "Sample flight"`.

Runs the same indexing path as user uploads, so the demo is real output, not hand-made data.
"""
import sys
import time

from . import db
from .config import settings
from .vision import OnnxDetector
from .worker import index_video


def seed(path, title):
    db.init()
    with db.transaction() as conn:
        if conn.execute("SELECT 1 FROM videos WHERE id = ?", (title,)).fetchone():
            return
        conn.execute("INSERT INTO videos (id, title, path, status, is_demo, created_at) "
                     "VALUES (?, ?, ?, 'processing', 1, ?)", (title, title, str(path), time.time()))
    index_video(title, path, OnnxDetector(settings.model_path, settings.conf_threshold))


if __name__ == "__main__":
    seed(sys.argv[1], sys.argv[2])
