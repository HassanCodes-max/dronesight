"""SQLite storage. All queries are parameterised, and every video read is scoped to its owner."""
import sqlite3
from contextlib import contextmanager

from .config import settings

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    path TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    duration REAL,
    width INTEGER,
    height INTEGER,
    is_demo INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    expires_at REAL
);
CREATE INDEX IF NOT EXISTS videos_status ON videos(status, created_at);
CREATE INDEX IF NOT EXISTS videos_owner ON videos(user_id, created_at);
CREATE TABLE IF NOT EXISTS moments (
    id INTEGER PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    t REAL NOT NULL,
    track_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    conf REAL NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    lat REAL,
    lon REAL
);
CREATE INDEX IF NOT EXISTS moments_video ON moments(video_id, t);
"""


def connect():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init():
    conn = connect()
    conn.executescript(SCHEMA)
    conn.close()


@contextmanager
def transaction():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def visible_video(conn, video_id, user_id):
    """Return a video the caller may see: their own, or a demo. Anything else looks like a 404."""
    return conn.execute(
        "SELECT * FROM videos WHERE id = ? AND (is_demo = 1 OR user_id = ?)",
        (video_id, user_id),
    ).fetchone()


def video_index(conn, video):
    """Every detection grouped by time. Search and the live boxes run in the browser from this."""
    samples = {}
    for row in conn.execute("SELECT t, x1, y1, x2, y2 FROM moments WHERE video_id = ? ORDER BY t", (video["id"],)):
        samples.setdefault(row["t"], []).append([round(row[k], 4) for k in ("x1", "y1", "x2", "y2")])
    return {"duration": video["duration"], "samples": [{"t": t, "boxes": boxes} for t, boxes in samples.items()]}
