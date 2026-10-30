"""API tests, written mainly as attacks on the access rules and limits."""
import os
import tempfile
import time
from dataclasses import replace

import numpy as np
import pytest

os.environ["DRONESIGHT_DATA"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient  # noqa: E402

from dronesight.app import config, db, main, search, security, worker  # noqa: E402
from dronesight.app.vision import IouTracker, decode, nms  # noqa: E402

PASSWORD = "correct horse battery"
MP4_HEADER = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


@pytest.fixture(autouse=True)
def fresh_state():
    security.limiter = security.RateLimiter()
    db.init()
    with db.transaction() as conn:
        conn.execute("DELETE FROM users")
        conn.execute("DELETE FROM videos")


@pytest.fixture
def client():
    with TestClient(main.app, base_url="http://testserver") as c:
        yield c


def signed_up(client, email="a@example.com"):
    assert client.post("/api/register", json={"email": email, "password": PASSWORD}).status_code == 201
    return client


def upload(client, body=MP4_HEADER):
    return client.put("/api/videos?title=Flight", content=body)


def make_ready(video_id, boxes_at=()):
    with db.transaction() as conn:
        conn.execute("UPDATE videos SET status = 'ready' WHERE id = ?", (video_id,))
        for track, t in enumerate(boxes_at):
            conn.execute("INSERT INTO moments (video_id, t, track_id, label, conf, x1, y1, x2, y2) "
                         "VALUES (?, ?, ?, 'person', 0.9, 0.1, 0.1, 0.2, 0.2)", (video_id, t, track))


def test_sql_injection_in_login_is_inert(client):
    signed_up(client)
    client.cookies.clear()
    r = client.post("/api/login", json={"email": "' OR 1=1 --", "password": PASSWORD})
    assert r.status_code == 401


def test_wrong_password_and_unknown_email_look_identical(client):
    signed_up(client)
    client.cookies.clear()
    a = client.post("/api/login", json={"email": "a@example.com", "password": "wrong password"})
    b = client.post("/api/login", json={"email": "nobody@example.com", "password": "wrong password"})
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json()


def test_login_brute_force_is_rate_limited(client):
    codes = [client.post("/api/login", json={"email": "a@example.com", "password": "wrong password"}).status_code
             for _ in range(config.settings.rate_login_per_minute + 1)]
    assert codes[-1] == 429


def test_forged_forwarded_header_cannot_dodge_the_rate_limit(client, monkeypatch):
    monkeypatch.setattr(security, "settings", replace(config.settings, trusted_proxy_header="x-forwarded-for"))
    limit = config.settings.rate_login_per_minute
    codes = [client.post("/api/login", json={"email": "a@example.com", "password": "wrong password"},
                         headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.9"}).status_code
             for i in range(limit + 1)]
    assert codes[-1] == 429


def test_passwords_and_session_tokens_are_not_stored_in_plain_text(client):
    signed_up(client)
    token = client.cookies[security.SESSION_COOKIE]
    with db.transaction() as conn:
        assert PASSWORD not in conn.execute("SELECT password_hash FROM users").fetchone()[0]
        assert token not in conn.execute("SELECT token_hash FROM sessions").fetchone()[0]


def test_session_cookie_flags(client):
    r = client.post("/api/register", json={"email": "c@example.com", "password": PASSWORD})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie


def test_upload_requires_login(client):
    assert upload(client).status_code == 401


def test_one_user_cannot_touch_another_users_video(client):
    signed_up(client)
    video_id = upload(client).json()["id"]
    make_ready(video_id)
    client.cookies.clear()
    signed_up(client, "b@example.com")
    assert client.get(f"/api/videos/{video_id}").status_code == 404
    assert client.get(f"/api/videos/{video_id}/stream").status_code == 404
    assert client.post(f"/api/videos/{video_id}/search", json={"question": "people"}).status_code == 404
    assert client.delete(f"/api/videos/{video_id}").status_code == 404
    assert video_id not in [v["id"] for v in client.get("/api/videos").json()]


def test_anonymous_visitor_cannot_see_private_video(client):
    signed_up(client)
    video_id = upload(client).json()["id"]
    client.cookies.clear()
    assert client.get(f"/api/videos/{video_id}").status_code == 404


def test_demo_cannot_be_deleted_by_a_user(client):
    signed_up(client)
    with db.transaction() as conn:
        conn.execute("INSERT INTO videos (id, title, path, status, is_demo, created_at) "
                     "VALUES ('demo', 'Demo', 'x', 'ready', 1, 0)")
    assert client.delete("/api/videos/demo").status_code == 404


def test_cross_site_write_is_rejected(client):
    signed_up(client)
    r = client.post("/api/logout", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_oversized_upload_is_refused_by_declared_size(client):
    signed_up(client)
    limit = config.settings.max_upload_mb * 1024 * 1024
    r = client.put("/api/videos", content=b"x", headers={"Content-Length": str(limit + 1)})
    assert r.status_code == 413


def test_upload_quota_is_enforced(client):
    signed_up(client)
    codes = [upload(client).status_code for _ in range(config.settings.uploads_per_user_per_day + 1)]
    assert codes[-1] == 429
    assert codes.count(202) == config.settings.uploads_per_user_per_day


def test_site_wide_cap_stops_a_bot_farm(client, monkeypatch):
    monkeypatch.setattr(security, "settings", replace(config.settings, site_uploads_per_day=1))
    signed_up(client)
    assert upload(client).status_code == 202
    client.cookies.clear()
    signed_up(client, "b@example.com")
    assert upload(client).status_code == 503


def test_hostile_title_stays_inert_data(client):
    signed_up(client)
    r = client.put("/api/videos", params={"title": "<script>alert(1)</script>../../etc/passwd"}, content=MP4_HEADER)
    video_id = r.json()["id"]
    with db.transaction() as conn:
        path = conn.execute("SELECT path FROM videos WHERE id = ?", (video_id,)).fetchone()["path"]
    assert "passwd" not in path and "script" not in path


def test_non_video_upload_is_rejected_by_content():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"<?php system($_GET['c']); ?>" * 10)
    assert not worker.looks_like_video(f.name)


def test_expired_videos_are_purged_with_their_files(client):
    signed_up(client)
    video_id = upload(client).json()["id"]
    with db.transaction() as conn:
        path = conn.execute("SELECT path FROM videos").fetchone()["path"]
        conn.execute("UPDATE videos SET expires_at = ?", (time.time() - 1,))
    assert worker.purge_expired() == 1
    assert not os.path.exists(path)
    assert client.get(f"/api/videos/{video_id}").status_code == 404


def test_interrupted_job_is_requeued_without_partial_results(client):
    signed_up(client)
    video_id = upload(client).json()["id"]
    make_ready(video_id, [1.0])
    with db.transaction() as conn:
        conn.execute("UPDATE videos SET status = 'processing'")
    worker.recover_interrupted()
    with db.transaction() as conn:
        assert conn.execute("SELECT status FROM videos").fetchone()[0] == "queued"
        assert conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0] == 0


def test_security_headers_are_on_every_response(client):
    for path in ["/", "/api/me", "/does-not-exist"]:
        headers = client.get(path).headers
        assert "default-src 'self'" in headers["content-security-policy"]
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"


def test_api_docs_are_not_public(client):
    for path in ["/docs", "/redoc", "/openapi.json"]:
        assert client.get(path).status_code == 404


def test_static_mount_does_not_serve_source_or_data(client):
    for path in ["/../main.py", "/%2e%2e/main.py", "/..%2fconfig.py"]:
        assert client.get(path).status_code == 404


def test_search_end_to_end(client):
    signed_up(client)
    video_id = upload(client).json()["id"]
    make_ready(video_id, [10.0, 10.0, 10.0, 40.0])
    r = client.post(f"/api/videos/{video_id}/search", json={"question": "groups of people"}).json()
    assert r["understood"] and len(r["clips"]) == 1
    assert r["clips"][0]["count"] == 3


def test_empty_result_warns_it_is_not_proof(client):
    signed_up(client)
    video_id = upload(client).json()["id"]
    make_ready(video_id)
    r = client.post(f"/api/videos/{video_id}/search", json={"question": "people"}).json()
    assert r["clips"] == [] and "not proof" in r["answer"]


@pytest.mark.parametrize("text, count, start, end", [
    ("people", 1, None, None),
    ("at least 2 people after 30 seconds", 2, 30, None),
    ("more than 2 people", 3, None, None),
    ("three people", 3, None, None),
    ("a crowd of people before 2 minutes", 3, None, 120),
])
def test_parse(text, count, start, end):
    q = search.parse(text)
    assert (q.min_count, q.start, q.end) == (count, start, end)


def test_parse_refuses_what_it_cannot_answer():
    assert search.parse("where are the cars") is None


def test_rate_limiter_window_resets():
    now = [0.0]
    limiter = security.RateLimiter(clock=lambda: now[0])
    assert limiter.allow("k", 1)[0]
    assert not limiter.allow("k", 1)[0]
    now[0] = 61.0
    assert limiter.allow("k", 1)[0]


def test_real_video_is_indexed_and_searchable(client, tmp_path):
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "flight.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(30):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    signed_up(client)
    video_id = upload(client, video.read_bytes()).json()["id"]
    with db.transaction() as conn:
        path = conn.execute("SELECT path FROM videos WHERE id = ?", (video_id,)).fetchone()["path"]

    def fake_detector(frame):
        return np.array([[0.1, 0.1, 0.2, 0.2]]), np.array([0.8])

    worker.index_video(video_id, path, fake_detector)
    state = client.get(f"/api/videos/{video_id}").json()
    assert state["status"] == "ready" and state["duration"] == pytest.approx(3.0)
    result = client.post(f"/api/videos/{video_id}/search", json={"question": "people"}).json()
    assert result["clips"][0]["count"] == 1 and result["clips"][0]["tracks"] == 1
    ranged = client.get(f"/api/videos/{video_id}/stream", headers={"Range": "bytes=0-99"})
    assert ranged.status_code == 206 and len(ranged.content) == 100


def test_corrupt_video_fails_cleanly(client):
    pytest.importorskip("cv2")
    signed_up(client)
    video_id = upload(client).json()["id"]
    with db.transaction() as conn:
        path = conn.execute("SELECT path FROM videos WHERE id = ?", (video_id,)).fetchone()["path"]
    worker.index_video(video_id, path, lambda frame: None)
    assert client.get(f"/api/videos/{video_id}").json()["status"] == "failed"


def test_nms_removes_duplicates():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=float)
    assert sorted(nms(boxes, np.array([0.9, 0.8, 0.7])).tolist()) == [0, 2]


def test_decode_maps_back_to_frame_coordinates():
    # A 100x50 frame in a 100x100 input: scale 1, 25 px bars above and below.
    output = np.array([[[50.0], [50.0], [100.0], [50.0], [0.9]]])
    boxes, scores = decode(output, 0.25, 1.0, 0, 25, 100, 50)
    assert boxes[0] == pytest.approx([0, 0, 1, 1])
    assert scores[0] == pytest.approx(0.9)


def test_tracker_keeps_identity_across_frames():
    tracker = IouTracker()
    first = tracker.update(np.array([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]]))
    second = tracker.update(np.array([[0.51, 0.5, 0.61, 0.6], [0.11, 0.1, 0.21, 0.2]]))
    assert second == [first[1], first[0]]
