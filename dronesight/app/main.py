"""DroneSight web API and static site. Run with `uvicorn dronesight.app.main:app`."""
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, search, security
from .config import settings

STATIC = Path(__file__).parent / "static"
MAX_JSON_BYTES = 16 * 1024

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@asynccontextmanager
async def lifespan(_app):
    db.init()
    settings.video_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="DroneSight", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=10, max_length=128)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=300)


@app.middleware("http")
async def protect(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        try:
            security.enforce_rate(request, "api", settings.rate_api_per_minute)
            if request.method in UNSAFE_METHODS:
                _check_origin(request)
                if request.url.path != "/api/videos":
                    _check_json_size(request)
        except HTTPException as exc:
            response = JSONResponse({"detail": exc.detail}, exc.status_code, headers=exc.headers)
        else:
            response = await call_next(request)
    else:
        response = await call_next(request)
    for name, value in security.SECURITY_HEADERS.items():
        response.headers[name] = value
    if settings.secure_cookies:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _check_origin(request):
    """Reject cross-site writes. SameSite cookies already block most; this covers old browsers."""
    origin = request.headers.get("origin")
    if origin and urlparse(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "This request came from another site.")


def _check_json_size(request):
    length = request.headers.get("content-length")
    if length and (not length.isdigit() or int(length) > MAX_JSON_BYTES):
        raise HTTPException(413, "Request is too large.")


def _set_session(response, token):
    response.set_cookie(security.SESSION_COOKIE, token, httponly=True, samesite="strict",
                        secure=settings.secure_cookies, max_age=settings.session_days * 86400, path="/")


@app.post("/api/register", status_code=201)
def register(body: Credentials, request: Request, response: Response):
    security.enforce_rate(request, "login", settings.rate_login_per_minute)
    email = body.email.strip()
    if not security.EMAIL_RE.match(email):
        raise HTTPException(422, "Please enter a valid email address.")
    password_hash = security.hash_password(body.password)
    with db.transaction() as conn:
        try:
            user_id = conn.execute("INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                                   (email, password_hash, time.time())).lastrowid
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Could not create that account. Try signing in instead.") from None
        _set_session(response, security.create_session(conn, user_id))
    return {"email": email}


@app.post("/api/login")
def login(body: Credentials, request: Request, response: Response):
    security.enforce_rate(request, "login", settings.rate_login_per_minute)
    with db.transaction() as conn:
        row = conn.execute("SELECT id, email, password_hash FROM users WHERE email = ?",
                           (body.email.strip(),)).fetchone()
        if not security.verify_password(row["password_hash"] if row else None, body.password):
            raise HTTPException(401, "Email or password is incorrect.")
        _set_session(response, security.create_session(conn, row["id"]))
    return {"email": row["email"]}


@app.post("/api/logout", status_code=204)
def logout(request: Request, response: Response):
    token = request.cookies.get(security.SESSION_COOKIE)
    if token:
        with db.transaction() as conn:
            security.delete_session(conn, token)
    response.delete_cookie(security.SESSION_COOKIE, path="/")


@app.get("/api/me")
def me(user=Depends(security.current_user)):
    return {"email": user["email"] if user else None}


def _public(row):
    return {k: row[k] for k in ("id", "title", "status", "progress", "message", "duration", "is_demo")}


def _uid(user):
    return user["id"] if user else None


@app.get("/api/videos")
def list_videos(user=Depends(security.current_user)):
    with db.transaction() as conn:
        rows = conn.execute("SELECT * FROM videos WHERE is_demo = 1 OR user_id = ? "
                            "ORDER BY is_demo, created_at DESC LIMIT 100", (_uid(user),)).fetchall()
    return [_public(r) for r in rows]


@app.put("/api/videos", status_code=202)
async def upload(request: Request, title: str = "Untitled flight", user=Depends(security.require_user)):
    """Raw streaming body: bytes are counted as they arrive, so an oversized upload stops early."""
    length = request.headers.get("content-length", "")
    limit = settings.max_upload_mb * 1024 * 1024
    if not length.isdigit():
        raise HTTPException(411, "Upload size is required.")
    if int(length) > limit:
        raise HTTPException(413, f"Videos can be up to {settings.max_upload_mb} MB.")
    with db.transaction() as conn:
        security.check_upload_quota(conn, user["id"])

    video_id = secrets.token_urlsafe(16)
    path = settings.video_dir / f"{video_id}.bin"  # the name is ours; user input never reaches a path
    received = 0
    try:
        with open(path, "wb") as f:
            async for chunk in request.stream():
                received += len(chunk)
                if received > limit:
                    raise HTTPException(413, f"Videos can be up to {settings.max_upload_mb} MB.")
                f.write(chunk)
        if received == 0:
            raise HTTPException(422, "That file is empty.")
        with db.transaction() as conn:
            security.check_upload_quota(conn, user["id"])  # re-checked: parallel uploads race the first check
            now = time.time()
            conn.execute(
                "INSERT INTO videos (id, user_id, title, path, status, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, 'queued', ?, ?)",
                (video_id, user["id"], title.strip()[:120] or "Untitled flight", str(path), now,
                 now + settings.retention_hours * 3600))
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {"id": video_id}


@app.get("/api/videos/{video_id}")
def get_video(video_id: str, user=Depends(security.current_user)):
    with db.transaction() as conn:
        row = db.visible_video(conn, video_id, _uid(user))
    if row is None:
        raise HTTPException(404, "Video not found.")
    return _public(row)


@app.get("/api/videos/{video_id}/stream")
def stream(video_id: str, user=Depends(security.current_user)):
    with db.transaction() as conn:
        row = db.visible_video(conn, video_id, _uid(user))
    if row is None or not Path(row["path"]).exists():
        raise HTTPException(404, "Video not found.")
    return FileResponse(row["path"], media_type="video/mp4", headers={"Cache-Control": "private, no-store"})


@app.post("/api/videos/{video_id}/search")
def search_video(video_id: str, body: Question, request: Request, user=Depends(security.current_user)):
    security.enforce_rate(request, "search", settings.rate_search_per_minute)
    with db.transaction() as conn:
        row = db.visible_video(conn, video_id, _uid(user))
        if row is None:
            raise HTTPException(404, "Video not found.")
        if row["status"] != "ready":
            raise HTTPException(409, "This video is still being processed.")
        return search.run(conn, video_id, body.question)


@app.delete("/api/videos/{video_id}", status_code=204)
def delete_video(video_id: str, user=Depends(security.require_user)):
    with db.transaction() as conn:
        row = conn.execute("SELECT path FROM videos WHERE id = ? AND user_id = ? AND is_demo = 0",
                           (video_id, user["id"])).fetchone()
        if row is None:
            raise HTTPException(404, "Video not found.")
        conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    Path(row["path"]).unlink(missing_ok=True)


@app.get("/healthz")
def healthz():
    return {"ok": True}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
