"""Authentication, rate limiting, quotas and response headers."""
import hashlib
import re
import secrets
import threading
import time
from collections import defaultdict, deque

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import HTTPException, Request

from . import db
from .config import settings

SESSION_COOKIE = "ds_session"
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,}$")
_hasher = PasswordHasher()
# Checked for unknown emails too, so response time does not reveal which accounts exist.
_DUMMY_HASH = _hasher.hash(secrets.token_hex(16))


def hash_password(password):
    return _hasher.hash(password)


def verify_password(stored, password):
    try:
        return _hasher.verify(stored or _DUMMY_HASH, password) and stored is not None
    except (VerificationError, InvalidHashError):
        return False


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(conn, user_id):
    """Only a hash is stored, so a database leak does not expose usable sessions."""
    token = secrets.token_urlsafe(32)
    conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
    conn.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                 (_token_hash(token), user_id, time.time() + settings.session_days * 86400))
    return token


def delete_session(conn, token):
    conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


def current_user(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT u.id, u.email FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (_token_hash(token), time.time()),
        ).fetchone()
    return dict(row) if row else None


def require_user(request: Request):
    user = current_user(request)
    if user is None:
        raise HTTPException(401, "Please sign in first.")
    return user


def client_ip(request: Request):
    # A forwarded header is only trusted when configured, because clients can forge it. Proxies
    # append to the right, so the leftmost entries are attacker-controlled and the last is not.
    header = settings.trusted_proxy_header
    if header and request.headers.get(header):
        return request.headers[header].split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Sliding window per key. Per-process: the CDN limits traffic before it reaches the app."""

    def __init__(self, clock=time.monotonic):
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()
        self._clock = clock

    def allow(self, key, limit, window=60.0):
        now = self._clock()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) >= limit:
                return False, max(1, int(hits[0] + window - now) + 1)
            hits.append(now)
            if len(self._hits) > 50_000:  # an IP-rotating flood must not exhaust memory
                self._hits = defaultdict(deque, {k: v for k, v in self._hits.items() if v})
            return True, 0


limiter = RateLimiter()


def enforce_rate(request, bucket, limit):
    allowed, retry_after = limiter.allow(f"{bucket}:{client_ip(request)}", limit)
    if not allowed:
        raise HTTPException(429, "Too many requests. Please wait a moment and try again.",
                            headers={"Retry-After": str(retry_after)})


def check_upload_quota(conn, user_id):
    """Per-user and site-wide caps, so one account or a bot farm cannot run up compute costs."""
    day_ago = time.time() - 86400
    mine = conn.execute("SELECT COUNT(*) FROM videos WHERE user_id = ? AND created_at > ?",
                        (user_id, day_ago)).fetchone()[0]
    if mine >= settings.uploads_per_user_per_day:
        raise HTTPException(429, f"You can upload {settings.uploads_per_user_per_day} videos a day. "
                                 "Please try again tomorrow.")
    count, seconds = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(duration), 0) FROM videos WHERE is_demo = 0 AND created_at > ?",
        (day_ago,)).fetchone()
    if count >= settings.site_uploads_per_day or seconds >= settings.site_processing_seconds_per_day:
        raise HTTPException(503, "We have reached today's processing limit. Please try again tomorrow.")


CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self'",
    "img-src 'self' data:",
    "media-src 'self'",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])
SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}
