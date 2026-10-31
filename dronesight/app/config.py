"""Runtime settings. Every limit lives here so the abuse controls are easy to audit."""
import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name, default):
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DRONESIGHT_DATA", "data/app")))
    secure_cookies: bool = os.environ.get("DRONESIGHT_ENV", "dev") == "production"
    trusted_proxy_header: str = os.environ.get("DRONESIGHT_CLIENT_IP_HEADER", "")

    session_days: int = _int("DRONESIGHT_SESSION_DAYS", 7)
    max_upload_mb: int = _int("DRONESIGHT_MAX_UPLOAD_MB", 200)
    max_video_seconds: int = _int("DRONESIGHT_MAX_VIDEO_SECONDS", 180)
    retention_hours: int = _int("DRONESIGHT_RETENTION_HOURS", 24)

    uploads_per_user_per_day: int = _int("DRONESIGHT_UPLOADS_PER_DAY", 3)
    site_uploads_per_day: int = _int("DRONESIGHT_SITE_UPLOADS_PER_DAY", 50)
    site_processing_seconds_per_day: int = _int("DRONESIGHT_SITE_SECONDS_PER_DAY", 3600)

    rate_login_per_minute: int = _int("DRONESIGHT_RATE_LOGIN", 5)
    rate_api_per_minute: int = _int("DRONESIGHT_RATE_API", 120)

    frame_stride_seconds: float = float(os.environ.get("DRONESIGHT_STRIDE_SECONDS", 0.25))
    conf_threshold: float = float(os.environ.get("DRONESIGHT_CONF", 0.25))

    @property
    def db_path(self):
        return self.data_dir / "dronesight.db"

    @property
    def video_dir(self):
        return self.data_dir / "videos"

    @property
    def model_path(self):
        return Path(os.environ.get("DRONESIGHT_MODEL", "models/person.onnx"))


settings = Settings()
