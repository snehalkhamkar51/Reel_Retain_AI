import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def get_config() -> dict:
    artifact_dir = Path(
        os.getenv("REELRETAIN_ARTIFACT_DIR", str(PROJECT_ROOT / "models"))
    ).expanduser()
    upload_dir = Path(
        os.getenv("REELRETAIN_UPLOAD_DIR", str(PROJECT_ROOT / "uploads"))
    ).expanduser()
    return {
        "HOST": os.getenv("REELRETAIN_HOST", "127.0.0.1"),
        "PORT": _positive_int("REELRETAIN_PORT", 5000),
        "MAX_CONTENT_LENGTH": _positive_int("REELRETAIN_MAX_CONTENT_LENGTH", 100 * 1024 * 1024),
        "ARTIFACT_DIR": artifact_dir,
        "UPLOAD_DIR": upload_dir,
        "AUDIO_TIMEOUT_SECONDS": _positive_int("REELRETAIN_AUDIO_TIMEOUT_SECONDS", 15),
        "VIDEO_TIMEOUT_SECONDS": _positive_int("REELRETAIN_VIDEO_TIMEOUT_SECONDS", 30),
        "GEMINI_API_KEY": os.getenv("REELRETAIN_GEMINI_API_KEY", ""),
        "GEMINI_MODEL": os.getenv("REELRETAIN_GEMINI_MODEL", "gemini-3.8-flash"),
        "GEMINI_TIMEOUT_SECONDS": _positive_int("REELRETAIN_GEMINI_TIMEOUT_SECONDS", 120),
        "GEMINI_POLL_INTERVAL_SECONDS": _positive_int("REELRETAIN_GEMINI_POLL_INTERVAL_SECONDS", 2),
        "TESTING": False,
    }
