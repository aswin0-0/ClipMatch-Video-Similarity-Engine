"""
config.py — Centralized configuration loader using python-dotenv.
All environment variables are parsed here and exposed as typed constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level up from this file if inside a package,
# or the same directory if run from project root).
load_dotenv(dotenv_path=Path(__file__).parent / ".env")


# ── MongoDB ──────────────────────────────────────────────────────────────────
MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "clipmatch")

# ── Directory paths ───────────────────────────────────────────────────────────
DB_VIDEOS_DIR: Path = Path(os.getenv("DB_VIDEOS_DIR", "./db_videos"))
PROCESSED_FRAMES_DIR: Path = Path(os.getenv("PROCESSED_FRAMES_DIR", "./processed_frames"))
QUERY_UPLOAD_DIR: Path = Path(os.getenv("QUERY_UPLOAD_DIR", "./tmp/queries"))

# ── NCC matching thresholds ───────────────────────────────────────────────────
NCC_COARSE_THRESHOLD: float = float(os.getenv("NCC_COARSE_THRESHOLD", "0.75"))
NCC_FINE_THRESHOLD: float = float(os.getenv("NCC_FINE_THRESHOLD", "0.85"))

# ── Frame extraction settings ─────────────────────────────────────────────────
COARSE_FPS: int = int(os.getenv("COARSE_FPS", "1"))
FINE_FPS: int = int(os.getenv("FINE_FPS", "24"))
FRAME_SIZE: int = int(os.getenv("FRAME_SIZE", "256"))


def ensure_directories() -> None:
    """Create required local directories if they don't exist."""
    for d in (DB_VIDEOS_DIR, PROCESSED_FRAMES_DIR, QUERY_UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)
