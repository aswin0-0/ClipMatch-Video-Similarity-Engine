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

# ── Matching optimization settings ──────────────────────────────────────────
FINE_SEARCH_WINDOW_SEC: float = float(os.getenv("FINE_SEARCH_WINDOW_SEC", "20"))
WINDOW_PADDING_SEC: float = float(os.getenv("WINDOW_PADDING_SEC", "5"))
ENABLE_MIRROR_MATCHING: bool = os.getenv("ENABLE_MIRROR_MATCHING", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ENABLE_PATCH_EARLY_REJECTION: bool = os.getenv(
    "ENABLE_PATCH_EARLY_REJECTION", "true"
).lower() in {"1", "true", "yes", "on"}
PATCH_GRID_SIZE: int = int(os.getenv("PATCH_GRID_SIZE", "4"))
PATCH_SCORE_THRESHOLD: float = float(os.getenv("PATCH_SCORE_THRESHOLD", "0.20"))
PATCH_MIN_PASS_RATIO: float = float(os.getenv("PATCH_MIN_PASS_RATIO", "0.50"))

# ── Frame extraction settings ─────────────────────────────────────────────────
COARSE_FPS: int = int(os.getenv("COARSE_FPS", "1"))
FINE_FPS: int = int(os.getenv("FINE_FPS", "24"))
FRAME_SIZE: int = int(os.getenv("FRAME_SIZE", "256"))
STORE_FINE_FRAMES: bool = os.getenv("STORE_FINE_FRAMES", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def ensure_directories() -> None:
    """Create required local directories if they don't exist."""
    for d in (DB_VIDEOS_DIR, PROCESSED_FRAMES_DIR, QUERY_UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)
