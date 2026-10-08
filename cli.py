"""
cli.py — Command Line Interface for ClipMatch.

Uses live frame extraction from video files instead of loading
pre-saved frames from disk. Much faster and more reliable.
"""

import asyncio
import logging
import time
from pathlib import Path

from config import (
    COARSE_FPS,
    FINE_FPS,
    DB_VIDEOS_DIR,
    NCC_COARSE_THRESHOLD,
    NCC_FINE_THRESHOLD,
    ensure_directories,
)
from services.matcher import match_query_clip_live

# ── Logging — show INFO+ for our code, suppress third-party noise ────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")

logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("motor").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("dns").setLevel(logging.WARNING)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


def discover_db_videos() -> dict[str, Path]:
    """Scan DB_VIDEOS_DIR and return video_id → path mapping."""
    mapping: dict[str, Path] = {}
    if not DB_VIDEOS_DIR.exists():
        logger.error("DB_VIDEOS_DIR does not exist: %s", DB_VIDEOS_DIR.resolve())
        return mapping

    for f in DB_VIDEOS_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            mapping[f.stem] = f

    return mapping


def main():
    print("=" * 60)
    print("  ClipMatch CLI  —  Video Matching via NCC")
    print("=" * 60)
    print(f"  Coarse FPS       : {COARSE_FPS}")
    print(f"  Fine FPS         : {FINE_FPS}")
    print(f"  Coarse Threshold : {NCC_COARSE_THRESHOLD}")
    print(f"  Fine Threshold   : {NCC_FINE_THRESHOLD}")
    print(f"  DB Videos Dir    : {DB_VIDEOS_DIR.resolve()}")
    print("=" * 60)

    # 1. Get query path from user
    query_input = input("\nEnter the path to your query clip file: ").strip()
    query_input = query_input.strip('"\'')

    query_path = Path(query_input)
    if not query_path.exists() or not query_path.is_file():
        print(f"\n[ERROR] File not found at '{query_path}'")
        return

    size_kb = query_path.stat().st_size / 1024
    print(f"\n[OK] Query file: {query_path.name} ({size_kb:.0f} KB)")

    # 1.5 Get method from user
    print("\nSelect Matching Method:")
    print("  1. Standard (Full 256x256 Frame NCC)")
    print("  2. Faster (Diagonal pixels only, ~256x faster)")
    method_choice = input("Enter choice (1 or 2) [default=1]: ").strip()
    method = "diagonal" if method_choice == "2" else "standard"
    print(f"\n[OK] Selected method: {method.upper()}")

    # 2. Discover DB videos (no MongoDB needed!)
    ensure_directories()
    db_video_paths = discover_db_videos()

    if not db_video_paths:
        print(f"\n[ERROR] No video files found in '{DB_VIDEOS_DIR.resolve()}'")
        return

    print(f"[OK] Found {len(db_video_paths)} DB videos.")
    print("\nStarting matching engine...\n")

    # 3. Run live matching (all frame extraction happens at query time)
    start_time = time.time()
    result = match_query_clip_live(
        query_path, db_video_paths, coarse_fps=COARSE_FPS, method=method
    )
    elapsed = time.time() - start_time

    # 4. Print results
    print("\n" + "=" * 60)

    if result.matched:
        print("  MATCH FOUND!")
        print(f"  Video ID   : {result.video_id}")
        print(f"  File Name  : {result.video_filename}")
        print(f"  Timestamp  : {result.timestamp_str} (at {result.timestamp_sec:.2f}s)")
        print(f"  Confidence : {result.confidence:.4f}")
    else:
        print("  NO MATCH FOUND (above fine threshold)")
        print(f"  Coarse threshold: {NCC_COARSE_THRESHOLD}")
        print(f"  Fine threshold  : {NCC_FINE_THRESHOLD}")

        # Show best coarse scores for debugging
        if result.coarse_scores:
            print("\n  Best coarse scores per video:")
            sorted_scores = sorted(
                result.coarse_scores.items(),
                key=lambda x: x[1][0],
                reverse=True,
            )
            for vid, (score, ts) in sorted_scores[:5]:
                print(f"    {vid:25s}  NCC={score:.4f}  @ t={ts:.1f}s")

    print(f"\n  Time elapsed: {elapsed:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
