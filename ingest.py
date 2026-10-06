"""
ingest.py — Offline database ingestion script for ClipMatch.

Run this script once (or whenever new videos are added) to:
  1. Scan DB_VIDEOS_DIR for video files.
  2. Extract frames at COARSE_FPS (1 FPS) and FINE_FPS (24 FPS).
  3. Save each frame as a lossless PNG to PROCESSED_FRAMES_DIR.
  4. Store frame metadata in MongoDB (collection: "frames").

MongoDB document schema:
  {
    "video_id":      str,    # Stem of the video filename (e.g. "vid_042")
    "frame_path":    str,    # Absolute path to the saved PNG file
    "timestamp_sec": float,  # Position in the source video (seconds)
    "fps":           int,    # Extraction rate used (1 or 24)
  }

Usage:
  python ingest.py [--videos-dir PATH] [--force]

  --videos-dir: Override DB_VIDEOS_DIR from .env (optional)
  --force:      Re-ingest videos that already exist in MongoDB (default: skip)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pymongo

from config import (
    COARSE_FPS,
    DB_VIDEOS_DIR,
    FINE_FPS,
    FRAME_SIZE,
    MONGO_DB_NAME,
    MONGO_URI,
    PROCESSED_FRAMES_DIR,
    ensure_directories,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}


# ── Frame preprocessing ───────────────────────────────────────────────────────

def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """BGR → grayscale → resize to FRAME_SIZE × FRAME_SIZE → uint8 [0, 255]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)


def _save_frame(frame: np.ndarray, dest: Path) -> bool:
    """Save a preprocessed uint8 grayscale frame as PNG. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dest), frame)
    if not ok:
        logger.error("Failed to write frame to '%s'", dest)
    return ok


# ── Per-video ingestion ───────────────────────────────────────────────────────

def _ingest_video(
    video_path: Path,
    video_id: str,
    target_fps: int,
    collection: pymongo.collection.Collection,
) -> int:
    """
    Extract frames from *video_path* at *target_fps*, save PNGs, and insert
    metadata documents into *collection*.

    Returns the number of frames successfully ingested.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return 0

    source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval: int = max(1, round(source_fps / target_fps))

    frame_dir = PROCESSED_FRAMES_DIR / video_id / f"{target_fps}fps"
    frame_dir.mkdir(parents=True, exist_ok=True)

    docs: list[dict] = []
    frame_index = 0
    saved_count = 0

    logger.info(
        "  Extracting @ %d FPS (source %.1f FPS, interval=%d) from '%s'",
        target_fps,
        source_fps,
        frame_interval,
        video_path.name,
    )

    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                timestamp_sec = frame_index / source_fps
                processed = _preprocess_frame(raw)

                frame_filename = f"frame_{frame_index:08d}.png"
                frame_dest = frame_dir / frame_filename

                if _save_frame(processed, frame_dest):
                    docs.append(
                        {
                            "video_id": video_id,
                            "frame_path": str(frame_dest.resolve()),
                            "timestamp_sec": round(timestamp_sec, 4),
                            "fps": target_fps,
                        }
                    )
                    saved_count += 1

            frame_index += 1
    finally:
        cap.release()

    if docs:
        collection.insert_many(docs, ordered=False)
        logger.info(
            "  Inserted %d frame docs @ %d FPS for video_id='%s'",
            len(docs),
            target_fps,
            video_id,
        )

    return saved_count


# ── Main ingestion flow ───────────────────────────────────────────────────────

def ingest(videos_dir: Path, force: bool = False) -> None:
    """
    Scan *videos_dir* for video files and ingest each one into MongoDB.

    Args:
        videos_dir: Directory containing raw DB video files.
        force:      If True, re-ingest even if the video_id already exists in MongoDB.
    """
    ensure_directories()

    # ── MongoDB setup ─────────────────────────────────────────────────────────
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    frames_col = db["frames"]

    # Create indexes for efficient querying
    frames_col.create_index([("video_id", pymongo.ASCENDING), ("fps", pymongo.ASCENDING)])
    frames_col.create_index([("fps", pymongo.ASCENDING)])
    logger.info("MongoDB indexes ensured on 'frames' collection.")

    # ── Discover videos ───────────────────────────────────────────────────────
    video_files = [
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        logger.warning("No video files found in '%s'. Nothing to ingest.", videos_dir)
        client.close()
        return

    logger.info("Found %d video(s) in '%s'.", len(video_files), videos_dir)

    total_frames = 0

    for video_path in sorted(video_files):
        video_id = video_path.stem  # e.g. "vid_042" from "vid_042.mp4"

        # ── Skip if already ingested (unless --force) ─────────────────────────
        if not force:
            existing = frames_col.count_documents({"video_id": video_id})
            if existing > 0:
                logger.info(
                    "Skipping '%s' (video_id='%s', %d docs exist). Use --force to re-ingest.",
                    video_path.name,
                    video_id,
                    existing,
                )
                continue
        else:
            # Remove stale documents for this video
            deleted = frames_col.delete_many({"video_id": video_id}).deleted_count
            if deleted:
                logger.info(
                    "  Removed %d stale docs for video_id='%s'.", deleted, video_id
                )

        logger.info("Ingesting '%s' (video_id='%s') …", video_path.name, video_id)

        # Ingest at both coarse and fine FPS rates
        for fps in (COARSE_FPS, FINE_FPS):
            n = _ingest_video(video_path, video_id, fps, frames_col)
            total_frames += n

    logger.info("Ingestion complete. Total frames stored: %d.", total_frames)
    client.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ClipMatch offline video ingestion script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=DB_VIDEOS_DIR,
        help=f"Directory containing source video files (default: {DB_VIDEOS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-ingest videos already present in MongoDB (deletes existing docs first).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    videos_dir: Path = args.videos_dir.resolve()
    if not videos_dir.exists():
        logger.error("Videos directory does not exist: %s", videos_dir)
        sys.exit(1)

    logger.info("=== ClipMatch Ingestion ===")
    logger.info("Videos dir    : %s", videos_dir)
    logger.info("Frames dir    : %s", PROCESSED_FRAMES_DIR.resolve())
    logger.info("MongoDB URI   : %s", MONGO_URI)
    logger.info("Coarse FPS    : %d", COARSE_FPS)
    logger.info("Fine FPS      : %d", FINE_FPS)
    logger.info("Frame size    : %dx%d", FRAME_SIZE, FRAME_SIZE)
    logger.info("Force re-ingest: %s", args.force)
    logger.info("=" * 40)

    ingest(videos_dir, force=args.force)
