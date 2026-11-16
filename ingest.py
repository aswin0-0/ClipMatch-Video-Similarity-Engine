
"""
ingest.py - Offline database ingestion script for ClipMatch.

Responsibilities:
  1. Scan DB_VIDEOS_DIR for source videos.
  2. Extract frames at COARSE_FPS (normally 1 FPS).
  3. Convert frames to grayscale and resize to FRAME_SIZE x FRAME_SIZE.
  4. Save frames as PNG files.
  5. Store frame metadata in MongoDB's "frames" collection.
  6. Track completed ingestion in "ingestion_status".

Fine frames are NOT stored. The matcher extracts fine frames directly
from the original video during the fine-matching stage.

Usage:
  python ingest.py [--videos-dir PATH] [--force]

  --videos-dir: Override DB_VIDEOS_DIR.
  --force:      Re-ingest videos even if already completed.
"""

from __future__ import annotations

import argparse
import logging
import shutil
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

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".webm",
    ".flv",
}


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale and resize to FRAME_SIZE x FRAME_SIZE."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return cv2.resize(
        gray,
        (FRAME_SIZE, FRAME_SIZE),
        interpolation=cv2.INTER_AREA,
    )


def _save_frame(frame: np.ndarray, destination: Path) -> bool:
    """Save a preprocessed grayscale frame as a PNG."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(destination), frame)

    if not success:
        logger.error("Failed to write frame: %s", destination)

    return success


def _ingest_video(
    video_path: Path,
    video_id: str,
    target_fps: int,
    collection: pymongo.collection.Collection,
) -> int:
    """
    Extract and save frames from one video.

    Frame metadata is inserted only after extraction finishes successfully.

    Returns:
        Number of successfully ingested frames.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if not source_fps or source_fps <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid source FPS reported for video: {video_path}"
        )

    frame_interval = max(1, round(source_fps / target_fps))

    frame_dir = (
        PROCESSED_FRAMES_DIR
        / video_id
        / f"{target_fps}fps"
    )

    # Remove frames from a previous incomplete attempt.
    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    documents: list[dict] = []

    frame_index = 0
    saved_count = 0

    logger.info(
        "Extracting at %d FPS "
        "(source FPS: %.2f, frame interval: %d): %s",
        target_fps,
        source_fps,
        frame_interval,
        video_path.name,
    )

    try:
        while True:
            success, raw_frame = cap.read()

            if not success:
                break

            if frame_index % frame_interval == 0:
                timestamp_sec = frame_index / source_fps

                processed_frame = _preprocess_frame(raw_frame)

                frame_filename = f"frame_{frame_index:08d}.png"
                frame_destination = frame_dir / frame_filename

                if not _save_frame(
                    processed_frame,
                    frame_destination,
                ):
                    raise RuntimeError(
                        f"Failed to save frame: {frame_destination}"
                    )

                documents.append(
                    {
                        "video_id": video_id,
                        "frame_path": str(
                            frame_destination.resolve()
                        ),
                        "timestamp_sec": round(timestamp_sec, 4),
                        "fps": target_fps,
                    }
                )

                saved_count += 1

            frame_index += 1

    finally:
        cap.release()

    if saved_count == 0:
        raise RuntimeError(
            f"No frames were extracted from: {video_path}"
        )

    # Insert metadata only after the complete extraction succeeds.
    collection.insert_many(documents, ordered=True)

    logger.info(
        "Inserted %d frame documents for video_id='%s'.",
        saved_count,
        video_id,
    )

    return saved_count


def _source_metadata(video_path: Path) -> dict:
    """Return source-file metadata used to detect changed videos."""
    stat = video_path.stat()

    return {
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def _is_ingestion_complete(
    video_id: str,
    video_path: Path,
    frames_collection: pymongo.collection.Collection,
    status_collection: pymongo.collection.Collection,
) -> bool:
    """
    Check whether a video has a completed ingestion record
    matching its current source-file metadata.
    """
    status = status_collection.find_one(
        {"_id": video_id}
    )

    if not status or status.get("status") != "completed":
        return False

    current_metadata = _source_metadata(video_path)

    if (
        status.get("source_size") != current_metadata["source_size"]
        or status.get("source_mtime_ns")
        != current_metadata["source_mtime_ns"]
    ):
        return False

    expected_count = status.get("frame_count", 0)

    actual_count = frames_collection.count_documents(
        {
            "video_id": video_id,
            "fps": COARSE_FPS,
        }
    )

    return expected_count > 0 and actual_count == expected_count


def ingest(videos_dir: Path, force: bool = False) -> None:
    """
    Ingest source videos into MongoDB.

    Only COARSE_FPS frames are persisted. Fine matching uses the
    original video files separately.
    """
    ensure_directories()

    videos_dir = videos_dir.resolve()

    if not videos_dir.exists() or not videos_dir.is_dir():
        raise FileNotFoundError(
            f"Videos directory does not exist: {videos_dir}"
        )

    client = pymongo.MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
    )

    try:
        # Verify MongoDB connectivity before starting.
        client.admin.command("ping")

        db = client[MONGO_DB_NAME]

        frames_collection = db["frames"]
        status_collection = db["ingestion_status"]

        # Indexes used by the matching and ingestion workflows.
        frames_collection.create_index(
            [
                ("video_id", pymongo.ASCENDING),
                ("fps", pymongo.ASCENDING),
            ]
        )

        frames_collection.create_index(
            [("fps", pymongo.ASCENDING)]
        )

        logger.info("MongoDB connection successful.")
        logger.info("MongoDB indexes ensured.")

        video_files = sorted(
            [
                path
                for path in videos_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in VIDEO_EXTENSIONS
            ]
        )

        if not video_files:
            logger.warning(
                "No supported video files found in: %s",
                videos_dir,
            )
            return

        logger.info(
            "Found %d source video(s).",
            len(video_files),
        )

        total_frames = 0
        completed_videos = 0
        skipped_videos = 0
        failed_videos = 0

        for video_path in video_files:
            video_id = video_path.stem

            logger.info(
                "Processing video: %s (video_id='%s')",
                video_path.name,
                video_id,
            )

            try:
                if not force and _is_ingestion_complete(
                    video_id,
                    video_path,
                    frames_collection,
                    status_collection,
                ):
                    logger.info(
                        "Skipping '%s': ingestion already complete.",
                        video_path.name,
                    )
                    skipped_videos += 1
                    continue

                # Remove old or incomplete metadata before retrying.
                deleted = frames_collection.delete_many(
                    {"video_id": video_id}
                ).deleted_count

                if deleted:
                    logger.info(
                        "Removed %d previous frame documents.",
                        deleted,
                    )

                status_collection.delete_one(
                    {"_id": video_id}
                )

                # Store only coarse frames.
                frame_count = _ingest_video(
                    video_path=video_path,
                    video_id=video_id,
                    target_fps=COARSE_FPS,
                    collection=frames_collection,
                )

                metadata = _source_metadata(video_path)

                # Mark completion only after frame insertion succeeds.
                status_collection.update_one(
                    {"_id": video_id},
                    {
                        "$set": {
                            "status": "completed",
                            "video_id": video_id,
                            "video_filename": video_path.name,
                            "fps": COARSE_FPS,
                            "frame_count": frame_count,
                            **metadata,
                        }
                    },
                    upsert=True,
                )

                total_frames += frame_count
                completed_videos += 1

                logger.info(
                    "Completed '%s': %d coarse frames.",
                    video_path.name,
                    frame_count,
                )

            except Exception:
                failed_videos += 1

                logger.exception(
                    "Failed to ingest video: %s",
                    video_path.name,
                )

        logger.info("=" * 55)
        logger.info("INGESTION SUMMARY")
        logger.info("Newly completed videos : %d", completed_videos)
        logger.info("Skipped videos         : %d", skipped_videos)
        logger.info("Failed videos          : %d", failed_videos)
        logger.info("New coarse frames      : %d", total_frames)
        logger.info("Stored FPS             : %d", COARSE_FPS)
        logger.info("Fine matching FPS      : %d (extracted live)", FINE_FPS)
        logger.info("=" * 55)

        if failed_videos:
            logger.warning(
                "Some videos failed. Run the script again to retry them."
            )

    finally:
        client.close()
        logger.info("MongoDB connection closed.")


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
        help=(
            "Directory containing source videos "
            f"(default: {DB_VIDEOS_DIR})"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Re-ingest videos even if already completed. "
            "Existing frame documents are deleted first."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    videos_dir = args.videos_dir.resolve()

    logger.info("=== ClipMatch Ingestion ===")
    logger.info("Videos directory : %s", videos_dir)
    logger.info(
        "Frames directory : %s",
        PROCESSED_FRAMES_DIR.resolve(),
    )
    logger.info("MongoDB database: %s", MONGO_DB_NAME)
    logger.info("Coarse FPS      : %d", COARSE_FPS)
    logger.info("Fine FPS        : %d (not stored)", FINE_FPS)
    logger.info(
        "Frame size      : %d x %d",
        FRAME_SIZE,
        FRAME_SIZE,
    )
    logger.info("Force re-ingest : %s", args.force)
    logger.info("=" * 55)

    try:
        ingest(videos_dir, force=args.force)
    except Exception:
        logger.exception("Ingestion process failed.")
        sys.exit(1)
