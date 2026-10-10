"""
services/frame_extractor.py — OpenCV-based frame extraction utilities.

All functions are pure, synchronous, and CPU-bound (intentional).
FastAPI will execute these in a thread pool via `def` endpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from config import FRAME_SIZE

logger = logging.getLogger(__name__)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert a raw BGR frame to a normalized float32 grayscale image.

    Steps:
      1. Convert BGR → Grayscale
      2. Resize to FRAME_SIZE × FRAME_SIZE
      3. Cast to float32 and normalize to [0, 1]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def _open_capture(video_path: str | Path) -> cv2.VideoCapture:
    """Open a VideoCapture and raise if the file cannot be read."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    return cap


def get_video_duration_sec(video_path: str | Path) -> float:
    """Return the source duration in seconds using OpenCV metadata."""
    cap = _open_capture(video_path)
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        return frame_count / source_fps if source_fps > 0 else 0.0
    finally:
        cap.release()


# ── Public frame-extraction API ───────────────────────────────────────────────

def extract_frames_at_fps(
    video_path: str | Path,
    target_fps: int,
) -> list[tuple[np.ndarray, float]]:
    """
    Extract and preprocess frames from *video_path* at *target_fps*.

    Returns a list of ``(preprocessed_frame, timestamp_sec)`` tuples.
    Only frames whose position aligns with the target interval are returned.

    Args:
        video_path:  Path to the source video file.
        target_fps:  Desired extraction rate (frames per second).

    Returns:
        List of (frame_array, timestamp_sec) sorted by timestamp.
    """
    cap = _open_capture(video_path)
    try:
        source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval: int = max(1, round(source_fps / target_fps))

        frames: list[tuple[np.ndarray, float]] = []
        frame_index: int = 0

        while True:
            ret, raw = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                timestamp_sec: float = frame_index / source_fps
                processed = _preprocess_frame(raw)
                frames.append((processed, timestamp_sec))

            frame_index += 1

        logger.debug(
            "Extracted %d frames from '%s' @ %d FPS (source %.1f FPS)",
            len(frames),
            video_path,
            target_fps,
            source_fps,
        )
        return frames
    finally:
        cap.release()


def extract_frames_in_window(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    target_fps: int,
) -> list[tuple[np.ndarray, float]]:
    """
    Extract frames from *video_path* within [start_sec, end_sec] at *target_fps*.

    Used by the fine-search pass to narrow the search window.

    Args:
        video_path:  Path to the DB video file.
        start_sec:   Window start (seconds).
        end_sec:     Window end   (seconds).
        target_fps:  Desired extraction rate.

    Returns:
        List of (frame_array, timestamp_sec) within the window.
    """
    cap = _open_capture(video_path)
    try:
        source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval: int = max(1, round(source_fps / target_fps))

        start_frame = int(start_sec * source_fps)
        end_frame = int(end_sec * source_fps)

        # Seek directly to the start frame for efficiency
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: list[tuple[np.ndarray, float]] = []
        frame_index: int = start_frame

        while frame_index <= end_frame:
            ret, raw = cap.read()
            if not ret:
                break

            if (frame_index - start_frame) % frame_interval == 0:
                timestamp_sec: float = frame_index / source_fps
                processed = _preprocess_frame(raw)
                frames.append((processed, timestamp_sec))

            frame_index += 1

        logger.debug(
            "Window extraction: %d frames from '%s' [%.1fs – %.1fs] @ %d FPS",
            len(frames),
            video_path,
            start_sec,
            end_sec,
            target_fps,
        )
        return frames
    finally:
        cap.release()


def load_frame_from_disk(frame_path: str | Path) -> np.ndarray:
    """
    Load a preprocessed frame that was saved to disk during ingestion.

    The image is stored as a uint8 PNG; we reload and re-normalize to float32.

    Args:
        frame_path: Absolute path to the PNG frame file.

    Returns:
        Normalized float32 ndarray of shape (FRAME_SIZE, FRAME_SIZE).

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError:           If OpenCV cannot decode the image.
    """
    p = Path(frame_path)
    if not p.exists():
        raise FileNotFoundError(f"Frame file not found: {p}")

    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot decode frame image: {p}")

    return img.astype(np.float32) / 255.0
