"""
services/matcher.py — Two-pass NCC video matching orchestration.

This module is the *only* place that knows about both frame extraction and NCC.
It exposes a single public function ``match_query_clip`` that:

  Pass 1 (Coarse):  Extract query frames @ 1 FPS, extract DB video frames @ 1 FPS
                    directly from the video files (live extraction, no disk I/O),
                    and identify candidate videos + rough timestamps.

  Pass 2 (Fine):    For each candidate above the coarse threshold, re-extract
                    query frames @ FINE_FPS, extract DB video frames from the
                    localized window @ FINE_FPS, and run sliding-window NCC to
                    pinpoint the best timestamp.

All I/O (MongoDB queries) is performed by the *caller* (the route handler) and
passed in as plain Python data structures — keeping this module pure and testable.

Return type: ``MatchResult`` dataclass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    COARSE_FPS,
    FINE_FPS,
    NCC_COARSE_THRESHOLD,
    NCC_FINE_THRESHOLD,
)
from services.frame_extractor import (
    extract_frames_at_fps,
    extract_frames_in_window,
    load_frame_from_disk,
)
from services.ncc import best_ncc_result, sliding_window_ncc

logger = logging.getLogger(__name__)

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """Outcome of a two-pass NCC match attempt."""
    matched: bool
    video_id: str = ""
    timestamp_sec: float = 0.0
    confidence: float = 0.0
    # Diagnostics: best score per video from coarse pass
    coarse_scores: dict = field(default_factory=dict)

    @property
    def timestamp_str(self) -> str:
        """Format timestamp as MM:SS."""
        minutes = int(self.timestamp_sec) // 60
        seconds = int(self.timestamp_sec) % 60
        return f"{minutes:02d}:{seconds:02d}"


# ── Type alias for MongoDB frame documents ────────────────────────────────────
# Each document: {video_id: str, frame_path: str, timestamp_sec: float, fps: int}
FrameDoc = dict[str, Any]


# ── Pass 1: Coarse search (live extraction) ───────────────────────────────────

def _coarse_search_live(
    query_frames: list[tuple[np.ndarray, float]],
    db_video_paths: dict[str, str | Path],
    coarse_fps: int = 1,
    method: str = "standard",
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, tuple[float, float]]]:
    """
    Live coarse search: extract frames from each DB video at coarse_fps
    and compare against query frames using sliding-window NCC.

    This avoids loading pre-saved frames from disk, which is slow for
    large databases. Instead, frames are extracted directly from video files.

    Args:
        query_frames: List of (frame, timestamp) from query clip at coarse_fps.
        db_video_paths: Mapping of video_id → path to video file.
        coarse_fps: FPS to extract DB frames at (default: 1).
        method:     'standard' or 'diagonal'.

    Returns:
        Tuple of:
          - candidates: video_id → list of (score, timestamp) above threshold
          - best_per_video: video_id → (best_score, best_timestamp) for diagnostics
    """
    query_frames_only = [f for f, _ in query_frames]
    n_query = len(query_frames_only)

    candidates: dict[str, list[tuple[float, float]]] = {}
    best_per_video: dict[str, tuple[float, float]] = {}

    total_videos = len(db_video_paths)

    for idx, (video_id, video_path) in enumerate(db_video_paths.items(), 1):
        logger.info(
            "Coarse [%d/%d]: extracting frames from '%s' @ %d FPS...",
            idx, total_videos, video_id, coarse_fps,
        )

        try:
            db_frames = extract_frames_at_fps(video_path, coarse_fps)
        except IOError as exc:
            logger.warning("Cannot open '%s': %s", video_path, exc)
            continue

        if len(db_frames) < n_query:
            logger.info(
                "Coarse [%d/%d]: '%s' has only %d frames (need %d query frames). Skipping.",
                idx, total_videos, video_id, len(db_frames), n_query,
            )
            continue

        logger.info(
            "Coarse [%d/%d]: comparing %d query frames vs %d DB frames (method=%s)...",
            idx, total_videos, n_query, len(db_frames), method,
        )

        scored = sliding_window_ncc(query_frames_only, db_frames, step=1, method=method)

        if scored:
            best_score, best_ts = max(scored, key=lambda x: x[0])
            best_per_video[video_id] = (best_score, best_ts)
            logger.info(
                "Coarse [%d/%d]: '%s' -> best NCC = %.4f @ t=%.1fs  (threshold=%.2f)",
                idx, total_videos, video_id, best_score, best_ts, NCC_COARSE_THRESHOLD,
            )

            above_threshold = [(s, t) for s, t in scored if s >= NCC_COARSE_THRESHOLD]
            if above_threshold:
                candidates[video_id] = above_threshold
                logger.info(
                    "Coarse HIT: '%s' — %d windows above threshold %.2f",
                    video_id, len(above_threshold), NCC_COARSE_THRESHOLD,
                )
        else:
            logger.info(
                "Coarse [%d/%d]: '%s' -> no valid windows.",
                idx, total_videos, video_id,
            )

    return candidates, best_per_video


# ── Pass 1 (legacy): Coarse search using pre-saved frames from disk ──────────

def _coarse_search(
    query_frames: list[np.ndarray],
    db_docs: list[FrameDoc],
) -> dict[str, list[tuple[float, float]]]:
    """
    Compare query frames against every DB document (pre-saved frames on disk).

    This is the legacy approach used by the API endpoint. For the CLI,
    use _coarse_search_live() instead for better performance.
    """
    from collections import defaultdict

    by_video: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)

    for doc in db_docs:
        try:
            frame = load_frame_from_disk(doc["frame_path"])
            by_video[doc["video_id"]].append((frame, float(doc["timestamp_sec"])))
        except (FileNotFoundError, IOError) as exc:
            logger.warning("Skipping frame '%s': %s", doc.get("frame_path"), exc)

    for vid in by_video:
        by_video[vid].sort(key=lambda x: x[1])

    candidates: dict[str, list[tuple[float, float]]] = {}

    for video_id, db_frames in by_video.items():
        scored = sliding_window_ncc(query_frames, db_frames, step=1)
        above_threshold = [(s, t) for s, t in scored if s >= NCC_COARSE_THRESHOLD]

        if above_threshold:
            candidates[video_id] = above_threshold

    return candidates


# ── Pass 2: Fine search ───────────────────────────────────────────────────────

def _fine_search(
    query_clip_path: str | Path,
    video_path: str | Path,
    candidate_windows: list[tuple[float, float]],
    window_padding_sec: float = 5.0,
    method: str = "standard",
) -> tuple[float, float] | None:
    """
    Perform high-FPS NCC over the localized candidate windows.

    Args:
        query_clip_path:   Path to the uploaded query video.
        video_path:        Path to the candidate DB video file.
        candidate_windows: List of (score, timestamp_sec) from coarse pass.
        window_padding_sec: Extra seconds added around each window boundary.
        method:             'standard' or 'diagonal'.

    Returns:
        (best_score, best_timestamp_sec) or None if no window exceeds NCC_FINE_THRESHOLD.
    """
    # Extract query frames at fine rate once
    query_frames_fine = extract_frames_at_fps(query_clip_path, FINE_FPS)
    query_frames_only = [f for f, _ in query_frames_fine]

    if not query_frames_only:
        logger.error("Fine search: no query frames extracted from '%s'", query_clip_path)
        return None

    overall_best: tuple[float, float] | None = None

    # Determine unique time windows to search (merge overlapping windows)
    timestamps = sorted({t for _, t in candidate_windows})

    for ts in timestamps:
        start = max(0.0, ts - window_padding_sec)
        end = ts + window_padding_sec

        db_frames_fine = extract_frames_in_window(video_path, start, end, FINE_FPS)
        if not db_frames_fine:
            continue

        scored = sliding_window_ncc(query_frames_only, db_frames_fine, step=1, method=method)
        best = best_ncc_result(scored)

        if best and best[0] >= NCC_FINE_THRESHOLD:
            if overall_best is None or best[0] > overall_best[0]:
                overall_best = best
                logger.info(
                    "Fine hit: video='%s', ts=%.2fs, score=%.4f",
                    video_path,
                    best[1],
                    best[0],
                )

    return overall_best


# ── Public API (live extraction — used by CLI) ────────────────────────────────

def match_query_clip_live(
    query_clip_path: str | Path,
    db_video_paths: dict[str, str | Path],
    coarse_fps: int = 1,
    method: str = "standard",
) -> MatchResult:
    """
    Run the two-pass NCC pipeline using live frame extraction.

    This function extracts frames directly from video files at query time,
    avoiding the need to load pre-saved frames from disk. Much faster and
    more reliable than the pre-saved approach.

    Args:
        query_clip_path: Path to the uploaded query clip file.
        db_video_paths:  Mapping of video_id → path to original DB video file.
        coarse_fps:      FPS for coarse extraction (default: 1).
        method:          'standard' or 'diagonal'.

    Returns:
        MatchResult with diagnostics.
    """
    logger.info("Starting LIVE coarse search for query: %s", query_clip_path)

    # ── Pass 1: Coarse (live extraction) ──────────────────────────────────────
    query_frames_coarse = extract_frames_at_fps(query_clip_path, coarse_fps)

    if not query_frames_coarse:
        logger.error("No frames extracted from query clip '%s'", query_clip_path)
        return MatchResult(matched=False)

    logger.info("Query clip: %d frames at %d FPS", len(query_frames_coarse), coarse_fps)

    candidates, best_per_video = _coarse_search_live(
        query_frames_coarse, db_video_paths, coarse_fps, method=method
    )

    # Log summary
    if best_per_video:
        sorted_videos = sorted(best_per_video.items(), key=lambda x: x[1][0], reverse=True)
        logger.info("=== Coarse Search Summary (top 5) ===")
        for vid, (score, ts) in sorted_videos[:5]:
            marker = " <-- CANDIDATE" if vid in candidates else ""
            logger.info("  %-25s  NCC=%.4f  @ t=%.1fs%s", vid, score, ts, marker)

    if not candidates:
        logger.info("Coarse search: no candidates found above threshold %.2f.", NCC_COARSE_THRESHOLD)
        return MatchResult(matched=False, coarse_scores=best_per_video)

    # ── Pass 2: Fine ──────────────────────────────────────────────────────────
    logger.info("Starting fine search over %d candidate video(s).", len(candidates))

    overall_best: tuple[float, float, str] | None = None  # (score, ts, video_id)

    for video_id, candidate_windows in candidates.items():
        video_path = db_video_paths.get(video_id)
        if not video_path:
            logger.warning("No video file path for video_id='%s'", video_id)
            continue

        fine_result = _fine_search(query_clip_path, video_path, candidate_windows, method=method)

        if fine_result:
            score, ts = fine_result
            if overall_best is None or score > overall_best[0]:
                overall_best = (score, ts, video_id)

    if overall_best:
        best_score, best_ts, best_vid = overall_best
        logger.info(
            "MATCH: video='%s', timestamp=%.2fs, confidence=%.4f",
            best_vid, best_ts, best_score,
        )
        return MatchResult(
            matched=True,
            video_id=best_vid,
            timestamp_sec=best_ts,
            confidence=round(best_score, 4),
            coarse_scores=best_per_video,
        )

    logger.info("Fine search: no match above threshold %.2f.", NCC_FINE_THRESHOLD)
    return MatchResult(matched=False, coarse_scores=best_per_video)


# ── Public API (pre-saved frames — used by FastAPI endpoint) ──────────────────

def match_query_clip(
    query_clip_path: str | Path,
    coarse_db_docs: list[FrameDoc],
    db_video_paths: dict[str, str | Path],
) -> MatchResult:
    """
    Run the two-pass NCC pipeline using pre-saved frames from disk.

    This is the original approach used by the FastAPI route handler.
    For CLI usage, prefer ``match_query_clip_live()`` instead.

    Args:
        query_clip_path:  Path to the uploaded query clip file.
        coarse_db_docs:   All MongoDB frame documents at COARSE_FPS.
        db_video_paths:   Mapping of video_id → path to original DB video file.

    Returns:
        MatchResult with ``matched=True`` on success.
    """
    logger.info("Starting coarse search for query: %s", query_clip_path)

    query_frames_coarse = [
        f for f, _ in extract_frames_at_fps(query_clip_path, COARSE_FPS)
    ]

    if not query_frames_coarse:
        logger.error("No frames extracted from query clip '%s'", query_clip_path)
        return MatchResult(matched=False)

    candidates = _coarse_search(query_frames_coarse, coarse_db_docs)

    if not candidates:
        logger.info("Coarse search: no candidates found.")
        return MatchResult(matched=False)

    logger.info("Starting fine search over %d candidate video(s).", len(candidates))

    overall_best: tuple[float, float, str] | None = None

    for video_id, candidate_windows in candidates.items():
        video_path = db_video_paths.get(video_id)
        if not video_path:
            logger.warning("No video file path registered for video_id='%s'", video_id)
            continue

        fine_result = _fine_search(query_clip_path, video_path, candidate_windows)

        if fine_result:
            score, ts = fine_result
            if overall_best is None or score > overall_best[0]:
                overall_best = (score, ts, video_id)

    if overall_best:
        best_score, best_ts, best_vid = overall_best
        logger.info(
            "Match found: video='%s', timestamp=%.2fs, confidence=%.4f",
            best_vid, best_ts, best_score,
        )
        return MatchResult(
            matched=True,
            video_id=best_vid,
            timestamp_sec=best_ts,
            confidence=round(best_score, 4),
        )

    logger.info("Fine search: no match above threshold.")
    return MatchResult(matched=False)
