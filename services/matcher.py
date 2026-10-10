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
    ENABLE_MIRROR_MATCHING,
    ENABLE_PATCH_EARLY_REJECTION,
    FINE_FPS,
    FINE_SEARCH_WINDOW_SEC,
    NCC_COARSE_THRESHOLD,
    NCC_FINE_THRESHOLD,
    PATCH_GRID_SIZE,
    PATCH_MIN_PASS_RATIO,
    PATCH_SCORE_THRESHOLD,
    WINDOW_PADDING_SEC,
)
from services.frame_extractor import (
    extract_frames_at_fps,
    extract_frames_in_window,
    get_video_duration_sec,
    load_frame_from_disk,
)
from services.ncc import best_ncc_result, sliding_window_ncc, sliding_window_ncc_oriented

logger = logging.getLogger(__name__)

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """Outcome of a two-pass NCC match attempt."""
    matched: bool
    video_id: str = ""
    video_filename: str = ""
    timestamp_sec: float = 0.0
    start_timestamp_sec: float | None = None
    end_timestamp_sec: float | None = None
    confidence: float = 0.0
    match_orientation: str = "normal"
    # Diagnostics: best score per video from coarse pass
    coarse_scores: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_timestamp_sec is None:
            self.start_timestamp_sec = self.timestamp_sec
        else:
            self.timestamp_sec = self.start_timestamp_sec
        if self.end_timestamp_sec is None:
            self.end_timestamp_sec = self.start_timestamp_sec

    @property
    def timestamp_str(self) -> str:
        """Backward-compatible alias for the formatted start timestamp."""
        return self.start_timestamp_str

    @staticmethod
    def _format_timestamp(timestamp_sec: float) -> str:
        minutes = int(timestamp_sec) // 60
        seconds = int(timestamp_sec) % 60
        return f"{minutes:02d}:{seconds:02d}"

    @property
    def start_timestamp_str(self) -> str:
        """Format the match start timestamp as MM:SS."""
        return self._format_timestamp(self.start_timestamp_sec or 0.0)

    @property
    def end_timestamp_str(self) -> str:
        """Format the exclusive match end timestamp as MM:SS."""
        return self._format_timestamp(self.end_timestamp_sec or 0.0)


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

        scored = sliding_window_ncc(
            query_frames_only,
            db_frames,
            step=1,
            method=method,
            enable_patch_early_rejection=ENABLE_PATCH_EARLY_REJECTION,
            patch_grid_size=PATCH_GRID_SIZE,
            patch_score_threshold=PATCH_SCORE_THRESHOLD,
            patch_min_pass_ratio=PATCH_MIN_PASS_RATIO,
        )

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
        scored = sliding_window_ncc(
            query_frames,
            db_frames,
            step=1,
            enable_patch_early_rejection=ENABLE_PATCH_EARLY_REJECTION,
            patch_grid_size=PATCH_GRID_SIZE,
            patch_score_threshold=PATCH_SCORE_THRESHOLD,
            patch_min_pass_ratio=PATCH_MIN_PASS_RATIO,
        )
        above_threshold = [(s, t) for s, t in scored if s >= NCC_COARSE_THRESHOLD]

        if above_threshold:
            candidates[video_id] = above_threshold

    return candidates


# ── Pass 2: Fine search ───────────────────────────────────────────────────────

def _fine_search(
    query_clip_path: str | Path,
    video_path: str | Path,
    candidate_windows: list[tuple[float, float]],
    query_duration_sec: float,
    window_padding_sec: float = WINDOW_PADDING_SEC,
    method: str = "standard",
    query_frames_fine: list[tuple[np.ndarray, float]] | None = None,
    mirror_matching: bool = ENABLE_MIRROR_MATCHING,
) -> tuple[float, float, str] | None:
    """
    Perform high-FPS NCC over the localized candidate windows.

    Args:
        query_clip_path:   Path to the uploaded query video.
        video_path:        Path to the candidate DB video file.
        candidate_windows: List of (score, timestamp_sec) from coarse pass.
        window_padding_sec: Extra seconds added around each window boundary.
        method:             'standard' or 'diagonal'.
        query_frames_fine:  Optional pre-extracted query frames for reuse.
        mirror_matching:    Compare normal and horizontally mirrored frames.

    Returns:
        (best_score, best_timestamp_sec, orientation) or None if no window
        exceeds NCC_FINE_THRESHOLD.
    """
    if query_frames_fine is None:
        query_frames_fine = extract_frames_at_fps(query_clip_path, FINE_FPS)
    query_frames_only = [f for f, _ in query_frames_fine]

    if not query_frames_only:
        logger.error("Fine search: no query frames extracted from '%s'", query_clip_path)
        return None

    overall_best: tuple[float, float, str] | None = None
    absolute_best_score = 0.0

    try:
        video_duration_sec = get_video_duration_sec(video_path)
    except IOError:
        video_duration_sec = 0.0

    search_duration = max(query_duration_sec, FINE_SEARCH_WINDOW_SEC)
    if video_duration_sec > 0:
        search_duration = min(search_duration, video_duration_sec)

    raw_windows = [
        (
            max(0.0, timestamp - window_padding_sec),
            max(0.0, timestamp - window_padding_sec) + search_duration,
        )
        for _, timestamp in candidate_windows
    ]
    raw_windows.sort()
    windows: list[tuple[float, float]] = []
    for start, end in raw_windows:
        if video_duration_sec > 0:
            end = min(video_duration_sec, end)
            start = min(start, end)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))

    for start, end in windows:
        db_frames_fine = extract_frames_in_window(video_path, start, end, FINE_FPS)
        if not db_frames_fine:
            continue

        scored = sliding_window_ncc_oriented(
            query_frames_only,
            db_frames_fine,
            step=1,
            method=method,
            mirror_matching=mirror_matching,
            enable_patch_early_rejection=ENABLE_PATCH_EARLY_REJECTION,
            patch_grid_size=PATCH_GRID_SIZE,
            patch_score_threshold=PATCH_SCORE_THRESHOLD,
            patch_min_pass_ratio=PATCH_MIN_PASS_RATIO,
        )
        best = max(scored, key=lambda item: item[0], default=None)

        if best:
            if best[0] > absolute_best_score:
                absolute_best_score = best[0]
            
            if best[0] >= NCC_FINE_THRESHOLD:
                if overall_best is None or best[0] > overall_best[0]:
                    overall_best = best
                    logger.info(
                        "Fine hit: video='%s', ts=%.2fs, score=%.4f, orientation=%s",
                        video_path,
                        best[1],
                        best[0],
                        best[2],
                    )

    if overall_best is None:
        logger.info(
            "Fine miss: video='%s', best score was %.4f (threshold=%.2f)",
            video_path,
            absolute_best_score,
            NCC_FINE_THRESHOLD,
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
    logger.info("Starting LIVE search (early stopping enabled) for query: %s", query_clip_path)

    query_frames_coarse = extract_frames_at_fps(query_clip_path, coarse_fps)
    if not query_frames_coarse:
        logger.error("No frames extracted from query clip '%s'", query_clip_path)
        return MatchResult(matched=False)

    logger.info("Query clip: %d frames at %d FPS", len(query_frames_coarse), coarse_fps)
    query_frames_only = [f for f, _ in query_frames_coarse]
    n_query = len(query_frames_only)
    try:
        query_duration_sec = get_video_duration_sec(query_clip_path)
    except IOError:
        query_duration_sec = (
            query_frames_coarse[-1][1] - query_frames_coarse[0][1] + 1 / coarse_fps
        )
    query_frames_fine = extract_frames_at_fps(query_clip_path, FINE_FPS)

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
            logger.info("Coarse [%d/%d]: '%s' has only %d frames. Skipping.", idx, total_videos, video_id, len(db_frames))
            continue

        logger.info("Coarse [%d/%d]: comparing %d query vs %d DB frames (method=%s)...", idx, total_videos, n_query, len(db_frames), method)
        scored = sliding_window_ncc(query_frames_only, db_frames, step=1, method=method)

        if not scored:
            continue

        best_score, best_ts = max(scored, key=lambda x: x[0])
        best_per_video[video_id] = (best_score, best_ts)
        
        above_threshold = [(s, t) for s, t in scored if s >= NCC_COARSE_THRESHOLD]
        if above_threshold:
            logger.info("Coarse HIT: '%s' — %d windows above threshold %.2f", video_id, len(above_threshold), NCC_COARSE_THRESHOLD)
            
            logger.info("Starting fine search for candidate '%s'...", video_id)
            fine_result = _fine_search(
                query_clip_path,
                video_path,
                above_threshold,
                query_duration_sec,
                method=method,
                query_frames_fine=query_frames_fine,
                mirror_matching=ENABLE_MIRROR_MATCHING,
            )
            
            if fine_result:
                fine_score, fine_ts, orientation = fine_result
                logger.info("EARLY STOP MATCH FOUND: video='%s', timestamp=%.2fs, confidence=%.4f", video_id, fine_ts, fine_score)
                return MatchResult(
                    matched=True,
                    video_id=video_id,
                    video_filename=Path(video_path).name,
                    timestamp_sec=fine_ts,
                    start_timestamp_sec=fine_ts,
                    end_timestamp_sec=fine_ts + query_duration_sec,
                    confidence=round(fine_score, 4),
                    match_orientation=orientation,
                    coarse_scores=best_per_video,
                )
        else:
            logger.info("Coarse miss: '%s' best NCC = %.4f", video_id, best_score)

    logger.info("Search complete: no match found across all %d videos.", total_videos)
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

    query_frames_with_ts = extract_frames_at_fps(query_clip_path, COARSE_FPS)
    query_frames_coarse = [f for f, _ in query_frames_with_ts]

    if not query_frames_coarse:
        logger.error("No frames extracted from query clip '%s'", query_clip_path)
        return MatchResult(matched=False)

    candidates = _coarse_search(query_frames_coarse, coarse_db_docs)

    if not candidates:
        logger.info("Coarse search: no candidates found.")
        return MatchResult(matched=False)

    logger.info("Starting fine search over %d candidate video(s).", len(candidates))

    try:
        query_duration_sec = get_video_duration_sec(query_clip_path)
    except IOError:
        query_duration_sec = (
            query_frames_with_ts[-1][1] - query_frames_with_ts[0][1] + 1 / COARSE_FPS
        )
    query_frames_fine = extract_frames_at_fps(query_clip_path, FINE_FPS)

    overall_best: tuple[float, float, str, str] | None = None

    for video_id, candidate_windows in candidates.items():
        video_path = db_video_paths.get(video_id)
        if not video_path:
            logger.warning("No video file path registered for video_id='%s'", video_id)
            continue

        fine_result = _fine_search(
            query_clip_path,
            video_path,
            candidate_windows,
            query_duration_sec,
            query_frames_fine=query_frames_fine,
            mirror_matching=ENABLE_MIRROR_MATCHING,
        )

        if fine_result:
            score, ts, orientation = fine_result
            if overall_best is None or score > overall_best[0]:
                overall_best = (score, ts, video_id, orientation)

    if overall_best:
        best_score, best_ts, best_vid, orientation = overall_best
        logger.info(
            "Match found: video='%s', timestamp=%.2fs, confidence=%.4f",
            best_vid, best_ts, best_score,
        )
        return MatchResult(
            matched=True,
            video_id=best_vid,
            timestamp_sec=best_ts,
            start_timestamp_sec=best_ts,
            end_timestamp_sec=best_ts + query_duration_sec,
            confidence=round(best_score, 4),
            match_orientation=orientation,
        )

    logger.info("Fine search: no match above threshold.")
    return MatchResult(matched=False)
