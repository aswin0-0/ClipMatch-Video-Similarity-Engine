"""
services/ncc.py — Normalized Cross-Correlation (NCC) engine.

NCC formula (zero-mean):
    NCC(f, t) = Σ[(f - μ_f)(t - μ_t)] / (σ_f · σ_t · N)

A score of 1.0 means perfect correlation; 0.0 means no correlation.
All functions are pure NumPy — no OpenCV dependencies.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ── Core NCC function ─────────────────────────────────────────────────────────

def ncc_score(frame_a: np.ndarray, frame_b: np.ndarray, method: str = "standard") -> float:
    """
    Compute the zero-mean Normalized Cross-Correlation between two frames.

    Both frames must be float32 arrays of the same shape.

    Args:
        frame_a: Query frame (float32, already normalized to [0,1]).
        frame_b: Database frame (float32, already normalized to [0,1]).
        method:  'standard' for full image NCC, 'diagonal' for just the main diagonal pixels.

    Returns:
        NCC score in [0.0, 1.0]. Returns 0.0 if either frame is constant.
    """
    if method == "diagonal":
        a = np.diag(frame_a).astype(np.float64)
        b = np.diag(frame_b).astype(np.float64)
    else:
        a = frame_a.flatten().astype(np.float64)
        b = frame_b.flatten().astype(np.float64)

    a -= a.mean()
    b -= b.mean()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        # Constant (flat) frame — undefined NCC, treat as no match
        return 0.0

    score = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp to [0, 1]: we only care about positive correlation
    return max(0.0, score)


# ── Sequence-level NCC matching ───────────────────────────────────────────────

def score_query_against_db_frames(
    query_frames: list[np.ndarray],
    db_frames: list[np.ndarray],
    method: str = "standard",
) -> float:
    """
    Compute an aggregate NCC score between a query sequence and a DB frame window.

    Strategy: pair each query frame with the corresponding DB frame (positional
    alignment) and average the per-pair NCC scores.  The number of pairs used is
    min(len(query_frames), len(db_frames)).

    Args:
        query_frames: Ordered list of preprocessed query clip frames.
        db_frames:    Ordered list of preprocessed DB video frames (same window).
        method:       'standard' or 'diagonal'.

    Returns:
        Mean NCC score across all paired frames (float in [0, 1]).
    """
    if not query_frames or not db_frames:
        return 0.0

    n_pairs = min(len(query_frames), len(db_frames))
    scores = [
        ncc_score(query_frames[i], db_frames[i], method=method)
        for i in range(n_pairs)
    ]
    mean_score = float(np.mean(scores))
    logger.debug("Sequence NCC: %.4f over %d pairs", mean_score, n_pairs)
    return mean_score


def sliding_window_ncc(
    query_frames: list[np.ndarray],
    db_frames: list[tuple[np.ndarray, float]],
    step: int = 1,
    method: str = "standard",
) -> list[tuple[float, float]]:
    """
    Slide the query sequence across the DB frame list and compute NCC at each position.

    Args:
        query_frames:  Ordered list of preprocessed query frames.
        db_frames:     List of (frame, timestamp_sec) for the DB window.
        step:          Stride between sliding window positions (default: 1).
        method:        'standard' or 'diagonal'.

    Returns:
        List of (ncc_score, timestamp_sec) for each window position,
        where timestamp_sec is the timestamp of the first DB frame in the window.
        Sorted by timestamp.
    """
    n_query = len(query_frames)
    n_db = len(db_frames)

    if n_query == 0 or n_db < n_query:
        return []

    results: list[tuple[float, float]] = []

    for i in range(0, n_db - n_query + 1, step):
        window_frames = [db_frames[i + j][0] for j in range(n_query)]
        window_ts = db_frames[i][1]
        score = score_query_against_db_frames(query_frames, window_frames, method=method)
        results.append((score, window_ts))

    return results


def best_ncc_result(
    scored_windows: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """
    Return the (score, timestamp_sec) pair with the highest NCC score.

    Args:
        scored_windows: Output from ``sliding_window_ncc``.

    Returns:
        Best (score, timestamp_sec) or None if the list is empty.
    """
    if not scored_windows:
        return None
    return max(scored_windows, key=lambda x: x[0])
