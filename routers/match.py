"""
routers/match.py — POST /match endpoint.

Responsibilities:
  1. Accept multipart/form-data video upload.
  2. Persist the query clip to QUERY_UPLOAD_DIR.
  3. Fetch coarse DB frame metadata from MongoDB (async).
  4. Build the db_video_paths mapping (video_id → file path).
  5. Delegate CPU-bound matching to the thread pool via run_in_executor,
     keeping the ASGI event loop unblocked.
  6. Clean up the temporary query file after matching.
  7. Return a structured JSON response.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from functools import partial
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config import COARSE_FPS, DB_VIDEOS_DIR, QUERY_UPLOAD_DIR
from database import get_database
from services.matcher import MatchResult, match_query_clip

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _save_upload(upload: UploadFile) -> Path:
    """
    Persist an uploaded file to QUERY_UPLOAD_DIR with a unique filename.

    Returns the absolute path to the saved file.
    """
    QUERY_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(upload.filename or "query.mp4").suffix or ".mp4"
    filename = f"{uuid.uuid4().hex}{suffix}"
    dest = QUERY_UPLOAD_DIR / filename

    content = await upload.read()
    dest.write_bytes(content)
    logger.info("Saved query clip to: %s (%d bytes)", dest, len(content))
    return dest


async def _fetch_coarse_docs() -> list[dict]:
    """
    Async: fetch all frame metadata documents ingested at COARSE_FPS from MongoDB.
    Returns a list of plain dicts (serialization-safe).
    """
    db = get_database()
    cursor = db["frames"].find(
        {"fps": COARSE_FPS},
        {"_id": 0, "video_id": 1, "frame_path": 1, "timestamp_sec": 1, "fps": 1},
    )
    return await cursor.to_list(length=None)


async def _fetch_db_video_paths() -> dict[str, Path]:
    """
    Async: fetch unique video_ids from MongoDB and map them to their
    expected file paths under DB_VIDEOS_DIR.

    Convention: video files are stored as ``<DB_VIDEOS_DIR>/<video_id>.<ext>``.
    We probe common video extensions to find the actual file.
    """
    db = get_database()
    video_ids: list[str] = await db["frames"].distinct("video_id")

    video_extensions = (".mp4", ".avi", ".mkv", ".mov", ".webm")
    mapping: dict[str, Path] = {}

    for vid in video_ids:
        for ext in video_extensions:
            candidate = DB_VIDEOS_DIR / f"{vid}{ext}"
            if candidate.exists():
                mapping[vid] = candidate
                break
        else:
            logger.warning(
                "No video file found for video_id='%s' in '%s'", vid, DB_VIDEOS_DIR
            )

    return mapping


def _cleanup(path: Path) -> None:
    """Silently remove the temporary query file."""
    try:
        path.unlink(missing_ok=True)
        logger.debug("Cleaned up temporary file: %s", path)
    except OSError as exc:
        logger.warning("Could not remove temp file '%s': %s", path, exc)


def _format_response(result: MatchResult) -> JSONResponse:
    """Serialize a MatchResult into the API response schema."""
    if result.matched:
        return JSONResponse(
            status_code=200,
            content={
                "status": "match_found",
                "video_id": result.video_id,
                "timestamp": result.timestamp_str,
                "start_timestamp": result.start_timestamp_str,
                "end_timestamp": result.end_timestamp_str,
                "start_timestamp_sec": result.start_timestamp_sec,
                "end_timestamp_sec": result.end_timestamp_sec,
                "confidence": result.confidence,
                "orientation": result.match_orientation,
            },
        )
    return JSONResponse(
        status_code=200,
        content={
            "status": "no_match",
            "message": "Query clip not found in database.",
        },
    )


async def _run_matching(
    query_path: Path,
    coarse_docs: list[dict],
    db_video_paths: dict[str, Path],
) -> MatchResult:
    """
    Run the synchronous ``match_query_clip`` in the default thread-pool executor
    so it does not block the asyncio event loop.
    """
    loop = asyncio.get_event_loop()
    fn = partial(match_query_clip, query_path, coarse_docs, db_video_paths)
    return await loop.run_in_executor(None, fn)


# ── Route handler ─────────────────────────────────────────────────────────────

@router.post(
    "/match",
    summary="Match a query video clip against the database",
    response_description="Match result with video_id, timestamp, and confidence",
    tags=["Matching"],
)
async def match_endpoint(
    file: UploadFile = File(..., description="Query video clip to match"),
) -> JSONResponse:
    """
    Upload a query video clip and find its best match in the database.

    - **Pass 1 (Coarse)**: 1-FPS NCC across all ingested DB frames.
    - **Pass 2 (Fine)**: 24-FPS sliding-window NCC within the candidate window.

    Returns JSON with `status`, `video_id`, `timestamp` (MM:SS), and `confidence`.
    """
    # ── Validate content type ─────────────────────────────────────────────────
    content_type = file.content_type or ""
    if not (
        content_type.startswith("video/")
        or content_type == "application/octet-stream"
    ):
        logger.warning("Rejected upload with content_type='%s'", content_type)
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type. Please upload a video file.",
        )

    query_path: Path | None = None
    try:
        # ── 1. Save upload ────────────────────────────────────────────────────
        query_path = await _save_upload(file)

        # ── 2. Fetch DB metadata (async, non-blocking) ────────────────────────
        coarse_docs = await _fetch_coarse_docs()
        if not coarse_docs:
            raise HTTPException(
                status_code=503,
                detail="Database is empty. Please run ingest.py first.",
            )

        db_video_paths = await _fetch_db_video_paths()

        # ── 3. CPU-bound matching (offloaded to thread pool) ──────────────────
        result: MatchResult = await _run_matching(query_path, coarse_docs, db_video_paths)

        return _format_response(result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during matching: %s", exc)
        raise HTTPException(status_code=500, detail="Internal matching error.") from exc
    finally:
        if query_path:
            _cleanup(query_path)
