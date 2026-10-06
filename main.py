"""
main.py — ClipMatch FastAPI application entry point.

Startup:
  - Creates required local directories (frames, uploads, db_videos).
  - Establishes the Motor (MongoDB) connection lazily.

Shutdown:
  - Closes the Motor client gracefully.

Routes:
  - GET  /          — Health check / API info
  - GET  /health    — Liveness probe (also verifies MongoDB connectivity)
  - POST /match     — Core video matching endpoint (see routers/match.py)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import MONGO_DB_NAME, MONGO_URI, ensure_directories
from database import close_client, get_database
from routers.match import router as match_router

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("clipmatch")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("ClipMatch starting up …")
    ensure_directories()
    logger.info("Directories ready. Connecting to MongoDB at %s …", MONGO_URI)

    # Trigger lazy client creation to surface connection errors early
    _ = get_database()
    logger.info("MongoDB connection initialised (db='%s').", MONGO_DB_NAME)

    yield  # ← Application runs here

    logger.info("ClipMatch shutting down …")
    await close_client()
    logger.info("MongoDB client closed.")


# ── Application factory ───────────────────────────────────────────────────────

app = FastAPI(
    title="ClipMatch",
    description=(
        "Video fingerprinting API that uses Normalized Cross-Correlation (NCC) "
        "to match a query clip against a database of stored videos and return "
        "the matching video ID and exact timestamp."
    ),
    version="1.0.0",
    contact={"name": "ClipMatch Team"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(match_router, prefix="")


# ── Built-in endpoints ────────────────────────────────────────────────────────

@app.get("/", tags=["Info"], summary="API root / welcome message")
async def root() -> JSONResponse:
    return JSONResponse(
        content={
            "service": "ClipMatch",
            "version": "1.0.0",
            "description": "NCC-based video fingerprinting API",
            "endpoints": {
                "match": "POST /match — Upload a query clip to find its match",
                "health": "GET  /health — Liveness & MongoDB connectivity check",
                "docs": "GET  /docs   — Interactive Swagger UI",
            },
        }
    )


@app.get("/health", tags=["Info"], summary="Liveness probe with MongoDB check")
async def health() -> JSONResponse:
    """
    Return HTTP 200 if the service is alive and MongoDB is reachable.
    Returns HTTP 503 if the database cannot be reached.
    """
    start = time.monotonic()
    try:
        db = get_database()
        await db.command("ping")
        latency_ms = round((time.monotonic() - start) * 1000, 2)
        return JSONResponse(
            content={
                "status": "ok",
                "database": "connected",
                "latency_ms": latency_ms,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Health check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "database": "unreachable",
                "detail": str(exc),
            },
        )


# ── Development runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
