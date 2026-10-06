# ClipMatch 🎬

**NCC-based video fingerprinting API** — Upload a short query clip and instantly find which stored video it came from and at exactly what timestamp.

---

## Architecture Overview

```
clipmatch/
├── main.py                   # FastAPI app factory + lifespan + health routes
├── config.py                 # Centralized .env config
├── database.py               # Motor async MongoDB client (lazy init)
├── ingest.py                 # Offline ingestion script (run once per new video)
├── routers/
│   └── match.py              # POST /match endpoint
├── services/
│   ├── frame_extractor.py    # OpenCV frame extraction & preprocessing
│   ├── ncc.py                # Pure NumPy NCC engine
│   └── matcher.py            # Two-pass coarse/fine NCC orchestration
├── .env                      # Environment configuration
├── requirements.txt
└── README.md
```

### Two-Pass NCC Matching

```
Query Clip Upload
      │
      ▼
┌─────────────────────────────────────┐
│  Pass 1 — Coarse Search (1 FPS)     │
│  • Extract query frames @ 1 FPS     │
│  • Load all DB frames @ 1 FPS       │
│  • Sliding-window NCC per video_id  │
│  • Threshold: NCC ≥ 0.75            │
└────────────────┬────────────────────┘
                 │ candidates (video_id + rough timestamp)
                 ▼
┌─────────────────────────────────────┐
│  Pass 2 — Fine Search (24 FPS)      │
│  • Extract query frames @ 24 FPS    │
│  • Extract DB window @ 24 FPS       │
│  • Sliding-window NCC over window   │
│  • Threshold: NCC ≥ 0.85            │
└────────────────┬────────────────────┘
                 │
                 ▼
         JSON Response
```

### Concurrency Design

The NCC computation is **CPU-bound**. The `match_endpoint` is declared `async def` but delegates heavy work to `asyncio.run_in_executor(None, ...)` which schedules it on Python's default `ThreadPoolExecutor`. This keeps the **ASGI event loop completely unblocked** during matching.

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- MongoDB running locally on port `27017`
- `ffmpeg` (optional, for video transcoding)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Edit `.env` to match your setup:

```dotenv
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=clipmatch

DB_VIDEOS_DIR=./db_videos
PROCESSED_FRAMES_DIR=./processed_frames
QUERY_UPLOAD_DIR=./tmp/queries

NCC_COARSE_THRESHOLD=0.75
NCC_FINE_THRESHOLD=0.85
COARSE_FPS=1
FINE_FPS=24
FRAME_SIZE=256
```

### 4. Add database videos

Place your video files in `./db_videos/`. The filename stem becomes the `video_id`:

```
db_videos/
  vid_001.mp4   →  video_id = "vid_001"
  vid_042.mp4   →  video_id = "vid_042"
```

### 5. Run ingestion (offline, one-time)

```bash
python ingest.py
```

**Options:**

| Flag | Description |
|------|-------------|
| `--videos-dir PATH` | Override `DB_VIDEOS_DIR` from `.env` |
| `--force` | Re-ingest videos already in MongoDB |

The script will:
- Extract frames at 1 FPS and 24 FPS for each video
- Save each frame as a 256×256 grayscale PNG to `processed_frames/<video_id>/<fps>fps/`
- Insert frame metadata documents into `MongoDB.clipmatch.frames`

### 6. Start the API server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or directly:

```bash
python main.py
```

---

## API Reference

### `POST /match`

Upload a query video clip for fingerprint matching.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | `UploadFile` | Query video clip (mp4, avi, mkv, mov, webm) |

**Response — Match Found (200):**

```json
{
  "status": "match_found",
  "video_id": "vid_042",
  "timestamp": "01:24",
  "confidence": 0.8823
}
```

**Response — No Match (200):**

```json
{
  "status": "no_match",
  "message": "Query clip not found in database."
}
```

**Response — DB Empty (503):**

```json
{
  "detail": "Database is empty. Please run ingest.py first."
}
```

### `GET /health`

Liveness probe.

```json
{
  "status": "ok",
  "database": "connected",
  "latency_ms": 1.23
}
```

### `GET /`

API info and endpoint listing.

### `GET /docs`

Interactive Swagger UI.

---

## Example — cURL

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/query_clip.mp4"
```

## Example — Python

```python
import requests

with open("query_clip.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/match",
        files={"file": ("query_clip.mp4", f, "video/mp4")},
    )

print(response.json())
# {'status': 'match_found', 'video_id': 'vid_042', 'timestamp': '01:24', 'confidence': 0.88}
```

---

## MongoDB Schema

**Collection: `frames`**

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | `str` | Video identifier (filename stem) |
| `frame_path` | `str` | Absolute path to the PNG frame on disk |
| `timestamp_sec` | `float` | Position in source video (seconds) |
| `fps` | `int` | Extraction rate (`1` = coarse, `24` = fine) |

**Indexes:**
- `(video_id, fps)` — compound index for targeted lookups
- `fps` — for filtering all coarse/fine docs

---

## Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `NCC_COARSE_THRESHOLD` | `0.75` | Lower → more candidates, slower fine pass |
| `NCC_FINE_THRESHOLD` | `0.85` | Lower → more false positives |
| `COARSE_FPS` | `1` | Higher → better coarse recall, more DB storage |
| `FINE_FPS` | `24` | Lower → faster but less precise timestamps |
| `FRAME_SIZE` | `256` | Higher → more accurate but slower NCC |
