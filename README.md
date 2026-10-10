# ClipMatch — Video Similarity Engine

ClipMatch is a fast, two-pass video matching engine that finds the exact timestamp of a short query clip within a large database of reference videos. It uses **Normalized Cross-Correlation (NCC)** on grayscale frames to determine similarity.

## System Logic

The system is designed to be highly accurate while minimizing unnecessary computation. It operates in two main phases:

### 1. Ingestion (Offline)
Before matching, the reference videos in your database must be ingested.
- The `ingest.py` script scans the `db_videos/` directory.
- It extracts frames at two rates: **Coarse (1 FPS)** and **Fine (24 FPS)**.
- Frames are preprocessed (grayscale, resized to 256x256, normalized) and saved as PNGs.
- Metadata (video ID, timestamp, frame paths) is pushed to a **MongoDB** database.

### 2. Matching (Online)
When a query clip is provided, the matching pipeline runs a **Two-Pass NCC** algorithm:

- **Pass 1: Coarse Search (Live Extraction)**
  - Frames are extracted live from the reference video files at 1 FPS and compared against the query clip at 1 FPS.
  - **Optimization:** You can choose the **Faster (Diagonal)** method, which computes NCC *only* on the diagonal pixels of the frames (256 pixels instead of 65,536). This provides a massive speedup (~256x faster) while still filtering out completely unrelated videos.
  - Windows of time that score above `NCC_COARSE_THRESHOLD` are flagged as candidates.

- **Pass 2: Fine Search**
  - For candidate windows found in Pass 1, the system extracts frames at 24 FPS from both the query clip and the localized 5-second window of the reference video.
  - A full 256x256 sliding-window NCC is performed.
  - The window with the highest score above `NCC_FINE_THRESHOLD` is returned as the exact match.

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+
- A MongoDB cluster (e.g., MongoDB Atlas)

### 2. Install Dependencies
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory (the system will use sensible defaults if not provided, but you must configure your MongoDB URI):

```dotenv
# MongoDB Configuration
MONGO_URI=mongodb+srv://<username>:<password>@cluster0...
MONGO_DB_NAME=clipmatch

# Path Configuration
DB_VIDEOS_DIR=./db_videos
PROCESSED_FRAMES_DIR=./processed_frames
QUERY_UPLOAD_DIR=./tmp/queries

# Matching Configuration
NCC_COARSE_THRESHOLD=0.45
NCC_FINE_THRESHOLD=0.60
COARSE_FPS=1
FINE_FPS=24
FRAME_SIZE=256
FINE_SEARCH_WINDOW_SEC=20
WINDOW_PADDING_SEC=5
ENABLE_MIRROR_MATCHING=false
STORE_FINE_FRAMES=false
```

Fine frames are extracted live during matching by default. Set
`STORE_FINE_FRAMES=true` only if persisted fine-frame PNGs are required for
offline analysis.

---

## How to Run

### Step 1: Ingest Database Videos
Place your reference video files (`.mp4`, `.mkv`, etc.) into the `./db_videos/` directory, then run the ingestion script:
```bash
python ingest.py
```
*(To forcefully re-ingest videos that are already in the DB, run `python ingest.py --force`)*

### Step 2: Match a Query Clip

**Option A: Command Line Interface (Recommended for local testing)**
Run the CLI to match a clip directly from your terminal. It bypasses the web server and runs the logic locally.
```bash
python cli.py
```
You will be prompted to enter the path to your query clip and choose between the **Standard** (Full NCC) or **Faster** (Diagonal NCC) matching methods.

**Option B: FastAPI Web Server**
Start the web server to expose the matching engine as a REST API.
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
You can then send a `multipart/form-data` POST request containing the video file to `http://localhost:8000/match`.
