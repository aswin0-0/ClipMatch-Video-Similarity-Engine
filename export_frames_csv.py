import csv
from pathlib import Path

import cv2

from config import (
    COARSE_FPS,
    DB_VIDEOS_DIR,
    FINE_FPS,
    NCC_COARSE_THRESHOLD,
    NCC_FINE_THRESHOLD,
    STORE_FINE_FRAMES,
)

output_file = Path("database_videos_metadata.csv")
video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}

fieldnames = [
    "video_id",
    "video_name",
    "video_path",
    "file_extension",
    "file_size_bytes",
    "width",
    "height",
    "source_fps",
    "frame_count",
    "duration_seconds",
    "storage_fps",
    "extraction_type",
    "persisted",
    "threshold",
]

with output_file.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for video_path in sorted(DB_VIDEOS_DIR.iterdir()):
        if not video_path.is_file() or video_path.suffix.lower() not in video_extensions:
            continue

        capture = cv2.VideoCapture(str(video_path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()

        duration_seconds = (
            round(frame_count / source_fps, 2) if source_fps > 0 else 0
        )

        common_metadata = {
            "video_id": video_path.stem,
            "video_name": video_path.name,
            "video_path": str(video_path.resolve()),
            "file_extension": video_path.suffix.lower(),
            "file_size_bytes": video_path.stat().st_size,
            "width": width,
            "height": height,
            "source_fps": round(source_fps, 2),
            "frame_count": frame_count,
            "duration_seconds": duration_seconds,
        }

        writer.writerow(
            {
                **common_metadata,
                "storage_fps": COARSE_FPS,
                "extraction_type": "coarse",
                "persisted": True,
                "threshold": NCC_COARSE_THRESHOLD,
            }
        )
        writer.writerow(
            {
                **common_metadata,
                "storage_fps": FINE_FPS,
                "extraction_type": "fine",
                "persisted": STORE_FINE_FRAMES,
                "threshold": NCC_FINE_THRESHOLD,
            }
        )

print(f"Created: {output_file.resolve()}")