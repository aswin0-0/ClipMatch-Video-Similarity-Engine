import csv
from pathlib import Path

import cv2

from config import (
    COARSE_FPS,
    DB_VIDEOS_DIR,
    FINE_FPS,
    NCC_COARSE_THRESHOLD,
    NCC_FINE_THRESHOLD,
    PROCESSED_FRAMES_DIR,
    video_sort_key,
)

coarse_output_file = Path("database_videos_coarse_metadata.csv")
fine_output_file = Path("database_videos_fine_metadata.csv")
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

def has_stored_frames(video_id: str, storage_fps: int) -> bool:
    frame_dir = PROCESSED_FRAMES_DIR / video_id / f"{storage_fps}fps"
    return any(frame_dir.glob("frame_*.png"))


with (
    coarse_output_file.open("w", newline="", encoding="utf-8") as coarse_csv,
    fine_output_file.open("w", newline="", encoding="utf-8") as fine_csv,
):
    coarse_writer = csv.DictWriter(coarse_csv, fieldnames=fieldnames)
    fine_writer = csv.DictWriter(fine_csv, fieldnames=fieldnames)
    coarse_writer.writeheader()
    fine_writer.writeheader()

    for video_path in sorted(DB_VIDEOS_DIR.iterdir(), key=video_sort_key):
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

        coarse_writer.writerow(
            {
                **common_metadata,
                "storage_fps": COARSE_FPS,
                "extraction_type": "coarse",
                "persisted": has_stored_frames(video_path.stem, COARSE_FPS),
                "threshold": NCC_COARSE_THRESHOLD,
            }
        )
        fine_writer.writerow(
            {
                **common_metadata,
                "storage_fps": FINE_FPS,
                "extraction_type": "fine",
                "persisted": has_stored_frames(video_path.stem, FINE_FPS),
                "threshold": NCC_FINE_THRESHOLD,
            }
        )

print(f"Created: {coarse_output_file.resolve()}")
print(f"Created: {fine_output_file.resolve()}")