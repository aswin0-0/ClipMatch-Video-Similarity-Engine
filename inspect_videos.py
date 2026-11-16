from pathlib import Path
import cv2

folder = Path(r"D:\ClipMatchData\reference_videos")

total_duration = 0
total_coarse = 0
total_fine = 0

for path in sorted(folder.glob("*.mp4")):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0:
        print("Could not read:", path.name)
        continue

    duration = frames / fps
    total_duration += duration
    total_coarse += int(duration * 1) + 1
    total_fine += int(duration * 24) + 1

    print(f"{path.name}: {duration:.1f}s | {fps:.2f} FPS")

print("\n--- Dataset Summary ---")
print("Videos:", len(list(folder.glob("*.mp4"))))
print(f"Total duration: {total_duration / 3600:.2f} hours")
print("Approx. coarse frames:", total_coarse)
print("Approx. fine frames:", total_fine)
