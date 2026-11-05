"""
benchmark.py — Automated threshold benchmarking & evaluation for ClipMatch.

Sweeps coarse and fine NCC thresholds across a test set of query clips
and exports a CSV report to determine the optimal threshold configuration
for maximum accuracy and F1-score.

Usage:
  python benchmark.py [--queries-dir PATH] [--method {diagonal,standard}]
                      [--tolerance-sec SECONDS] [--ground-truth-csv PATH]
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    COARSE_FPS,
    DB_VIDEOS_DIR,
    ENABLE_MIRROR_MATCHING,
    ENABLE_PATCH_EARLY_REJECTION,
    FINE_FPS,
    FINE_SEARCH_WINDOW_SEC,
    PATCH_GRID_SIZE,
    PATCH_MIN_PASS_RATIO,
    PATCH_SCORE_THRESHOLD,
    WINDOW_PADDING_SEC,
    video_sort_key,
)
from services.frame_extractor import (
    extract_frames_at_fps,
    extract_stored_frames_at_fps,
    extract_stored_frames_from_db,
    get_video_duration_sec,
)
from services.matcher import _fine_search
from services.ncc import sliding_window_ncc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


@dataclass
class QueryBenchmarkResult:
    """Raw matching profile for a single query clip."""
    query_name: str
    true_video_id: str | None
    true_start_sec: float | None
    true_end_sec: float | None
    pred_video_id: str
    pred_coarse_score: float
    pred_fine_score: float
    pred_timestamp_sec: float
    pred_orientation: str
    elapsed_sec: float


def parse_ground_truth_from_name(
    filename: str, available_db_videos: list[str]
) -> tuple[str | None, float | None, float | None]:
    """
    Attempt to extract expected video ID and timestamps from filename.
    Examples:
      - 'query_video7(0_56-1_11).mp4' -> ('Query_Video7', 56.0, 71.0)
      - 'Query_Video14(45-60).mp4'     -> ('Query_Video14', 45.0, 60.0)
      - 'query_video12.mp4'            -> ('Query_Video12', None, None)
    """
    stem = Path(filename).stem

    # 1. Parse timestamps: (MM_SS-MM_SS) or (SS-SS)
    start_sec: float | None = None
    end_sec: float | None = None

    time_match_colon = re.search(r"\((\d+)[_:.](\d+)\s*-\s*(\d+)[_:.](\d+)\)", stem)
    if time_match_colon:
        m1, s1, m2, s2 = map(int, time_match_colon.groups())
        start_sec = float(m1 * 60 + s1)
        end_sec = float(m2 * 60 + s2)
    else:
        time_match_sec = re.search(r"\((\d+)\s*-\s*(\d+)\)", stem)
        if time_match_sec:
            start_sec = float(time_match_sec.group(1))
            end_sec = float(time_match_sec.group(2))

    # 2. Extract base name without timestamp
    base_name = re.sub(r"\(.*?\)", "", stem).strip()
    clean_base = re.sub(r"[^a-zA-Z0-9]", "", base_name).lower()

    # 3. Match against known database video IDs
    matched_video_id: str | None = None
    for db_vid in available_db_videos:
        clean_db = re.sub(r"[^a-zA-Z0-9]", "", db_vid).lower()
        if clean_base == clean_db or clean_base.endswith(clean_db) or clean_db.endswith(clean_base):
            matched_video_id = db_vid
            break

    return matched_video_id, start_sec, end_sec


def load_ground_truth_csv(csv_path: Path) -> dict[str, tuple[str, float | None, float | None]]:
    """
    Load ground truth annotations from a CSV file.
    Expected headers: query_filename, true_video_id, [true_start_sec, true_end_sec]
    """
    mapping: dict[str, tuple[str, float | None, float | None]] = {}
    if not csv_path.exists():
        return mapping

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("query_filename", "").strip()
            vid = row.get("true_video_id", "").strip()
            start = float(row["true_start_sec"]) if row.get("true_start_sec") else None
            end = float(row["true_end_sec"]) if row.get("true_end_sec") else None
            if fname and vid:
                mapping[fname] = (vid, start, end)

    return mapping


def evaluate_query(
    query_path: Path,
    db_video_paths: dict[str, Path],
    method: str = "diagonal",
) -> tuple[str, float, float, float, str, float]:
    """
    Run one full matching pass over all DB videos to find the unconditional
    best candidate video, coarse score, fine score, and timestamp.
    """
    start_time = time.time()
    query_frames_coarse = extract_frames_at_fps(query_path, COARSE_FPS)
    if not query_frames_coarse:
        return "", 0.0, 0.0, 0.0, "normal", time.time() - start_time

    query_frames_only = [f for f, _ in query_frames_coarse]
    n_query = len(query_frames_only)
    query_duration_sec = n_query / COARSE_FPS

    # Coarse search across all DB videos
    best_per_video: dict[str, tuple[float, float, list[tuple[float, float]]]] = {}

    for video_id, video_path in db_video_paths.items():
        try:
            db_frames = extract_stored_frames_from_db(video_id, COARSE_FPS)
        except Exception:
            try:
                db_frames = extract_stored_frames_at_fps(video_id, video_path, COARSE_FPS)
            except Exception:
                try:
                    db_frames = extract_frames_at_fps(video_path, COARSE_FPS)
                except Exception:
                    continue

        if len(db_frames) < n_query:
            continue

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
            best_per_video[video_id] = (best_score, best_ts, scored)

    if not best_per_video:
        return "", 0.0, 0.0, 0.0, "normal", time.time() - start_time

    # Rank candidates by coarse score
    ranked_candidates = sorted(best_per_video.items(), key=lambda x: x[1][0], reverse=True)

    # Pre-extract fine query frames once
    query_frames_fine = extract_frames_at_fps(query_path, FINE_FPS)

    best_candidate_vid = ""
    best_coarse_score = 0.0
    best_fine_score = 0.0
    best_fine_ts = 0.0
    best_orientation = "normal"

    # Evaluate fine search on top 3 candidates (unconditional peak score)
    for video_id, (coarse_score, best_ts, scored) in ranked_candidates[:3]:
        video_path = db_video_paths[video_id]
        # Top 5 windows for this video
        top_windows = sorted(scored, key=lambda x: x[0], reverse=True)[:5]
        candidate_windows = [(s, t) for s, t in top_windows]

        fine_result = _fine_search(
            query_clip_path=query_path,
            video_path=video_path,
            candidate_windows=candidate_windows,
            query_duration_sec=query_duration_sec,
            window_padding_sec=WINDOW_PADDING_SEC,
            method=method,
            query_frames_fine=query_frames_fine,
            mirror_matching=ENABLE_MIRROR_MATCHING,
            enable_patch_early_rejection=ENABLE_PATCH_EARLY_REJECTION,
            fine_threshold=0.0,  # capture absolute peak score
        )

        if fine_result and fine_result[0] > best_fine_score:
            best_fine_score = fine_result[0]
            best_fine_ts = fine_result[1]
            best_orientation = fine_result[2]
            best_candidate_vid = video_id
            best_coarse_score = coarse_score

    elapsed = time.time() - start_time
    return (
        best_candidate_vid,
        best_coarse_score,
        best_fine_score,
        best_fine_ts,
        best_orientation,
        elapsed,
    )


def run_benchmark(
    queries_dir: Path,
    db_videos_dir: Path,
    ground_truth_csv_path: Path | None = None,
    method: str = "diagonal",
    tolerance_sec: float = 3.0,
    output_sweep_csv: Path = Path("threshold_benchmark.csv"),
    output_details_csv: Path = Path("benchmark_query_details.csv"),
):
    print("=" * 65)
    print("  ClipMatch — Threshold Evaluation & Optimization Engine")
    print("=" * 65)
    print(f"  Queries Directory : {queries_dir.resolve()}")
    print(f"  DB Directory      : {db_videos_dir.resolve()}")
    print(f"  Matching Method   : {method.upper()}")
    print(f"  Timestamp Error   : ±{tolerance_sec:.1f}s")
    print(f"  Output Sweep CSV  : {output_sweep_csv.name}")
    print(f"  Output Details CSV: {output_details_csv.name}")
    print("=" * 65)

    if not queries_dir.exists():
        queries_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Created queries folder: {queries_dir.resolve()}")
        print("Please place your test query clips in this directory and re-run.")
        return

    # Discover DB videos
    db_video_paths: dict[str, Path] = {}
    for f in sorted(db_videos_dir.iterdir(), key=video_sort_key):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            db_video_paths[f.stem] = f

    if not db_video_paths:
        print(f"\n[ERROR] No database videos found in {db_videos_dir.resolve()}")
        return

    print(f"\n[OK] Found {len(db_video_paths)} DB videos.")

    # Discover Query files
    query_files = [
        f for f in sorted(queries_dir.iterdir(), key=video_sort_key)
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not query_files:
        print(f"\n[ERROR] No query video files found in '{queries_dir.resolve()}'.")
        print("Place query clips (e.g. 'query_video7(0_56-1_11).mp4') in that folder.")
        return

    print(f"[OK] Found {len(query_files)} test query clips.\n")

    # Load Ground Truth
    gt_map = load_ground_truth_csv(ground_truth_csv_path) if ground_truth_csv_path else {}
    available_vids = list(db_video_paths.keys())

    query_results: list[QueryBenchmarkResult] = []

    print("Phase 1: Evaluating Raw Match Scores Across All Queries...")
    print("-" * 65)

    for idx, qpath in enumerate(query_files, 1):
        # Resolve Ground Truth
        if qpath.name in gt_map:
            true_vid, true_start, true_end = gt_map[qpath.name]
        else:
            true_vid, true_start, true_end = parse_ground_truth_from_name(qpath.name, available_vids)

        ts_info = f" @ {true_start:.1f}s" if true_start is not None else ""
        print(f"[{idx}/{len(query_files)}] Processing: {qpath.name}")
        print(f"      Ground Truth : {true_vid or 'Unknown'}{ts_info}")

        pred_vid, coarse_score, fine_score, pred_ts, orientation, elapsed = evaluate_query(
            qpath, db_video_paths, method=method
        )

        print(
            f"      Match Output : {pred_vid} (Coarse={coarse_score:.4f}, "
            f"Fine={fine_score:.4f} @ {pred_ts:.1f}s) [{elapsed:.1f}s]"
        )

        query_results.append(
            QueryBenchmarkResult(
                query_name=qpath.name,
                true_video_id=true_vid,
                true_start_sec=true_start,
                true_end_sec=true_end,
                pred_video_id=pred_vid,
                pred_coarse_score=coarse_score,
                pred_fine_score=fine_score,
                pred_timestamp_sec=pred_ts,
                pred_orientation=orientation,
                elapsed_sec=elapsed,
            )
        )

    # ── Export Query Details CSV ──────────────────────────────────────────────
    with open(output_details_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_file",
            "true_video_id",
            "true_start_sec",
            "true_end_sec",
            "predicted_video_id",
            "predicted_timestamp_sec",
            "coarse_ncc_score",
            "fine_ncc_score",
            "orientation",
            "elapsed_seconds",
        ])
        for q in query_results:
            writer.writerow([
                q.query_name,
                q.true_video_id or "UNKNOWN",
                f"{q.true_start_sec:.2f}" if q.true_start_sec is not None else "",
                f"{q.true_end_sec:.2f}" if q.true_end_sec is not None else "",
                q.pred_video_id,
                f"{q.pred_timestamp_sec:.2f}",
                f"{q.pred_coarse_score:.4f}",
                f"{q.pred_fine_score:.4f}",
                q.pred_orientation,
                f"{q.elapsed_sec:.2f}",
            ])

    print(f"\n[OK] Saved individual query details to: {output_details_csv.resolve()}")

    # ── Phase 2: In-Memory Threshold Sweep ───────────────────────────────────
    print("\nPhase 2: Sweeping Threshold Combinations & Computing Accuracy...")
    print("-" * 65)

    coarse_thresholds = np.round(np.arange(0.40, 0.81, 0.05), 2)
    fine_thresholds = np.round(np.arange(0.50, 0.91, 0.05), 2)

    sweep_results: list[dict[str, Any]] = []

    for c_th in coarse_thresholds:
        for f_th in fine_thresholds:
            tp = 0
            fp = 0
            fn = 0

            for q in query_results:
                matched = (q.pred_coarse_score >= c_th) and (q.pred_fine_score >= f_th)

                if q.true_video_id is None:
                    # No ground truth label available; cannot score validity
                    continue

                if matched:
                    # Check video ID
                    vid_correct = (q.pred_video_id.lower() == q.true_video_id.lower())
                    
                    # Check timestamp if ground truth timestamp exists
                    ts_correct = True
                    if q.true_start_sec is not None:
                        ts_diff = abs(q.pred_timestamp_sec - q.true_start_sec)
                        ts_correct = (ts_diff <= tolerance_sec)

                    if vid_correct and ts_correct:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1

            total_valid_queries = tp + fp + fn
            if total_valid_queries == 0:
                continue

            accuracy = (tp / total_valid_queries) * 100.0
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1_score = (
                (2.0 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            sweep_results.append({
                "coarse_threshold": c_th,
                "fine_threshold": f_th,
                "total_queries": total_valid_queries,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "accuracy_pct": round(accuracy, 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
            })

    # Sort sweep results by F1-score descending, then accuracy descending
    sweep_results.sort(key=lambda r: (r["f1_score"], r["accuracy_pct"]), reverse=True)

    # ── Export Sweep CSV ──────────────────────────────────────────────────────
    with open(output_sweep_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "coarse_threshold",
            "fine_threshold",
            "total_queries",
            "true_positives",
            "false_positives",
            "false_negatives",
            "accuracy_pct",
            "precision",
            "recall",
            "f1_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_results)

    print(f"[OK] Saved threshold sweep analysis to: {output_sweep_csv.resolve()}")

    # ── Terminal Summary Table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  TOP 5 OPTIMAL THRESHOLD CONFIGURATIONS")
    print("=" * 80)
    print(
        f"  {'Coarse Thresh':<14} {'Fine Thresh':<13} {'Accuracy':<12} "
        f"{'Precision':<11} {'Recall':<10} {'F1-Score':<10} {'TP/FP/FN'}"
    )
    print("-" * 80)

    for row in sweep_results[:5]:
        counts = f"{row['true_positives']}/{row['false_positives']}/{row['false_negatives']}"
        print(
            f"  {row['coarse_threshold']:<14.2f} {row['fine_threshold']:<13.2f} "
            f"{row['accuracy_pct']:>6.2f}%     {row['precision']:<11.4f} "
            f"{row['recall']:<10.4f} {row['f1_score']:<10.4f} {counts}"
        )

    print("=" * 80)

    if sweep_results:
        best = sweep_results[0]
        print("\n[RECOMMENDED SETTINGS]")
        print(f"  Set in .env file:")
        print(f"    NCC_COARSE_THRESHOLD={best['coarse_threshold']:.2f}")
        print(f"    NCC_FINE_THRESHOLD={best['fine_threshold']:.2f}")
        print(f"  Expected Accuracy: {best['accuracy_pct']:.1f}% | F1-Score: {best['f1_score']:.4f}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ClipMatch automated threshold evaluation and benchmarking."
    )
    parser.add_argument(
        "--queries-dir",
        type=Path,
        default=Path("./test_queries"),
        help="Directory containing query clips (default: ./test_queries)",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=DB_VIDEOS_DIR,
        help=f"Directory containing database videos (default: {DB_VIDEOS_DIR})",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=None,
        help="Optional CSV specifying ground truth video and timestamps",
    )
    parser.add_argument(
        "--method",
        choices=["diagonal", "standard"],
        default="diagonal",
        help="NCC matching method (default: diagonal)",
    )
    parser.add_argument(
        "--tolerance-sec",
        type=float,
        default=3.0,
        help="Allowed timestamp deviation in seconds for a hit (default: 3.0)",
    )
    parser.add_argument(
        "--output-sweep",
        type=Path,
        default=Path("threshold_benchmark.csv"),
        help="Output CSV for threshold sweep (default: threshold_benchmark.csv)",
    )
    parser.add_argument(
        "--output-details",
        type=Path,
        default=Path("benchmark_query_details.csv"),
        help="Output CSV for query raw results (default: benchmark_query_details.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        queries_dir=args.queries_dir,
        db_videos_dir=args.db_dir,
        ground_truth_csv_path=args.ground_truth_csv,
        method=args.method,
        tolerance_sec=args.tolerance_sec,
        output_sweep_csv=args.output_sweep,
        output_details_csv=args.output_details,
    )

