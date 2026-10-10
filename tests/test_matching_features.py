import unittest
from unittest.mock import patch

import numpy as np

from services.matcher import MatchResult, _fine_search
from services.ncc import (
    ncc_score,
    score_query_against_db_frames_oriented,
)


class MatchingFeaturesTests(unittest.TestCase):
    def setUp(self):
        base = np.arange(256 * 256, dtype=np.float32).reshape(256, 256) / 65536.0
        self.frame = base

    def test_ncc_handles_brightness_shift(self):
        brighter = np.clip(self.frame + 0.2, 0.0, 1.0)
        self.assertGreater(ncc_score(self.frame, brighter), 0.95)

    def test_mirror_matching_reports_orientation(self):
        mirrored = np.fliplr(self.frame)
        score, orientation = score_query_against_db_frames_oriented(
            [self.frame],
            [mirrored],
            mirror_matching=True,
        )
        self.assertGreater(score, 0.99)
        self.assertEqual(orientation, "mirrored")

    def test_patch_gate_rejects_unrelated_frame(self):
        score, orientation = score_query_against_db_frames_oriented(
            [self.frame],
            [np.zeros_like(self.frame)],
            enable_patch_early_rejection=True,
        )
        self.assertEqual(score, 0.0)
        self.assertEqual(orientation, "normal")

    def test_match_result_preserves_timestamp_alias(self):
        result = MatchResult(
            matched=True,
            timestamp_sec=12.5,
            end_timestamp_sec=20.0,
        )
        self.assertEqual(result.start_timestamp_sec, 12.5)
        self.assertEqual(result.timestamp_sec, 12.5)
        self.assertEqual(result.start_timestamp_str, "00:12")
        self.assertEqual(result.end_timestamp_str, "00:20")
        self.assertEqual(result.timestamp_str, result.start_timestamp_str)

    @patch("services.matcher.sliding_window_ncc_oriented")
    @patch("services.matcher.extract_frames_in_window")
    @patch("services.matcher.get_video_duration_sec", return_value=100.0)
    def test_fine_search_is_bounded(
        self,
        _duration,
        extract_window,
        score_windows,
    ):
        extract_window.return_value = [(self.frame, 5.0)]
        score_windows.return_value = [(0.9, 5.0, "normal")]

        result = _fine_search(
            "query.mp4",
            "database.mp4",
            [(0.8, 10.0)],
            query_duration_sec=4.0,
            query_frames_fine=[(self.frame, 0.0)],
        )

        self.assertIsNotNone(result)
        start, end = extract_window.call_args.args[1:3]
        self.assertLessEqual(end - start, 20.0)
        self.assertEqual(result[2], "normal")


if __name__ == "__main__":
    unittest.main()
