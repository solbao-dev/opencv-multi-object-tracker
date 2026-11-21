"""실제 OpenCV와 임시 영상을 사용해 입력부터 저장까지 검증합니다."""

import json
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from object_tracker.app import AppConfig, run
from object_tracker.models import Detection
from object_tracker.motion import MotionDetector
from object_tracker.overlay import draw_overlay
from object_tracker.target import MatchResult, TargetMatcher
from object_tracker.tracker import Tracker
from object_tracker.video_io import write_image
from tools.generate_demo import make_target
from tools.review_events import resolve_video, review


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.target = self.root / "기준 이미지.png"
        write_image(self.target, make_target())

    def tearDown(self):
        self.temp.cleanup()

    def make_video(self):
        path = self.root / "input.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 360))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(15):
                frame = np.zeros((360, 640, 3), np.uint8)
                if index >= 5:
                    frame[100:280, 30 + index:286 + index] = make_target()
                writer.write(frame)
        finally:
            writer.release()
        return path

    def test_real_video_end_to_end_and_readable_output(self):
        path = self.make_video()
        result = run(AppConfig(str(path), self.target, output=self.root / "out",
                               warmup_frames=3, headless=True, save_video=True))
        logs = [json.loads(line) for line in (result / "frames.jsonl").read_text().splitlines()]
        self.assertEqual([row["frame"] for row in logs], list(range(1, 16)))
        self.assertTrue(any(row["target"]["detected"] for row in logs))
        self.assertTrue(any(t["missed"] == 0 for row in logs for t in row["tracks"]))
        capture = cv2.VideoCapture(str(result / "annotated.avi"))
        self.assertTrue(capture.isOpened())
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 15)
        capture.release()

    def test_pause_does_not_advance_frames_or_learning_and_snapshot_works(self):
        path = self.make_video()
        # 1프레임 처리 -> p -> 정지 중 s -> 정지 유지 -> p -> 2프레임 처리 -> q
        with patch("object_tracker.app.cv2.namedWindow"), patch("object_tracker.app.cv2.imshow"), \
             patch("object_tracker.app.cv2.destroyAllWindows"), \
             patch("object_tracker.app.cv2.getWindowProperty", return_value=1), \
             patch("object_tracker.app.cv2.waitKey", side_effect=[ord("p"), ord("s"), -1, ord("p"), ord("q")]):
            result = run(AppConfig(str(path), self.target, output=self.root / "out"))
        logs = (result / "frames.jsonl").read_text().splitlines()
        self.assertEqual(len(logs), 2)
        snapshots = list((result / "snapshots").glob("*.png"))
        self.assertEqual(len(snapshots), 1)
        self.assertIn("000001", snapshots[0].name)

    def test_three_learning_rates_are_applied(self):
        with patch("object_tracker.app.cv2.namedWindow"), patch("object_tracker.app.cv2.imshow"), \
             patch("object_tracker.app.cv2.destroyAllWindows"), \
             patch("object_tracker.app.cv2.getWindowProperty", return_value=1), \
             patch("object_tracker.app.cv2.waitKey", side_effect=[ord("1"), ord("2"), ord("3"), ord("q")]):
            result = run(AppConfig(str(self.make_video()), self.target, output=self.root / "out"))
        rates = [json.loads(line)["learning_rate"] for line in
                 (result / "frames.jsonl").read_text().splitlines()]
        self.assertEqual(rates, [0.01, 0.001, 0.01, 0.1])

    def test_mog2_warmup_area_filter_and_moving_rectangle(self):
        detector = MotionDetector(min_area=200, warmup_frames=5)
        blank = np.zeros((240, 320, 3), np.uint8)
        for _ in range(5):
            detections, _ = detector.detect(blank)
            self.assertEqual(detections, [])
        frame = blank.copy()
        frame[80:150, 90:150] = 255
        frame[10:12, 10:12] = 255
        detections, mask = detector.detect(frame)
        self.assertEqual(len(detections), 1)
        self.assertTrue(100 < detections[0].center[0] < 140)
        self.assertEqual(mask.dtype, np.uint8)

    def test_orb_positive_negative_and_featureless_reference(self):
        matcher = TargetMatcher(self.target)
        frame = np.full((360, 640, 3), 42, np.uint8)
        frame[100:280, 120:376] = make_target()
        self.assertTrue(matcher.match(frame).detected)
        self.assertFalse(matcher.match(np.zeros_like(frame)).detected)
        blank = self.root / "blank.png"
        write_image(blank, np.zeros((180, 256, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "keypoints"):
            TargetMatcher(blank)

    def test_overlay_does_not_modify_analysis_input(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        original = frame.copy()
        tracks = Tracker().update([Detection((100, 120, 40, 60))])
        display = draw_overlay(frame, tracks, MatchResult(), 1, 20, 0.01)
        self.assertTrue(np.array_equal(frame, original))
        self.assertFalse(np.array_equal(display, frame))

    def test_invalid_source_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "Cannot open source"):
            run(AppConfig(str(self.root / "missing.avi"), self.target, headless=True))

    def test_frame_limit(self):
        result = run(AppConfig(str(self.make_video()), self.target, output=self.root / "out",
                               headless=True, max_frames=2))
        self.assertEqual(len((result / "frames.jsonl").read_text().splitlines()), 2)

    def test_invalid_motion_settings(self):
        for kwargs in ({"learning_rate": float("nan")}, {"learning_rate": -0.1},
                       {"min_area": float("inf")}, {"min_area": float("nan")},
                       {"min_area": 0}, {"warmup_frames": -1}):
            with self.assertRaises(ValueError):
                MotionDetector(**kwargs)

    def test_manual_review_saves_counts_and_preserves_other_events(self):
        events = self.root / "events.json"
        events.write_text(json.dumps([
            {"video": "input.avi", "condition": "single", "event": 1, "start": 1, "end": 5},
            {"video": "input.avi", "condition": "single", "event": 2, "start": 6, "end": 10}]))
        output = self.root / "observations.csv"
        with patch("tools.review_events.cv2.namedWindow"), \
             patch("tools.review_events.cv2.imshow"), \
             patch("tools.review_events.cv2.destroyAllWindows"), \
             patch("tools.review_events.cv2.getWindowProperty", return_value=1), \
             patch("tools.review_events.cv2.waitKey", side_effect=[ord(k) for k in "xfoss"]):
            review(self.make_video(), events, "single", output)
        with output.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[0]["id_switches"], rows[0]["failures"]), ("1", "1"))
        self.assertIn("50%", rows[0]["notes"])
        self.assertEqual((rows[1]["id_switches"], rows[1]["failures"]), ("0", "0"))

    def test_review_resolves_baseline_video_from_run_index(self):
        directory = self.root / "baseline"
        directory.mkdir()
        video = directory / "annotated.avi"
        self.make_video().rename(video)
        index = self.root / "run_index.json"
        index.write_text(json.dumps({"baseline/overlap": str(directory)}), encoding="utf-8")
        self.assertEqual(resolve_video(None, "overlap", index), video)

    def test_review_explicit_video_does_not_require_run_index(self):
        video = self.make_video()
        self.assertEqual(resolve_video(video, "custom", self.root / "missing.json"), video)

    def test_review_missing_or_invalid_baseline_has_reproduction_hint(self):
        index = self.root / "run_index.json"
        for contents in (None, "invalid json", "{}",
                         json.dumps({"baseline/overlap": str(self.root / "missing")})):
            if contents is not None:
                index.write_text(contents, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run_experiments"):
                resolve_video(None, "overlap", index)


if __name__ == "__main__":
    unittest.main()
