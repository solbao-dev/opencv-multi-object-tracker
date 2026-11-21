"""과제의 연속 프레임 기준과 가림 예외가 집계에 적용되는지 확인합니다."""

import unittest

from object_tracker.evaluation import measure_event


def fixture(ids, occluded=()):
    truth, logs = [], []
    for index, track_id in enumerate(ids):
        truth.append({"objects": [{"name": "A", "bbox": [0, 0, 40, 40],
                                   "occlusion": 0.5 if index in occluded else 0}]})
        logs.append({"tracks": [] if track_id is None else [
            {"id": track_id, "bbox": [0, 0, 40, 40], "missed": 0}]})
    return truth, logs


class EvaluationTests(unittest.TestCase):
    def measure(self, ids, occluded=()):
        truth, logs = fixture(ids, occluded)
        return measure_event(truth, logs, 1, len(ids))

    def test_switch_needs_five_continuous_frames_and_is_counted_once(self):
        self.assertEqual(self.measure([1] + [2] * 4)["id_switches"], 0)
        self.assertEqual(self.measure([1] + [2] * 20)["id_switches"], 1)

    def test_missing_needs_ten_and_separate_episodes_count_separately(self):
        self.assertEqual(self.measure([1] + [None] * 9)["failures"], 0)
        self.assertEqual(self.measure([1] + [None] * 20)["failures"], 1)
        self.assertEqual(self.measure([None] * 10 + [1] + [None] * 10)["failures"], 2)

    def test_occlusion_at_exactly_half_breaks_missing_streak(self):
        self.assertEqual(self.measure([1] + [None] * 12, occluded=[6])["failures"], 0)

    def test_switch_streak_is_broken_by_missing_frame(self):
        self.assertEqual(self.measure([1] + [2] * 4 + [None] + [2] * 4)["id_switches"], 0)

    def test_remembered_unobserved_track_is_a_miss(self):
        truth, logs = fixture([1] * 10)
        for row in logs:
            row["tracks"][0]["missed"] = 1
        self.assertEqual(measure_event(truth, logs, 1, 10)["failures"], 1)


if __name__ == "__main__":
    unittest.main()
