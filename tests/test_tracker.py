"""ID 오류로 이어질 수 있는 수명/경합/거리 경계 조건을 검증합니다."""

import unittest

from object_tracker.models import Detection
from object_tracker.tracker import Tracker


def at(x, y=20):
    return Detection((x, y, 10, 10))


class TrackerTests(unittest.TestCase):
    def test_identity_survives_small_motion_and_detection_order_change(self):
        tracker = Tracker(max_distance=30)
        tracker.update([at(10), at(100)])
        tracks = tracker.update([at(105), at(15)])
        self.assertEqual([(t.id, t.bbox[0]) for t in tracks], [(1, 15), (2, 105)])

    def test_one_detection_cannot_update_two_ids(self):
        tracker = Tracker()
        tracker.update([at(10), at(30)])
        tracks = tracker.update([at(20)])
        self.assertEqual(sum(t.visible for t in tracks), 1)
        self.assertEqual(sum(t.missed == 1 for t in tracks), 1)

    def test_second_nearest_remains_available_after_closest_is_taken(self):
        tracker = Tracker(max_distance=40)
        tracker.update([at(0), at(30)])
        tracker.update([at(10), at(50)])
        self.assertEqual(tracker.tracks[1].bbox[0], 10)
        self.assertEqual(tracker.tracks[2].bbox[0], 50)

    def test_missing_tolerance_and_never_reused_id(self):
        tracker = Tracker(max_missed=2)
        tracker.update([at(10)])
        tracker.update([])
        self.assertFalse(tracker.tracks[1].visible)
        tracker.update([])
        self.assertIn(1, tracker.tracks)
        tracker.update([])
        self.assertNotIn(1, tracker.tracks)
        self.assertEqual(tracker.update([at(10)])[0].id, 2)

    def test_short_disappearance_reuses_id(self):
        tracker = Tracker(max_missed=2)
        tracker.update([at(10)])
        tracker.update([])
        self.assertEqual(tracker.update([at(15)])[0].id, 1)

    def test_distant_object_creates_new_id_at_gate_boundary(self):
        tracker = Tracker(max_distance=10)
        tracker.update([at(10)])
        self.assertEqual(tracker.update([at(20)])[0].id, 1)
        tracker.update([at(31)])
        self.assertEqual([t.id for t in tracker.tracks.values() if t.visible], [2])

    def test_history_is_bounded(self):
        tracker = Tracker(trail_length=3)
        for x in range(20):
            tracker.update([at(x)])
        self.assertEqual(list(tracker.tracks[1].trail), [(22, 25), (23, 25), (24, 25)])

    def test_unobserved_identity_cannot_be_recovered_at_crossing(self):
        tracker = Tracker()
        tracker.update([at(10), at(90)])
        tracker.update([at(40), at(60)])
        # 실제 A는 오른쪽으로 계속 이동했어도 거리만 보면 왼쪽 감지가 ID 1입니다.
        tracker.update([at(40), at(60)])
        self.assertEqual(tracker.tracks[1].bbox[0], 40)

    def test_invalid_parameters(self):
        for kwargs in ({"max_distance": 0}, {"max_distance": float("nan")},
                       {"max_missed": -1}, {"trail_length": 1}):
            with self.assertRaises(ValueError):
                Tracker(**kwargs)


if __name__ == "__main__":
    unittest.main()
