"""거리만 이용한 Greedy 추적. OpenCV와 화면 출력에 의존하지 않습니다."""

from __future__ import annotations

from collections import deque
from math import hypot, isfinite
from typing import Dict, List, Sequence

from .models import Detection, Track


class Tracker:
    """ID 발급, 일대일 연결, 누락 수명, 궤적만 책임집니다.

    모든 (기존 ID, 새 감지) 쌍을 가까운 순서로 살펴봅니다. 이미 선택한 ID나
    감지는 다시 쓰지 않습니다. 전체 최적해를 구하는 Hungarian 방식이 아니므로
    서로 겹친 객체의 원래 정체성을 보장할 수 없습니다.
    """

    def __init__(self, max_distance: float = 70, max_missed: int = 15,
                 trail_length: int = 64) -> None:
        if not isfinite(max_distance) or max_distance <= 0:
            raise ValueError("max_distance must be a positive finite number")
        if max_missed < 0 or trail_length < 2:
            raise ValueError("max_missed >= 0 and trail_length >= 2 are required")
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.trail_length = trail_length
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1  # 삭제한 번호를 재사용하지 않아 로그를 혼동하지 않습니다.

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        """새 프레임을 딱 한 번 반영합니다. 일시정지 중에는 호출하지 않습니다."""
        candidates = []
        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                distance = hypot(track.center[0] - detection.center[0],
                                 track.center[1] - detection.center[1])
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, index))

        used_tracks, used_detections = set(), set()
        # 거리가 같을 때 ID와 감지 순서로 결정하여 매번 같은 결과를 냅니다.
        for _, track_id, index in sorted(candidates):
            if track_id in used_tracks or index in used_detections:
                continue
            track = self.tracks[track_id]
            track.bbox = detections[index].bbox
            track.missed = 0
            track.trail.append(track.center)
            used_tracks.add(track_id)
            used_detections.add(index)

        for track_id in list(self.tracks):
            if track_id not in used_tracks:
                track = self.tracks[track_id]
                track.missed += 1
                # max_missed번의 누락까지 보관하고 다음 누락에서 삭제합니다.
                if track.missed > self.max_missed:
                    del self.tracks[track_id]

        for index, detection in enumerate(detections):
            if index not in used_detections:
                track = Track(self._next_id, detection.bbox,
                              trail=deque([detection.center], maxlen=self.trail_length))
                self.tracks[track.id] = track
                self._next_id += 1
        return list(self.tracks.values())
