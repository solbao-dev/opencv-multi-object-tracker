"""감지기와 추적기가 주고받는 데이터. 영상 처리 알고리즘은 넣지 않습니다."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

Point = Tuple[int, int]
Box = Tuple[int, int, int, int]  # 왼쪽 위 x, y, 너비, 높이 (픽셀)


@dataclass(frozen=True)
class Detection:
    """이번 프레임에서 찾은 움직임 영역. 아직 객체 ID는 없습니다."""

    bbox: Box

    @property
    def center(self) -> Point:
        x, y, width, height = self.bbox
        return (x + width // 2, y + height // 2)


@dataclass
class Track:
    """여러 프레임에 걸친 한 ID의 상태. missed > 0이면 현재 미검출입니다."""

    id: int
    bbox: Box
    missed: int = 0
    trail: Deque[Point] = field(default_factory=lambda: deque(maxlen=64))

    @property
    def center(self) -> Point:
        return Detection(self.bbox).center

    @property
    def visible(self) -> bool:
        return self.missed == 0
