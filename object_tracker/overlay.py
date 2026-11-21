"""계산한 결과를 복사한 프레임 위에 그립니다. 분석 입력은 훼손하지 않습니다."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .models import Track
from .target import MatchResult


def draw_overlay(frame: np.ndarray, tracks: Sequence[Track], target: MatchResult,
                 frame_number: int, fps: float, learning_rate: float,
                 warming_up: bool = False, paused: bool = False) -> np.ndarray:
    result = frame.copy()
    for track in tracks:
        if not track.visible:
            continue  # 보관 중인 옛 위치를 현재 검출처럼 그리지 않습니다.
        x, y, width, height = track.bbox
        cv2.rectangle(result, (x, y), (x + width, y + height), (70, 230, 100), 2)
        cv2.circle(result, track.center, 4, (0, 235, 255), -1)
        cv2.putText(result, "ID:{}".format(track.id), (x, max(18, y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 230, 100), 2)
        if len(track.trail) >= 2:
            points = np.array(track.trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(result, [points], False, (80, 80, 255), 2, cv2.LINE_AA)
    if target.polygon is not None:
        cv2.polylines(result, [target.polygon.astype(np.int32)], True, (255, 220, 30), 2)

    # 작은 영상에서도 글자가 읽히도록 상단/하단의 단색 띠에 상태를 표시합니다.
    cv2.rectangle(result, (0, 0), (result.shape[1], 66), (28, 30, 35), -1)
    label = "FPS: {:.1f} | Frame: {} | LR: {}".format(fps, frame_number, learning_rate)
    cv2.putText(result, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA)
    status = "PAUSED" if paused else "WARMING UP" if warming_up else "RUNNING"
    cv2.putText(result, "{} | ORB: {} matches / {} inliers".format(
        status, target.good_matches, target.inliers), (10, 49),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 205, 235), 1, cv2.LINE_AA)
    if target.detected:
        cv2.putText(result, "TARGET DETECTED", (10, 94), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 60, 255), 2, cv2.LINE_AA)
    bottom = result.shape[0] - 8
    cv2.rectangle(result, (0, bottom - 21), (result.shape[1], result.shape[0]),
                  (28, 30, 35), -1)
    cv2.putText(result, "q: quit  s: snapshot  p: pause  1/2/3: learning rate",
                (10, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230),
                1, cv2.LINE_AA)
    return result
