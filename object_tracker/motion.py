"""MOG2 -> 그림자 제거 -> 열림/닫힘 -> 윤곽선 -> 감지 목록."""

from __future__ import annotations

from math import isfinite
from typing import List, Tuple

import cv2
import numpy as np

from .models import Detection

LEARNING_RATES = (0.001, 0.01, 0.1)


class MotionDetector:
    """각 픽셀의 시간적 변화를 학습해 움직임 영역을 찾습니다.

    MOG2는 한 픽셀의 배경 밝기를 여러 Gaussian 분포로 표현합니다.
    learning_rate가 크면 새 모습을 빨리 배경으로 받아들입니다.
    사람/물건의 종류를 인식하는 탐지기는 아닙니다.
    """

    def __init__(self, learning_rate: float = 0.01, min_area: float = 400,
                 warmup_frames: int = 30) -> None:
        if (not 0 <= learning_rate <= 1 or not isfinite(min_area)
                or min_area <= 0 or warmup_frames < 0):
            raise ValueError("Invalid motion detector settings")
        self.learning_rate = learning_rate
        self.min_area = min_area
        self.warmup_frames = warmup_frames
        self.frames_seen = 0
        self._background = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=True)
        self._opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    @property
    def warming_up(self) -> bool:
        return self.frames_seen <= self.warmup_frames

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        # 첫 프레임은 배경을 모르는 상태이므로 화면 전체가 전경일 수 있습니다.
        raw = self._background.apply(frame, learningRate=self.learning_rate)
        self.frames_seen += 1
        # MOG2의 그림자 값 127을 제외하고 확실한 전경 255만 남깁니다.
        _, mask = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._opening)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._closing)
        if self.warming_up:
            return [], np.zeros_like(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        detections = [Detection(cv2.boundingRect(contour)) for contour in contours
                      if cv2.contourArea(contour) >= self.min_area]
        # OpenCV 내부의 윤곽선 반환 순서 대신 좌표 순서로 고정합니다.
        return sorted(detections, key=lambda item: item.bbox), mask
