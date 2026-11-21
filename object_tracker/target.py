"""등록 이미지 한 장의 ORB 특징을 매 프레임 검색합니다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    """Windows 한글 경로도 읽도록 imread 대신 바이트를 디코딩합니다."""
    if not path.is_file():
        raise ValueError("Image does not exist: {}".format(path))
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image: {}".format(path))
    return image


@dataclass(frozen=True)
class MatchResult:
    frame_keypoints: int = 0
    good_matches: int = 0
    inliers: int = 0
    detected: bool = False
    polygon: Optional[np.ndarray] = None


class TargetMatcher:
    """ORB 기술자 비교와 기하 검증만 담당합니다. 추적 ID에는 관여하지 않습니다."""

    def __init__(self, image_path: Path, min_matches: int = 12) -> None:
        self.reference = read_image(image_path)
        self.min_matches = min_matches
        self._orb = cv2.ORB_create(nfeatures=1200, edgeThreshold=15, fastThreshold=10)
        gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        self.keypoints, self._descriptors = self._orb.detectAndCompute(gray, None)
        if self._descriptors is None or len(self.keypoints) < min_matches:
            raise ValueError("Target needs at least {} textured keypoints; use a larger, "
                             "sharply focused image with patterns/text".format(min_matches))
        # ORB는 0/1 비트열이므로 서로 다른 비트 수인 Hamming 거리를 씁니다.
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def match(self, frame: np.ndarray) -> MatchResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 2:
            return MatchResult(frame_keypoints=len(keypoints))
        pairs = self._matcher.knnMatch(self._descriptors, descriptors, k=2)
        # 1등과 2등이 비슷하면 모호한 대응입니다. 확실한 1등만 남깁니다.
        good = [pair[0] for pair in pairs if len(pair) == 2
                and pair[0].distance < 0.75 * pair[1].distance]
        # 여러 기준점이 같은 프레임 점을 가리키는 중복 대응을 제거합니다.
        unique = {match.trainIdx: match for match in
                  sorted(good, key=lambda item: item.distance, reverse=True)}
        good = list(unique.values())
        if len(good) < self.min_matches:
            return MatchResult(len(keypoints), len(good))
        source = np.float32([self.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        destination = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(source, destination, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        if matrix is None or not np.isfinite(matrix).all():
            return MatchResult(len(keypoints), len(good), inliers)
        height, width = self.reference.shape[:2]
        corners = np.float32([(0, 0), (width - 1, 0),
                              (width - 1, height - 1), (0, height - 1)]).reshape(-1, 1, 2)
        polygon = cv2.perspectiveTransform(corners, matrix)
        if not np.isfinite(polygon).all():
            return MatchResult(len(keypoints), len(good), inliers)
        # 우연한 몇 점의 매칭만으로 성공이라고 표시하지 않습니다.
        area = abs(cv2.contourArea(polygon))
        frame_area = frame.shape[0] * frame.shape[1]
        plausible = (cv2.isContourConvex(polygon) and 100 <= area <= frame_area * 1.5
                     and np.max(np.abs(polygon)) < max(frame.shape[:2]) * 5)
        detected = bool(inliers >= 8 and inliers / len(good) >= 0.5 and plausible)
        return MatchResult(len(keypoints), len(good), inliers, detected,
                           polygon if detected else None)
