"""영상/이미지/프레임 로그의 입출력. 알고리즘과 분리해 자원을 정리합니다."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from .models import Track
from .target import MatchResult


def write_image(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".png", frame)
    if not success:
        raise OSError("Cannot encode image: {}".format(path))
    encoded.tofile(str(path))


class VideoSource:
    """VideoCapture 하나의 수명과 입력 FPS 정보를 관리합니다."""

    def __init__(self, source: Union[str, int]) -> None:
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError("Cannot open source: {} (check path/camera access)".format(source))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self.is_camera = isinstance(source, int)

    def close(self) -> None:
        self.capture.release()


class OutputSession:
    """실행별 폴더에 프레임 로그, 스냅샷, 선택적인 결과 영상을 저장합니다."""

    def __init__(self, root: Path, save_video: bool, source_fps: float) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.directory = root / stamp
        self.directory.mkdir(parents=True, exist_ok=False)
        self._log = (self.directory / "frames.jsonl").open("w", encoding="utf-8")
        self._writer: Optional[cv2.VideoWriter] = None
        self._save_video = save_video
        self._fps = source_fps

    def record(self, frame: np.ndarray, tracks: Sequence[Track], target: MatchResult,
               frame_number: int, processing_fps: float, learning_rate: float,
               warming_up: bool) -> None:
        if self._save_video and self._writer is None:
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(str(self.directory / "annotated.avi"),
                                           cv2.VideoWriter_fourcc(*"MJPG"),
                                           self._fps, (width, height))
            if not self._writer.isOpened():
                raise OSError("Cannot create MJPG output video")
        if self._writer is not None:
            self._writer.write(frame)
        row = {"frame": frame_number, "time_s": (frame_number - 1) / self._fps,
               "processing_fps": round(processing_fps, 2), "learning_rate": learning_rate,
               "warming_up": warming_up,
               "tracks": [{"id": t.id, "bbox": list(t.bbox), "missed": t.missed}
                          for t in tracks],
               "target": {"keypoints": target.frame_keypoints,
                          "good_matches": target.good_matches, "inliers": target.inliers,
                          "detected": target.detected}}
        self._log.write(json.dumps(row) + "\n")

    def snapshot(self, frame: np.ndarray, frame_number: int) -> Path:
        # 같은 프레임에서 s를 여러 번 눌러도 기존 사진을 덮어쓰지 않습니다.
        stamp = datetime.now().strftime("%H%M%S-%f")
        path = self.directory / "snapshots" / "frame-{:06d}-{}.png".format(frame_number, stamp)
        write_image(path, frame)
        return path

    def close(self) -> None:
        self._log.close()
        if self._writer is not None:
            self._writer.release()
