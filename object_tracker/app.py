"""탑다운으로 읽는 본체: 입력 -> 감지 -> 추적/재인식 -> 표시 -> 제어."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional, Union

import cv2

from .motion import LEARNING_RATES, MotionDetector
from .overlay import draw_overlay
from .target import TargetMatcher
from .tracker import Tracker
from .video_io import OutputSession, VideoSource

WINDOW = "OpenCV Multi Object Tracker"


@dataclass(frozen=True)
class AppConfig:
    source: Union[str, int]
    target: Path
    output: Path = Path("outputs")
    learning_rate: float = 0.01
    min_area: float = 400
    max_distance: float = 70
    max_missed: int = 15
    warmup_frames: int = 30
    width: int = 960
    headless: bool = False
    save_video: bool = False
    show_mask: bool = False
    max_frames: Optional[int] = None


def run(config: AppConfig) -> Path:
    """한 실행을 조율합니다. 각 알고리즘은 별도 객체가 담당합니다."""
    # 잘못된 기준 이미지는 카메라를 열기 전에 발견합니다.
    matcher = TargetMatcher(config.target)
    motion = MotionDetector(config.learning_rate, config.min_area, config.warmup_frames)
    tracker = Tracker(config.max_distance, config.max_missed)
    source = VideoSource(config.source)
    output = None
    try:
        output = OutputSession(config.output, config.save_video, source.fps)
        metadata = asdict(config)
        metadata.update(source_fps=source.fps, opencv_version=cv2.__version__, frame_base=1)
        (output.directory / "run.json").write_text(
            json.dumps(metadata, default=str, indent=2), encoding="utf-8")
        if not config.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

        frame_number, paused, fps = 0, False, 0.0
        previous_tick = None
        frame = mask = tracks = target = None
        while True:
            tick = perf_counter()
            if not paused:
                ok, frame = source.capture.read()
                if not ok:
                    if frame_number == 0:
                        raise ValueError("Source opened but contains no decodable frames")
                    break
                frame_number += 1
                if frame.shape[1] > config.width:
                    scale = config.width / frame.shape[1]
                    frame = cv2.resize(frame, (config.width, round(frame.shape[0] * scale)))

                detections, mask = motion.detect(frame)
                tracks = tracker.update(detections)
                target = matcher.match(frame)  # ORB는 움직임 검출과 독립적으로 매 프레임 실행
                if previous_tick is not None:
                    current_fps = 1.0 / max(tick - previous_tick, 1e-9)
                    fps = current_fps if fps == 0 else 0.9 * fps + 0.1 * current_fps
                previous_tick = tick

            display = draw_overlay(frame, tracks, target, frame_number, fps,
                                   motion.learning_rate, motion.warming_up, paused)
            if not paused:
                output.record(display, tracks, target, frame_number, fps,
                              motion.learning_rate, motion.warming_up)
            if config.max_frames is not None and frame_number >= config.max_frames:
                break
            if config.headless:
                continue

            cv2.imshow(WINDOW, display)
            if config.show_mask:
                cv2.imshow("Foreground Mask", mask)
            # 파일은 입력 FPS에 맞춰 재생하고, 웹캠/일시정지는 짧게 키를 기다립니다.
            elapsed_ms = (perf_counter() - tick) * 1000
            delay = 20 if paused else 1 if source.is_camera else max(
                1, round(1000 / source.fps - elapsed_ms))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                print("Snapshot:", output.snapshot(display, frame_number))
            elif key == ord("p"):
                paused = not paused
                # 정지한 시간을 FPS에 섞지 않습니다. 영상/모델/ID 수명도 정지합니다.
                previous_tick = None
                fps = 0.0
            elif key in (ord("1"), ord("2"), ord("3")):
                motion.learning_rate = LEARNING_RATES[key - ord("1")]
                print("Learning rate:", motion.learning_rate)
        return output.directory
    finally:
        # 예외, EOF, q, 창 닫기 모두 같은 경로로 정리합니다.
        source.close()
        if output is not None:
            output.close()
        if not config.headless:
            cv2.destroyAllWindows()
