"""외부 다운로드 없이 24초짜리 표준 실험 영상 네 개를 재현합니다.

사람이 등장하는 실제 CCTV가 아닌 합성 물체 영상입니다. 정답 좌표를 알 수
있어 자동 보조 계측이 가능하지만, 실제 환경 성능으로 일반화하면 안 됩니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from object_tracker.video_io import write_image

FPS, FRAMES, SIZE = 20, 480, (640, 360)
COUNTS = {"single": 10, "overlap": 10, "stop": 5, "lighting": 5}


def make_target() -> np.ndarray:
    rng = np.random.default_rng(32)
    image = np.full((180, 256, 3), 235, np.uint8)
    for _ in range(130):
        x, y = rng.integers([5, 5], [250, 175])
        color = tuple(int(v) for v in rng.integers(20, 210, 3))
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 8)), color, -1)
    cv2.rectangle(image, (4, 4), (251, 175), (15, 15, 15), 3)
    cv2.putText(image, "CODYSSEY", (13, 80), cv2.FONT_HERSHEY_DUPLEX,
                1.03, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(image, "MISSION 3-2", (27, 125), cv2.FONT_HERSHEY_DUPLEX,
                0.75, (10, 10, 10), 2, cv2.LINE_AA)
    return image


def background() -> np.ndarray:
    image = np.full((SIZE[1], SIZE[0], 3), 42, np.uint8)
    for x in range(0, SIZE[0], 40):
        cv2.line(image, (x, 0), (x, SIZE[1] - 1), (48, 48, 48), 1)
    return image


def boxes_for(condition: str, local: int, length: int):
    t = local / (length - 1)
    x = 80 + round(480 * t)
    y = 214 + round(24 * np.sin(t * np.pi * 2))
    if condition == "overlap":
        return [("A", (x - 32, 178, 64, 80)),
                ("B", (560 - round(480 * t) - 32, 186, 64, 80))]
    if condition == "stop":
        # 중앙에서 약 2초 정지: 배경으로 흡수되는 모습을 관찰합니다.
        progress = t * 2.5 if t < 0.2 else 0.5 if t < 0.7 else 0.5 + (t - 0.7) / 0.6
        x, y = 80 + round(480 * progress), 220
    return [("A", (x - 32, y - 40, 64, 80))]


def overlap_fraction(back, front) -> float:
    x, y, w, h = back
    xx, yy, ww, hh = front
    area = max(0, min(x + w, xx + ww) - max(x, xx)) * max(
        0, min(y + h, yy + hh) - max(y, yy))
    return area / (w * h)


def writer_for(path: Path) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    if not writer.isOpened():
        raise OSError("Cannot create video: {}".format(path))
    return writer


def generate(root: Path = Path("data")) -> None:
    directory = root / "demo"
    directory.mkdir(parents=True, exist_ok=True)
    write_image(root / "target.png", make_target())
    events = []
    for condition, count in COUNTS.items():
        length = (FRAMES - 60) // count
        writer = writer_for(directory / (condition + ".avi"))
        truth = []
        try:
            for index in range(FRAMES):
                frame = background()
                row = {"frame": index + 1, "objects": []}
                if index >= 60:
                    event = (index - 60) // length + 1
                    local = (index - 60) % length
                    boxes = boxes_for(condition, local, length)
                    for name, bbox in boxes:
                        x, y, w, h = bbox
                        color = (70, 180, 235) if name == "A" else (235, 130, 70)
                        cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), color, -1)
                        cv2.putText(frame, name, (x + 13, y + 49), cv2.FONT_HERSHEY_DUPLEX,
                                    1.15, (20, 20, 20), 2)
                        occlusion = overlap_fraction(bbox, boxes[1][1]) if len(boxes) == 2 and name == "A" else 0
                        row["objects"].append({"name": name, "bbox": bbox, "occlusion": occlusion})
                    if condition == "lighting" and length // 3 <= local < 2 * length // 3:
                        frame = cv2.add(frame, np.full_like(frame, 85))
                    if local == 0:
                        events.append({"video": condition + ".avi", "condition": condition,
                                       "event": event, "start": index + 1, "end": index + length})
                cv2.putText(frame, "{} | frame {}".format(condition, index + 1), (12, 335),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)
                writer.write(frame)
                truth.append(row)
        finally:
            writer.release()
        (directory / (condition + "-truth.json")).write_text(json.dumps(truth), encoding="utf-8")
    (directory / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")

    writer = writer_for(directory / "target_demo.avi")
    try:
        target = make_target()
        for index in range(160):
            frame = background()
            if index >= 40:
                x = 25 + (index - 40) * 2
                frame[100:280, x:x + 256] = target
            writer.write(frame)
    finally:
        writer.release()
    print("Created four 24s scenarios, 30 events, target image and 8s recognition demo:", directory)


if __name__ == "__main__":
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--root", type=Path, default=Path("data"))
    generate(command.parse_args().root)
