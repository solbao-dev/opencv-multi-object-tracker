"""발표와 관찰 기록에 사용할 연속 프레임 접촉 시트를 만듭니다."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from object_tracker.video_io import write_image


def make_sheets() -> None:
    root = Path("evidence")
    index = json.loads((root / "run_index.json").read_text())
    events = json.loads((root / "event_details.json").read_text())
    directory = root / "review_sheets"
    directory.mkdir(exist_ok=True)
    # 실패/ID 변경은 그 연속 구간을 한 장에서 확인하게 합니다.
    # 0건인 이벤트도 시작/중간/끝의 추적 상태를 빠짐없이 싣습니다.
    for condition in ("single", "overlap", "stop", "lighting"):
        selected = [e for e in events if e["condition"] == condition]
        capture = cv2.VideoCapture(str(Path(index["baseline/" + condition]) / "annotated.avi"))
        for event in selected:
            details = event["details"]
            if details:
                samples = [event["start"] + 3]
                for detail in details:
                    samples += list(range(detail["start"], detail["confirmed"] + 1))
                samples += [event["end"] - 3]
            else:
                samples = list(range(event["start"], event["end"] + 1, 4))
            # 전체 객체와 ID를 읽을 수 있게 세로 위/아래 여백만 잘라냅니다.
            samples = list(dict.fromkeys(samples))
            cards = []
            for number in samples:
                capture.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Cannot read {} frame {}".format(condition, number))
                card = cv2.resize(frame[110:295], (384, 111))
                card = cv2.copyMakeBorder(card, 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 27, 30))
                cv2.putText(card, "{} E{} F{}".format(condition, event["event"], number), (8, 17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1)
                cards.append(card)
            while len(cards) % 3:
                cards.append(np.full_like(cards[0], 25))
            sheet = np.vstack([np.hstack(cards[i:i + 3]) for i in range(0, len(cards), 3)])
            write_image(directory / "{}-{:02d}.jpg".format(condition, event["event"]), sheet)
        capture.release()
        # 30개 이벤트를 종류별로 모아 모든 반복의 전/후 상태를 함께 확인합니다.
        overview = []
        capture = cv2.VideoCapture(str(Path(index["baseline/" + condition]) / "annotated.avi"))
        for event in selected:
            row = []
            for number in (event["start"] + 4, (event["start"] + event["end"]) // 2,
                           event["end"] - 4):
                capture.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Missing overview frame")
                card = cv2.resize(frame[100:295], (384, 117))
                card = cv2.copyMakeBorder(card, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 27, 30))
                cv2.putText(card, "E{} F{}".format(event["event"], number), (8, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1)
                row.append(card)
            overview.append(np.hstack(row))
        capture.release()
        write_image(directory / (condition + "-overview.jpg"), np.vstack(overview))
    print("Created 30 event sheets and four overview sheets:", directory)


if __name__ == "__main__":
    make_sheets()
