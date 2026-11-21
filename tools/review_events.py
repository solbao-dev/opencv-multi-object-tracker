"""추적 결과를 프레임 단위로 탐색하고 이벤트별 관찰 CSV를 저장합니다.

기본 실험의 결과 영상을 자동으로 선택하며 --video로 다른 영상을 지정할 수
있습니다. 계수는 5/10프레임의 지속 조건과 50% 이상 가림 제외를 기준으로 합니다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import cv2


def resolve_video(video: Optional[Path], condition: str,
                  index_path: Path = Path("evidence/run_index.json")) -> Path:
    """명시한 영상을 우선하고, 생략하면 조건에 맞는 기본 실험 결과를 찾습니다."""
    if video is not None:
        return video
    hint = "Run python -m tools.run_experiments first, or pass --video."
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        path = Path(index["baseline/" + condition]) / "annotated.avi"
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("Cannot select baseline video. " + hint) from error
    if not path.is_file():
        raise ValueError("Baseline video does not exist: {}. {}".format(path, hint))
    return path


def review(video: Path, events_path: Path, condition: str, output: Path) -> None:
    events = [e for e in json.loads(events_path.read_text(encoding="utf-8"))
              if e["condition"] == condition]
    if not events:
        raise ValueError("No events for condition: " + condition)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("Cannot open " + str(video))
    window = "Manual Event Review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    index, number, switches, failures, excluded = 0, events[0]["start"], 0, 0, False
    # 기존 관찰을 보존하고, 같은 video/condition/event를 저장할 때만 갱신합니다.
    rows = []
    if output.exists():
        with output.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    try:
        while index < len(events):
            event = events[index]
            capture.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Cannot decode frame {}".format(number))
            panel = cv2.copyMakeBorder(frame, 0, 104, 0, 0, cv2.BORDER_CONSTANT, value=(28, 30, 35))
            texts = ["Event {}/{} | frames {}-{} | now {}".format(index + 1, len(events), event["start"], event["end"], number),
                     "switches:{} failures:{} occlusion>=50%:{}".format(switches, failures, excluded),
                     "a/d: -/+1  j/l: -/+10  x: switch  f: failure  o: occlusion",
                     "r: reset counts   s: save & next event   q: quit (current unsaved)"]
            for line, text in enumerate(texts):
                cv2.putText(panel, text, (10, frame.shape[0] + 22 + line * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.imshow(window, panel)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("a"), ord("d"), ord("j"), ord("l")):
                step = {ord("a"): -1, ord("d"): 1, ord("j"): -10, ord("l"): 10}[key]
                number = max(event["start"], min(event["end"], number + step))
            elif key == ord("x"):
                switches += 1
            elif key == ord("f"):
                failures += 1
            elif key == ord("o"):
                excluded = not excluded
            elif key == ord("r"):
                switches, failures, excluded = 0, 0, False
            elif key == ord("s"):
                row = {"video": event["video"], "condition": condition, "event": event["event"],
                       "start_frame": event["start"], "end_frame": event["end"],
                       "id_switches": switches, "failures": failures,
                       "method": "manual_frame_review",
                       "notes": "occlusion >=50% excluded" if excluded else "no >=50% occlusion marked"}
                rows = [old for old in rows if (old["video"], old["condition"], str(old["event"])) !=
                        (row["video"], condition, str(row["event"]))]
                rows.append(row)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                index += 1
                if index < len(events):
                    number = events[index]["start"]
                    switches, failures, excluded = 0, 0, False
                print("Saved event", event["event"], flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--video", type=Path, help="Annotated video; defaults to the baseline run")
    command.add_argument("--events", type=Path, default=Path("data/demo/events.json"))
    command.add_argument("--condition", required=True)
    command.add_argument("--output", type=Path, default=Path("evidence/manual_observations.csv"))
    args = command.parse_args()
    try:
        review(resolve_video(args.video, args.condition), args.events, args.condition, args.output)
    except (ValueError, OSError, cv2.error) as error:
        command.exit(1, "Error: {}\n".format(error))
