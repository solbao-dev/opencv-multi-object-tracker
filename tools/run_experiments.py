"""기본 실행과 같은 run()으로 데이터를 만들고 재현 가능한 수치를 저장합니다."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

from object_tracker.app import AppConfig, run
from object_tracker.evaluation import measure_event
from object_tracker.target import TargetMatcher
from object_tracker.video_io import write_image
from tools.generate_demo import COUNTS, generate


def write_csv(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_experiment(root: Path) -> list:
    matcher = TargetMatcher(Path("data/target.png"))
    reference = matcher.reference
    h, w = reference.shape[:2]
    rows = []
    images = []
    for label, angle, covered in [("front", 0, 0), ("rotation_30", 30, 0),
                                   ("rotation_60", 60, 0), ("occlusion_30", 0, 0.3),
                                   ("occlusion_50", 0, 0.5)]:
        # 캔버스 여유를 확보해 회전으로 잘리는 현상과 회전 자체를 분리합니다.
        canvas = np.full((420, 520, 3), 42, np.uint8)
        canvas[120:120 + h, 130:130 + w] = reference
        if angle:
            matrix = cv2.getRotationMatrix2D((260, 210), angle, 1)
            canvas = cv2.warpAffine(canvas, matrix, (520, 420), borderValue=(42, 42, 42))
        if covered:
            canvas[120:120 + h, 130:130 + round(w * covered)] = 42
        result = matcher.match(canvas)
        row = {"condition": label, "reference_keypoints": len(matcher.keypoints),
               "frame_keypoints": result.frame_keypoints, "matched_keypoints": result.good_matches,
               "inliers": result.inliers,
               "match_rate_percent": round(100 * result.good_matches / len(matcher.keypoints), 2),
               "detected": result.detected}
        rows.append(row)
        cv2.putText(canvas, "{} | good {} | inliers {}".format(label, result.good_matches,
                    result.inliers), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
        write_image(root / "features" / (label + ".png"), canvas)
        images.append(cv2.resize(canvas, (390, 315)))
    blank = np.full_like(images[0], 42)
    write_image(root / "feature_conditions.jpg", np.vstack([
        np.hstack(images[:3]), np.hstack(images[3:] + [blank])]))
    write_csv(root / "feature_results.csv", rows)
    return rows


def main() -> None:
    root = Path("evidence")
    root.mkdir(exist_ok=True)
    required = [Path("data/target.png"), Path("data/demo/events.json")]
    required.extend(Path("data/demo/{}.avi".format(condition)) for condition in COUNTS)
    # 필요한 데모 자산이 없으면 생성하여 처음 실행에도 같은 입력을 사용합니다.
    if not all(path.is_file() for path in required):
        generate()
    events = json.loads(Path("data/demo/events.json").read_text())
    summaries, observations, details, run_index = [], [], [], {}
    # 거리와 누락 허용 횟수를 조절해 위치 기반 추적의 한계를 비교합니다.
    experiments = [("baseline", 0.01, 70, 15), ("slow_learning", 0.001, 70, 15),
                   ("fast_learning", 0.1, 70, 15), ("small_gate", 0.01, 20, 15),
                   ("long_retention", 0.01, 70, 40)]
    for name, rate, distance, missed in experiments:
        for condition in COUNTS:
            config = AppConfig(source="data/demo/{}.avi".format(condition), target=Path("data/target.png"),
                               output=root / "runs" / name / condition, learning_rate=rate,
                               max_distance=distance, max_missed=missed, width=640,
                               headless=True, save_video=name == "baseline")
            directory = run(config)
            run_index["{}/{}".format(name, condition)] = directory.as_posix()
            logs = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
            truth = json.loads(Path("data/demo/{}-truth.json".format(condition)).read_text())
            measurements = []
            for event in [e for e in events if e["condition"] == condition]:
                measured = measure_event(truth, logs, event["start"], event["end"])
                measurements.append(measured)
                if name == "baseline":
                    observations.append({"video": event["video"], "condition": condition,
                        "event": event["event"], "start_frame": event["start"], "end_frame": event["end"],
                        "id_switches": measured["id_switches"], "failures": measured["failures"],
                        "excluded_object_frames": measured["excluded_object_frames"],
                        "method": "synthetic_ground_truth_assisted", "notes": "See event_details.json; manual review is separate"})
                    details.append(dict(event, **measured))
            summaries.append({"experiment": name, "condition": condition, "learning_rate": rate,
                "max_distance": distance, "max_missed": missed, "tests": len(measurements),
                "id_switches": sum(m["id_switches"] for m in measurements),
                "failures": sum(m["failures"] for m in measurements),
                "failed_events": sum(m["failures"] > 0 for m in measurements),
                "failure_rate_percent": round(100 * sum(m["failures"] > 0 for m in measurements) / len(measurements), 2),
                "median_processing_fps": round(statistics.median(row["processing_fps"] for row in logs[1:]), 2)})
            print(name, condition, summaries[-1], flush=True)
    write_csv(root / "tracking_results.csv", summaries)
    write_csv(root / "automatic_observations.csv", observations)
    (root / "event_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (root / "run_index.json").write_text(json.dumps(run_index, indent=2), encoding="utf-8")
    feature_experiment(root)
    print("Evidence saved:", root)


if __name__ == "__main__":
    main()
