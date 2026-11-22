"""모폴로지 커널 크기 효과를 기존 합성 입력으로 보충 측정합니다.

프로젝트 루트에서 python -m tools.measure_morphology 명령으로 실행합니다.
기존 감지기·추적기·평가 함수를 재사용하고 결과는 evidence/morphology_results.json에 저장합니다.
"""

import json
import platform
from pathlib import Path

import cv2
import numpy as np

from object_tracker.evaluation import measure_event
from object_tracker.motion import MotionDetector
from object_tracker.tracker import Tracker


def measure_kernels():
    """커널 크기만 바꾼 두 영상의 자동 보조 측정 결과 목록을 반환합니다.

    같은 입력·학습률·면적·추적 기준을 유지하고 매 실행의 배경 모델을 새로
    생성합니다. 파일 읽기 실패 시 RuntimeError를 발생시킵니다.
    """
    events = json.loads(Path('data/demo/events.json').read_text())
    rows = []
    for condition in ('overlap', 'lighting'):
        truth = json.loads(Path(f'data/demo/{condition}-truth.json').read_text())
        selected = [event for event in events if event['condition'] == condition]
        for opening, closing in ((1, 7), (3, 7), (7, 7), (3, 3), (3, 15)):
            detector = MotionDetector()
            # 실행 객체의 커널만 교체해 기존 감지·추적 함수를 그대로 사용합니다.
            detector._opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening, opening))
            detector._closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing, closing))
            tracker = Tracker()
            capture = cv2.VideoCapture(f'data/demo/{condition}.avi')
            if not capture.isOpened():
                raise RuntimeError(f'Cannot open {condition}')
            logs = []
            shortfall_frames = two_visible_frames = foreground_pixels = 0
            try:
                for actual in truth:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError('Unexpected end of video')
                    detections, mask = detector.detect(frame)
                    tracks = tracker.update(detections)
                    logs.append({'tracks': [
                        {'id': track.id, 'bbox': list(track.bbox), 'missed': track.missed}
                        for track in tracks]})
                    foreground_pixels += cv2.countNonZero(mask)
                    visible = sum(obj['occlusion'] < 0.5 for obj in actual['objects'])
                    if visible == 2:
                        two_visible_frames += 1
                        shortfall_frames += len(detections) < 2
            finally:
                capture.release()
            measured = [measure_event(truth, logs, event['start'], event['end'])
                        for event in selected]
            rows.append({
                'condition': condition, 'opening': opening, 'closing': closing,
                'frames': len(logs), 'events': len(selected),
                'id_switches': sum(item['id_switches'] for item in measured),
                'failures': sum(item['failures'] for item in measured),
                'failed_events': sum(item['failures'] > 0 for item in measured),
                'two_visible_frames': two_visible_frames,
                'fewer_than_two_detections_frames': shortfall_frames,
                'foreground_pixels_sum': foreground_pixels,
            })
    return rows


if __name__ == '__main__':
    results = {
        'environment': {'python': platform.python_version(), 'opencv': cv2.__version__,
                        'numpy': np.__version__, 'platform': platform.platform()},
        'method': 'Supplemental synthetic-input automatic measurements; fresh models per run',
        'kernels': measure_kernels(),
    }
    destination = Path('evidence/morphology_results.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(destination)
