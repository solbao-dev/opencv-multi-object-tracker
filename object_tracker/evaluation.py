"""측정 보조 계측: 정답과 프레임 로그를 비교하되 수동 관찰과 구별합니다."""

from __future__ import annotations

from math import hypot
from typing import Dict, List


def assign_truth(objects: List[dict], tracks: List[dict]) -> Dict[str, int]:
    """평가에만 쓰는 일대일 중심점 연결. 추적 알고리즘에는 정답을 주지 않습니다."""
    pairs = []
    for obj in objects:
        if obj["occlusion"] >= 0.5:
            continue
        x, y, w, h = obj["bbox"]
        for track in tracks:
            if track["missed"]:
                continue
            xx, yy, ww, hh = track["bbox"]
            distance = hypot(x + w / 2 - xx - ww / 2, y + h / 2 - yy - hh / 2)
            # 거대 전경 박스나 옆 객체를 정답으로 오인하지 않도록 크기도 확인합니다.
            if distance <= 45 and ww <= w * 2.5 and hh <= h * 2.5:
                pairs.append((distance, obj["name"], track["id"]))
    assigned, used = {}, set()
    for _, name, track_id in sorted(pairs):
        if name not in assigned and track_id not in used:
            assigned[name] = track_id
            used.add(track_id)
    return assigned


def measure_event(truth: List[dict], logs: List[dict], start: int, end: int) -> dict:
    """연속 5/10프레임 기준을 적용하고 지속되는 한 사건은 한 번만 셉니다.

    프레임 번호는 1부터, 양 끝 포함. 가림 >=50%는 실패 연속 구간을 끊습니다.
    최초 연결 ID를 기준으로 시작하고, 다른 ID가 5회 연속이면 switch로 확정한 뒤
    기준을 새 ID로 옮깁니다. 단순히 ID 숫자가 큰 것을 실패로 세지 않습니다.
    """
    states, switches, failures, excluded = {}, 0, 0, 0
    details = []
    for number in range(start, end + 1):
        objects = truth[number - 1]["objects"]
        assigned = assign_truth(objects, logs[number - 1]["tracks"])
        for obj in objects:
            name = obj["name"]
            state = states.setdefault(name, {"id": None, "candidate": None,
                                             "changed": 0, "missing": 0})
            if obj["occlusion"] >= 0.5:
                excluded += 1
                state.update(missing=0, changed=0, candidate=None)
                continue
            observed = assigned.get(name)
            if observed is None:
                state["missing"] += 1
                state.update(changed=0, candidate=None)
                if state["missing"] == 10:
                    failures += 1
                    details.append({"kind": "failure", "object": name,
                                    "start": number - 9, "confirmed": number})
                continue
            state["missing"] = 0
            if state["id"] is None:
                state["id"] = observed
            if observed == state["id"]:
                state.update(candidate=None, changed=0)
                continue
            state["changed"] = state["changed"] + 1 if state["candidate"] == observed else 1
            state["candidate"] = observed
            if state["changed"] == 5:
                switches += 1
                details.append({"kind": "switch", "object": name, "start": number - 4,
                                "confirmed": number, "old": state["id"], "new": observed})
                state.update(id=observed, candidate=None, changed=0)
    return {"id_switches": switches, "failures": failures,
            "excluded_object_frames": excluded, "details": details}
