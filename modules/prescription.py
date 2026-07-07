"""관리 처방(권고) 추천 로직.

처방은 '검토' 수준의 권고이며, 실제 시행은 현장조사·전문가·지자체 확인이 필요하다.
처방의 category 는 지원사업(support_programs.csv) 카테고리와 일치시켜 매칭이 되도록 한다.
"""

from __future__ import annotations

import pandas as pd

from .scoring import _to_int, _to_float, _truthy

# '높음' 판정 임계값 (샘플 데이터 분포 기준의 합리적 고정값)
ABSORPTION_HIGH = 6.0     # 연간 탄소흡수량(tCO2/년)
TIMBER_HIGH = 200.0       # 임목 잠재량(m3)


def recommend_prescriptions(row: pd.Series, scoring_result: dict) -> list[dict]:
    """산림 조건과 점수 결과를 바탕으로 관리 처방 후보 목록을 생성한다.

    각 처방 dict: {"category", "action", "reason", "priority"}
    category 는 support_programs.csv 의 category 와 매칭된다.
    """
    risk = scoring_result.get("risk_level", "중간")
    high_priority = risk in ("높음", "다소 높음")

    age = _to_int(row.get("age_class"))
    crown = str(row.get("crown_density")).strip()
    landslide = str(row.get("landslide_risk")).strip()
    fire = str(row.get("fire_risk")).strip()
    slope = _to_float(row.get("slope_deg"))
    absorption = _to_float(row.get("annual_absorption_tco2"))
    timber = _to_float(row.get("timber_potential_m3"))
    road_dist = _to_float(row.get("forest_road_distance_m"))
    protected = _truthy(row.get("protected_area"))

    prescriptions: list[dict] = []

    # 1) 고영급 + 고밀도 → 숲가꾸기
    if age >= 5 and crown == "밀":
        prescriptions.append({
            "category": "숲가꾸기",
            "action": "선택적 간벌 또는 숲가꾸기 검토",
            "reason": "고영급·고밀도(밀) 임분으로 과밀 해소 및 생육 개선 검토 필요성이 있습니다.",
            "priority": "높음" if high_priority else "중간",
        })

    # 2) 산사태 위험 + 급경사 → 산사태 예방
    if landslide == "높음" and slope >= 25:
        prescriptions.append({
            "category": "산사태 예방",
            "action": "산사태 위험지 모니터링 및 사방사업 검토",
            "reason": "산사태 위험도가 높고 경사가 가팔라 재해 예방 차원의 검토가 필요합니다.",
            "priority": "높음",
        })

    # 3) 산불 위험 높음 → 산불 예방
    if fire == "높음":
        prescriptions.append({
            "category": "산불 예방",
            "action": "산불 예방을 위한 연료물질 관리 검토",
            "reason": "산불 위험도가 높아 가연성 연료물질 관리·예방 조치 검토가 필요합니다.",
            "priority": "높음" if high_priority else "중간",
        })

    # 4) 탄소흡수량 높음 → 탄소관리 후보
    if absorption >= ABSORPTION_HIGH:
        prescriptions.append({
            "category": "탄소관리",
            "action": "탄소관리(흡수량 유지·증진) 후보지 검토",
            "reason": f"연간 탄소흡수량이 {absorption:.1f} tCO2 수준으로, 탄소관리 후보지 검토 가치가 있습니다.",
            "priority": "중간",
        })

    # 5) 임목 잠재량 높음 + 완경사 → 조림·갱신(국산재 공급 후보)
    if timber >= TIMBER_HIGH and slope < 25:
        prescriptions.append({
            "category": "조림·갱신",
            "action": "국산재 공급 후보지 검토(수확·갱신)",
            "reason": f"임목 잠재량이 {timber:.0f} m3 수준이고 경사가 완만하여 국산재 공급 후보로 검토할 수 있습니다.",
            "priority": "중간",
        })

    # 6) 보호구역 → 현장조사/행정 검토
    if protected:
        prescriptions.append({
            "category": "현장조사",
            "action": "보호구역 관련 행정 검토 및 현장조사 필요",
            "reason": "보호구역에 해당하여 관리 행위 전 관련 행정 검토와 현장조사가 필요합니다.",
            "priority": "중간",
        })

    # 7) 임도 거리 멂 → 현장 접근성/조사 비용 검토
    if road_dist >= 500:
        prescriptions.append({
            "category": "현장조사",
            "action": "현장 접근성 개선 또는 조사 비용 고려",
            "reason": f"임도와의 거리가 {road_dist:.0f} m로 멀어 현장 접근성·조사 비용을 함께 고려해야 합니다.",
            "priority": "중간",
        })

    # 8) 종합 점수가 관리 필요 이상이면 상담 채널 안내 처방 추가
    if scoring_result.get("score", 0) >= 60:
        prescriptions.append({
            "category": "산림조합 상담",
            "action": "산림조합·지자체 상담 검토",
            "reason": "관리공백 가능성이 높게 진단되어 산림조합 또는 지자체 상담을 통한 관리 계획 수립을 권장합니다.",
            "priority": "높음" if high_priority else "중간",
        })

    # 처방이 하나도 없으면 기본 모니터링 권고
    if not prescriptions:
        prescriptions.append({
            "category": "현장조사",
            "action": "정기 모니터링 유지",
            "reason": "현재 데이터 기준 즉시 처방 필요성은 낮으나, 변화 감지를 위한 정기 모니터링을 권장합니다.",
            "priority": "낮음",
        })

    # 우선순위 정렬(높음 > 중간 > 낮음)
    order = {"높음": 0, "중간": 1, "낮음": 2}
    prescriptions.sort(key=lambda p: order.get(p["priority"], 1))
    return prescriptions
