"""관리공백 가능성 점수화 로직.

특정 산림이 '방치되었다'고 확정하지 않는다.
산림공간정보·관리이력 데이터를 기반으로 '관리공백 가능성(우선관리 필요도)'을
사전 진단하기 위한 점수만 산출한다.
"""

from __future__ import annotations

import pandas as pd


# --------------------------------------------------------------------------- #
# 안전한 형 변환 유틸 (CSV 로 읽으면 bool 이 "True"/"False" 문자열이 될 수 있음)
# --------------------------------------------------------------------------- #
def _truthy(value) -> bool:
    """다양한 표현의 참/거짓 값을 안전하게 bool 로 변환한다."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y", "t", "예", "참")
    if pd.isna(value):
        return False
    return bool(value)


def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# 점수 규칙: (설명 라벨, 조건함수(row)->bool, 가중치)
# 기획서 5장 표를 그대로 코드화한 것이며, 규칙 추가/수정이 쉽도록 리스트로 둔다.
# --------------------------------------------------------------------------- #
SCORING_RULES = [
    ("최근 관리 이력 없음",       lambda r: not _truthy(r.get("has_recent_management")),        25),
    ("관리 이력 오래됨(10년 이상)", lambda r: _to_int(r.get("management_history_years")) >= 10,    15),
    ("고영급 임분(5영급 이상)",    lambda r: _to_int(r.get("age_class")) >= 5,                    15),
    ("수관밀도 높음(밀)",         lambda r: str(r.get("crown_density")).strip() == "밀",          15),
    ("임도 거리 멂(500m 이상)",   lambda r: _to_float(r.get("forest_road_distance_m")) >= 500,    15),
    ("경사 큼(25도 이상)",        lambda r: _to_float(r.get("slope_deg")) >= 25,                  10),
    ("산사태 위험 높음",          lambda r: str(r.get("landslide_risk")).strip() == "높음",        10),
    ("산불 위험 높음",            lambda r: str(r.get("fire_risk")).strip() == "높음",             10),
    ("부재산주",                 lambda r: _truthy(r.get("owner_absentee")),                     10),
    ("사유림(자율관리 부담)",      lambda r: str(r.get("owner_type")).strip() == "사유",            5),
    ("보호구역",                 lambda r: _truthy(r.get("protected_area")),                      5),
]
# owner_type(소유구분) 활용:
#  - '사유'림은 관리주체가 분산·영세하여 관리공백 위험이 구조적으로 높음 → +5
#  - '부재산주'(사유 중 자율관리 곤란 추정)는 별도 +10
#  - 국·공유는 가점 없음(공공이 관리). 국·공유만 보려면 앱에서 owner_type 필터 사용.


def _grade_and_risk(score: int) -> tuple[str, str]:
    """0~100 점수를 등급/위험수준 라벨로 변환한다."""
    if score >= 80:
        return "우선관리 필요", "높음"
    if score >= 60:
        return "관리 필요", "다소 높음"
    if score >= 40:
        return "모니터링 필요", "중간"
    return "관리 우선순위 낮음", "낮음"


def calculate_care_gap_score(row: pd.Series) -> dict:
    """단일 산림 폴리곤(row)의 관리공백 가능성 점수를 계산한다.

    Returns
    -------
    dict : {
        "score": int(0~100),
        "grade": str,
        "risk_level": str,
        "factors": [{"factor": str, "score": int}, ...]   # 기여 점수 내림차순
    }
    """
    factors: list[dict] = []
    raw_score = 0

    for label, condition, weight in SCORING_RULES:
        try:
            hit = bool(condition(row))
        except Exception:
            # 데이터 누락/형식 오류는 '해당 없음'으로 처리해 점수화가 중단되지 않게 한다.
            hit = False
        if hit:
            raw_score += weight
            factors.append({"factor": label, "score": weight})

    # 0~100 으로 상한 제한
    score = max(0, min(100, raw_score))
    grade, risk_level = _grade_and_risk(score)

    # 기여 점수 큰 순으로 정렬
    factors.sort(key=lambda f: f["score"], reverse=True)

    return {
        "score": score,
        "grade": grade,
        "risk_level": risk_level,
        "factors": factors,
    }


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """전체 폴리곤 DataFrame 에 점수/등급/위험수준 컬럼을 붙여 반환한다(지자체 탭용)."""
    results = df.apply(calculate_care_gap_score, axis=1)
    out = df.copy()
    out["care_gap_score"] = [r["score"] for r in results]
    out["care_gap_grade"] = [r["grade"] for r in results]
    out["care_gap_risk"] = [r["risk_level"] for r in results]
    return out
