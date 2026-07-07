"""관리공백 가능성 점수의 근거를 자연어로 설명한다.

표현 규칙:
- '방치 확정', '불법 방치' 등 단정 표현 금지
- '관리공백 가능성', '우선관리 필요성', '사전진단' 표현 사용
"""

from __future__ import annotations

import pandas as pd


# 점수 요인 라벨 -> 자연어 절(clause) 매핑
_FACTOR_CLAUSE = {
    "최근 관리 이력 없음": "최근 산림사업 이력이 확인되지 않고",
    "관리 이력 오래됨(10년 이상)": "마지막 관리로부터 10년 이상 경과한 것으로 보이며",
    "고영급 임분(5영급 이상)": "5영급 이상의 고영급 임분으로 관리 시점이 도래하였고",
    "수관밀도 높음(밀)": "수관밀도가 '밀'로 과밀 상태이며",
    "임도 거리 멂(500m 이상)": "임도와의 거리가 멀어 현장 관리 접근성이 낮고",
    "경사 큼(25도 이상)": "경사가 가파른 지형이며",
    "산사태 위험 높음": "산사태 위험도가 높고",
    "산불 위험 높음": "산불 위험도가 높으며",
    "부재산주": "부재산주 가능성이 있어 자율적 관리가 어려울 수 있고",
    "보호구역": "보호구역에 해당하여 행정적 관리 검토가 필요하며",
}


def generate_factor_explanation(scoring_result: dict, row: pd.Series) -> str:
    """점수에 영향을 준 주요 요인(상위 3~5개)을 자연어 설명으로 변환한다."""
    factors = scoring_result.get("factors", [])
    grade = scoring_result.get("grade", "사전진단")
    score = scoring_result.get("score", 0)

    region = str(row.get("region", "해당 지역"))
    species = str(row.get("species", "해당 수종"))

    if not factors:
        return (
            f"{region}의 {species} 임분은 현재 데이터 기준으로 관리공백 가능성을 높이는 "
            f"뚜렷한 요인이 확인되지 않았습니다. 다만 현장 여건은 달라질 수 있으므로 "
            f"정기 모니터링을 권장합니다. (사전진단 결과)"
        )

    # 상위 3~5개 요인만 사용
    top = factors[:5] if len(factors) > 3 else factors
    clauses = [_FACTOR_CLAUSE.get(f["factor"], f["factor"]) for f in top]

    # 절들을 자연스럽게 연결
    if len(clauses) == 1:
        body = clauses[0]
    else:
        body = ", ".join(clauses[:-1]) + ", " + clauses[-1]

    # 등급에 따른 마무리 문장
    closing = {
        "우선관리 필요": "이에 따라 우선관리 필요성이 높게 평가되었습니다.",
        "관리 필요": "이에 따라 관리 필요성이 있는 것으로 평가되었습니다.",
        "모니터링 필요": "이에 따라 지속적인 모니터링이 필요한 것으로 평가되었습니다.",
        "관리 우선순위 낮음": "다만 종합적으로는 관리 우선순위가 낮은 편으로 평가되었습니다.",
    }.get(grade, "관리공백 가능성에 대한 사전진단 결과입니다.")

    return (
        f"해당 산림({region}, {species})은 {body}, {closing} "
        f"(관리공백 가능성 점수 {score}점 · 사전진단)"
    )
