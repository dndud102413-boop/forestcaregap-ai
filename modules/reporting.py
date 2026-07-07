"""산주 설득용 AI 리포트 생성.

리포트는 사전진단 참고자료이며, 법적·행정적 확정 문서가 아니다.
반드시 마지막에 사전진단 안내 문구를 포함한다.
"""

from __future__ import annotations

import pandas as pd

# 모든 리포트 말미에 반드시 들어가는 고정 안내 문구
DISCLAIMER = (
    "※ 본 결과는 특정 산림의 방치 여부를 법적·행정적으로 확정하는 것이 아니라, "
    "산림공간정보와 관리이력 데이터를 기반으로 관리공백 가능성이 높은 산림을 "
    "사전 선별하기 위한 의사결정 참고자료입니다."
)


def _risk_if_unmanaged(row: pd.Series, scoring_result: dict) -> list[str]:
    """관리하지 않을 경우 예상되는 위험을 데이터 기반으로 나열한다(확정 아님)."""
    risks: list[str] = []
    if str(row.get("landslide_risk")).strip() == "높음":
        risks.append("산사태 위험 구간으로, 미관리 시 재해 피해 가능성이 커질 수 있습니다.")
    if str(row.get("fire_risk")).strip() == "높음":
        risks.append("산불 위험이 높아, 연료물질 누적 시 산불 확산 위험이 증가할 수 있습니다.")
    if str(row.get("crown_density")).strip() == "밀":
        risks.append("과밀 임분은 생장 저하·병해충 취약성·고사목 증가로 이어질 수 있습니다.")
    if scoring_result.get("score", 0) >= 60:
        risks.append("관리 시점을 놓칠 경우 향후 관리 비용이 증가하고 임분 가치가 저하될 수 있습니다.")
    if not risks:
        risks.append("현재 데이터 기준 즉각적 위험은 낮으나, 미관리 장기화 시 상태 악화 가능성은 존재합니다.")
    return risks


def generate_landowner_report(
    row: pd.Series,
    scoring_result: dict,
    explanation: str,
    prescriptions: list[dict],
    matched_programs: pd.DataFrame,
) -> str:
    """산주용 설득 리포트(텍스트)를 생성한다."""
    pid = row.get("polygon_id", "-")
    region = row.get("region", "-")
    species = row.get("species", "-")
    area = row.get("area_ha", "-")
    age = row.get("age_class", "-")
    score = scoring_result.get("score", 0)
    grade = scoring_result.get("grade", "-")

    lines: list[str] = []
    lines.append("=" * 56)
    lines.append("       ForestCareGap AI · 산림 관리공백 사전진단 리포트")
    lines.append("=" * 56)

    # 1. 산림 기본 정보
    lines.append("\n[1] 산림 기본 정보")
    lines.append(f"  · 폴리곤 ID : {pid}")
    lines.append(f"  · 위치(지역): {region}")
    lines.append(f"  · 수종 / 영급: {species} / {age}영급")
    lines.append(f"  · 면적      : {area} ha")

    # 2. 관리공백 가능성 점수
    lines.append("\n[2] 관리공백 가능성 점수 (사전진단)")
    lines.append(f"  · 점수 : {score} / 100")
    lines.append(f"  · 등급 : {grade}")
    lines.append(f"  · 위험수준 : {scoring_result.get('risk_level', '-')}")

    # 3. 주요 원인
    lines.append("\n[3] 주요 원인 (설명)")
    lines.append(f"  {explanation}")

    # 4. 관리하지 않을 경우 예상 위험
    lines.append("\n[4] 관리하지 않을 경우 예상되는 위험")
    for r in _risk_if_unmanaged(row, scoring_result):
        lines.append(f"  · {r}")

    # 5. 추천 관리 처방
    lines.append("\n[5] 추천 관리 처방 (검토 권고)")
    if prescriptions:
        for i, p in enumerate(prescriptions, 1):
            lines.append(f"  {i}. [{p['priority']}] {p['action']} ({p['category']})")
            lines.append(f"     - 사유: {p['reason']}")
    else:
        lines.append("  · 즉시 처방 필요성은 낮음 (정기 모니터링 권장)")

    # 6. 지원사업 검토 후보
    lines.append("\n[6] 지원사업 검토 후보 (신청 가능 확정 아님)")
    if matched_programs is not None and not matched_programs.empty:
        for _, prog in matched_programs.iterrows():
            lines.append(f"  · {prog.get('program_name', '-')} [{prog.get('category', '-')}]")
            lines.append(f"     - {prog.get('description', '')}")
            lines.append(f"     - 문의: {prog.get('contact_channel', '-')} / {prog.get('검토 상태', '검토 후보')}")
    else:
        lines.append("  · 매칭된 지원사업 검토 후보가 없습니다. (산림조합/지자체 상담 권장)")

    # 7. 상담 안내
    lines.append("\n[7] 산림조합·지자체 상담 안내")
    lines.append("  · 본 사전진단 결과를 바탕으로 관할 산림조합 또는 지자체 산림부서와 상담을")
    lines.append("    진행하시면, 현장조사 및 지원사업 적용 가능 여부를 확인할 수 있습니다.")

    # 8. 사전진단 안내 문구(필수)
    lines.append("\n[8] 안내")
    lines.append(f"  {DISCLAIMER}")
    lines.append("=" * 56)

    return "\n".join(lines)
