"""처방을 지원사업 '검토 후보'와 연결한다.

표현 규칙:
- '지원사업 신청 가능 확정' 금지
- '검토 후보', '상담 필요', '지자체 확인 필요'로 표현
"""

from __future__ import annotations

import pandas as pd


def match_support_programs(prescriptions: list[dict], programs_df: pd.DataFrame) -> pd.DataFrame:
    """처방 목록의 category 와 지원사업 category 를 매칭해 검토 후보를 반환한다.

    Returns
    -------
    pd.DataFrame : 매칭된 지원사업 + '연계 처방' / '검토 상태' 컬럼
                   (매칭 결과가 없으면 빈 DataFrame)
    """
    if programs_df is None or programs_df.empty or not prescriptions:
        return pd.DataFrame(
            columns=[
                "program_name", "category", "연계 처방", "description",
                "contact_channel", "검토 상태", "disclaimer",
            ]
        )

    # category -> 대표 처방 action 매핑(같은 카테고리 처방이 여러 개면 첫 항목)
    cat_to_action: dict[str, str] = {}
    for p in prescriptions:
        cat_to_action.setdefault(p["category"], p["action"])

    categories = list(cat_to_action.keys())
    matched = programs_df[programs_df["category"].isin(categories)].copy()

    if matched.empty:
        return pd.DataFrame(
            columns=[
                "program_name", "category", "연계 처방", "description",
                "contact_channel", "검토 상태", "disclaimer",
            ]
        )

    matched["연계 처방"] = matched["category"].map(cat_to_action)
    # 모든 매칭 결과는 '확정'이 아니라 '검토 후보' 상태로 표기
    matched["검토 상태"] = "검토 후보 (지자체·산림조합 확인 필요)"

    cols = [
        "program_name", "category", "연계 처방", "description",
        "contact_channel", "검토 상태", "disclaimer",
    ]
    cols = [c for c in cols if c in matched.columns]
    return matched[cols].reset_index(drop=True)
