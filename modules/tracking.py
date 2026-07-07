"""관리 후 성과 추적 로직.

관리 전/후 관리공백 가능성 점수 변화를 비교해 개선 정도를 사전진단 수준으로 평가한다.
"""

from __future__ import annotations


def calculate_management_outcome(before_score: int, after_score: int) -> dict:
    """관리 전/후 점수를 비교해 성과 등급과 메시지를 산출한다.

    Returns
    -------
    dict : {"score_reduction": int, "outcome_grade": str, "message": str}
    """
    try:
        before = int(before_score)
        after = int(after_score)
    except (TypeError, ValueError):
        return {
            "score_reduction": 0,
            "outcome_grade": "데이터 부족",
            "message": "관리 전/후 점수 데이터가 올바르지 않아 성과를 평가할 수 없습니다.",
        }

    reduction = before - after  # 점수가 줄수록(=관리공백 가능성 감소) 개선

    if reduction >= 20:
        grade = "크게 개선"
    elif reduction >= 10:
        grade = "개선"
    elif reduction >= 1:
        grade = "일부 개선"
    else:
        grade = "변화 없음 또는 추가 관리 필요"

    if reduction > 0:
        message = (
            f"관리공백 가능성 점수가 {reduction}점 감소하여 관리 상태 개선 가능성이 "
            f"확인되었습니다. ('{grade}', 사전진단)"
        )
    else:
        message = (
            f"관리공백 가능성 점수가 {abs(reduction)}점 변화(감소 없음)하여 추가 관리 "
            f"또는 현장 재점검이 필요할 수 있습니다. ('{grade}', 사전진단)"
        )

    return {
        "score_reduction": reduction,
        "outcome_grade": grade,
        "message": message,
    }
