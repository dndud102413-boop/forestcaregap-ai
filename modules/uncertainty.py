# -*- coding: utf-8 -*-
"""예측 불확실성(신뢰도) — 점수 하나가 아니라 '얼마나 확신하는지'를 같이 준다.

두 축을 결합한다(둘 다 honest):
  ① 모델 내부 불일치 — RandomForest 400그루의 관리확률 표준편차(std).
     트리들이 갈릴수록(=std 큼) 그 필지 판단의 불확실성이 크다(앙상블 분산 근사).
  ② 지역 일반화 신뢰 — 검증의 '지역별 honest AUC'(시군구 공간 홀드아웃).
     약한 지역(예: 삼척 0.50)은 그 자체로 신뢰도를 낮춘다.

출력: ai_confidence ∈ {높음, 보통, 낮음}.  triage 도구가 '확신 높은 곳부터' 가도록.
주의: 이는 확률의 정답 보장이 아니라 '상대적 확신도' 표시다.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def tree_disagreement(X, rf) -> "np.ndarray":
    """RF 각 트리의 P(관리) 표준편차(메모리 절약 위해 누적합으로 계산)."""
    n = X.shape[0]; T = len(rf.estimators_)
    s = np.zeros(n); s2 = np.zeros(n)
    for est in rf.estimators_:
        p = est.predict_proba(X)[:, 1]
        s += p; s2 += p * p
    mean = s / T
    var = np.clip(s2 / T - mean * mean, 0.0, None)
    return np.sqrt(var)


def region_auc_map(validation_report: dict) -> dict:
    """validation_report.json → {시군구: honest AUC}."""
    if not validation_report:
        return {}
    pra = validation_report.get("per_region_auc", {})
    return {str(k): (v.get("auc") if isinstance(v, dict) else None) for k, v in pra.items()}


def assign_confidence(std_arr, region_arr, rmap: dict,
                      weak_auc: float = 0.6, ok_auc: float = 0.7) -> "np.ndarray":
    """트리 불일치(전역 3분위) + 지역 honest AUC → {높음/보통/낮음}.

    규칙(보수적·정직): 약한 지역(AUC<weak_auc)은 최고 '보통'으로 제한,
    매우 약하면 '낮음'. 그 외엔 트리 불일치가 작을수록 높은 신뢰.
    """
    std = np.asarray(std_arr, dtype=float)
    # 전역 3분위 경계(작을수록 확신↑)
    q1, q2 = np.nanpercentile(std, [33, 66])
    base = np.where(std <= q1, "높음", np.where(std <= q2, "보통", "낮음"))
    out = np.array(base, dtype=object)
    for i, reg in enumerate(region_arr):
        a = rmap.get(str(reg))
        if a is None:
            continue
        if a < weak_auc:                          # 약한 지역: 신뢰 강등
            out[i] = "낮음" if (a < weak_auc - 0.05 or out[i] == "낮음") else "보통"
        elif a < ok_auc and out[i] == "높음":     # 보통 지역: 높음→보통
            out[i] = "보통"
    return out


# 전역 트리 std 3분위(562k 실측: 0.093/0.140 — precompute 의 ai_confidence 가 정확본,
# 이 상수는 앱 즉석 1필지용 폴백 임계)
STD_Q33, STD_Q66 = 0.093, 0.140


def confidence_single(std: float, region_auc, weak_auc: float = 0.6, ok_auc: float = 0.7) -> str:
    """단일 필지 신뢰도(고정 임계 + 지역 게이트). assign_confidence 와 동일 논리."""
    level = "높음" if std <= STD_Q33 else ("보통" if std <= STD_Q66 else "낮음")
    if region_auc is not None:
        if region_auc < weak_auc:
            level = "낮음" if (region_auc < weak_auc - 0.05 or level == "낮음") else "보통"
        elif region_auc < ok_auc and level == "높음":
            level = "보통"
    return level


CONF_ICON = {"높음": "🟢", "보통": "🟡", "낮음": "🔴"}


def confidence_note(level: str, region_auc) -> str:
    icon = CONF_ICON.get(level, "")
    ra = (f"지역 honest AUC {region_auc:.2f}" if isinstance(region_auc, (int, float)) else "지역 신뢰도 미상")
    tail = {
        "높음": "트리 합의·지역 일반화 모두 양호 — 우선 현장확인 후보로 신뢰.",
        "보통": "일부 불확실 — 현장확인 권장.",
        "낮음": "모델 불확실(트리 의견 분산 또는 약한 지역) — 현장확인을 특히 우선.",
    }.get(level, "")
    return f"{icon} 신뢰도 {level} ({ra}) — {tail}"
