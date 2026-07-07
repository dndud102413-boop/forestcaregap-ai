# -*- coding: utf-8 -*-
"""PU(Positive-Unlabeled) 학습 보정 — 라벨 불완전성을 정면으로 다룬다.

문제: 우리 라벨 has_recent_management 의 '1'은 '관리 기록이 있음'(확실한 Positive)이지만
      '0'은 '관리 안 됨'이 아니라 '기록이 없음'(Unlabeled — 실제론 관리됐을 수도).
      이를 단순 이진분류로 0=음성 취급하면 관리확률이 체계적으로 과소추정된다.

핵심(Elkan & Noto, KDD 2008, SCAR 가정):
  관측라벨 s, 진짜라벨 y 일 때 P(s=1|y=1)=c (상수, '라벨 빈도').
  분류기 g(x)=P(s=1|x) 를 학습하면  P(y=1|x) = g(x) / c.
  c 는 '확실한 양성'들에서의 g 평균으로 추정한다(교차적합으로 낙관 편향 제거).

활용: ① 진짜 관리율(prevalence) 추정 = 관측율 / c  → "기록 누락 보정"
      ② PU 보정 관리확률 = min(1, g/c)  → 관리공백 점수의 절대수준 보정
주의: SCAR 하에서 c 로 나누는 것은 단조변환이라 '순위(AUC)'는 불변 — PU 의 기여는
      랭킹이 아니라 '유병률 추정 + 절대확률 보정 + 방법론 정당성'이다.
"""
from __future__ import annotations
import numpy as np


def estimate_c(g_on_positives: "np.ndarray") -> float:
    """라벨 빈도 c = P(s=1|y=1) ≈ 확실한 양성에서의 g(x) 평균."""
    g = np.asarray(g_on_positives, dtype=float)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return 1.0
    return float(np.clip(g.mean(), 1e-6, 1.0))


def pu_correct(prob_s: "np.ndarray", c: float) -> "np.ndarray":
    """관측확률 g(x)=P(s=1|x) → PU 보정 관리확률 P(y=1|x)=min(1, g/c)."""
    c = max(float(c), 1e-6)
    return np.clip(np.asarray(prob_s, dtype=float) / c, 0.0, 1.0)


def estimated_prevalence(observed_pos_rate: float, c: float) -> float:
    """진짜 관리율 추정 = 관측 관리율 / c (≥ 관측율)."""
    c = max(float(c), 1e-6)
    return float(min(1.0, observed_pos_rate / c))
