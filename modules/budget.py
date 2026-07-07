# -*- coding: utf-8 -*-
"""예산 우선순위 보조 점수 (앱·사전계산 공용 모듈)."""
from __future__ import annotations
import pandas as pd
from .prescription import ABSORPTION_HIGH, TIMBER_HIGH


def hazard_score(row: pd.Series) -> float:
    """재해위험 점수(0~100): 산사태 등급 기반 + 산불 높음 보정."""
    base = {"높음": 100, "중간": 60, "낮음": 30}.get(str(row.get("landslide_risk")).strip(), 30)
    if str(row.get("fire_risk")).strip() == "높음":
        base += 20
    return float(min(100, base))


def effect_score(row: pd.Series) -> float:
    """관리효과 점수(0~100): 탄소흡수·임목잠재·수관밀도 기반."""
    score = 0
    if float(row.get("annual_absorption_tco2", 0) or 0) >= ABSORPTION_HIGH:
        score += 40
    if float(row.get("timber_potential_m3", 0) or 0) >= TIMBER_HIGH:
        score += 40
    if str(row.get("crown_density")).strip() == "밀":
        score += 20
    return float(min(100, score))


def access_score(row: pd.Series) -> float:
    """접근성 점수(0~100): 임도 거리 기반."""
    d = float(row.get("forest_road_distance_m", 9999) or 9999)
    if d < 300:
        return 100.0
    if d < 800:
        return 60.0
    return 30.0


def budget_priority_score(care_gap: float, row: pd.Series) -> float:
    """예산 우선순위 점수 = 관리공백0.45 + 재해0.25 + 관리효과0.20 + 접근성0.10."""
    return round(
        care_gap * 0.45
        + hazard_score(row) * 0.25
        + effect_score(row) * 0.20
        + access_score(row) * 0.10,
        1,
    )
