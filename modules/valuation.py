# -*- coding: utf-8 -*-
"""산림 경제가치 환산 모듈 (추정).

산출:
  - 탄소 자산가치(저장량 × 배출권가)         [원, 스톡]
  - 목재 자산가치(임목축적 × 입목 단가)        [원, 스톡]
  - 연간 탄소가치(흡수량 × 배출권가)          [원/년, 흐름]
  - 연간 공익기능 가치(면적 × 공익 평가액)      [원/년, 흐름]
  - 관리공백 위험노출 가치(자산 × 관리공백점수)  [원]  ← 서비스 핵심 지표

계수는 공개 통계 기반 '기본값'이며 조정 가능(시장가·연도별 변동). '확정액'이 아니라 추정.

출처(근사):
  - 탄소 배출권(KAU): 한국거래소(KRX) 배출권시장 시세
  - 입목/원목 가격: 산림청 임업통계연보·한국임업진흥원
  - 공익기능 평가액: 국립산림과학원 「산림의 공익기능 가치 평가」(전국 평균 환산)
"""
from __future__ import annotations
import pandas as pd

# ----- 조정 가능한 단가/계수 (기본값) ------------------------------------- #
CARBON_PRICE_KRW_PER_TCO2 = 9_000          # 배출권(KAU) 근사 단가 (원/tCO2)
TIMBER_PRICE_KRW_PER_M3   = 70_000         # 입목 기준 대표 단가 (원/m3)
PUBLIC_FUNCTION_KRW_PER_HA_YR = 41_000_000  # 산림 공익기능 평가 전국평균 (원/ha/년)

_CONIFERS = {"소나무", "잣나무", "낙엽송", "리기다소나무", "편백", "곰솔", "삼나무", "전나무", "가문비나무", "해송"}


def _timber_mult(species: str) -> float:
    s = str(species)
    if s in _CONIFERS:
        return 1.1          # 침엽수(용재) 가산
    if s in ("제지", "죽림", "정보없음"):
        return 0.3          # 미립목지 등 감산
    return 0.9              # 활엽수 등


def add_value_columns(df: pd.DataFrame,
                      carbon_price: float = CARBON_PRICE_KRW_PER_TCO2,
                      timber_price: float = TIMBER_PRICE_KRW_PER_M3,
                      public_per_ha: float = PUBLIC_FUNCTION_KRW_PER_HA_YR) -> pd.DataFrame:
    """경제가치 컬럼(val_*)을 추가해 반환한다."""
    out = df.copy()
    num = lambda c: pd.to_numeric(out.get(c), errors="coerce").fillna(0.0)
    mult = out.get("species", pd.Series(index=out.index, dtype=object)).map(_timber_mult).fillna(0.9)

    out["val_carbon_asset"] = num("carbon_storage_tco2") * carbon_price           # 원
    out["val_timber"]       = num("timber_potential_m3") * timber_price * mult     # 원
    out["val_total_asset"]  = out["val_carbon_asset"] + out["val_timber"]          # 원

    out["val_carbon_annual"] = num("annual_absorption_tco2") * carbon_price        # 원/년
    out["val_public_annual"] = num("area_ha") * public_per_ha                      # 원/년
    out["val_annual"]        = out["val_carbon_annual"] + out["val_public_annual"]  # 원/년

    gap = num("ai_gap_score") if "ai_gap_score" in out.columns else num("care_gap_score")
    out["val_at_risk"] = (out["val_total_asset"] * gap / 100.0).round(0)           # 원
    for c in ["val_carbon_asset", "val_timber", "val_total_asset",
              "val_carbon_annual", "val_public_annual", "val_annual"]:
        out[c] = out[c].round(0)
    return out


def format_won(x) -> str:
    """원 단위 금액을 조/억/만원으로 보기 좋게 변환."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e12:
        return f"{sign}{x/1e12:,.1f}조원"
    if x >= 1e8:
        return f"{sign}{x/1e8:,.1f}억원"
    if x >= 1e4:
        return f"{sign}{x/1e4:,.0f}만원"
    return f"{sign}{x:,.0f}원"
