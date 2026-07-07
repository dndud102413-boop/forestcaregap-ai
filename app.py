"""ForestCareGap AI · 산림 관리공백 사전진단 의사결정 지원 서비스 (프로토타입)

핵심 흐름:
    관리공백 탐지 → 원인 설명 → 관리 처방 → 지원사업 연결
    → 산주 설득 → 지자체 예산 우선순위 → 관리 후 성과 추적

주의: 본 서비스는 특정 산림의 '방치'를 확정하지 않는다.
      모든 결과는 '관리공백 가능성 사전진단'이며 현장조사·행정 확인이 필요하다.

실행:  streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st

# app.py 가 위치한 폴더를 import 경로에 추가(어디서 실행해도 modules 패키지 인식)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

from modules.scoring import calculate_care_gap_score, score_dataframe
from modules.explanation import generate_factor_explanation
from modules.prescription import recommend_prescriptions, ABSORPTION_HIGH, TIMBER_HIGH
from modules.support_matching import match_support_programs
from modules.reporting import generate_landowner_report, DISCLAIMER
from modules.tracking import calculate_management_outcome

# AI(ML) 엔진 — 선택적 로드(모델 파일/라이브러리 없으면 자동 비활성)
try:
    from modules import ml_engine
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

from modules.valuation import format_won  # 경제가치 표기
from modules import pdf_report             # 한글 PDF 다운로드(reportlab)
from modules import llm_report             # LLM 산주 리포트(Gemini, 룰기반 폴백)
from modules import rag_report             # RAG 맞춤 안내문(지원사업 근거검색 + LLM/폴백)
from modules import uncertainty            # 예측 신뢰도(트리 불일치 + 지역 honest AUC)

# 지도 라이브러리(GeoPandas)는 환경 문제가 잦으므로 '선택' 으로만 확인한다.
try:
    import geopandas as _gpd  # noqa: F401
    GEOPANDAS_AVAILABLE = True
except Exception:
    GEOPANDAS_AVAILABLE = False

# Plotly 도 선택. 없으면 streamlit 내장 차트로 대체한다.
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# pydeck — 지도 시각화(선택)
try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except Exception:
    PYDECK_AVAILABLE = False


# --------------------------------------------------------------------------- #
# 샘플 데이터 생성 (실제 공공데이터가 없어도 작동)
# --------------------------------------------------------------------------- #
GANGWON_REGIONS = [
    "춘천시", "원주시", "강릉시", "동해시", "태백시", "속초시", "삼척시",
    "홍천군", "횡성군", "영월군", "평창군", "정선군", "철원군", "화천군",
    "양구군", "인제군", "고성군", "양양군",
]
SPECIES = ["소나무", "잣나무", "낙엽송", "리기다소나무", "굴참나무", "신갈나무", "편백", "상수리나무"]
OWNER_TYPES = ["개인", "개인", "개인", "법인", "종중", "공유"]  # 사유림 비중 높게
ASPECTS = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"]
RISK = ["낮음", "중간", "높음"]
DIAMETER = ["소경목", "중경목", "대경목"]
DENSITY = ["소", "중", "밀"]

# 관리유형(KMeans 군집) → 설명·권장조치. ml_engine 의 실제 군집명 기준(과장 없이).
SEGMENT_INFO = {
    "재해취약형": ("산불·산사태 위험이 상대적으로 높은 유형", "예방사업·재해위험 저감 관리 우선 검토"),
    "외진 관리취약형": ("임도 거리가 멀고 경사가 높아 관리 접근성이 낮은 유형", "현장 접근성 확인, 작업로·임도 접근 가능성 검토"),
    "경영활성형": ("관리 활동이 상대적으로 활발한 유형", "현 관리 유지·정기 모니터링"),
    "보전우선형": ("보호가치가 높아 보전이 우선되는 유형", "보전 모니터링, 개입은 신중히 검토"),
    "일반관리형": ("뚜렷한 특이 신호가 없는 일반 유형", "정기 모니터링, 주변 관리변화 추적"),
}


def generate_sample_forest_data(n: int = 40, seed: int = 42) -> pd.DataFrame:
    """실데이터가 없을 때 사용할 샘플 산림 폴리곤 데이터를 재현 가능하게 생성한다."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n + 1):
        age = int(rng.integers(1, 8))                  # 1~7 영급
        hist_years = int(rng.integers(0, 26))          # 마지막 관리 후 경과 연수
        has_recent = bool(hist_years <= 4)             # 최근 4년 내 관리 여부
        slope = round(float(rng.uniform(5, 40)), 1)
        area = round(float(rng.uniform(0.5, 25.0)), 2)
        crown = str(rng.choice(DENSITY, p=[0.3, 0.4, 0.3]))
        absorption = round(float(rng.uniform(1.0, 12.0)), 1)
        timber = round(float(area * rng.uniform(8, 22)), 0)  # 면적 비례 임목 잠재량
        rows.append({
            "polygon_id": f"GW-{i:04d}",
            "region": str(rng.choice(GANGWON_REGIONS)),
            "owner_type": str(rng.choice(OWNER_TYPES)),
            "species": str(rng.choice(SPECIES)),
            "age_class": age,
            "diameter_class": DIAMETER[min(2, age // 3)],
            "crown_density": crown,
            "area_ha": area,
            "elevation_m": int(rng.integers(80, 1300)),
            "slope_deg": slope,
            "aspect": str(rng.choice(ASPECTS)),
            "forest_road_distance_m": int(rng.integers(30, 1500)),
            "road_distance_m": int(rng.integers(100, 5000)),
            "management_history_years": hist_years,
            "has_recent_management": has_recent,
            "landslide_risk": str(rng.choice(RISK, p=[0.45, 0.35, 0.20])),
            "fire_risk": str(rng.choice(RISK, p=[0.4, 0.4, 0.2])),
            "protected_area": bool(rng.random() < 0.15),
            "owner_absentee": bool(rng.random() < 0.40),
            "carbon_storage_tco2": round(float(area * rng.uniform(15, 35)), 1),
            "annual_absorption_tco2": absorption,
            "timber_potential_m3": timber,
        })
    return pd.DataFrame(rows)


# CSV 로 읽으면 문자열이 되는 bool 컬럼을 실제 bool 로 정규화
_BOOL_COLS = ["has_recent_management", "protected_area", "owner_absentee"]


def _normalize_bools(df: pd.DataFrame) -> pd.DataFrame:
    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: str(v).strip().lower() in ("true", "1", "yes", "t", "예", "참")
                if not isinstance(v, bool) else v
            )
    return df


@st.cache_data(show_spinner=False)
def load_forest_data() -> tuple[pd.DataFrame, str]:
    """폴리곤 데이터를 로드한다.

    우선순위: 사전계산 parquet → 실데이터 CSV → 샘플 CSV → 자동생성.
    반환: (DataFrame, 데이터출처 라벨)
    """
    scored_pq = os.path.join(DATA_DIR, "forest_scored.parquet")
    real = os.path.join(DATA_DIR, "real_forest_polygons.csv")
    path = os.path.join(DATA_DIR, "sample_forest_polygons.csv")
    if os.path.exists(scored_pq):
        df = _normalize_bools(pd.read_parquet(scored_pq))
        return df, f"실데이터 · 강원 전수 {len(df):,}필지 (사전계산)"
    if os.path.exists(real):
        return _normalize_bools(pd.read_csv(real)), "실데이터(강원 공공데이터)"
    if os.path.exists(path):
        return _normalize_bools(pd.read_csv(path)), "샘플 데이터"
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate_sample_forest_data()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return _normalize_bools(df), "샘플 데이터(자동생성)"


@st.cache_data(show_spinner=False)
def load_support_programs() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "support_programs.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # 파일이 없으면 빈 프레임(앱이 죽지 않도록)
    return pd.DataFrame(columns=[
        "program_id", "program_name", "category", "target_condition",
        "description", "contact_channel", "disclaimer",
    ])


@st.cache_data(show_spinner=False)
def load_tracking() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "management_tracking.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=[
        "polygon_id", "before_score", "after_score", "management_action",
        "management_date", "field_survey_done", "owner_contacted",
        "support_program_linked", "notes",
    ])


# 예산 우선순위 보조 점수 — 앱·사전계산 공용 모듈에서 가져온다.
from modules.budget import (
    hazard_score, effect_score, access_score, budget_priority_score,
)


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="ForestCareGap AI", page_icon="🌲", layout="wide")

# --------------------------------------------------------------------------- #
# 깔끔한 화이트 모던 테마 (기업형) — 여백·카드·다크 밴드 중심
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@latest/dist/web/static/pretendard.css');
:root{
  --bg:#FFFFFF; --soft:#F4F5F6; --card:#FFFFFF; --ink:#15171A; --text:#2A2D31;
  --muted:#6E7176; --border:#E6E7E9; --accent:#202327; --accent-d:#0E1012; --band:#202327;
}
html, body, .stApp, [class*="css"]{
  font-family:'Pretendard','Inter',-apple-system,system-ui,sans-serif !important;
}
.stApp{ background:var(--bg); color:var(--text); }
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stMain"] .block-container{ max-width:1160px; padding-top:2.4rem; padding-bottom:5rem; }

/* 타이포 — 깔끔한 솔리드 (앤틱X) */
h1,h2,h3,h4{ color:var(--ink); letter-spacing:-0.02em; font-weight:800; }
h1{ font-size:2.25rem; }
h2{ font-size:1.45rem; margin-top:.6rem; padding-bottom:.35rem; }
h3{ font-size:1.12rem; }
/* 섹션 제목 좌측 액센트 바 */
[data-testid="stHeading"] h2{ border-left:4px solid var(--accent); padding-left:.6rem; }
[data-testid="stCaptionContainer"], .stCaption{ color:var(--muted) !important; }

/* 메트릭 = 흰 카드 + 옅은 보더/그림자 */
[data-testid="stMetric"]{
  background:var(--card); border:1px solid var(--border); border-radius:14px;
  padding:16px 18px; box-shadow:0 1px 2px rgba(16,33,44,.04), 0 8px 24px rgba(16,33,44,.05);
  transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
[data-testid="stMetric"]:hover{ transform:translateY(-2px); border-color:#D5DBE1;
  box-shadow:0 10px 28px rgba(16,33,44,.10); }
[data-testid="stMetricValue"]{ font-weight:800; color:var(--ink); }
[data-testid="stMetricLabel"] p{ color:var(--muted); font-weight:600; letter-spacing:.01em; }

/* 탭 — 미니멀, 액센트 언더라인 */
.stTabs [data-baseweb="tab-list"]{ gap:8px; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"]{
  background:transparent; color:var(--muted); font-weight:600; padding:10px 6px;
}
.stTabs [data-baseweb="tab"]:hover{ color:var(--ink); }
.stTabs [aria-selected="true"]{ color:var(--ink) !important; border-bottom:2px solid var(--accent); }

/* 표 — 옅은 보더 + 라운드 */
[data-testid="stDataFrame"], [data-testid="stTable"]{
  border:1px solid var(--border); border-radius:12px; overflow:hidden;
  box-shadow:0 1px 2px rgba(16,33,44,.04);
}

/* 버튼 — 솔리드 그린, 깔끔 */
.stButton>button, [data-testid="stDownloadButton"]>button{
  background:var(--accent); color:#FFFFFF; border:0; border-radius:10px;
  font-weight:700; padding:.55rem 1.15rem; box-shadow:0 2px 8px rgba(20,22,26,.18);
}
.stButton>button:hover, [data-testid="stDownloadButton"]>button:hover{ background:var(--accent-d); }

/* 알림 박스 — 라이트 */
[data-testid="stAlert"]{ border-radius:12px; border:1px solid var(--border); }

/* 입력 위젯 */
[data-baseweb="select"]>div{ background:var(--card); border-color:var(--border)!important; border-radius:10px; }
[data-baseweb="input"]{ border-radius:10px; }

/* 진행바 */
.stProgress > div > div > div > div{ background:var(--accent); }

/* 구분선/스크롤바 */
hr{ border-color:var(--border); }
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:#D7DDE3; border-radius:8px; }
::-webkit-scrollbar-thumb:hover{ background:#C2CAD2; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

forest_df, data_source = load_forest_data()
programs_df = load_support_programs()
tracking_df = load_tracking()


@st.cache_data(show_spinner=False)
def load_gangwon_geo():
    """강원 시군구 경계(단계구분도용). 없으면 None."""
    p = os.path.join(DATA_DIR, "gangwon_municipalities.geojson")
    if not (GEOPANDAS_AVAILABLE and os.path.exists(p)):
        return None
    import geopandas as gpd
    return gpd.read_file(p)


@st.cache_data(show_spinner=False)
def load_validation_report():
    """validation_suite.py 가 만든 honest 검증 리포트(JSON). 없으면 None."""
    p = os.path.join(DATA_DIR, "validation_report.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_ml_bundle():
    return ml_engine.load() if ML_AVAILABLE else None


@st.cache_data(show_spinner=False)
def ai_enrich(df: pd.DataFrame, _src: str):
    b = load_ml_bundle()
    return ml_engine.predict_dataframe(df, b) if b is not None else None


ml_bundle = load_ml_bundle()
# 사전계산 parquet 에 AI 컬럼이 있으면 그대로 사용(빠름), 없으면 실시간 추론
if "ai_gap_score" in forest_df.columns:
    ai_df = forest_df
elif ml_bundle is not None:
    ai_df = ai_enrich(forest_df, data_source)
else:
    ai_df = None
PRECOMPUTED = "care_gap_score" in forest_df.columns

st.title("🌲 ForestCareGap AI")
st.caption(
    f"관리공백 가능성 사전진단 기반 산림관리 의사결정 지원 서비스 (프로토타입) · "
    f"데이터: {data_source} · 분석 폴리곤 {len(forest_df):,}개"
)
st.info(
    "본 서비스는 특정 산림의 방치 여부를 **법적·행정적으로 확정하지 않습니다**. "
    "모든 결과는 산림공간정보·관리이력 데이터 기반의 **관리공백 가능성 사전진단**이며, "
    "현장조사와 지자체·산림조합 확인이 필요합니다.",
    icon="ℹ️",
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① 서비스 소개",
    "② 개별 산림 진단",
    "③ 지자체 우선관리 대시보드",
    "④ 관리 후 성과 추적",
    "⑤ 기술 설명",
])

# ----------------------------- 탭 1. 서비스 소개 ---------------------------- #
with tab1:
    st.header("서비스 소개")
    st.subheader("ForestCareGap AI")
    st.markdown(
        "**한 줄 요약** — 관리공백 가능성이 높은 사유림을 사전 선별하고, 원인·처방·지원사업·"
        "산주 리포트·지자체 예산 우선순위·관리 성과까지 연결하는 AI 의사결정 지원 서비스."
    )
    st.markdown("#### 핵심 흐름")
    st.success(
        "관리공백 탐지 → 원인 설명 → 관리 처방 → 지원사업 연결 → "
        "산주 설득 → 지자체 예산 우선순위 → 관리 후 성과 추적"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("분석 대상 폴리곤", f"{len(forest_df):,} 개")
    c2.metric("연계 지원사업(검토 후보)", f"{len(programs_df):,} 종")
    c3.metric("성과 추적 사례", f"{len(tracking_df):,} 건")

    with st.expander("📢 발표용 핵심 문구 (발표자 스크립트)", expanded=False):
        st.markdown(
            "1. **무엇** — ForestCareGap AI는 강원도 약 56만 개 산림 필지를 대상으로, 실제 숲가꾸기·조림 "
            "시행 이력과 지형·임상·재해위험·접근성·토양·경제가치 데이터를 결합해 **관리공백 가능성**을 "
            "사전진단합니다.\n\n"
            "2. **무엇이 아닌가** — 방치를 확정하는 시스템이 아니라, 공개데이터 기반으로 **현장 확인이 필요한 "
            "우선관리 후보지를 줄여주는 의사결정 지원 도구**입니다.\n\n"
            "3. **정직한 검증** — 랜덤 교차검증 AUC는 0.98로 높게 나오지만 공간 군집으로 부풀려진 값입니다. "
            "지역 그룹 기반 **공간 교차검증으로 누수를 통제해 약 0.29의 거품을 제거**하고, **honest AUC 0.688**로 "
            "보고합니다.\n\n"
            "4. **행정 가치** — 점수 **상위 20%만 확인해도 전체 관리대상의 절반을 포착**합니다. 한정된 인력의 "
            "현장확인 우선순위 도구입니다.\n\n"
            "5. **예산 의사결정** — 강원 '우선관리 필요' 약 21만 필지 전체 관리엔 약 **1.1조**가 필요합니다. "
            "예산 시뮬레이터로 **가장 시급한 곳부터 배분**하도록 돕습니다.\n\n"
            "6. **데이터 정직성** — 입지토양 피처는 honest 검증으로 +0.014 향상되어 채택, 임도망 밀도는 "
            "효과 없어 기각했습니다. **무분별하게 넣지 않고 검증 후 반영**했습니다."
        )
        st.caption("※ 수치는 검증 리포트 기준 — 발표 시 그대로 인용 가능합니다.")

    st.warning(
        "⚠️ 본 화면의 모든 결과는 **사전진단**입니다. '방치 확정'·'불법 방치'·'신청 가능 확정' 등의 "
        "단정 표현을 사용하지 않으며, 최종 판단은 현장조사·행정 확인을 거칩니다.",
        icon="⚠️",
    )

# ----------------------------- 탭 2. 개별 진단 ------------------------------ #
with tab2:
    st.header("개별 산림 진단")
    # 전수(56만) 환경에서도 가벼운 선택: 우선순위 상위 목록 또는 ID 직접 입력
    if "val_at_risk" in forest_df.columns:
        rank_col, rank_label = "val_at_risk", "위험노출 가치"
    elif "ai_gap_score" in forest_df.columns:
        rank_col, rank_label = "ai_gap_score", "관리공백 점수"
    elif "care_gap_score" in forest_df.columns:
        rank_col, rank_label = "care_gap_score", "관리공백 점수"
    else:
        rank_col, rank_label = "polygon_id", "ID"
    mode = st.radio("필지 선택 방식", ["우선순위 상위에서 선택", "폴리곤 ID 직접 입력"],
                    horizontal=True, key="sel_mode")
    if mode == "폴리곤 ID 직접 입력":
        pid_in = st.text_input("폴리곤 ID (예: GW-000001)", value=str(forest_df["polygon_id"].iloc[0]))
        hit = forest_df[forest_df["polygon_id"] == pid_in.strip()]
        if hit.empty:
            st.warning("해당 ID를 찾지 못했습니다. 목록에서 선택하거나 정확한 ID를 입력하세요.")
            row = forest_df.iloc[0]
        else:
            row = hit.iloc[0]
    else:
        regions_opt = ["전체"] + sorted(forest_df["region"].dropna().unique().tolist())
        sreg = st.selectbox("지역(선택)", regions_opt, key="diag_region")
        pool = forest_df if sreg == "전체" else forest_df[forest_df["region"] == sreg]
        pool = pool.sort_values(rank_col, ascending=False).head(300)
        pid = st.selectbox(f"폴리곤 ID — {rank_label} 상위 300", pool["polygon_id"].tolist(), key="diag_pid")
        row = pool[pool["polygon_id"] == pid].iloc[0]
    pid = row["polygon_id"]

    # 산림 기본정보
    st.subheader("산림 기본 정보")
    info_cols = st.columns(4)
    info_cols[0].metric("지역", str(row["region"]))
    info_cols[1].metric("수종", str(row["species"]))
    info_cols[2].metric("영급", f"{row['age_class']}영급")
    info_cols[3].metric("면적", f"{row['area_ha']} ha")
    info_cols2 = st.columns(4)
    info_cols2[0].metric("고도", f"{row['elevation_m']} m")
    info_cols2[1].metric("경사", f"{row['slope_deg']}°")
    info_cols2[2].metric("임도거리", f"{row['forest_road_distance_m']} m")
    info_cols2[3].metric("수관밀도", str(row["crown_density"]))

    # 점수 계산
    scoring_result = calculate_care_gap_score(row)
    explanation = generate_factor_explanation(scoring_result, row)
    prescriptions = recommend_prescriptions(row, scoring_result)
    matched = match_support_programs(prescriptions, programs_df)

    # ── 📋 필지 진단카드 (심사용 요약) ──────────────────────────────────────── #
    st.subheader("📋 필지 진단카드")
    if ai_df is not None and "ai_gap_score" in forest_df.columns:
        airow_c = ai_df[ai_df["polygon_id"] == pid].iloc[0]
        gap_score = float(airow_c["ai_gap_score"])
        seg_name = str(airow_c.get("ai_segment_name", "일반관리형"))
        pct_top = float((forest_df["ai_gap_score"] >= gap_score).mean()) * 100
        try:
            card_reasons = [c["요인"] for c in ml_engine.explain_row(row, ml_bundle, topk=3)] if ml_bundle else []
        except Exception:
            card_reasons = []
        score_label = "관리공백 점수 (AI)"
    else:
        gap_score = float(scoring_result["score"])
        seg_name = "-"
        _base = forest_df["care_gap_score"] if "care_gap_score" in forest_df.columns else pd.Series([gap_score])
        pct_top = float((_base >= gap_score).mean()) * 100
        card_reasons = [f["factor"] for f in scoring_result["factors"][:3]]
        score_label = "관리공백 점수 (규칙)"

    if pct_top <= 5:
        grade_c, badge = "매우 높음", "🔴"
    elif pct_top <= 20:
        grade_c, badge = "높음", "🟠"
    elif pct_top <= 50:
        grade_c, badge = "보통", "🟡"
    else:
        grade_c, badge = "낮음", "🟢"

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric(score_label, f"{gap_score:.0f} / 100",
               help="값이 클수록 구조적 관리공백 가능성↑ — '확률'이 아닌 우선순위 점수입니다.")
    cc2.metric("우선순위 등급", f"{badge} {grade_c}", help="강원 전체 필지 대비 백분위 기반")
    cc3.metric("전체 중 위치", f"상위 {pct_top:.1f}%", help="이 점수보다 높은 필지의 비율")

    seg_desc, seg_action = SEGMENT_INFO.get(seg_name, ("일반 관리 대상 유형", "정기 모니터링, 주변 관리변화 추적"))
    st.markdown(
        f"- **시군구**: {row['region']}  ·  **수종/영급**: {row['species']} {row['age_class']}영급  ·  "
        f"**면적**: {row['area_ha']}ha\n"
        f"- **관리유형(AI)**: {seg_name} — {seg_desc}\n"
        + (f"- **주요 근거 Top 3**: {' · '.join(card_reasons)}\n" if card_reasons else "")
        + f"- **권장 조치(검토)**: {seg_action}"
    )
    # 🛡 지역 모델 신뢰도 — 검증의 지역별 honest AUC 로 약한 지역 정직 표시
    _vr_card = load_validation_report()
    _rauc = None
    if _vr_card and isinstance(_vr_card.get("per_region_auc"), dict):
        _rentry = _vr_card["per_region_auc"].get(str(row["region"]))
        _rauc = _rentry.get("auc") if isinstance(_rentry, dict) else None
    if _rauc is not None and _rauc < 0.6:
        st.warning(
            f"⚠️ **{row['region']}는 모델 신뢰도가 낮은 지역입니다** (지역 honest AUC {_rauc:.2f}). "
            "이 지역은 AI 점수보다 **현장 확인을 우선**하시길 권장합니다.", icon="⚠️")
    elif _rauc is not None:
        st.caption(f"🛡 지역 모델 신뢰도: **{row['region']}** honest AUC {_rauc:.2f} "
                   f"({'양호' if _rauc >= 0.7 else '보통'}) — 신뢰도가 낮은 지역은 현장확인을 더 권장합니다.")

    # 🎯 이 필지의 예측 신뢰도 — 트리 불일치(앙상블 분산) + 지역 honest AUC 결합
    _conf = row.get("ai_confidence") if "ai_confidence" in row.index else None
    if _conf is None and ml_bundle is not None:
        try:
            _Xc = ml_engine.build_features(pd.DataFrame([row])).fillna(ml_bundle["median"]).values
            _std1 = float(uncertainty.tree_disagreement(_Xc, ml_bundle["rf"])[0])
            _conf = uncertainty.confidence_single(_std1, _rauc)
        except Exception:
            _conf = None
    if _conf:
        st.caption("🎯 이 필지 예측 신뢰도: " + uncertainty.confidence_note(_conf, _rauc))

    st.caption(
        "⚠️ 본 결과는 '방치 확정'이 아니라 공개데이터·공간모델 기반 **관리공백 가능성 사전진단**입니다. "
        "확률이 아닌 **우선순위 점수**이며, 실제 사업 대상 여부는 **현장조사·행정자료 확인**이 필요합니다."
    )
    st.divider()

    # 점수 카드
    st.subheader("관리공백 가능성 점수 (사전진단)")
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    sc1.metric("점수", f"{scoring_result['score']} / 100")
    sc2.metric("등급", scoring_result["grade"])
    sc3.progress(scoring_result["score"] / 100.0, text=f"위험수준: {scoring_result['risk_level']}")
    if scoring_result["factors"]:
        st.caption("점수 기여 요인")
        st.dataframe(pd.DataFrame(scoring_result["factors"]), hide_index=True, width="stretch")

    # 🤖 AI 정밀진단 (준지도 학습 + 설명가능 AI)
    ai_contribs = None
    if ai_df is not None:
        airow = ai_df[ai_df["polygon_id"] == pid].iloc[0]
        st.subheader("🤖 AI 정밀진단")
        st.caption(
            "실제 산림경영활동(숲가꾸기·조림) 시행 여부를 학습한 RandomForest 모델 · "
            "관리확률이 낮을수록 구조적 관리공백 가능성↑"
        )
        a1, a2, a3 = st.columns(3)
        a1.metric("AI 관리공백 점수", f"{airow['ai_gap_score']:.0f} / 100")
        a2.metric("AI 추정 관리확률", f"{airow['ai_management_prob'] * 100:.0f}%")
        a3.metric("관리유형(AI 군집)", airow["ai_segment_name"])
        ai_contribs = ml_engine.explain_row(row, ml_bundle, topk=5)
        st.caption("AI 근거 — 트리경로 기반 기여분해 (SHAP 동일 취지의 가법적 설명, 공백↑ = 위험 가중 요인)")
        st.dataframe(pd.DataFrame(ai_contribs), hide_index=True, width="stretch")

    # 💰 경제가치 평가
    if "val_total_asset" in row.index:
        st.subheader("💰 경제가치 평가 (추정)")
        st.caption("탄소·목재·공익기능 가치 환산 — 시장가·계수 기반 추정(조정 가능), 확정액 아님")
        e1, e2, e3 = st.columns(3)
        e1.metric("자산가치 (탄소+목재)", format_won(row["val_total_asset"]))
        e2.metric("연간 가치 (탄소흡수+공익)", format_won(row["val_annual"]) + "/년")
        e3.metric("관리공백 위험노출 가치", format_won(row["val_at_risk"]),
                  help="자산가치 × 관리공백점수 — 방치 시 위험에 노출되는 가치")

    # 원인 설명
    st.subheader("주요 원인 설명")
    st.write(explanation)

    # 처방
    st.subheader("추천 관리 처방 (검토 권고)")
    st.dataframe(pd.DataFrame(prescriptions), hide_index=True, width="stretch")

    # 지원사업 검토 후보
    st.subheader("지원사업 검토 후보 (신청 가능 확정 아님)")
    if not matched.empty:
        st.dataframe(matched, hide_index=True, width="stretch")
    else:
        st.write("매칭된 지원사업 검토 후보가 없습니다. (산림조합·지자체 상담 권장)")

    # 산주용 리포트
    st.subheader("산주용 AI 리포트")
    report = generate_landowner_report(row, scoring_result, explanation, prescriptions, matched)
    if ai_df is not None:
        airow = ai_df[ai_df["polygon_id"] == pid].iloc[0]
        ai_lines = "\n".join(
            f"     - {c['요인']}({c['값']}) {c['방향']} {c['기여(공백↑)']:+.3f}" for c in (ai_contribs or [])
        )
        report += (
            "\n\n[AI 정밀진단 — 준지도 학습 모델]\n"
            f"  · AI 관리공백 점수: {airow['ai_gap_score']:.0f}/100\n"
            f"  · AI 추정 관리확률: {airow['ai_management_prob'] * 100:.0f}%\n"
            f"  · 관리유형(AI 군집): {airow['ai_segment_name']}\n"
            f"  · 주요 근거(기여분해):\n{ai_lines}\n"
            "  ※ 데이터 학습 기반 추정치이며 현장조사로 확인이 필요합니다."
        )
    if "val_total_asset" in row.index:
        report += (
            "\n\n[경제가치 평가 — 추정]\n"
            f"  · 자산가치(탄소+목재): {format_won(row['val_total_asset'])}\n"
            f"  · 연간 가치(탄소흡수+공익): {format_won(row['val_annual'])}/년\n"
            f"  · 관리공백 위험노출 가치: {format_won(row['val_at_risk'])}\n"
            "  ※ 시장가·계수 기반 추정이며 확정액이 아닙니다."
        )
    st.text_area("리포트 미리보기", report, height=480)
    dlc1, dlc2 = st.columns(2)
    if pdf_report.available():
        try:
            _pdf = pdf_report.text_report_pdf(
                "ForestCareGap 관리공백 사전진단 리포트", report,
                subtitle=f"필지 {pid} · {row['region']} · 사전진단(현장확인 필요)",
                disclaimer=DISCLAIMER)
            dlc1.download_button("📄 PDF 다운로드", data=_pdf,
                                 file_name=f"ForestCareGap_{pid}.pdf", mime="application/pdf")
        except Exception as _e:
            dlc1.caption(f"PDF 생성 오류({type(_e).__name__}) — txt 이용")
    else:
        dlc1.caption("PDF 미지원 환경(폰트 없음) — txt 이용")
    dlc2.download_button("📄 텍스트 (.txt)", data=report.encode("utf-8"),
                         file_name=f"ForestCareGap_{pid}.txt", mime="text/plain")

    # 🤖 AI 자연어 리포트 (LLM) — 구조화 진단을 산주 눈높이 자연어로
    st.subheader("🤖 AI 자연어 리포트 (LLM)")
    if llm_report.available():
        st.caption("Gemini 연결됨 — 진단 결과를 산주 눈높이 자연어로 생성합니다.")
    else:
        st.caption("LLM 키 미설정 — 버튼 클릭 시 룰기반 자연어 리포트로 생성됩니다. "
                   "(환경변수 GEMINI_API_KEY 설정 + `pip install google-genai` 시 Gemini 자연어 생성)")
    if st.button("🤖 AI 자연어 리포트 생성", key=f"llmbtn_{pid}"):
        _ctx = {
            "region": row["region"], "species": row["species"], "age_class": row["age_class"],
            "area_ha": row["area_ha"], "road_dist": row.get("forest_road_distance_m"),
            "slope": row.get("slope_deg"), "gap_score": f"{gap_score:.0f}",
            "pct_top": f"{pct_top:.1f}", "segment": seg_name, "reasons": card_reasons,
        }
        st.session_state[f"llm_{pid}"] = llm_report.generate(_ctx)
    _llm = st.session_state.get(f"llm_{pid}")
    if _llm:
        badge = "🟢 Gemini 생성" if _llm["source"] == "gemini" else "⚪ 룰기반 생성 (키 설정 시 Gemini)"
        st.caption(f"생성 방식: {badge}")
        st.info(_llm["text"])
        if pdf_report.available():
            try:
                _lpdf = pdf_report.text_report_pdf(
                    "ForestCareGap · AI 자연어 리포트", _llm["text"],
                    subtitle=f"필지 {pid} · {row['region']} · 사전진단(현장확인 필요)",
                    disclaimer=DISCLAIMER)
                st.download_button("📄 AI 리포트 PDF", data=_lpdf, key=f"llmpdf_{pid}",
                                   file_name=f"ForestCareGap_AI리포트_{pid}.pdf", mime="application/pdf")
            except Exception:
                pass

    # 🔎 RAG 맞춤 안내문 (지원사업 근거검색 + 생성)
    st.subheader("🔎 RAG 맞춤 안내문 (근거검색 + 생성)")
    st.caption(
        "산림 지원사업·관리기준 지식베이스에서 이 필지에 맞는 근거를 검색(TF-IDF, 오프라인)해, "
        "**검색된 근거 안에서만** 산주 안내문을 생성합니다 — 환각을 줄인 근거기반 생성(RAG). "
        + ("Gemini 연결됨." if llm_report.available() else "키 미설정 시 검색기반 폴백으로 생성됩니다."))
    if st.button("🔎 RAG 안내문 생성", key=f"ragbtn_{pid}"):
        _prot = str(row.get("protected_area")).strip().lower() in ("true", "1", "yes", "예", "참")
        _rctx = {
            "region": row["region"], "species": row["species"], "age_class": row["age_class"],
            "area_ha": row["area_ha"], "road_dist": row.get("forest_road_distance_m"),
            "slope": row.get("slope_deg"), "gap_score": f"{gap_score:.0f}", "pct_top": f"{pct_top:.1f}",
            "segment": seg_name, "reasons": card_reasons,
            "crown_density": row.get("crown_density"), "landslide": row.get("landslide_risk"),
            "fire": row.get("fire_risk"), "protected": _prot,
        }
        st.session_state[f"rag_{pid}"] = rag_report.generate(_rctx)
    _rag = st.session_state.get(f"rag_{pid}")
    if _rag:
        badge = "🟢 Gemini 생성(근거주입)" if _rag["source"] == "gemini" else "⚪ 검색기반 생성 (키 설정 시 Gemini)"
        st.caption(f"생성 방식: {badge}")
        if _rag.get("retrieved"):
            st.markdown("**🔎 검색된 근거 (지원사업·관리기준):**")
            st.dataframe(pd.DataFrame([
                {"근거": d["title"], "분류": d["category"], "유사도": d["score"], "출처": d["source"]}
                for d in _rag["retrieved"]]), hide_index=True, width="stretch")
        st.info(_rag["text"])
        if pdf_report.available():
            try:
                _rpdf = pdf_report.text_report_pdf(
                    "ForestCareGap · RAG 맞춤 안내문", _rag["text"],
                    subtitle=f"필지 {pid} · {row['region']} · 근거검색 기반 · 사전진단(현장확인 필요)",
                    disclaimer=DISCLAIMER)
                st.download_button("📄 RAG 안내문 PDF", data=_rpdf, key=f"ragpdf_{pid}",
                                   file_name=f"ForestCareGap_RAG_{pid}.pdf", mime="application/pdf")
            except Exception:
                pass

# ----------------------------- 탭 3. 지자체 대시보드 ------------------------ #
with tab3:
    st.header("지자체 우선관리 대시보드")

    if PRECOMPUTED:
        # 사전계산된 점수(care_gap_score·budget·ai_*)를 그대로 사용 → 56만 건도 즉시
        scored = forest_df
    else:
        scored = score_dataframe(forest_df)
        scored["budget_priority_score"] = [
            budget_priority_score(r["care_gap_score"], r) for _, r in scored.iterrows()
        ]
        if ai_df is not None:
            scored = scored.merge(
                ai_df[["polygon_id", "ai_gap_score", "ai_segment_name"]], on="polygon_id", how="left"
            )

    # 필터
    fcol1, fcol2 = st.columns(2)
    regions = ["전체"] + sorted(scored["region"].unique().tolist())
    sel_region = fcol1.selectbox("지역 필터", regions, key="dash_region")
    only_priority = fcol2.checkbox("우선관리 필요 등급만 보기", value=False)

    view = scored.copy()
    if sel_region != "전체":
        view = view[view["region"] == sel_region]
    if only_priority:
        view = view[view["care_gap_grade"] == "우선관리 필요"]

    # 요약 지표
    has_ai = "ai_gap_score" in view.columns
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("대상 폴리곤", f"{len(view):,} 개")
    m2.metric("평균 관리공백 점수", f"{view['care_gap_score'].mean():.1f}" if len(view) else "-")
    m3.metric("우선관리 필요", f"{(view['care_gap_grade'] == '우선관리 필요').sum():,} 개")
    m4.metric("AI 평균 공백점수", f"{view['ai_gap_score'].mean():.1f}" if has_ai and len(view) else "-")

    # 💰 경제가치 합계 (선택 범위)
    if "val_total_asset" in view.columns and len(view):
        st.subheader("💰 산림 경제가치 (선택 범위 합계)")
        ev1, ev2, ev3 = st.columns(3)
        ev1.metric("자산가치 (탄소+목재)", format_won(view["val_total_asset"].sum()))
        ev2.metric("⚠️ 관리공백 위험노출 가치", format_won(view["val_at_risk"].sum()),
                   help="자산가치 × 관리공백점수 — 방치 시 위험에 노출되는 산림가치 총액")
        ev3.metric("연간 공익기능 가치", format_won(view["val_public_annual"].sum()) + "/년",
                   help="국립산림과학원 공익기능 평가 전국평균 기준 참고치")
        st.caption("※ 시장가·계수 기반 추정치(조정 가능)이며 확정액이 아닙니다.")

    # 💵 예산 시뮬레이터 — 예산 대비 관리 효과 (강원 전체 기준)
    if {"val_at_risk", "budget_priority_score", "area_ha"}.issubset(scored.columns):
        st.subheader("💵 예산 시뮬레이터")
        st.caption(
            "투입 예산을 넣으면 **예산 우선순위가 높은 필지부터 관리**한다고 가정해 "
            "'몇 필지 관리 가능 · 위험노출 가치 얼마 해소 · 어디에 집중되나'를 추정합니다. "
            "강원 전체 기준 · 비용·가치는 조정 가능한 추정치이며 확정액이 아닙니다."
        )
        bcol1, bcol2 = st.columns(2)
        budget_eok = bcol1.slider("투입 예산 (억원)", min_value=5, max_value=1000, value=100, step=5)
        unit_cost_man = bcol2.number_input(
            "관리 단가 (만원/ha · 숲가꾸기 근사)", min_value=50, max_value=1000, value=250, step=10,
            help="산림청 숲가꾸기 사업비 근사 단가. 조정 가능한 추정치입니다.")
        budget_won = budget_eok * 1e8
        unit_won = unit_cost_man * 1e4

        sim = scored[["polygon_id", "region", "area_ha", "val_at_risk",
                      "budget_priority_score", "care_gap_grade"]].copy()
        _area = pd.to_numeric(sim["area_ha"], errors="coerce")
        _area = _area.fillna(_area.median())
        sim["cost"] = _area * unit_won
        sim = sim.sort_values("budget_priority_score", ascending=False)
        sim["cum_cost"] = sim["cost"].cumsum()
        sim["cum_val"] = pd.to_numeric(sim["val_at_risk"], errors="coerce").fillna(0).cumsum()
        chosen = sim[sim["cum_cost"] <= budget_won]
        n_chosen = len(chosen)
        used = float(chosen["cost"].sum())
        val_addressed = float(pd.to_numeric(chosen["val_at_risk"], errors="coerce").fillna(0).sum())
        val_total = float(pd.to_numeric(sim["val_at_risk"], errors="coerce").fillna(0).sum())
        hi_mask = sim["care_gap_grade"] == "우선관리 필요"
        hi_total = int(hi_mask.sum())
        hi_total_cost = float(sim.loc[hi_mask, "cost"].sum())
        budget_share = 100 * budget_won / hi_total_cost if hi_total_cost else 0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("관리 가능 필지", f"{n_chosen:,} 개",
                  help=("필지당 평균 " + format_won(used / n_chosen)) if n_chosen else "-")
        r2.metric("사용 예산", format_won(used), help=f"투입 {format_won(budget_won)} 중")
        r3.metric("해소 위험노출 가치", format_won(val_addressed))
        r4.metric("전체 관리 소요(추정)", format_won(hi_total_cost),
                  help=f"'우선관리 필요' {hi_total:,}개 전체 관리 추정비")

        st.info(
            f"강원 '우선관리 필요' **{hi_total:,}개**를 모두 관리하려면 약 **{format_won(hi_total_cost)}**이 "
            f"필요합니다 — 현재 예산은 그 중 약 **{budget_share:.1f}%**. 그래서 **가장 시급한 필지부터 "
            f"우선 배분**하는 것이 이 시뮬레이터의 목적입니다.", icon="🎯")

        if n_chosen:
            top_reg = (chosen.groupby("region")["val_at_risk"].sum()
                       .sort_values(ascending=False).head(3))
            st.caption("집중 시군구(해소 가치 기준): "
                       + " · ".join(f"**{k}** {format_won(v)}" for k, v in top_reg.items()))

        sc1, sc2 = st.columns(2)
        with sc1:
            st.caption("시군구별 관리 가능 필지 수")
            reg_cnt = chosen["region"].value_counts().head(12) if n_chosen else pd.Series(dtype=int)
            if PLOTLY_AVAILABLE and len(reg_cnt):
                fig = px.bar(x=reg_cnt.values, y=reg_cnt.index, orientation="h",
                             labels={"x": "필지 수", "y": ""})
                fig.update_layout(height=330, margin=dict(t=8, b=8), template="plotly_white",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#2B3947", yaxis=dict(autorange="reversed"))
                fig.update_traces(marker_color="#202327")
                st.plotly_chart(fig, width="stretch")
            elif len(reg_cnt):
                st.bar_chart(reg_cnt)
        with sc2:
            st.caption("예산 ↑ 에 따른 누적 위험해소 가치 (붉은선 = 현재 예산)")
            step = max(1, len(sim) // 400)
            curve = sim.iloc[::step].copy()
            curve["예산_억원"] = curve["cum_cost"] / 1e8
            curve["해소가치_억원"] = curve["cum_val"] / 1e8
            curve = curve[curve["예산_억원"] <= budget_eok * 3]
            if PLOTLY_AVAILABLE and len(curve):
                fig = px.area(curve, x="예산_억원", y="해소가치_억원",
                              labels={"예산_억원": "투입 예산(억원)", "해소가치_억원": "누적 해소가치(억원)"})
                fig.add_vline(x=budget_eok, line_dash="dash", line_color="#F87171")
                fig.update_layout(height=330, margin=dict(t=8, b=8), template="plotly_white",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#2B3947")
                fig.update_traces(line_color="#202327", fillcolor="rgba(32,35,39,.10)")
                st.plotly_chart(fig, width="stretch")
            elif len(curve):
                st.line_chart(curve.set_index("예산_억원")["해소가치_억원"])
        st.caption(
            "※ '우선순위 상위부터 관리'를 가정한 시뮬레이션입니다. 실제 집행은 현장조사·행정절차·"
            "산주 동의가 필요하며, 단가·가치는 추정치(조정 가능)로 확정액이 아닙니다."
        )

    # 🗺 관리공백 지도 (시군구 단계구분도 / 고위험 필지 핫스팟)
    geo = load_gangwon_geo()
    if PYDECK_AVAILABLE and len(view):
        st.subheader("🗺 관리공백 지도")
        choices = []
        if geo is not None:
            choices.append("시군구 단계구분도")
        if {"lon", "lat"}.issubset(view.columns):
            choices.append("고위험 필지 핫스팟")
        map_type = st.radio("지도 유형", choices, horizontal=True, key="map_type") if choices else None

        if map_type == "시군구 단계구분도":
            metric_label = st.selectbox(
                "색 기준", ["평균 관리공백 점수", "평균 AI 공백점수", "위험노출 가치(억원)"], key="choro_metric")
            mm = {"평균 관리공백 점수": ("care_gap_score", "mean"),
                  "평균 AI 공백점수": ("ai_gap_score", "mean"),
                  "위험노출 가치(억원)": ("val_at_risk", "sum")}
            col, agg = mm[metric_label]
            if col in scored.columns:
                reg = getattr(scored.groupby("region")[col], agg)()
                if col == "val_at_risk":
                    reg = reg / 1e8  # 억원
                g = geo.merge(reg.rename("metric").reset_index().rename(columns={"region": "name"}),
                              on="name", how="left")
                g["metric"] = g["metric"].fillna(reg.mean()).round(1)
                mn, mx = g["metric"].min(), g["metric"].max()

                def _col(v):
                    t = (v - mn) / (mx - mn) if mx > mn else 0.5
                    return [int(225 - t * 45), int(220 - t * 175), int(218 - t * 175), 195]
                g["fill_color"] = g["metric"].map(_col)
                gj = json.loads(g.to_json())
                layer = pdk.Layer(
                    "GeoJsonLayer", gj, get_fill_color="properties.fill_color",
                    get_line_color=[255, 255, 255], line_width_min_pixels=1,
                    pickable=True, stroked=True, filled=True,
                )
                vs = pdk.ViewState(latitude=37.82, longitude=128.3, zoom=6.8)
                tip = {"html": "<b>{name}</b><br/>" + metric_label + ": {metric}"}
                st.caption(f"시군구별 {metric_label} · 🔴 높음 → 🟢 낮음")
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=vs, tooltip=tip, map_style=None))

        elif map_type == "고위험 필지 핫스팟":
            map_key = "ai_gap_score" if has_ai else "care_gap_score"
            n_show = min(5000, len(view))
            st.caption(f"관리공백 점수 상위 {n_show:,}개 필지 · 🔴높음 → 🟢낮음 (점 클릭 시 상세)")
            mdf = view.dropna(subset=["lon", "lat"]).sort_values(map_key, ascending=False).head(n_show).copy()
            s = mdf[map_key].clip(0, 100)
            mdf["r"] = (205 + s * 0.4).clip(0, 255).astype(int)
            mdf["g"] = (205 - s * 1.7).clip(0, 255).astype(int)
            mdf["b"] = (205 - s * 1.7).clip(0, 255).astype(int)
            if "ai_segment_name" not in mdf.columns:
                mdf["ai_segment_name"] = "-"
            layer = pdk.Layer(
                "ScatterplotLayer", data=mdf,
                get_position=["lon", "lat"], get_fill_color=["r", "g", "b", 170],
                get_radius=150, radius_min_pixels=2, radius_max_pixels=9, pickable=True,
            )
            vs = pdk.ViewState(latitude=37.82, longitude=128.3, zoom=7.1)
            tip = {"html": "<b>{polygon_id}</b> · {region}<br/>수종 {species}<br/>"
                           "관리공백점수 {" + map_key + "}<br/>유형 {ai_segment_name}"}
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=vs, tooltip=tip, map_style=None))

    # AI 관리유형(세그먼트) 분석 — 유형별 프로파일 + 설명 + 권장조치
    if has_ai and len(view):
        st.subheader("AI 관리유형(세그먼트) 분석")
        st.caption("KMeans 4유형 자동 분류 — 각 유형의 특성과 권장 조치(군집 평균 기반, 과장 없음)")
        seg_grp = view.groupby("ai_segment_name")
        seg_tbl = pd.DataFrame({
            "필지 수": seg_grp["polygon_id"].count(),
            "평균 AI공백점수": seg_grp["ai_gap_score"].mean().round(1),
            "평균 면적(ha)": seg_grp["area_ha"].mean().round(1),
        })
        if "forest_road_distance_m" in view.columns:
            seg_tbl["평균 임도거리(m)"] = seg_grp["forest_road_distance_m"].mean().round(0)
        if "val_at_risk" in view.columns:
            seg_tbl["위험노출가치(억원)"] = (seg_grp["val_at_risk"].sum() / 1e8).round(0)
        seg_tbl["비율(%)"] = (100 * seg_tbl["필지 수"] / seg_tbl["필지 수"].sum()).round(1)
        seg_tbl = seg_tbl.reset_index().rename(columns={"ai_segment_name": "관리유형"})
        seg_tbl["설명"] = seg_tbl["관리유형"].map(lambda s: SEGMENT_INFO.get(s, ("일반 관리 대상 유형", ""))[0])
        seg_tbl["권장 조치"] = seg_tbl["관리유형"].map(lambda s: SEGMENT_INFO.get(s, ("", "정기 모니터링"))[1])
        order = ["관리유형", "필지 수", "비율(%)", "평균 AI공백점수", "평균 면적(ha)"]
        order += [c for c in ["평균 임도거리(m)", "위험노출가치(억원)"] if c in seg_tbl.columns]
        order += ["설명", "권장 조치"]
        seg_tbl = seg_tbl.sort_values("필지 수", ascending=False)
        st.dataframe(seg_tbl[order], hide_index=True, width="stretch")
        if PLOTLY_AVAILABLE:
            dist = seg_tbl.set_index("관리유형")["필지 수"]
            figs = px.pie(values=dist.values, names=dist.index, hole=0.45)
            figs.update_layout(height=300, margin=dict(t=10, b=10), template="plotly_white",
                               paper_bgcolor="rgba(0,0,0,0)", font_color="#2B3947",
                               colorway=["#202327", "#4B5159", "#7A828B", "#A8AEB6", "#D2D6DB"])
            st.plotly_chart(figs, width="stretch")

    # 전체 테이블 (대용량 대비 상위 500개만 표시)
    st.subheader("관리공백 상위 필지")
    show_cols = [
        "polygon_id", "region", "species", "age_class", "care_gap_score",
        "care_gap_grade", "budget_priority_score",
    ]
    if has_ai:
        show_cols += ["ai_gap_score", "ai_segment_name"]
    sort_key = "ai_gap_score" if has_ai else "care_gap_score"
    st.caption(f"대상 {len(view):,}필지 중 점수 상위 500개")
    st.dataframe(
        view[show_cols].sort_values(sort_key, ascending=False).head(500),
        hide_index=True, width="stretch",
    )

    # 📋 현장조사 워크리스트 (현장용 export)
    st.subheader("📋 현장조사 워크리스트")
    st.caption("선택 지역의 우선순위 상위 필지를 현장 점검용 명단(CSV/PDF)으로 내보냅니다 — 좌표·근거·조치 포함.")
    wc1, wc2, wc3 = st.columns(3)
    w_region = wc1.selectbox("지역", ["강원 전체"] + sorted(scored["region"].dropna().unique().tolist()), key="wl_region")
    w_n = wc2.number_input("필지 수", min_value=10, max_value=500, value=50, step=10, key="wl_n")
    _sort_opts = []
    if "ai_gap_score" in scored.columns: _sort_opts.append("AI 관리공백 점수")
    if "budget_priority_score" in scored.columns: _sort_opts.append("예산 우선순위")
    if "care_gap_score" in scored.columns: _sort_opts.append("규칙 점수")
    w_sort = wc3.selectbox("정렬 기준", _sort_opts or ["polygon_id"], key="wl_sort")
    _scmap = {"AI 관리공백 점수": "ai_gap_score", "예산 우선순위": "budget_priority_score", "규칙 점수": "care_gap_score"}
    _sc = _scmap.get(w_sort, "polygon_id")

    wsrc = scored if w_region == "강원 전체" else scored[scored["region"] == w_region]
    wsrc = wsrc.sort_values(_sc, ascending=False).head(int(w_n))
    score_col = "ai_gap_score" if "ai_gap_score" in scored.columns else "care_gap_score"
    try:
        t95, t80, t50 = scored[score_col].quantile([0.95, 0.80, 0.50]).tolist()
    except Exception:
        t95 = t80 = t50 = 0

    def _wgrade(s):
        return "매우높음" if s >= t95 else "높음" if s >= t80 else "보통" if s >= t50 else "낮음"
    _ramap = {}
    _vrw = load_validation_report()
    if _vrw and isinstance(_vrw.get("per_region_auc"), dict):
        _ramap = {k: v.get("auc") for k, v in _vrw["per_region_auc"].items() if isinstance(v, dict)}
    _segs = (wsrc["ai_segment_name"].astype(str).values if "ai_segment_name" in wsrc.columns else ["-"] * len(wsrc))
    # 필지별 예측 신뢰도(트리 불일치 + 지역 AUC) — 사전계산 컬럼 우선, 없으면 즉석 계산
    if "ai_confidence" in wsrc.columns:
        _confs = wsrc["ai_confidence"].astype(str).values
    else:
        _confs = ["-"] * len(wsrc)
        try:
            if ml_bundle is not None and len(wsrc):
                _Xw = ml_engine.build_features(wsrc).fillna(ml_bundle["median"]).values
                _stdw = uncertainty.tree_disagreement(_Xw, ml_bundle["rf"])
                _confs = [uncertainty.confidence_single(float(s), _ramap.get(str(r)))
                          for s, r in zip(_stdw, wsrc["region"].values)]
        except Exception:
            pass
    worklist = pd.DataFrame({
        "순위": list(range(1, len(wsrc) + 1)),
        "필지ID": wsrc["polygon_id"].astype(str).values,
        "시군구": wsrc["region"].astype(str).values,
        "위도": (wsrc["lat"].round(5).values if "lat" in wsrc.columns else ["-"] * len(wsrc)),
        "경도": (wsrc["lon"].round(5).values if "lon" in wsrc.columns else ["-"] * len(wsrc)),
        "AI공백점수": wsrc[score_col].round(0).astype(int).values,
        "등급": [_wgrade(s) for s in wsrc[score_col].values],
        "관리유형": _segs,
        "권장조치": [SEGMENT_INFO.get(str(s), ("", "현장 확인 우선·정기 모니터링"))[1] for s in _segs],
        "신뢰도": _confs,
        "지역AUC": [(round(_ramap[str(r)], 2) if _ramap.get(str(r)) is not None else "-") for r in wsrc["region"].values],
    })
    st.dataframe(worklist, hide_index=True, width="stretch")
    dwc1, dwc2 = st.columns(2)
    dwc1.download_button("📥 워크리스트 CSV", data=worklist.to_csv(index=False).encode("utf-8-sig"),
                         file_name=f"현장조사_워크리스트_{w_region}.csv", mime="text/csv", key="wl_csv")
    if pdf_report.available():
        try:
            _wpdf = pdf_report.table_pdf(
                f"현장조사 워크리스트 — {w_region}", worklist,
                subtitle=f"ForestCareGap AI · 우선순위 상위 {len(worklist)}필지 · 사전진단(현장확인 필요)",
                disclaimer="관리공백 가능성 사전진단 · 확정 아님 · 좌표는 참고용이며 현장조사·행정 확인 필요",
                max_rows=int(w_n))
            dwc2.download_button("📄 워크리스트 PDF", data=_wpdf, key="wl_pdf",
                                 file_name=f"현장조사_워크리스트_{w_region}.pdf", mime="application/pdf")
        except Exception:
            pass
    st.caption("※ 좌표는 필지 대표점(참고용). 점수·등급은 우선순위이며 '방치 확정'이 아닙니다.")

    # 등급별 개수 차트
    st.subheader("관리공백 등급별 개수")
    grade_counts = (
        view["care_gap_grade"].value_counts()
        .reindex(["우선관리 필요", "관리 필요", "모니터링 필요", "관리 우선순위 낮음"])
        .fillna(0).astype(int)
    )
    if PLOTLY_AVAILABLE and len(view):
        fig = px.bar(
            x=grade_counts.index, y=grade_counts.values,
            labels={"x": "등급", "y": "개수"}, color=grade_counts.index,
            color_discrete_sequence=["#B23B3B", "#6E7176", "#9CA3AC", "#CDD2D7"],
        )
        fig.update_layout(
            showlegend=False, height=320, margin=dict(t=10, b=10),
            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#2B3947",
        )
        fig.update_xaxes(gridcolor="rgba(16,33,44,.06)")
        fig.update_yaxes(gridcolor="rgba(16,33,44,.06)")
        st.plotly_chart(fig, width="stretch")
    else:
        st.bar_chart(grade_counts)

    # 시군구 우선관리 순위 (전수 집계)
    st.subheader("시군구 우선관리 순위")
    grp = scored.groupby("region")
    region_tbl = pd.DataFrame({
        "전체 필지": grp["polygon_id"].count(),
        "고위험 필지": grp["care_gap_grade"].apply(lambda s: int((s == "우선관리 필요").sum())),
        "평균 관리공백": grp["care_gap_score"].mean().round(1),
    })
    region_tbl["고위험 비율(%)"] = (100 * region_tbl["고위험 필지"] / region_tbl["전체 필지"]).round(1)
    _hi = scored[scored["care_gap_grade"] == "우선관리 필요"]
    region_tbl["고위험 면적(ha)"] = _hi.groupby("region")["area_ha"].sum().round(0)
    if "val_at_risk" in scored.columns:
        region_tbl["위험노출가치(억원)"] = (grp["val_at_risk"].sum() / 1e8).round(0)
    if "ai_segment_name" in scored.columns:
        region_tbl["대표 유형"] = grp["ai_segment_name"].agg(
            lambda s: s.mode().iloc[0] if len(s.mode()) else "-")
    region_tbl = region_tbl.reset_index().rename(columns={"region": "시군구"}).fillna(0)

    _vr_reg = load_validation_report()
    if _vr_reg and isinstance(_vr_reg.get("per_region_auc"), dict):
        _ramap = {k: v.get("auc") for k, v in _vr_reg["per_region_auc"].items() if isinstance(v, dict)}
        region_tbl["모델신뢰도(AUC)"] = region_tbl["시군구"].map(_ramap).round(2)

    sort_choices = [c for c in ["위험노출가치(억원)", "고위험 비율(%)", "고위험 필지", "평균 관리공백"]
                    if c in region_tbl.columns]
    sort_by = st.selectbox("정렬 기준", sort_choices, key="region_sort")
    region_tbl = region_tbl.sort_values(sort_by, ascending=False)
    st.dataframe(region_tbl, hide_index=True, width="stretch")
    if pdf_report.available():
        try:
            _rpdf = pdf_report.table_pdf(
                "시군구 우선관리 순위", region_tbl,
                subtitle="ForestCareGap AI · 강원 전체 · 추정치(조정 가능)",
                disclaimer="관리공백 가능성 사전진단 · 확정 아님 · 현장조사·행정 확인 필요")
            st.download_button("📄 시군구 순위 PDF 다운로드", data=_rpdf,
                               file_name="ForestCareGap_시군구순위.pdf", mime="application/pdf")
        except Exception:
            pass

    rc1, rc2 = st.columns(2)
    with rc1:
        st.caption("시군구별 고위험 필지 수")
        bar1 = region_tbl.set_index("시군구")["고위험 필지"].sort_values(ascending=False)
        if PLOTLY_AVAILABLE and len(bar1):
            figb = px.bar(x=bar1.index, y=bar1.values, labels={"x": "", "y": "고위험 필지"})
            figb.update_layout(height=300, margin=dict(t=8, b=8), template="plotly_white",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#2B3947")
            figb.update_traces(marker_color="#B23B3B")
            st.plotly_chart(figb, width="stretch")
        else:
            st.bar_chart(bar1)
    with rc2:
        if "위험노출가치(억원)" in region_tbl.columns:
            st.caption("시군구별 위험노출 가치 (억원)")
            bar2 = region_tbl.set_index("시군구")["위험노출가치(억원)"].sort_values(ascending=False)
            if PLOTLY_AVAILABLE and len(bar2):
                figv = px.bar(x=bar2.index, y=bar2.values, labels={"x": "", "y": "억원"})
                figv.update_layout(height=300, margin=dict(t=8, b=8), template="plotly_white",
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color="#2B3947")
                figv.update_traces(marker_color="#202327")
                st.plotly_chart(figv, width="stretch")
            else:
                st.bar_chart(bar2)
    st.caption("※ 전수 집계 · 위험노출가치는 추정치(조정 가능)이며 확정액이 아닙니다.")

    # 예산 우선순위 Top 10
    st.subheader("예산 우선순위 Top 10")
    st.caption("예산 우선순위 = 관리공백×0.45 + 재해위험×0.25 + 관리효과×0.20 + 접근성×0.10")
    top10 = (
        scored.sort_values("budget_priority_score", ascending=False)
        .head(10)[[
            "polygon_id", "region", "care_gap_score", "care_gap_grade",
            "budget_priority_score",
        ]]
        .reset_index(drop=True)
    )
    top10.index = top10.index + 1
    st.dataframe(top10, width="stretch")

# ----------------------------- 탭 4. 성과 추적 ------------------------------ #
with tab4:
    st.header("관리 후 성과 추적")
    if tracking_df.empty:
        st.write("성과 추적 데이터가 없습니다.")
    else:
        tpid = st.selectbox("폴리곤 ID 선택", tracking_df["polygon_id"].tolist(), key="track_pid")
        trow = tracking_df[tracking_df["polygon_id"] == tpid].iloc[0]
        outcome = calculate_management_outcome(trow["before_score"], trow["after_score"])

        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("관리 전 점수", f"{trow['before_score']}")
        oc2.metric("관리 후 점수", f"{trow['after_score']}",
                   delta=f"-{outcome['score_reduction']}" if outcome["score_reduction"] > 0 else "0")
        oc3.metric("성과 등급", outcome["outcome_grade"])

        st.subheader("관리 활동 정보")
        st.write(f"- **관리 활동**: {trow['management_action']}")
        st.write(f"- **관리 일자**: {trow['management_date']}")
        st.write(f"- **현장조사 완료**: {trow['field_survey_done']}")
        st.write(f"- **산주 연락**: {trow['owner_contacted']}")
        st.write(f"- **연계 지원사업**: {trow['support_program_linked']}")
        st.write(f"- **비고**: {trow['notes']}")

        st.subheader("성과 메시지")
        if outcome["score_reduction"] >= 10:
            st.success(outcome["message"])
        elif outcome["score_reduction"] >= 1:
            st.info(outcome["message"])
        else:
            st.warning(outcome["message"])

# ----------------------------- 탭 5. 기술 설명 ------------------------------ #
with tab5:
    st.header("기술 설명")

    # 실제 학습된 AI 모델 카드
    if ml_bundle is not None:
        st.subheader("🤖 탑재된 AI 모델 (실측)")
        auc = ml_bundle.get("cv_auc", float("nan"))
        auc_sp = ml_bundle.get("cv_auc_spatial", float("nan"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("모델", "RandomForest")
        c2.metric("AUC (랜덤 CV)", f"{auc:.3f}" if auc == auc else "-",
                  help="일반 5겹 교차검증 — 같은 지역 내 예측 성능")
        c3.metric("AUC (공간 CV)", f"{auc_sp:.3f}" if auc_sp == auc_sp else "-",
                  help="시군구 블록 홀드아웃 — 미관측 지역 일반화(정직한 보수적 수치)")
        c4.metric("학습 표본", f"{ml_bundle.get('n', 0):,} 필지")
        st.caption(
            "준지도 학습(실제 산림경영활동 시행여부 라벨) + 공간 lag 피처(이웃 관리율). "
            "관리는 공간적으로 군집하므로 **랜덤 CV는 높고(낙관), 공간 CV는 보수적(정직)** — 둘 다 공개합니다."
        )
        imp = ml_bundle.get("importance", {})
        if imp:
            st.caption("AI 변수 중요도 Top")
            st.dataframe(
                pd.DataFrame(list(imp.items())[:8], columns=["변수", "중요도"]),
                hide_index=True, width="stretch",
            )
        names = ml_bundle.get("cluster_names", {})
        if names:
            st.caption("AI 관리유형(KMeans 4군집): " + " · ".join(sorted(set(names.values()))))
        st.divider()

    # 📏 honest 검증 리포트 (validation_suite.py 산출)
    vr = load_validation_report()
    if vr:
        st.subheader("📏 모델 검증 (honest · 공간 교차검증)")
        lc = vr.get("leakage_check", {})
        ha = vr.get("honest_auc", {})
        v1, v2, v3 = st.columns(3)
        v1.metric("honest AUC (공간 CV)", f"{ha.get('fold_mean', float('nan')):.3f}",
                  help="시군구 그룹 홀드아웃 — 미관측 지역 일반화. 메인 성능지표.")
        v2.metric("naive(누수) AUC", f"{lc.get('naive_with_leakage', float('nan')):.3f}",
                  help="공간 누수를 통제하지 않은 낙관치(참고용, 신뢰하지 않음)")
        v3.metric("제거한 거품", f"{lc.get('inflation_prevented', 0):.3f}",
                  help="누수를 폴드마다 차단해 방지한 AUC 부풀림(낙관−정직)")
        st.info(
            "0.98처럼 보이는 점수는 **공간 자기상관 누수**로 부풀려진 값입니다. 우리는 폴드마다 이웃을 "
            "train 에서만 재계산해 누수를 차단하고 **정직한 0.69**를 보고합니다 — 이것이 본 모델의 신뢰 근거입니다.",
            icon="🛡️")

        st.markdown("##### 우선순위 도구로서의 적중률 (Top-K)")
        rl = vr.get("ranking_lift", {})
        lift_rows = [{"상위 구간": k, "적중률": f"{r['precision'] * 100:.1f}%",
                      "평균대비 lift": f"{r['lift']}배", "관리대상 포착률": f"{r['recall'] * 100:.0f}%"}
                     for k, r in rl.items()]
        if lift_rows:
            st.dataframe(pd.DataFrame(lift_rows), hide_index=True, width="stretch")
            st.caption("점수 상위 20%만 확인해도 전체 관리대상 필지의 절반을 포착 → 한정 인력의 현장확인 우선순위 도구.")

        bl = vr.get("baselines", {})
        ab = vr.get("ablation", {})
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("##### 단순 규칙 대비")
            st.dataframe(pd.DataFrame([
                {"방법": "무작위", "honest AUC": bl.get("random")},
                {"방법": "구조적 규칙", "honest AUC": bl.get("structural_rules_clean")},
                {"방법": "AI (RandomForest)", "honest AUC": bl.get("ml_full")},
            ]), hide_index=True, width="stretch")
            st.caption(f"AI가 단순 규칙보다 +{bl.get('ml_vs_rules_gain', 0):.2f} 우수.")
        with b2:
            st.markdown("##### 피처 기여 (ablation)")
            st.dataframe(pd.DataFrame([
                {"구성": "이웃관리율 단독", "honest AUC": ab.get("neighbor_only")},
                {"구성": "이웃관리율 제거", "honest AUC": ab.get("without_neighbor")},
                {"구성": "전체 피처", "honest AUC": ab.get("full")},
            ]), hide_index=True, width="stretch")
            st.caption("지형·임상·토양이 이웃신호보다 더 기여 → 단순 공간 베끼기가 아님.")

        pr = vr.get("per_region_auc", {})
        ps = vr.get("per_region_summary", {})
        if pr:
            st.markdown("##### 지역별 honest AUC (안정성·약점 공개)")
            prdf = pd.DataFrame([{"시군구": k, "honest_auc": v["auc"]} for k, v in pr.items()])
            if PLOTLY_AVAILABLE:
                figp = px.bar(prdf.sort_values("honest_auc"), x="honest_auc", y="시군구",
                              orientation="h", labels={"honest_auc": "honest AUC", "시군구": ""})
                figp.add_vline(x=0.5, line_dash="dash", line_color="#9AA4AE")
                figp.update_layout(height=430, margin=dict(t=8, b=8), template="plotly_white",
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color="#2B3947")
                figp.update_traces(marker_color="#202327")
                st.plotly_chart(figp, width="stretch")
            else:
                st.dataframe(prdf, hide_index=True, width="stretch")
            st.caption(f"지역 편차 {ps.get('worst')}~{ps.get('best')} (점선 0.5 = 무작위). "
                       "일부 지역(예: 삼척)은 약해 향후 지역보정이 과제 — 한계를 정직하게 공개합니다.")

        og = vr.get("ownership_subgroup")
        if og:
            st.markdown("##### 국유/사유 서브그룹 (소유 유형별 정직성 점검)")
            st.dataframe(pd.DataFrame([
                {"소유 추정": k, "honest AUC": v.get("auc"), "필지수": v.get("n"),
                 "관리시행률(%)": v.get("관리시행률(%)")} for k, v in og.items()
            ]), hide_index=True, width="stretch")
            st.caption("국유림 신호로 소유 유형을 나눠 성능을 점검 — 특정 소유 유형에 치우치지 않는지 정직하게 공개.")

        # 🧪 PU(Positive-Unlabeled) 학습 — 라벨 불완전성 정량화
        import json as _json
        _pup = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "experiment_pu_learning.json")
        if os.path.exists(_pup):
            try:
                _pu = _json.load(open(_pup, encoding="utf-8"))
                st.markdown("##### 🧪 PU(Positive-Unlabeled) 학습 — 라벨 불완전성 정량화")
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("라벨빈도 c (Elkan-Noto)", f"{_pu['elkan_noto_c']:.2f}",
                           help="실제 관리된 산림 중 '기록이 남은' 비율(SCAR 가정). 1이면 라벨 완전.")
                pc2.metric("추정 진짜 관리율", f"{_pu['estimated_true_prevalence'] * 100:.2f}%",
                           help=f"관측 {_pu['observed_pos_rate'] * 100:.1f}% ÷ c — 기록 누락 보정")
                pc3.metric("Bagging-PU 랭킹(honest)", f"{_pu['bagging_pu_spatial']['mean']:.3f}",
                           delta=f"{_pu['delta_spatial']:+.4f} vs 현행", delta_color="off")
                st.caption(
                    f"우리 라벨의 '0'은 '관리 안 됨'이 아니라 '기록 없음'(Unlabeled)입니다. PU 학습으로 "
                    f"라벨빈도 c={_pu['elkan_noto_c']:.2f} 추정 → 관리 기록은 실제 관리의 약 {_pu['elkan_noto_c'] * 100:.0f}%만 포착, "
                    f"**진짜 관리율은 ~{_pu['estimated_true_prevalence'] * 100:.1f}%**로 추정(관측 {_pu['observed_pos_rate'] * 100:.1f}%). "
                    "Bagging-PU 재학습으로 랭킹을 검증했으나 현행과 동등(개선 없음) → 랭킹은 견고한 현행을 유지하고, "
                    "PU는 **라벨 불완전성을 정직하게 정량화**하는 데 사용합니다. (experiment_pu_learning.py)")
            except Exception:
                pass

        st.success(
            "**확률 보정 적용** — 위 신뢰도곡선처럼 원시 모델 확률은 과대확신 경향이 있어, isotonic 보정을 "
            "**표시되는 'AI 추정 관리확률'에 적용**했습니다. 순위(랭킹)는 그대로이고 확률값이 실제 관측률에 가깝게 정직해집니다.",
            icon="✅")
        st.caption(
            "측정: validation_suite.py · 공간 GroupKFold(시군구 19) · 폴드별 이웃관리율 재계산(누수 차단) · "
            f"양성률 {vr.get('base_rate', 0) * 100:.0f}% · n={vr.get('n', 0):,}")
        st.divider()

    # 📚 데이터 활용 및 실험 이력 (채택/기각) — 데이터활용 점수 핵심
    st.subheader("📚 데이터 활용 및 실험 이력")
    st.caption("공개데이터 다종을 융합하고, 신규 피처는 '실험 후 honest 공간검증'으로 채택/기각했습니다.")
    data_table = pd.DataFrame([
        {"데이터": "수치임상도", "사용 변수": "수종·영급·경급·수관밀도", "역할": "핵심 입력", "상태": "✅ 채택"},
        {"데이터": "DEM(수치표고)", "사용 변수": "고도·경사·향", "역할": "지형", "상태": "✅ 채택"},
        {"데이터": "임도망도", "사용 변수": "최근접 임도거리", "역할": "접근성", "상태": "✅ 채택"},
        {"데이터": "임도망도 (밀도·구간수)", "사용 변수": "1km연장·반경내 구간수", "역할": "망 접근도", "상태": "❌ 기각 (honest −0.001)"},
        {"데이터": "일반도로 (연속수치지형도 도로중심선)", "사용 변수": "최근접 일반도로 거리", "역할": "정밀 접근성", "상태": "✅ 채택 (honest +0.015)"},
        {"데이터": "산사태위험지도", "사용 변수": "위험등급", "역할": "재해위험", "상태": "✅ 채택"},
        {"데이터": "산림기능구분도", "사용 변수": "자연환경보전림", "역할": "보전 제약", "상태": "✅ 채택"},
        {"데이터": "산림입지토양도", "사용 변수": "모암·토양형·사면형", "역할": "토양", "상태": "✅ 채택 (honest +0.014)"},
        {"데이터": "산림입지토양도 (토성·토심·기후대 등 5종)", "사용 변수": "—", "역할": "—", "상태": "❌ 기각 (노이즈)"},
        {"데이터": "산림사업 이력 (숲가꾸기·조림)", "사용 변수": "시행 여부", "역할": "학습 라벨", "상태": "✅ 채택"},
        {"데이터": "이웃관리율 (파생 피처)", "사용 변수": "반경 20이웃 관리율", "역할": "공간 패턴", "상태": "✅ 채택 (중요도 0.79)"},
        {"데이터": "국유림 경영계획부 이력 (소반 좌표)", "사용 변수": "최근접 국유림 거리·근접", "역할": "소유(국유) 신호", "상태": "✅ 채택 (honest +0.008)"},
        {"데이터": "경제림육성단지도 (ecoFrst)", "사용 변수": "단지 소속·목재생산림·최근접 거리", "역할": "경영우선 정책구역", "상태": "❌ 기각 (honest +0.0025, 기준 미달)"},
        {"데이터": "경제가치 계수 (국립산림과학원)", "사용 변수": "탄소·목재·공익", "역할": "가치 환산", "상태": "✅ 채택"},
    ])
    st.dataframe(data_table, hide_index=True, width="stretch")
    st.success(
        "**실험 후 채택/기각** — 입지토양(+0.014)·국유림 소유신호(+0.008)·일반도로 거리(+0.015)는 honest 검증으로 "
        "채택, **임도망 밀도/구간수는 −0.001로 기각**(임도거리와 중복), 토양 5종은 노이즈로 제외했습니다. "
        "같은 '도로'라도 새 정보를 주는 일반도로는 채택하고 중복인 임도밀도는 버리는 — **'정직하게 검증한 뒤 "
        "반영'**이 본 모델의 데이터 활용 원칙입니다.", icon="🧪")
    st.divider()

    st.markdown(
        f"""
#### 입력 데이터
- (프로토타입) 샘플 산림 폴리곤 CSV — 임상도·DEM·임도망·산림사업 이력 등 **공공데이터 연결을 가정한 22개 속성**
- 확장 시: 수치임상도, 수치표고모델(DEM), 산림입지토양도, 임도망도, 산림사업이력, 산사태·산불 위험지도 등

#### 진단 로직 (이중 구조)
- **규칙 기반(설명용 기준선)**: 최근 관리이력 없음(+25), 관리이력 10년+(+15), 고영급(+15),
  수관밀도 '밀'(+15), 임도거리 500m+(+15), 급경사(+10), 산사태/산불(+10/+10), 부재산주(+10),
  사유림(+5), 보호구역(+5) → 0~100 등급화
- **AI 모델(준지도 학습)**: 실제 산림경영활동 시행 여부를 학습 → '관리확률' 추정,
  (1−관리확률)을 관리공백 점수로 산출. 규칙과 교차검증해 신뢰도 보강
- 임계값(탄소 ≥ {ABSORPTION_HIGH} tCO2/년, 임목 ≥ {TIMBER_HIGH} m³)은 처방·예산 점수에 사용

#### 설명가능 AI (XAI) — 구현됨
- **트리경로 기반 기여분해**(treeinterpreter 방식, SHAP과 동일 취지의 가법적 설명)로
  필지별 '관리공백 기여 요인'을 항상 제시 → 블랙박스 지양
- 규칙 factor + AI 기여도 + 자연어 설명을 함께 제공

#### 관리유형 군집 (KMeans) — 구현됨
- 산림을 4개 유형(재해취약/외진 관리취약/경영활성/보전우선 등)으로 자동 분류 → 맞춤 처방 연결

#### 방법론 고도화 — 구현됨
- **공간 분리(Spatial Block) 교차검증**: 시군구 GroupKFold + 폴드별 이웃관리율 재계산으로 누수 차단 → honest AUC
- **PU(Positive-Unlabeled) 학습**: 라벨 '0'=기록없음(관리안됨 아님)을 정면으로 다뤄 라벨빈도 c≈0.86 추정,
  진짜 관리율 ~4.6% 보정. Bagging-PU로 랭킹 견고성 검증
- **예측 신뢰도(불확실성)**: 트리 400그루 불일치 + 지역 honest AUC를 결합해 필지별 신뢰도(높음/보통/낮음) 표시
- **RAG 맞춤 안내문**: 지원사업·관리기준 지식베이스 검색(TF-IDF) → 근거 안에서만 LLM 생성(환각 억제, 키 없으면 폴백)

#### 향후 고도화
- 위성 식생지수(NDVI) 등 원격탐사 모달리티 융합, 지역 보정, 처방-성과 인과 추정

#### 한계 및 주의사항
- 본 프로토타입은 **샘플 데이터** 기반이며 실제 산림 상태를 단정하지 않습니다.
- 모든 결과는 **관리공백 가능성 사전진단**이며, 신청·시행은 **현장조사·지자체·산림조합 확인**이 필요합니다.
- GeoPandas 등 공간 라이브러리는 선택 사항이며, 미설치 환경에서도 표·점수·리포트 중심으로 작동합니다.

---
**환경 점검**: GeoPandas {'사용 가능 ✅' if GEOPANDAS_AVAILABLE else '미설치(표 중심으로 동작) ⚠️'}
· Plotly {'사용 가능 ✅' if PLOTLY_AVAILABLE else '미설치(내장 차트로 대체) ⚠️'}
"""
    )
    st.caption(DISCLAIMER)
