# -*- coding: utf-8 -*-
"""ForestCareGap AI · ML 엔진 (준지도 학습 + 설명가능 AI + 관리유형 군집)

설계 (독창성):
  1) 준지도 학습 — 실제 산림경영활동(숲가꾸기·조림) '시행 여부'를 라벨로
     RandomForest 를 학습해 '관리가 일어나는 산림의 특성'을 데이터로 학습한다.
     → 미관리·고가치 필지 = 구조적 '관리공백' (라벨 없는 공백을 학습으로 추정).
  2) 설명가능 AI — 트리 경로 기반 기여분해(treeinterpreter 방식, SHAP 과 동일 취지의
     가법적 기여도)로 '왜 이 점수인가'를 필지별로 설명한다(외부 의존성 없음).
  3) 관리유형 군집 — KMeans 로 산림을 4개 유형으로 자동 분류해 맞춤 처방을 연결한다.

주의: '방치 확정'이 아니라 '관리공백 가능성 사전진단'. 모든 출력은 확률·점수·유형이다.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import BallTree
from sklearn.isotonic import IsotonicRegression

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "ml_engine.joblib")

CONIFERS = {"소나무","잣나무","낙엽송","리기다소나무","편백","곰솔","삼나무","전나무","가문비나무","해송"}
SOUTH = {"남","남서","남동"}
_ORD_DENS = {"소":0,"중":1,"밀":2}
_ORD_DIAM = {"치수":0,"소경목":1,"중경목":2,"대경목":3}
_ORD_RISK = {"낮음":0,"중간":1,"높음":2}

# (내부키, 한글라벨) — 설명·중요도 표기에 사용. 누수 컬럼(관리이력·소유·도로복제)은 제외.
FEATURES = [
    ("forest_road_distance_m", "임도 거리"),
    ("slope_deg",              "경사"),
    ("elevation_m",            "고도"),
    ("area_ha",                "면적"),
    ("age_class",              "영급"),
    ("dens_o",                 "수관밀도"),
    ("diam_o",                 "경급"),
    ("landslide_o",            "산사태위험"),
    ("fire_o",                 "산불위험"),
    ("aspect_south",           "남향"),
    ("is_conifer",             "침엽수"),
    ("protected_i",            "보호구역"),
    ("carbon_storage_tco2",    "탄소저장량"),
    ("annual_absorption_tco2", "연간흡수량"),
    ("timber_potential_m3",    "목재잠재량"),
    ("neighbor_mgmt_rate",     "이웃 관리율"),   # 공간 lag — 주변 필지의 관리시행 비율
    # 입지토양도(모암·토양형·사면형) — honest 이중 CV 재검증 +0.0163 으로 채택(4변수).
    # factorize 정수코드(build_dataset 에서 562k 일괄 인코딩, parquet 에 고정 저장).
    ("soil_PRRCK_LARG",        "모암(대분류)"),
    ("soil_PRRCK_MDDL",        "모암(중분류)"),
    ("soil_SLTP_CD",           "토양형"),
    ("soil_SLANT_TYP",         "사면형"),
    # 국유림 소유 신호(경영계획부 소반 최근접) — honest 이중 CV +0.0148 로 채택.
    ("gukyu_dist_m",           "국유림 거리"),
    ("gukyu_in500",            "국유림 500m내"),
    ("gukyu_in1km",            "국유림 1km내"),
    # 일반도로(연속수치지형도 도로중심선) 접근성 — honest 이중 CV +0.0149 채택
    ("road2_dist_m",           "일반도로 거리"),
    ("road2_in100",            "도로 100m내"),
    ("road2_in300",            "도로 300m내"),
]
NBR_K = 20  # 이웃 수
SOIL_FEAT_KEYS = ["soil_PRRCK_LARG", "soil_PRRCK_MDDL", "soil_SLTP_CD", "soil_SLANT_TYP"]
GUKYU_FEAT_KEYS = ["gukyu_dist_m", "gukyu_in500", "gukyu_in1km"]
ROAD2_FEAT_KEYS = ["road2_dist_m", "road2_in100", "road2_in300"]
FEAT_KEYS = [k for k, _ in FEATURES]
FEAT_LABELS = {k: v for k, v in FEATURES}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """원천 22컬럼 DataFrame → 모델 입력 피처(숫자) DataFrame."""
    x = pd.DataFrame(index=df.index)
    num = lambda c: pd.to_numeric(df.get(c), errors="coerce")
    x["forest_road_distance_m"] = num("forest_road_distance_m")
    x["slope_deg"]   = num("slope_deg")
    x["elevation_m"] = num("elevation_m")
    x["area_ha"]     = num("area_ha")
    x["age_class"]   = num("age_class")
    x["dens_o"] = df.get("crown_density").map(_ORD_DENS) if "crown_density" in df else np.nan
    x["diam_o"] = df.get("diameter_class").map(_ORD_DIAM) if "diameter_class" in df else np.nan
    x["landslide_o"] = df.get("landslide_risk").map(_ORD_RISK) if "landslide_risk" in df else np.nan
    x["fire_o"]      = df.get("fire_risk").map(_ORD_RISK) if "fire_risk" in df else np.nan
    x["aspect_south"] = df.get("aspect").isin(SOUTH).astype(float) if "aspect" in df else 0.0
    x["is_conifer"]   = df.get("species").isin(CONIFERS).astype(float) if "species" in df else 0.0
    x["protected_i"]  = df.get("protected_area").map(
        lambda v: 1.0 if str(v).strip().lower() in ("true","1","yes","예","참") or v is True else 0.0
    ) if "protected_area" in df else 0.0
    x["carbon_storage_tco2"]    = num("carbon_storage_tco2")
    x["annual_absorption_tco2"] = num("annual_absorption_tco2")
    x["timber_potential_m3"]    = num("timber_potential_m3")
    x["neighbor_mgmt_rate"]     = num("neighbor_mgmt_rate").fillna(0.0) if "neighbor_mgmt_rate" in df else 0.0
    # 토양 정수코드(없으면 -1). 결측/구버전 CSV 에도 안전.
    for sc in SOIL_FEAT_KEYS:
        x[sc] = pd.to_numeric(df[sc], errors="coerce").fillna(-1.0) if sc in df.columns else -1.0
    # 국유림 소유 신호. dist 결측은 caller median 대체, 이진은 0.
    x["gukyu_dist_m"] = pd.to_numeric(df["gukyu_dist_m"], errors="coerce") if "gukyu_dist_m" in df.columns else np.nan
    x["gukyu_in500"]  = pd.to_numeric(df["gukyu_in500"], errors="coerce").fillna(0.0) if "gukyu_in500" in df.columns else 0.0
    x["gukyu_in1km"]  = pd.to_numeric(df["gukyu_in1km"], errors="coerce").fillna(0.0) if "gukyu_in1km" in df.columns else 0.0
    # 일반도로 접근성. dist 결측은 caller median 대체, 이진은 0.
    x["road2_dist_m"] = pd.to_numeric(df["road2_dist_m"], errors="coerce") if "road2_dist_m" in df.columns else np.nan
    x["road2_in100"]  = pd.to_numeric(df["road2_in100"], errors="coerce").fillna(0.0) if "road2_in100" in df.columns else 0.0
    x["road2_in300"]  = pd.to_numeric(df["road2_in300"], errors="coerce").fillna(0.0) if "road2_in300" in df.columns else 0.0
    return x[FEAT_KEYS]


def _label_int(s) -> "np.ndarray":
    return s.map(lambda v: 1 if (v is True or str(v).strip().lower() in ("true", "1", "yes", "예", "참")) else 0).astype(int).values


def compute_neighbor_rate(df: pd.DataFrame, k: int = NBR_K, train_mask=None) -> "np.ndarray":
    """각 필지의 '이웃 관리율' = 가장 가까운 k개 이웃의 관리시행 평균(자기 제외).

    train_mask 가 주어지면 이웃 후보를 train 으로 한정(누수 방지/CV용).
    """
    if not {"lat", "lon"}.issubset(df.columns):
        return np.zeros(len(df), dtype=np.float32)
    coords = np.radians(df[["lat", "lon"]].to_numpy(dtype=float))
    y = _label_int(df["has_recent_management"])
    src = np.arange(len(df)) if train_mask is None else np.where(train_mask)[0]
    tree = BallTree(coords[src], metric="haversine")
    dist, nn = tree.query(coords, k=min(k + 1, len(src)))
    ynn = y[src][nn]
    wgt = (dist > 1e-9).astype(np.float32)        # 거리0(자기) 가중 0
    return ((ynn * wgt).sum(1) / np.clip(wgt.sum(1), 1, None)).astype(np.float32)


def _label_clusters(centroids_real: pd.DataFrame, mgmt_rate: pd.Series) -> dict:
    """군집 중심 특성으로 관리유형 이름을 자동 부여(독창적 세그먼트)."""
    names = {}
    used = set()
    # 우선순위 규칙: 재해취약 > 관리취약(외짐) > 경영활성(접근양호+관리율) > 보전우선
    g = centroids_real
    order = []
    order.append((g["landslide_o"].idxmax(), "재해취약형"))
    order.append((g["forest_road_distance_m"].idxmax(), "외진 관리취약형"))
    order.append((mgmt_rate.idxmax(), "경영활성형"))
    order.append((g["protected_i"].idxmax(), "보전우선형"))
    for cid, nm in order:
        if cid not in used:
            names[cid] = nm; used.add(cid)
    for cid in g.index:           # 남는 군집
        if cid not in names:
            names[cid] = "일반관리형"
    return names


def _make_rf():
    return RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=8,
        class_weight="balanced", n_jobs=-1, random_state=42)


def _cv_auc(Xf, y, df, splits, groups=None, return_oof=False):
    """폴드마다 이웃관리율을 train 라벨로 재계산(누수 방지)해 AUC 측정.
    return_oof=True 면 (평균AUC, out-of-fold 확률배열) 반환(확률 보정용)."""
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return (float("nan"), np.full(len(y), np.nan)) if return_oof else float("nan")
    has_nbr = {"lat", "lon"}.issubset(df.columns)
    nbr_idx = FEAT_KEYS.index("neighbor_mgmt_rate")
    aucs = []
    oof = np.full(len(y), np.nan)
    it = splits.split(Xf, y, groups) if groups is not None else splits.split(Xf, y)
    for tr, te in it:
        Xt = Xf.copy()
        if has_nbr:
            mask = np.zeros(len(df), dtype=bool); mask[tr] = True
            Xt[:, nbr_idx] = compute_neighbor_rate(df, train_mask=mask)
        m = _make_rf(); m.fit(Xt[tr], y[tr])
        p = m.predict_proba(Xt[te])[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p))
    return (float(np.mean(aucs)), oof) if return_oof else float(np.mean(aucs))


def train_from_frame(df: pd.DataFrame, out_path: str = MODEL_PATH) -> dict:
    """좌표·이웃관리율을 포함한 DataFrame 으로 학습(이웃관리율 적용 경로)."""
    df = df.copy()
    if {"lat", "lon"}.issubset(df.columns) and "neighbor_mgmt_rate" not in df.columns:
        df["neighbor_mgmt_rate"] = compute_neighbor_rate(df)
    X = build_features(df)
    med = X.median(numeric_only=True)
    Xf = X.fillna(med).values.astype(np.float64)
    y = _label_int(df["has_recent_management"])

    # 이중 교차검증: 랜덤(낙관) + 공간블록(정직, 미관측 지역 일반화)
    auc_rand, oof_rand = _cv_auc(Xf, y, df, StratifiedKFold(5, shuffle=True, random_state=42), return_oof=True)
    auc_sp = float("nan")
    if "region" in df.columns and df["region"].nunique() >= 5:
        auc_sp = _cv_auc(Xf, y, df, GroupKFold(n_splits=5), groups=df["region"].astype(str).values)

    rf = _make_rf(); rf.fit(Xf, y)
    # 확률 보정(isotonic) — 랜덤 CV OOF 로 적합해 과대확신 교정(순위 불변, 표시 확률만 보정)
    calibrator = None
    try:
        ok = ~np.isnan(oof_rand)
        if ok.sum() > 100 and y[ok].sum() > 10:
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_rand[ok], y[ok])
    except Exception:
        calibrator = None
    scaler = StandardScaler().fit(Xf)
    km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(scaler.transform(Xf))
    cent_real = pd.DataFrame(Xf, columns=FEAT_KEYS).groupby(km.labels_).mean()
    mgmt_rate = pd.Series(y).groupby(km.labels_).mean()
    cluster_names = {int(k): v for k, v in _label_clusters(cent_real, mgmt_rate).items()}
    importance = dict(sorted(zip([FEAT_LABELS[k] for k in FEAT_KEYS], rf.feature_importances_.tolist()),
                             key=lambda t: t[1], reverse=True))
    bundle = {
        "rf": rf, "scaler": scaler, "kmeans": km, "median": med,
        "cluster_names": cluster_names, "features": FEAT_KEYS,
        "cv_auc": auc_rand, "cv_auc_spatial": auc_sp, "importance": importance,
        "pos_rate": float(y.mean()), "n": int(len(y)), "calibrator": calibrator,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(bundle, out_path)
    return {"cv_auc": auc_rand, "cv_auc_spatial": auc_sp, "n": len(y), "pos": int(y.sum()),
            "importance_top": list(importance.items())[:6], "clusters": cluster_names}


def train_and_save(csv_path: str, out_path: str = MODEL_PATH) -> dict:
    """CSV 경로로 학습(좌표 없으면 이웃관리율=0 → 베이스 모델). 호환용."""
    return train_from_frame(pd.read_csv(csv_path), out_path)


def load(path: str = MODEL_PATH):
    return joblib.load(path) if os.path.exists(path) else None


def predict_dataframe(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """df 에 AI 컬럼 추가: ai_management_prob, ai_gap_score(0~100), ai_segment, ai_segment_name."""
    X = build_features(df).fillna(bundle["median"])
    p = bundle["rf"].predict_proba(X.values)[:, 1]            # 관리(경영) 확률
    # 관리공백 점수 = (1 - 관리확률) 을 데이터 내 min-max 로 0~100 스케일
    inv = 1.0 - p
    lo, hi = inv.min(), inv.max()
    gap = 100 * (inv - lo) / (hi - lo) if hi > lo else inv * 0
    seg = bundle["kmeans"].predict(bundle["scaler"].transform(X.values))
    out = df.copy()
    cal = bundle.get("calibrator")
    p_show = cal.predict(p) if cal is not None else p   # 보정 확률(표시용). gap 순위는 raw 기준 불변
    out["ai_management_prob"] = np.round(p_show, 3)
    out["ai_gap_score"] = np.round(gap, 1)
    out["ai_segment"] = seg
    out["ai_segment_name"] = [bundle["cluster_names"].get(int(s), "일반관리형") for s in seg]
    return out


def explain_row(df_row: pd.Series, bundle: dict, topk: int = 5) -> list[dict]:
    """트리 경로 기반 가법적 기여분해로 '관리공백' 기여 요인 상위 topk 를 반환.

    각 요인의 기여도 = 관리확률을 '낮추는'(=공백을 키우는) 방향이면 양수.
    """
    rf = bundle["rf"]
    X = build_features(pd.DataFrame([df_row])).fillna(bundle["median"]).values.astype(np.float32)
    contribs = np.zeros(len(FEAT_KEYS))
    for est in rf.estimators_:
        t = est.tree_
        path = est.decision_path(X).indices       # 루트→리프 노드 순서
        for a, b in zip(path[:-1], path[1:]):
            f = t.feature[a]
            if f < 0:
                continue
            pa = t.value[a][0]; pa = pa / pa.sum()
            pb = t.value[b][0]; pb = pb / pb.sum()
            contribs[f] += (pb[1] - pa[1])        # P(관리) 변화량
    contribs /= len(rf.estimators_)
    gap_contrib = -contribs                        # 공백 방향(관리확률 감소 = 공백 증가)
    order = np.argsort(-np.abs(gap_contrib))[:topk]
    res = []
    for i in order:
        res.append({
            "요인": FEAT_LABELS[FEAT_KEYS[i]],
            "값": _fmt_val(df_row, FEAT_KEYS[i]),
            "기여(공백↑)": round(float(gap_contrib[i]), 3),
            "방향": "공백↑" if gap_contrib[i] > 0 else "공백↓",
        })
    return res


def _fmt_val(row: pd.Series, key: str):
    m = {"forest_road_distance_m": ("forest_road_distance_m", "m"),
         "slope_deg": ("slope_deg", "°"), "elevation_m": ("elevation_m", "m"),
         "area_ha": ("area_ha", "ha")}
    if key in m:
        c, u = m[key]
        try: return f"{float(row.get(c)):.0f}{u}"
        except Exception: return "-"
    raw = {"dens_o": "crown_density", "diam_o": "diameter_class",
           "landslide_o": "landslide_risk", "fire_o": "fire_risk",
           "age_class": "age_class", "aspect_south": "aspect", "is_conifer": "species",
           "protected_i": "protected_area", "carbon_storage_tco2": "carbon_storage_tco2",
           "annual_absorption_tco2": "annual_absorption_tco2", "timber_potential_m3": "timber_potential_m3"}
    return str(row.get(raw.get(key, key), "-"))


if __name__ == "__main__":
    import json
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv = os.path.join(base, "data", "real_forest_polygons.csv")
    rpt = train_and_save(csv)
    print(json.dumps(rpt, ensure_ascii=False, indent=2))
