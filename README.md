# 🌲 ForestCareGap AI

**관리공백 가능성 사전진단 기반 산림관리 의사결정 지원 서비스 (프로토타입)**

---

## 1. 프로젝트 개요

ForestCareGap AI는 **관리공백 가능성이 높은 사유림을 사전 선별**하고, 그 원인을 설명하며,
관리 처방·지원사업 검토 후보·산주 설득 리포트·지자체 예산 우선순위·관리 후 성과 추적까지
하나의 흐름으로 제공하는 AI 기반 의사결정 지원 서비스입니다.

```
관리공백 탐지 → 원인 설명 → 관리 처방 → 지원사업 연결
→ 산주 설득 → 지자체 예산 우선순위 → 관리 후 성과 추적
```

> ⚠️ 본 서비스는 특정 산림의 방치 여부를 **법적·행정적으로 확정하지 않습니다.**
> 모든 결과는 **관리공백 가능성 사전진단**이며, 현장조사·지자체·산림조합 확인이 필요합니다.

---

## 2. 설치 방법

```bash
pip install -r requirements.txt
```

> GeoPandas/Shapely 설치가 어려운 환경에서는 두 패키지를 제외해도 됩니다.
> 앱은 공간 라이브러리 없이도 **표·점수·순위·리포트 중심**으로 정상 작동합니다.

---

## 3. 실행 방법

```bash
streamlit run app.py
```

- 최초 실행 시 `data/sample_forest_polygons.csv`가 없으면 **자동 생성**됩니다.
- 별도의 공공데이터가 없어도 샘플 데이터로 전체 기능이 동작합니다.

---

## 4. 데이터 구조

```text
forestcaregap_ai/
├─ app.py                     # Streamlit 앱 (5개 탭)
├─ requirements.txt
├─ README.md
├─ data/
│  ├─ sample_forest_polygons.csv   # 산림 폴리곤(자동 생성, 22개 속성)
│  ├─ support_programs.csv         # 지원사업 검토 후보
│  └─ management_tracking.csv      # 관리 전/후 성과 추적
├─ modules/
│  ├─ scoring.py            # 관리공백 가능성 점수화
│  ├─ explanation.py        # 점수 근거 자연어 설명
│  ├─ prescription.py       # 관리 처방 추천
│  ├─ support_matching.py   # 지원사업 검토 후보 매칭
│  ├─ reporting.py          # 산주용 리포트 생성
│  └─ tracking.py           # 관리 후 성과 추적
└─ assets/screenshots/
```

**`sample_forest_polygons.csv` 주요 컬럼** — polygon_id, region, owner_type, species,
age_class, diameter_class, crown_density, area_ha, elevation_m, slope_deg, aspect,
forest_road_distance_m, road_distance_m, management_history_years, has_recent_management,
landslide_risk, fire_risk, protected_area, owner_absentee, carbon_storage_tco2,
annual_absorption_tco2, timber_potential_m3

---

## 5. 점수화 로직

| 변수 | 조건 | 점수 |
|------|------|----:|
| 최근 관리 이력 없음 | has_recent_management == False | +25 |
| 관리 이력 오래됨 | management_history_years ≥ 10 | +15 |
| 고영급 | age_class ≥ 5 | +15 |
| 수관밀도 높음 | crown_density == "밀" | +15 |
| 임도 거리 멂 | forest_road_distance_m ≥ 500 | +15 |
| 경사 큼 | slope_deg ≥ 25 | +10 |
| 산사태 위험 높음 | landslide_risk == "높음" | +10 |
| 산불 위험 높음 | fire_risk == "높음" | +10 |
| 부재산주 | owner_absentee == True | +10 |
| 보호구역 | protected_area == True | +5 |

- 최종 점수는 **0~100점**으로 제한
- **0~39** 관리 우선순위 낮음 · **40~59** 모니터링 필요 · **60~79** 관리 필요 · **80~100** 우선관리 필요

**예산 우선순위 점수** = 관리공백×0.45 + 재해위험×0.25 + 관리효과×0.20 + 접근성×0.10

---

## 6. 주요 기능

1. **개별 산림 진단** — 점수·등급·원인 설명·처방·지원사업 후보·산주 리포트
2. **지자체 우선관리 대시보드** — 전체 점수표, 우선관리 필터, 지역별 평균, 예산 우선순위 Top 10, 등급별 차트
3. **관리 후 성과 추적** — 관리 전/후 점수 비교 및 개선 등급
4. **설명가능 AI** — 점수 기여 요인을 항상 함께 제시
5. **확장 가능 구조** — 실제 임상도/DEM/임도망/산림사업 이력 데이터 연결 가능

---

## 7. 한계

- 본 프로토타입은 **샘플 데이터** 기반이며 실제 산림 상태를 단정하지 않습니다.
- 점수화는 현재 **규칙 기반(가중합)**으로, 데이터 분포·임계값은 시연용 가정입니다.
- 모든 결과는 **사전진단**이며 신청·시행은 현장조사·지자체·산림조합 확인이 필요합니다.

---

## 8. 향후 고도화 계획

- 규칙 기반 가중치를 **실제 관리이력 라벨 기반 ML 모델**(RandomForest/GBM)로 대체
- **공간 분리(Spatial Block) 교차검증**으로 일반화 성능 확보
- 산림 법령·지원사업 공고 **RAG** 연결 → 정확한 '지자체 확인 필요' 안내
- 산주 맞춤 설득 리포트 **LLM 자연어 생성**
- 임상도·DEM·임도망 등 **실제 산림 공공데이터 연동** 및 폴리곤 지도 시각화

---

## 표현 규칙 (중요)

화면·리포트에서 다음 표현은 **사용하지 않습니다**: 방치 확정 / 불법 방치 / 행정 판단 완료 /
지원사업 신청 가능 확정 / 현장조사 불필요.
대신 **관리공백 가능성 / 우선관리 필요도 / 사전진단 / 현장조사 필요 / 지원사업 검토 후보 /
지자체 확인 필요 / 산림조합 상담 추천** 으로 표현합니다.
