# -*- coding: utf-8 -*-
"""RAG(검색증강생성) 산주 맞춤 안내문 — '지어내지 않고 근거를 찾아서' 설명한다.

흐름: 필지 진단(상황) → ① 지식베이스에서 관련 지원사업·관리기준 검색(TF-IDF, 오프라인)
      → ② 검색된 근거를 LLM 프롬프트에 주입해 산주 안내문 생성(Gemini, 키 없으면 검색기반 폴백)

설계:
  · 검색은 외부 임베딩 API 없이 char n-gram TF-IDF 코사인으로(한국어·오프라인·결정론적).
  · LLM 은 '검색된 근거 안에서만' 말하도록 강하게 지시(환각 억제) + 사전진단 디스클레이머.
  · 키/SDK 없거나 실패해도 검색 근거로 폴백 텍스트를 만들어 항상 동작(앱 안 죽음).

⚠️ 지식베이스(KB)는 공개 지원사업 '예시'다. 조문·예산·요건의 구체 수치는 연도별 지자체
   공고·법령으로 교체/검증해야 한다(여기선 일반적 설명 + 출처 표기, 허위 수치 금지).
"""
from __future__ import annotations
import os
import numpy as np

from .llm_report import _get_key, _MODEL, _fallback_text

# ── 지식베이스(KB): 산림 지원사업·관리기준 스니펫 (예시·확장 대상) ──────────────
# 각 항목: id, title, category, when(적용 상황 키워드), text(근거), source
KB = [
    {"id": "KB-숲가꾸기", "title": "숲가꾸기(간벌·가지치기) 지원", "category": "숲가꾸기",
     "when": "고영급 고밀도 과밀 임분 간벌 생장 침엽수 밀",
     "text": "고영급·과밀(수관밀도 '밀') 임분은 솎아베기(간벌)·가지치기 등 숲가꾸기로 생장과 임분 건강을 개선한다. 비용 일부를 지자체·산림조합이 지원 검토.",
     "source": "산림청 숲가꾸기 사업(연도별 지자체 공고)"},
    {"id": "KB-조림갱신", "title": "조림·갱신 및 묘목 지원", "category": "조림·갱신",
     "when": "수확 갱신 조림 완경사 임목 잠재량 국산재 벌기령",
     "text": "벌기령 도달·임목 잠재량이 큰 완경사지는 수확 후 조림·갱신으로 국산재 공급과 산림 순환경영을 도모. 조림비·묘목 지원 검토 대상.",
     "source": "산림청 조림 사업(연도별 지자체 공고)"},
    {"id": "KB-사방", "title": "사방사업(산사태 예방)", "category": "산사태 예방",
     "when": "산사태 위험 급경사 재해 사방댐 토사",
     "text": "산사태 위험등급이 높고 경사가 급한 산지는 사방댐·산지사방 등 재해예방 사업 검토 대상. 재해위험 평가 결과로 우선순위가 결정된다.",
     "source": "산림청·지자체 사방사업"},
    {"id": "KB-산불", "title": "산불예방 숲가꾸기(내화수림)", "category": "산불 예방",
     "when": "산불 위험 침엽수 연료물질 내화수림 가연성",
     "text": "산불 위험이 높은 침엽 임분은 연료물질(낙엽·고사목) 관리와 내화수림대 조성으로 산불 확산 위험을 낮춘다.",
     "source": "지자체 산불방지 사업"},
    {"id": "KB-탄소", "title": "산림탄소 상쇄·관리", "category": "탄소관리",
     "when": "탄소 흡수 잠재 기후 공익 상쇄",
     "text": "탄소흡수 잠재력이 큰 임분은 산림탄소 흡수량 유지·증진 프로그램(산림탄소상쇄 등) 등록·인증 요건 충족 시 참여 검토 가능.",
     "source": "한국임업진흥원·산림탄소센터"},
    {"id": "KB-경영상담", "title": "산림경영계획 수립·상담", "category": "산림조합 상담",
     "when": "사유림 관리 필요 경영계획 부재산주 자문 상담",
     "text": "관리 필요도가 높은 사유림은 산림경영계획 수립과 관리 자문을 지역 산림조합에서 상담받을 수 있다. 경영계획 수립 시 각종 지원사업 연계가 쉬워진다.",
     "source": "지역 산림조합"},
    {"id": "KB-현장조사", "title": "현장조사·산림조사 지원", "category": "현장조사",
     "when": "접근성 낮음 임도 멂 보호구역 정밀조사 행정",
     "text": "임도 접근성이 낮거나 보호구역에 해당하면 관리 행위 전 현장 정밀조사·행정 확인이 필요하다. 조사 일정·비용은 사전 협의 대상.",
     "source": "관할 시군 산림부서"},
    {"id": "KB-보호구역", "title": "보호구역 행정 검토", "category": "현장조사",
     "when": "보호구역 자연환경보전림 규제 행정 검토 제약",
     "text": "보호구역(자연환경보전림 등)은 관리 행위에 행정 검토·제약이 따른다. 시행 전 관할 부서와 가능 행위를 확인해야 한다.",
     "source": "산림보호구역 관련 규정(관할 부서 확인)"},
]

_VEC = None
_MAT = None


def _ensure_index():
    """KB TF-IDF 인덱스(char n-gram, 한국어·오프라인) 1회 구축."""
    global _VEC, _MAT
    if _MAT is not None:
        return
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = [f"{d['title']} {d['category']} {d['when']} {d['text']}" for d in KB]
    _VEC = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    _MAT = _VEC.fit_transform(corpus)


def _query_from_ctx(ctx: dict) -> str:
    """필지 진단 → 검색 질의문(상황 키워드 합성)."""
    parts = [str(ctx.get("segment", "")), str(ctx.get("species", "")),
             f"{ctx.get('age_class','')}영급"]
    if str(ctx.get("crown_density", "")).strip() == "밀":
        parts.append("고밀도 과밀 밀 간벌 숲가꾸기")
    if str(ctx.get("landslide", "")).strip() == "높음":
        parts.append("산사태 위험 급경사 사방")
    if str(ctx.get("fire", "")).strip() == "높음":
        parts.append("산불 위험 연료물질")
    try:
        if float(ctx.get("road_dist") or 0) >= 500:
            parts.append("임도 멂 접근성 낮음 현장조사")
    except Exception:
        pass
    if ctx.get("protected"):
        parts.append("보호구역 행정 검토")
    parts.extend(str(r) for r in ctx.get("reasons", [])[:3])
    return " ".join(p for p in parts if p.strip())


def retrieve(ctx: dict, k: int = 3) -> list[dict]:
    """상황에 맞는 KB 근거 상위 k개(점수 포함)."""
    _ensure_index()
    from sklearn.metrics.pairwise import cosine_similarity
    q = _VEC.transform([_query_from_ctx(ctx)])
    sims = cosine_similarity(q, _MAT)[0]
    order = np.argsort(-sims)[:k]
    out = []
    for i in order:
        if sims[i] <= 0:
            continue
        d = dict(KB[i]); d["score"] = round(float(sims[i]), 3)
        out.append(d)
    return out


def _build_prompt(ctx: dict, docs: list[dict]) -> str:
    ev = "\n".join(f"  [{i+1}] {d['title']} (출처: {d['source']})\n      {d['text']}"
                   for i, d in enumerate(docs)) or "  (관련 근거 없음)"
    reasons = " / ".join(str(r) for r in ctx.get("reasons", [])[:3]) or "여러 입지·임상 요인"
    return f"""너는 한국 산림청 산림경영 전문가다. 아래 '검색된 근거' 안에서만 사실을 사용해
임야 소유자(산주)에게 보내는 맞춤 안내문을 작성하라. 근거에 없는 제도·수치·조문은 절대 지어내지 마라.

[필지 진단]
- 위치(시군구): {ctx.get('region')} · 수종/영급: {ctx.get('species')} {ctx.get('age_class')}영급 · 면적 {ctx.get('area_ha')}ha
- 관리공백 점수: {ctx.get('gap_score')}/100 (상위 {ctx.get('pct_top')}%) · 관리유형(AI): {ctx.get('segment')}
- 주요 근거(Top3): {reasons}

[검색된 근거 — 이 안에서만 인용]
{ev}

[작성 지침]
- 5~7문장, 산주 눈높이. 1) 현재 상태 2) 왜 우선 확인이 필요한지 3) 위 근거에 기반한 권장 조치·지원사업.
- 권장 지원사업을 언급할 때 반드시 위 근거의 제목과 출처를 함께 표기하라(예: "숲가꾸기 사업(출처: …)").
- '방치 확정' 단정 금지 → '관리공백 가능성', '현장 확인이 필요한 우선 후보'로 표현.
- 점수는 확정이 아닌 우선순위임을 밝히고, 마지막에
  '본 안내는 공개데이터·AI 기반 사전진단이며 신청·시행은 현장조사·지자체·산림조합 확인이 필요합니다.'를 덧붙여라.
"""


def _fallback_with_evidence(ctx: dict, docs: list[dict]) -> str:
    """LLM 없이 검색 근거로 구성하는 안내문(항상 동작)."""
    base = _fallback_text(ctx)
    if not docs:
        return base
    lines = ["", "[검색된 지원사업·관리 근거]"]
    for d in docs:
        lines.append(f" · {d['title']} — {d['text']} (출처: {d['source']})")
    lines.append("(본 안내는 공개데이터·AI 기반 사전진단이며 신청·시행은 현장조사·지자체·산림조합 확인이 필요합니다.)")
    return base + "\n" + "\n".join(lines)


def generate(ctx: dict, k: int = 3) -> dict:
    """RAG 안내문 생성. 반환 {text, source, model, retrieved:[...], [error]}."""
    docs = retrieve(ctx, k=k)
    key = _get_key()
    if not key:
        return {"text": _fallback_with_evidence(ctx, docs), "source": "fallback",
                "model": None, "retrieved": docs}
    try:
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=_MODEL, contents=_build_prompt(ctx, docs))
        text = (getattr(resp, "text", "") or "").strip()
        if not text:
            raise ValueError("빈 응답")
        return {"text": text, "source": "gemini", "model": _MODEL, "retrieved": docs}
    except Exception as e:  # noqa: BLE001
        return {"text": _fallback_with_evidence(ctx, docs), "source": "fallback",
                "model": None, "retrieved": docs, "error": str(e)}
