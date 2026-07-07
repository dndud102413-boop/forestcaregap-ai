# -*- coding: utf-8 -*-
"""LLM 산주 맞춤 리포트 (Gemini) + 룰기반 폴백.

관리공백 진단 결과(점수·근거·유형)를 임야 소유자(산주) 눈높이의 자연어로 생성한다.
API 키가 없거나 호출이 실패해도 **결정론적 폴백 텍스트**로 항상 동작(앱 안 죽음).

  · 키: 환경변수 GEMINI_API_KEY 또는 GOOGLE_API_KEY
  · 모델: 환경변수 GEMINI_MODEL (기본 gemini-2.0-flash)
  · SDK: google-genai (`pip install google-genai`) — 팀 기존 forest_reco/llm.py 와 동일 방식

설계 원칙: '방치 확정' 금지, '관리공백 가능성 사전진단·우선순위'로만 표현, 수치 날조 금지.
"""
from __future__ import annotations
import os

_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def _get_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def available() -> bool:
    """Gemini 키 + SDK 가 모두 준비됐는지(=실제 LLM 생성 가능)."""
    if not _get_key():
        return False
    try:
        from google import genai  # noqa: F401
        return True
    except Exception:
        return False


def _build_prompt(ctx: dict) -> str:
    reasons = " / ".join(str(r) for r in ctx.get("reasons", [])[:3]) or "여러 입지·임상 요인"
    return f"""너는 한국 산림청 산림경영에 정통한 전문가다. 아래는 한 산림 필지의
'관리공백 가능성 사전진단' 결과다. 임야 소유자(산주)가 이해하기 쉽게 자연어로 설명하라.

[필지 정보]
- 위치(시군구): {ctx.get('region')}
- 수종/영급: {ctx.get('species')} {ctx.get('age_class')}영급, 면적 {ctx.get('area_ha')}ha
- 임도거리: {ctx.get('road_dist')}m, 경사: {ctx.get('slope')}°

[AI 진단]
- 관리공백 점수: {ctx.get('gap_score')}/100 (전체 강원 필지 중 상위 {ctx.get('pct_top')}%)
- 관리유형(AI 군집): {ctx.get('segment')}
- 주요 근거(Top3): {reasons}

[작성 지침]
- 독자: 임야 소유자(산주). 전문용어는 최소화하고 실용적으로 5~7문장.
- 순서: 1) 이 산림이 현재 어떤 상태로 보이는지 2) 왜 우선 확인이 필요한지
  3) 권장 조치(현장 확인 우선, 숲가꾸기 등 관리 검토) 순으로.
- ⚠️ 절대 '방치 확정'이라 단정하지 말 것. '관리공백 가능성', '현장 확인이 필요한 우선 후보'로 표현.
- 점수는 확정이 아니라 우선순위임을 분명히 하고, 데이터에 없는 수치는 지어내지 말 것.
- 마지막에 '본 진단은 공개데이터·AI 기반 사전진단이며 현장조사·행정 확인이 필요합니다.'를 덧붙여라.
"""


def _fallback_text(ctx: dict) -> str:
    """LLM 없이 구조화 결과로 만드는 한국어 자연어 설명(항상 동작)."""
    reasons = [str(r) for r in ctx.get("reasons", [])[:3]]
    rtxt = ", ".join(reasons) if reasons else "여러 입지·임상 요인"
    return (
        f"{ctx.get('region')}에 위치한 이 산림(수종 {ctx.get('species')}, {ctx.get('age_class')}영급, "
        f"면적 {ctx.get('area_ha')}ha)은 관리공백 가능성 점수 {ctx.get('gap_score')}점으로, "
        f"강원 전체 필지 중 상위 {ctx.get('pct_top')}%에 해당해 현장 확인이 우선 필요한 후보로 진단됩니다. "
        f"이렇게 진단된 주요 근거는 {rtxt}입니다. 관리유형은 '{ctx.get('segment')}'에 해당합니다. "
        f"다만 이는 방치를 확정하는 것이 아니라 우선순위가 상대적으로 높다는 의미이며, "
        f"임도 접근성·임분 상태를 현장에서 확인한 뒤 숲가꾸기 등 관리 시행을 검토하시길 권장합니다.\n"
        f"(본 진단은 공개데이터·AI 기반 사전진단이며 현장조사·행정 확인이 필요합니다.)"
    )


def generate(ctx: dict) -> dict:
    """자연어 리포트 생성. 반환 {text, source('gemini'|'fallback'), model, [error]}."""
    key = _get_key()
    if not key:
        return {"text": _fallback_text(ctx), "source": "fallback", "model": None}
    try:
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=_MODEL, contents=_build_prompt(ctx))
        text = (getattr(resp, "text", "") or "").strip()
        if not text:
            raise ValueError("빈 응답")
        return {"text": text, "source": "gemini", "model": _MODEL}
    except Exception as e:  # noqa: BLE001
        return {"text": _fallback_text(ctx), "source": "fallback", "model": None, "error": str(e)}
