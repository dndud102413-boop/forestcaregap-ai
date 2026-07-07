# -*- coding: utf-8 -*-
"""한글 PDF 생성 (reportlab + 한글 TTF 등록).

txt 대비 보관·공유에 유리. 폰트는 후보 목록으로 탐색:
  프로젝트 동봉(assets/*.ttf) → Windows Malgun Gothic → Linux 나눔/데자뷰.
폰트를 못 찾으면 available()=False 로 떨어져 앱이 txt 만 제공(앱은 죽지 않음).
"""
from __future__ import annotations
import os
import io

_FONT_NAME: str | None = None
_TRIED = False

_FONT_CANDS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "NanumGothic.ttf"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "Pretendard-Regular.ttf"),
    r"C:\Windows\Fonts\malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/notosanscjk/NotoSansCJK-Regular.ttc",
]


def _register_font() -> str | None:
    global _FONT_NAME, _TRIED
    if _TRIED:
        return _FONT_NAME
    _TRIED = True
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        return None
    for c in _FONT_CANDS:
        if os.path.exists(c):
            try:
                pdfmetrics.registerFont(TTFont("KFont", c))
                _FONT_NAME = "KFont"
                return _FONT_NAME
            except Exception:
                continue
    return None


def available() -> bool:
    """reportlab + 한글 폰트가 모두 준비됐는지."""
    try:
        import reportlab  # noqa: F401
    except Exception:
        return False
    return _register_font() is not None


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt(v) -> str:
    """정수형 float(예: 5196.0)은 정수로 표기."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def text_report_pdf(title: str, body: str, subtitle: str = "", disclaimer: str = "") -> bytes:
    """제목 + 본문 텍스트(줄바꿈·들여쓰기 보존)를 깔끔한 A4 PDF bytes 로 반환."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.styles import ParagraphStyle

    font = _register_font() or "Helvetica"
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm, bottomMargin=16 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm, title=title)
    st_title = ParagraphStyle("t", fontName=font, fontSize=16, leading=20,
                              textColor=colors.HexColor("#15171A"), spaceAfter=2)
    st_sub = ParagraphStyle("s", fontName=font, fontSize=9.5, leading=13,
                            textColor=colors.HexColor("#6E7176"), spaceAfter=6)
    st_body = ParagraphStyle("b", fontName=font, fontSize=9.5, leading=15,
                             textColor=colors.HexColor("#2A2D31"))
    st_disc = ParagraphStyle("d", fontName=font, fontSize=8, leading=11,
                             textColor=colors.HexColor("#6E7176"))
    story = [Paragraph(_esc(title), st_title)]
    if subtitle:
        story.append(Paragraph(_esc(subtitle), st_sub))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#202327"), spaceAfter=8))
    for line in body.split("\n"):
        lead = len(line) - len(line.lstrip(" "))
        safe = ("&nbsp;" * lead) + _esc(line.lstrip(" ")) if line.strip() else "&nbsp;"
        story.append(Paragraph(safe, st_body))
    if disclaimer:
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E6E7E9"), spaceAfter=4))
        story.append(Paragraph(_esc(disclaimer), st_disc))
    doc.build(story)
    return buf.getvalue()


def table_pdf(title: str, df, subtitle: str = "", disclaimer: str = "", max_rows: int = 60) -> bytes:
    """DataFrame 을 가로 A4 표 PDF bytes 로 반환(헤더 차콜, 줄무늬)."""
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import ParagraphStyle

    font = _register_font() or "Helvetica"
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), topMargin=14 * mm, bottomMargin=12 * mm,
                            leftMargin=12 * mm, rightMargin=12 * mm, title=title)
    st_h = ParagraphStyle("t", fontName=font, fontSize=14, leading=18, textColor=colors.HexColor("#15171A"))
    st_sub = ParagraphStyle("s", fontName=font, fontSize=9, leading=12, textColor=colors.HexColor("#6E7176"), spaceAfter=6)
    st_cell = ParagraphStyle("c", fontName=font, fontSize=8, leading=10, textColor=colors.HexColor("#2A2D31"))
    st_head = ParagraphStyle("h", fontName=font, fontSize=8, leading=10, textColor=colors.white)
    df2 = df.head(max_rows)
    data = [[Paragraph(_esc(c), st_head) for c in df2.columns]]
    for _, r in df2.iterrows():
        data.append([Paragraph(_esc(_fmt(v)), st_cell) for v in r.tolist()])
    story = [Paragraph(_esc(title), st_h)]
    if subtitle:
        story.append(Paragraph(_esc(subtitle), st_sub))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#202327"), spaceAfter=6))
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#202327")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E6E7E9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F5F6")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    if disclaimer:
        story.append(Spacer(1, 6))
        story.append(Paragraph(_esc(disclaimer), st_sub))
    doc.build(story)
    return buf.getvalue()
