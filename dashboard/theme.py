"""Visual layer: colours, CSS and small HTML building blocks. No data logic here."""
from __future__ import annotations

from html import escape
from typing import Optional

import altair as alt

BG = "#0B0F17"
PANEL = "#121826"
BORDER = "#1F2937"
TEXT = "#E5E7EB"
MUTED = "#8B95A7"
ACCENT = "#3B82F6"
POSITIVE = "#22C55E"
NEGATIVE = "#EF4444"
WARNING = "#F59E0B"
NEUTRAL = "#94A3B8"

STATUS_COLORS = {"Outperforming": POSITIVE, "Underperforming": NEGATIVE,
                 "Inconclusive": NEUTRAL}
FONT = "'Heebo', 'Segoe UI', 'Arial Hebrew', Arial, sans-serif"
KIND_DASH = {"Agent": [1, 0], "Control": [6, 3], "Benchmark": [2, 2]}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;600;700&display=swap');
/* ---- right-to-left layout (Hebrew UI) ---- */
html, body, [data-testid="stApp"], [data-testid="stSidebar"], .stMarkdown, button, input,
textarea, select, label {{ font-family: {FONT}; }}
[data-testid="stMain"] .block-container, [data-testid="stSidebar"] {{ direction: rtl; }}
.stMarkdown, [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"],
[data-testid="stExpander"] summary {{ text-align: right; }}
/* tables, charts, sliders, code and JSON keep left-to-right rendering */
[data-testid="stDataFrame"], [data-testid="stVegaLiteChart"], [data-testid="stSlider"] > div:last-child,
[data-testid="stJson"], [data-testid="stCode"], [data-testid="stNumberInputContainer"],
.soa-meta, .soa-ltr {{ direction: ltr; }}
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarHeader"] {{ direction: ltr; }}
.block-container {{ padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1500px; }}
[data-testid="stSidebar"] {{ background: #0E1420; border-left: 1px solid {BORDER}; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.soa-banner {{
  display: flex; gap: .75rem; align-items: center; padding: .55rem .9rem; margin-bottom: 1rem;
  border: 1px solid rgba(245,158,11,.35); background: rgba(245,158,11,.08);
  border-radius: 8px; color: #FCD34D; font-size: .85rem;
}}
.soa-banner b {{ color: #FBBF24; white-space: nowrap; }}
.soa-header {{ display: flex; justify-content: space-between; align-items: flex-end;
  margin-bottom: .9rem; gap: 1rem; flex-wrap: wrap; }}
.soa-header h1 {{ font-size: 1.65rem; margin: 0; padding: 0; }}
.soa-header .sub {{ color: {MUTED}; font-size: .85rem; margin-top: .2rem; max-width: 760px; }}
.soa-meta {{ color: {MUTED}; font-size: .75rem; font-family: ui-monospace, monospace;
  text-align: left; }}
.soa-card {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
  padding: .9rem 1rem; height: 100%;
}}
.soa-card .label {{ color: {MUTED}; font-size: .8rem; font-weight: 500; }}
.soa-card .value {{ font-size: 1.6rem; font-weight: 650; margin-top: .25rem;
  font-variant-numeric: tabular-nums; white-space: nowrap; }}
.soa-card .value {{ direction: ltr; unicode-bidi: plaintext; text-align: right; }}
.soa-card .value.long {{ font-size: 1.25rem; padding-top: .3rem; }}
.soa-card .foot {{ color: {MUTED}; font-size: .78rem; margin-top: .15rem; }}
.soa-pill {{ display: inline-block; padding: .1rem .55rem; border-radius: 999px;
  font-size: .72rem; font-weight: 600; letter-spacing: .04em; }}
.soa-flag {{ padding: .45rem .7rem; border-right: 3px solid {WARNING};
  background: rgba(245,158,11,.06); margin-bottom: .4rem; border-radius: 4px;
  font-size: .9rem; }}
.soa-reason {{ padding: .55rem .8rem; border: 1px solid {BORDER}; border-radius: 8px;
  margin-bottom: .45rem; background: {PANEL}; font-size: .88rem; }}
.soa-reason b {{ color: {TEXT}; }}
.soa-kv {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .5rem; }}
.soa-kv div {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
  padding: .5rem .7rem; }}
.soa-kv span {{ display: block; color: {MUTED}; font-size: .76rem; }}
.soa-kv strong {{ direction: ltr; unicode-bidi: isolate; }}
.soa-kv strong {{ font-size: 1rem; font-variant-numeric: tabular-nums; }}
@media (max-width: 900px) {{ .soa-kv {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
</style>
"""


def inject_css() -> None:
    import streamlit as st

    st.html(CSS)


def tone(value: float) -> str:
    return POSITIVE if value > 0 else NEGATIVE if value < 0 else TEXT


def card(label: str, value: str, foot: str = "", color: Optional[str] = None) -> str:
    style = f' style="color:{color}"' if color else ""
    size = " long" if len(value) > 8 else ""
    foot_html = f'<div class="foot">{escape(foot)}</div>' if foot else ""
    return (f'<div class="soa-card"><div class="label">{escape(label)}</div>'
            f'<div class="value{size}"{style}>{escape(value)}</div>{foot_html}</div>')


def pill(text: str, color: str) -> str:
    return (f'<span class="soa-pill" style="color:{color};border:1px solid {color}55;'
            f'background:{color}1A">{escape(text)}</span>')


def synthetic_banner(dataset_version: str, signal_mode: str) -> str:
    return (f'<div class="soa-banner"><b>נתונים סינתטיים</b><span>כל המחירים, הנתונים '
            f'הפיננסיים והתשואות בעמוד זה נוצרו בסימולטור דטרמיניסטי (דאטה '
            f'<span class="soa-ltr">{escape(dataset_version)}</span>, {escape(signal_mode)}). '
            f'אלה אינם נתוני שוק אמיתיים. למחקר בלבד: אין מסחר ואין כאן ייעוץ השקעות.'
            f'</span></div>')


def header(title: str, subtitle: str, meta: str) -> str:
    return (f'<div class="soa-header"><div><h1>{escape(title)}</h1>'
            f'<div class="sub">{escape(subtitle)}</div></div>'
            f'<div class="soa-meta">{meta}</div></div>')


def kv_grid(items: dict) -> str:
    cells = "".join(f"<div><span>{escape(k)}</span><strong>{escape(v)}</strong></div>"
                    for k, v in items.items())
    return f'<div class="soa-kv">{cells}</div>'


def chart(c: alt.Chart, height: int = 320) -> alt.Chart:
    """Shared chart styling (transparent background, muted grid)."""
    return (c.properties(height=height, background="transparent")
             .configure_axis(gridColor=BORDER, domainColor=BORDER, labelColor=MUTED,
                             titleColor=MUTED, tickColor=BORDER, labelFont=FONT,
                             titleFont=FONT, titleFontSize=12)
             .configure_legend(labelColor=TEXT, titleColor=MUTED, orient="bottom",
                               columns=4, symbolType="stroke", labelFont=FONT,
                               labelLimit=220)
             .configure_view(stroke=None))
