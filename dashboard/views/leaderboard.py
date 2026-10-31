from __future__ import annotations

import altair as alt
import streamlit as st

from dashboard import analytics as an
from dashboard import i18n as he
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import page_header

STATUS_COLORS_HE = {he.v(k): c for k, c in theme.STATUS_COLORS.items()}


def _status_style(value: str) -> str:
    return f"color: {STATUS_COLORS_HE.get(value, theme.TEXT)}; font-weight: 600"


def render(ctx: DashboardContext) -> None:
    exp = ctx.exp
    page_header(ctx, "דירוג סוכנים",
                f"מדורג לפי {he.RANK_KEYS.get(exp.results['rank_by'])}. האלפא שנתית ומחושבת "
                "מול פרוקסי השוק. סטטוס מובהק דורש ⁦|t| ≥ 2⁩.")
    df = an.agent_leaderboard(exp)
    view = he.frame(df, ["Type", "Status"])
    c = he.col
    styled = (view.style
              .format({c("Opportunity Score"): "{:.1f}",
                       c("Alpha"): "{:+.1%}", c("Alpha t-stat"): "{:.2f}",
                       c("Win Rate"): "{:.1%}", c("Sharpe"): "{:.2f}",
                       c("Max Drawdown"): "{:.1%}", c("Total Return"): "{:+.1%}",
                       c("Beats Random"): "{:.0%}"}, na_rep="-")
              .map(_status_style, subset=[c("Status")])
              .map(lambda x: f"color: {theme.tone(x)}", subset=[c("Alpha"), c("Total Return")])
              .map(lambda x: f"color: {theme.MUTED}" if x == he.v("Control") else "",
                   subset=[c("Type")]))
    st.dataframe(styled, hide_index=True, width="stretch", height=38 + 35 * len(df))

    right, left = st.columns(2)
    with right:
        st.markdown("#### אלפא מול השוק, כולל מובהקות")
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("Agent:N", sort=list(df["Agent"]), title=None,
                    axis=alt.Axis(labelLimit=220, labelOverlap=False)),
            x=alt.X("Alpha:Q", axis=alt.Axis(format="%"), title="אלפא שנתית"),
            color=alt.condition(alt.datum.Alpha >= 0, alt.value(theme.POSITIVE),
                                alt.value(theme.NEGATIVE)),
            opacity=alt.condition(f"abs(datum['Alpha t-stat']) >= {an.ALPHA_SIGNIFICANT_T}",
                                  alt.value(1.0), alt.value(0.45)),
            tooltip=[alt.Tooltip("Agent", title="סוכן"),
                     alt.Tooltip("Alpha:Q", title="אלפא", format="+.2%"),
                     alt.Tooltip("Alpha t-stat:Q", title="t-stat", format=".2f")])
        st.altair_chart(theme.chart(bars, 320), width="stretch")
        st.caption("עמודה שקופה למחצה: ⁦|t| < 2⁩, כלומר לא ניתן להבדיל את התוצאה ממזל.")
    with left:
        st.markdown("#### סיכון מול תשואה")
        points = df.assign(Kind=df["Type"].map(he.v))
        dots = alt.Chart(points).mark_circle(size=140, opacity=.9).encode(
            x=alt.X("Max Drawdown:Q", axis=alt.Axis(format="%"), title="ירידה מקסימלית"),
            y=alt.Y("Sharpe:Q", title="שארפ (שנתי)"),
            color=alt.Color("Kind:N", title=None,
                            scale=alt.Scale(domain=he.values(["Strategy", "Control"]),
                                            range=[theme.ACCENT, theme.WARNING])),
            tooltip=[alt.Tooltip("Agent", title="סוכן"),
                     alt.Tooltip("Sharpe:Q", title="שארפ", format=".2f"),
                     alt.Tooltip("Max Drawdown:Q", title="ירידה מקסימלית", format=".1%")])
        labels = dots.mark_text(align="left", dx=9, fontSize=11, color=theme.MUTED).encode(
            text="Agent:N", color=alt.value(theme.MUTED))
        st.altair_chart(theme.chart(dots + labels, 320), width="stretch")

    with st.expander("איך לקרוא את הטבלה"):
        st.markdown(
            "- **ציון הזדמנות**: הציון הממוצע (0-100) של המניות שהסוכן בחר בפועל, ולא של כל "
            "המניות. ריק עבור הסוכן האקראי, כי הציון שלו הוא רעש.\n"
            "- **אלפא**: החותך ברגרסיה מול פרוקסי השוק, בחישוב שנתי. זו התשואה שהסוכן השיג "
            "מעבר לחשיפה שלו לשוק.\n"
            "- **סטטוס**: *עדיף על השוק* או *נחות מהשוק* רק כאשר ⁦|t| ≥ 2⁩. אחרת *לא מובהק*, "
            "כלומר לא ניתן להבדיל ממזל.\n"
            "- **עוקף תיקים אקראיים**: אחוז תיקי ה-Top-N האקראיים (מדד הבחירה האקראית) שהשיגו "
            "תשואה כוללת נמוכה יותר.\n"
            "- **Random Agent** הוא סוכן ביקורת (פלסבו) שעובד באותו מנגנון בדיוק כמו "
            "האסטרטגיות. אסטרטגיה שלא עוקפת אותו לא מראה יתרון.\n"
            "- **עסקאות**: כל בחירה של מניה מוחזקת חודש אחד, במשקל שווה, עדיין בלי עמלות "
            "ובלי החלקה (slippage).")
