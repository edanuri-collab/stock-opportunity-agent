from __future__ import annotations

from html import escape

import altair as alt
import streamlit as st

from dashboard import analytics as an
from dashboard import i18n as he
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import fmt_date, page_header, timestamp_select


def render(ctx: DashboardContext) -> None:
    exp = ctx.exp
    page_header(ctx, "פרטי מניה", "תצוגה נכון לנקודת זמן: כל מה שמוצג כאן מבוסס רק על מידע "
                                  "שהיה ידוע בתאריך הנבחר.")
    tickers = sorted(exp.dataset.tickers)
    if st.session_state.get("detail_ticker") not in tickers:
        st.session_state["detail_ticker"] = tickers[0]
    c1, c2, _ = st.columns([1.2, 1.2, 3])
    ticker = c1.selectbox("טיקר", tickers, key="detail_ticker")
    when = timestamp_select("נכון לתאריך", ctx.timestamps, "detail_ts", c2)

    snap = an.snapshot_of(exp, ticker, when)
    views = an.agent_views(exp, ticker, when, ctx.scores)
    strategies = views[views["Type"] == "Strategy"]
    picked = strategies[strategies["Picked"]]
    consensus = strategies["Opportunity Score"].mean()

    st.markdown(f"### {escape(snap.company_name)} &nbsp;`{snap.ticker}`")
    st.caption(f"{he.v(snap.sector)} / {he.v(snap.industry)} · נכון ל-{fmt_date(when)}")
    k = st.columns(4)
    k[0].html(theme.card("ציון קונצנזוס", f"{consensus:.1f}",
                         f"ממוצע {len(strategies)} אסטרטגיות, סטיית תקן "
                         f"{strategies['Opportunity Score'].std(ddof=0):.1f}"))
    top = strategies.iloc[0]
    k[1].html(theme.card("הסוכן המוביל", top["Agent"], f"ציון {top['Opportunity Score']:.1f}"))
    k[2].html(theme.card("נבחרה ע\"י", f"{len(picked)} / {len(strategies)}",
                         f"אסטרטגיות שמחזיקות אותה ב-Top {exp.parameters['top_n']}",
                         theme.POSITIVE if len(picked) else theme.NEUTRAL))
    prof = an.factor_profile(snap)
    risk = prof.set_index("Factor").loc["Risk", "Score"]
    k[3].html(theme.card("ציון סיכון", f"{risk:.0f}", "0 = נמוך, 100 = גבוה",
                         theme.NEGATIVE if risk >= 60 else theme.WARNING if risk >= 35
                         else theme.POSITIVE))

    st.markdown("#### מדדים פיננסיים")
    st.html(theme.kv_grid(an.financial_metrics(snap)))

    right, left = st.columns([1, 1.4])
    with right:
        st.markdown("#### פרופיל גורמים")
        prof = prof.assign(Label=prof["Factor"].str.lower().map(he.FACTORS))
        bars = alt.Chart(prof).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("Label:N", sort=list(prof["Label"]), title=None),
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100]), title="ציון"),
            color=alt.condition(alt.datum.Factor == "Risk", alt.value(theme.NEGATIVE),
                                alt.value(theme.ACCENT)),
            tooltip=[alt.Tooltip("Label", title="גורם"),
                     alt.Tooltip("Score:Q", title="ציון", format=".1f")])
        st.altair_chart(theme.chart(bars, 300), width="stretch")
    with left:
        st.markdown("#### ניתוח הסוכנים")
        table = he.frame(views[["Agent", "Type", "Opportunity Score", "Rank", "Picked"]],
                         ["Type"])
        st.dataframe(
            table, hide_index=True, width="stretch", height=38 + 35 * len(views),
            column_config={he.col("Opportunity Score"): st.column_config.ProgressColumn(
                format="%.1f", min_value=0, max_value=100),
                he.col("Picked"): st.column_config.CheckboxColumn()})

    reasons, risks = st.columns([1.4, 1])
    with reasons:
        st.markdown("#### הסיבות להמלצה")
        source = picked if len(picked) else strategies.head(2)
        if not len(picked):
            st.caption("אף סוכן אסטרטגיה לא בוחר במניה הזו בתאריך הזה. "
                       "מוצגות שתי ההערכות החיוביות ביותר.")
        for row in source.to_dict("records"):
            st.html(f'<div class="soa-reason">{escape(row["Reasoning"])}</div>')
    with risks:
        st.markdown("#### סיכונים")
        flags = an.risk_flags(snap)
        if flags:
            st.html("".join(f'<div class="soa-flag">{escape(f)}</div>' for f in flags))
        else:
            st.caption("לא הופעל אף דגל סיכון.")
        st.caption("הדגלים הם ספים פשוטים על נתוני המניה: תנודתיות, מינוף, תזרים, גודל, שורט, "
                   "ירידה מהשיא וצמיחה.")

    st.markdown("#### היסטוריית ציונים")
    hist = an.score_history(exp, ticker, when, ctx.scores)
    hist = hist[~hist["Agent"].isin(an.CONTROL_AGENTS)]
    lines = alt.Chart(hist).mark_line(point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Score:Q", title="ציון", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Agent:N", title=None),
        tooltip=[alt.Tooltip("Agent", title="סוכן"), alt.Tooltip("Date:T", title="תאריך"),
                 alt.Tooltip("Score:Q", title="ציון", format=".1f")])
    price = an.price_history(exp, ticker, when)
    price_line = alt.Chart(price).mark_area(
        line={"color": theme.ACCENT}, color=theme.ACCENT, opacity=.15).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Price:Q", title="מחיר", scale=alt.Scale(zero=False),
                axis=alt.Axis(format="$,.0f")),
        tooltip=[alt.Tooltip("Date:T", title="תאריך"),
                 alt.Tooltip("Price:Q", title="מחיר", format="$,.2f")])
    h1, h2 = st.columns([1.6, 1])
    h1.altair_chart(theme.chart(lines, 300), width="stretch")
    h2.altair_chart(theme.chart(price_line, 300), width="stretch")
    st.caption("ציונים ומחיר עד התאריך הנבחר בלבד. חודשים עתידיים לא מוצגים, ואף נתון כאן "
               "לא משתמש בתשואות שהתממשו.")
