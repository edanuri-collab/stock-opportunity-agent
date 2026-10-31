from __future__ import annotations

import streamlit as st

from dashboard import analytics as an
from dashboard import i18n as he
from dashboard.context import DashboardContext
from dashboard.views import fmt_date, page_header, timestamp_select

SORTABLE = ("Opportunity Score", "Value Score", "Catalyst Score", "Momentum Score",
            "Risk Score", "Market Cap", "Price", "Picked By", "Ticker")
TABLE_KEY = "opp_table"
ANY_AGENT = "Any agent"


def _open_detail() -> None:
    """Row-selection callback: jump to Stock Detail for the selected ticker."""
    rows = st.session_state[TABLE_KEY].selection.rows
    tickers = st.session_state.get("opp_tickers", [])
    if rows and rows[0] < len(tickers):
        st.session_state["detail_ticker"] = tickers[rows[0]]
        st.session_state["detail_ts"] = st.session_state["opp_ts"]
        st.session_state["page"] = "Stock Detail"


def render(ctx: DashboardContext) -> None:
    page_header(ctx, "הזדמנויות במניות",
                "ציוני הגורמים של כל מניה והציון הגבוה ביותר שקיבלה מסוכן אסטרטגיה. "
                "בחר שורה כדי לפתוח את דף המניה.")

    f1, f2, f3, f4 = st.columns([2, 2, 2, 1.2])
    when = timestamp_select("נכון לתאריך", ctx.timestamps, "opp_ts", f4)
    full = an.opportunities_frame(ctx.exp, when, ctx.scores)
    query = f1.text_input("חיפוש", placeholder="טיקר או שם חברה", key="opp_query")
    sectors = f2.multiselect("סקטור", sorted(full["Sector"].unique()), key="opp_sectors",
                             format_func=he.v, placeholder="כל הסקטורים")
    agent = f3.selectbox("הומלצה ע\"י", [ANY_AGENT] + an.strategy_agents(ctx.exp),
                         key="opp_agent",
                         format_func=lambda a: "כל סוכן" if a == ANY_AGENT else a)

    g1, g2, g3, g4, g5 = st.columns([2, 2, 2, 1.2, 1.2])
    min_score = g1.slider("ציון הזדמנות מינימלי", 0, 100, 0, key="opp_min")
    max_risk = g2.slider("ציון סיכון מקסימלי", 0, 100, 100, key="opp_risk")
    sort_by = g3.selectbox("מיון לפי", SORTABLE, key="opp_sort", format_func=he.col)
    descending = g4.toggle("סדר יורד", value=True, key="opp_desc")
    picked_only = g5.toggle("נבחרו בלבד", value=False, key="opp_picked",
                            help="רק מניות שנמצאות ב-Top-N של לפחות סוכן אסטרטגיה אחד")

    df = an.filter_opportunities(full, query, sectors,
                                 None if agent == ANY_AGENT else agent,
                                 min_score, max_risk, picked_only)
    df = df.sort_values([sort_by, "Ticker"], ascending=[not descending, True]) \
           .reset_index(drop=True)
    st.session_state["opp_tickers"] = list(df["Ticker"])
    view = he.frame(df.assign(**{"Market Cap": df["Market Cap"] / 1e9}), ["Sector"])

    st.caption(f"{len(df)} מתוך {len(full)} מניות, {fmt_date(when)}. "
               f"מספר בוחרים = כמה סוכני אסטרטגיה מחזיקים את המניה ב-Top "
               f"{ctx.exp.parameters['top_n']} שלהם. הסוכן האקראי לא נספר.")
    c = he.col
    num = st.column_config.NumberColumn
    st.dataframe(
        view, hide_index=True, width="stretch", height=560, key=TABLE_KEY,
        on_select=_open_detail, selection_mode="single-row",
        column_config={
            c("Price"): num(format="$%.2f"), c("Market Cap"): num(format="$%.1fB"),
            c("Value Score"): num(format="%.1f"), c("Catalyst Score"): num(format="%.1f"),
            c("Momentum Score"): num(format="%.1f"), c("Risk Score"): num(format="%.1f"),
            c("Opportunity Score"): st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f"),
            c("Agents"): st.column_config.TextColumn(width="medium"),
        })
    st.download_button("ייצוא ל-CSV", he.frame(df, ["Sector"]).to_csv(index=False)
                       .encode("utf-8-sig"),
                       file_name=f"opportunities_{fmt_date(when)}.csv", mime="text/csv")
