from __future__ import annotations

import streamlit as st

from dashboard import analytics as an
from dashboard.context import DashboardContext
from dashboard.views import fmt_date, page_header, timestamp_select

SORTABLE = ("Opportunity Score", "Value Score", "Catalyst Score", "Momentum Score",
            "Risk Score", "Market Cap", "Price", "Picked By", "Ticker")
TABLE_KEY = "opp_table"


def _open_detail() -> None:
    """Row-selection callback: jump to Stock Detail for the selected ticker."""
    rows = st.session_state[TABLE_KEY].selection.rows
    tickers = st.session_state.get("opp_tickers", [])
    if rows and rows[0] < len(tickers):
        st.session_state["detail_ticker"] = tickers[rows[0]]
        st.session_state["detail_ts"] = st.session_state["opp_ts"]
        st.session_state["page"] = "Stock Detail"


def render(ctx: DashboardContext) -> None:
    page_header(ctx, "Stock Opportunities",
                "Each stock's factor scores and its best score from any strategy agent. "
                "Select a row to open the stock's detail page.")
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1.2])
    when = timestamp_select("As of", ctx.timestamps, "opp_ts", f4)
    full = an.opportunities_frame(ctx.exp, when, ctx.scores)
    query = f1.text_input("Search", placeholder="Ticker or company", key="opp_query")
    sectors = f2.multiselect("Sector", sorted(full["Sector"].unique()), key="opp_sectors")
    agent = f3.selectbox("Recommended by", ["Any agent"] + an.strategy_agents(ctx.exp),
                         key="opp_agent")

    g1, g2, g3, g4, g5 = st.columns([2, 2, 2, 1.2, 1.2])
    min_score = g1.slider("Min opportunity score", 0, 100, 0, key="opp_min")
    max_risk = g2.slider("Max risk score", 0, 100, 100, key="opp_risk")
    sort_by = g3.selectbox("Sort by", SORTABLE, key="opp_sort")
    descending = g4.toggle("Descending", value=True, key="opp_desc")
    picked_only = g5.toggle("Picked only", value=False, key="opp_picked",
                            help="Only stocks in at least one strategy agent's top-N")

    df = an.filter_opportunities(full, query, sectors,
                                 None if agent == "Any agent" else agent,
                                 min_score, max_risk, picked_only)
    df = df.sort_values([sort_by, "Ticker"], ascending=[not descending, True]) \
           .reset_index(drop=True)
    st.session_state["opp_tickers"] = list(df["Ticker"])
    view = df.assign(**{"Market Cap": df["Market Cap"] / 1e9})

    st.caption(f"{len(df)} of {len(full)} stocks, {fmt_date(when)}. "
               f"Picked By = number of strategy agents with the stock in their top "
               f"{ctx.exp.parameters['top_n']}. The Random Agent is excluded.")
    score = dict(min_value=0, max_value=100, format="%.1f")
    st.dataframe(
        view, hide_index=True, width="stretch", height=560, key=TABLE_KEY,
        on_select=_open_detail, selection_mode="single-row",
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Market Cap": st.column_config.NumberColumn(format="$%.1fB"),
            "Value Score": st.column_config.NumberColumn(format="%.1f"),
            "Catalyst Score": st.column_config.NumberColumn(format="%.1f"),
            "Momentum Score": st.column_config.NumberColumn(format="%.1f"),
            "Risk Score": st.column_config.NumberColumn(format="%.1f"),
            "Opportunity Score": st.column_config.ProgressColumn(**score),
            "Agents": st.column_config.TextColumn(width="medium"),
        })
    st.download_button("Export CSV", df.to_csv(index=False).encode(),
                       file_name=f"opportunities_{fmt_date(when)}.csv", mime="text/csv")
