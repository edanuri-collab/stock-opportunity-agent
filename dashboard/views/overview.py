from __future__ import annotations

import altair as alt
import streamlit as st

from dashboard import analytics as an
from dashboard import components as ui
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import fmt_date, page_header
from engine.benchmarks import MARKET


def render(ctx: DashboardContext) -> None:
    exp, status = ctx.exp, ctx.status
    s = an.overview_stats(exp)
    page_header(ctx, "Overview", "Quant research lab: same data, same timestamp, many agents, "
                                 "benchmarked against market and luck.")

    c = st.columns(6)
    c[0].html(theme.card("Stocks Scanned", f"{s['stocks_scanned']:,}",
                         f"{s['snapshots_analyzed']:,} snapshots / {status.periods} months"))
    c[1].html(theme.card("Opportunities Found", f"{s['opportunities']}",
                         f"{s['high_conviction']} picked by 3+ agents, {fmt_date(s['latest_timestamp'])}"))
    c[2].html(theme.card("Active Agents", f"{s['active_agents']}",
                         f"{s['strategy_agents']} strategies + 1 random control"))
    c[3].html(theme.card("Best Alpha (vs market)", f"{s['best_alpha']:+.1%}",
                         f"{s['best_agent']}, t = {s['best_t']:.2f}", theme.tone(s["best_alpha"])))
    c[4].html(theme.card("Last Updated", fmt_date(status.as_of),
                         f"run {s['run_timestamp'][:16].replace('T', ' ')} UTC"))
    c[5].html(theme.card("Data Source", status.label,
                         f"no live feed, dataset {status.dataset_version}", theme.WARNING))

    st.markdown("#### System performance")
    p = st.columns(5)
    n = s["strategy_agents"]
    p[0].html(theme.card("Positive alpha", f"{s['beat_market']} / {n}", "strategies vs market"))
    p[1].html(theme.card("Significant alpha", f"{s['significant']} / {n}", "t-stat >= 2",
                         theme.POSITIVE if s["significant"] else theme.NEUTRAL))
    p[2].html(theme.card("Beat Random Agent", f"{s['beat_random_agent']} / {n}",
                         "total return vs placebo"))
    p[3].html(theme.card("Avg Sharpe", f"{s['avg_sharpe']:.2f}", "strategy agents, annualised",
                         theme.tone(s["avg_sharpe"])))
    p[4].html(theme.card("Market Return", f"{s['market_return']:+.1%}",
                         f"cap-weighted synthetic proxy, {status.periods} months",
                         theme.tone(s["market_return"])))
    if s["significant"] == 0:
        st.caption("No strategy clears |t| >= 2 in this run, so none is distinguishable from "
                   "luck. That is the honest reading for a short synthetic sample.")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Equity curve")
        eq = ui.equity_frame(exp)
        leaders = [m.agent for m in exp.metrics if not an.is_control(m.agent)][:3]
        keep = leaders + ["Random Agent", MARKET]
        long = an.long_curves(eq[keep], "Equity")
        line = alt.Chart(long).mark_line(strokeWidth=2).encode(
            x=alt.X("Date:T", title=None), y=alt.Y("Equity:Q", title="Equity (start = 1.0)",
                                                    scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", title=None, sort=keep),
            strokeDash=alt.StrokeDash("Kind:N", legend=None,
                                      scale=alt.Scale(domain=list(theme.KIND_DASH),
                                                      range=list(theme.KIND_DASH.values()))),
            tooltip=["Series", alt.Tooltip("Date:T"), alt.Tooltip("Equity:Q", format=".3f")])
        st.altair_chart(theme.chart(line, 300), width="stretch")
        st.caption("Top 3 strategies by rank, the Random Agent (dashed) and the market proxy "
                   "(dotted).")
    with right:
        st.markdown(f"#### Top opportunities, {fmt_date(ctx.latest)}")
        opp = an.opportunities_frame(exp, ctx.latest, ctx.scores).head(10)
        st.dataframe(
            opp[["Ticker", "Company", "Opportunity Score", "Top Agent", "Picked By"]],
            hide_index=True, width="stretch", height=388,
            column_config={"Opportunity Score": st.column_config.ProgressColumn(
                format="%.1f", min_value=0, max_value=100)})
