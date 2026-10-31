from __future__ import annotations

from html import escape

import altair as alt
import streamlit as st

from dashboard import analytics as an
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import fmt_date, page_header, timestamp_select


def render(ctx: DashboardContext) -> None:
    exp = ctx.exp
    page_header(ctx, "Stock Detail", "Point-in-time view: everything below uses data known "
                                     "on the as-of date only.")
    tickers = sorted(exp.dataset.tickers)
    if st.session_state.get("detail_ticker") not in tickers:
        st.session_state["detail_ticker"] = tickers[0]
    c1, c2, _ = st.columns([1.2, 1.2, 3])
    ticker = c1.selectbox("Ticker", tickers, key="detail_ticker")
    when = timestamp_select("As of", ctx.timestamps, "detail_ts", c2)

    snap = an.snapshot_of(exp, ticker, when)
    views = an.agent_views(exp, ticker, when, ctx.scores)
    strategies = views[views["Type"] == "Strategy"]
    picked = strategies[strategies["Picked"]]
    consensus = strategies["Opportunity Score"].mean()

    st.markdown(f"### {escape(snap.company_name)} &nbsp;`{snap.ticker}`")
    st.caption(f"{snap.sector} / {snap.industry} - as of {fmt_date(when)}")
    k = st.columns(4)
    k[0].html(theme.card("Consensus Score", f"{consensus:.1f}",
                         f"mean of {len(strategies)} strategies, "
                         f"std {strategies['Opportunity Score'].std(ddof=0):.1f}"))
    top = strategies.iloc[0]
    k[1].html(theme.card("Top Agent", top["Agent"], f"score {top['Opportunity Score']:.1f}"))
    k[2].html(theme.card("Picked By", f"{len(picked)} / {len(strategies)}",
                         f"strategies with it in top {exp.parameters['top_n']}",
                         theme.POSITIVE if len(picked) else theme.NEUTRAL))
    risk = an.factor_profile(snap).set_index("Factor").loc["Risk", "Score"]
    k[3].html(theme.card("Risk Score", f"{risk:.0f}", "0 = low, 100 = high",
                         theme.NEGATIVE if risk >= 60 else theme.WARNING if risk >= 35
                         else theme.POSITIVE))

    st.markdown("#### Financial metrics")
    st.html(theme.kv_grid(an.financial_metrics(snap)))

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("#### Factor profile")
        prof = an.factor_profile(snap)
        bars = alt.Chart(prof).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("Factor:N", sort=list(prof["Factor"]), title=None),
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Score (0-100)"),
            color=alt.condition(alt.datum.Factor == "Risk", alt.value(theme.NEGATIVE),
                                alt.value(theme.ACCENT)),
            tooltip=["Factor", alt.Tooltip("Score:Q", format=".1f")])
        st.altair_chart(theme.chart(bars, 300), width="stretch")
    with right:
        st.markdown("#### Agent analysis")
        st.dataframe(
            views[["Agent", "Type", "Opportunity Score", "Rank", "Picked"]],
            hide_index=True, width="stretch", height=38 + 35 * len(views),
            column_config={"Opportunity Score": st.column_config.ProgressColumn(
                format="%.1f", min_value=0, max_value=100),
                "Picked": st.column_config.CheckboxColumn()})

    reasons, risks = st.columns([1.4, 1])
    with reasons:
        st.markdown("#### Why it is recommended")
        source = picked if len(picked) else strategies.head(2)
        if not len(picked):
            st.caption("No strategy agent picks this stock on this date. "
                       "Showing the two most favourable views.")
        for row in source.to_dict("records"):
            st.html(f'<div class="soa-reason"><b>{escape(row["Agent"])}</b> '
                    f'({row["Opportunity Score"]:.1f}): {escape(row["Reasoning"])}</div>')
    with risks:
        st.markdown("#### Risks")
        flags = an.risk_flags(snap)
        if flags:
            st.html("".join(f'<div class="soa-flag">{escape(f)}</div>' for f in flags))
        else:
            st.caption("No rule-based risk flags triggered.")
        st.caption("Flags are simple thresholds on the snapshot (volatility, leverage, cash "
                   "flow, size, short interest, drawdown, growth).")

    st.markdown("#### Score history")
    hist = an.score_history(exp, ticker, when, ctx.scores)
    hist = hist[~hist["Agent"].isin(an.CONTROL_AGENTS)]
    lines = alt.Chart(hist).mark_line(point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Agent:N", title=None),
        tooltip=["Agent", alt.Tooltip("Date:T"), alt.Tooltip("Score:Q", format=".1f")])
    price = an.price_history(exp, ticker, when)
    price_line = alt.Chart(price).mark_area(
        line={"color": theme.ACCENT}, color=theme.ACCENT, opacity=.15).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Price:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="$,.0f")),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Price:Q", format="$,.2f")])
    h1, h2 = st.columns([1.6, 1])
    h1.altair_chart(theme.chart(lines, 300), width="stretch")
    h2.altair_chart(theme.chart(price_line, 300), width="stretch")
    st.caption("Scores and price up to the as-of date only. Future months are not shown, "
               "and nothing here uses realised returns.")
