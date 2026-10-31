from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import analytics as an
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import page_header


def _status_style(value: str) -> str:
    return f"color: {theme.STATUS_COLORS.get(value, theme.TEXT)}; font-weight: 600"


def render(ctx: DashboardContext) -> None:
    exp = ctx.exp
    page_header(ctx, "Agents Leaderboard",
                f"Ranked by {exp.results['rank_by']}. Alpha is annualised vs the market proxy; "
                "status needs |t| >= 2.")
    df = an.agent_leaderboard(exp)
    styled = (df.style
              .format({"Opportunity Score": lambda v: "-" if pd.isna(v) else f"{v:.1f}",
                       "Alpha": "{:+.1%}", "Alpha t-stat": "{:.2f}",
                       "Win Rate": "{:.1%}", "Sharpe": "{:.2f}", "Max Drawdown": "{:.1%}",
                       "Total Return": "{:+.1%}", "Beats Random": "{:.0%}"})
              .map(_status_style, subset=["Status"])
              .map(lambda v: f"color: {theme.tone(v)}", subset=["Alpha", "Total Return"])
              .map(lambda v: f"color: {theme.MUTED}" if v == "Control" else "", subset=["Type"]))
    st.dataframe(styled, hide_index=True, width="stretch", height=38 + 35 * len(df))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Alpha vs market, with significance")
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("Agent:N", sort=list(df["Agent"]), title=None,
                    axis=alt.Axis(labelLimit=220, labelOverlap=False)),
            x=alt.X("Alpha:Q", axis=alt.Axis(format="%"), title="Annualised alpha"),
            color=alt.condition(alt.datum.Alpha >= 0, alt.value(theme.POSITIVE),
                                alt.value(theme.NEGATIVE)),
            opacity=alt.condition(f"abs(datum['Alpha t-stat']) >= {an.ALPHA_SIGNIFICANT_T}",
                                  alt.value(1.0), alt.value(0.45)),
            tooltip=["Agent", alt.Tooltip("Alpha:Q", format="+.2%"),
                     alt.Tooltip("Alpha t-stat:Q", format=".2f"), "Status"])
        st.altair_chart(theme.chart(bars, 320), width="stretch")
        st.caption("Faded bars: |t| < 2, not distinguishable from luck.")
    with right:
        st.markdown("#### Risk vs reward")
        dots = alt.Chart(df).mark_circle(size=140, opacity=.9).encode(
            x=alt.X("Max Drawdown:Q", axis=alt.Axis(format="%"), title="Max drawdown"),
            y=alt.Y("Sharpe:Q", title="Sharpe (annualised)"),
            color=alt.Color("Type:N", title=None,
                            scale=alt.Scale(domain=["Strategy", "Control"],
                                            range=[theme.ACCENT, theme.WARNING])),
            tooltip=["Agent", alt.Tooltip("Sharpe:Q", format=".2f"),
                     alt.Tooltip("Max Drawdown:Q", format=".1%")])
        labels = dots.mark_text(align="left", dx=9, fontSize=11, color=theme.MUTED).encode(
            text="Agent:N", color=alt.value(theme.MUTED))
        st.altair_chart(theme.chart(dots + labels, 320), width="stretch")

    with st.expander("How to read this table"):
        st.markdown(
            "- **Opportunity Score**: average score (0-100) of the stocks the agent actually "
            "picked, not of the whole universe. Blank for the Random Agent (its score is "
            "hash noise).\n"
            "- **Alpha**: regression intercept vs the market proxy, annualised. Return the "
            "agent earned beyond its market exposure.\n"
            "- **Status**: *Outperforming* / *Underperforming* only when |t| >= 2, otherwise "
            "*Inconclusive* (not distinguishable from luck).\n"
            "- **Beats Random**: share of random top-N portfolios (Random Selection benchmark) "
            "with a lower total return.\n"
            "- **Random Agent** is a placebo control with the same mechanics as the strategies. "
            "A strategy that cannot beat it shows no edge.\n"
            "- **Trades**: one pick held one month. Equal weight, no costs or slippage yet.")
