from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import analytics as an
from dashboard import components as ui
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import page_header


def _curve_chart(long: pd.DataFrame, value: str, order: list, fmt: str, title: str,
                 area: bool = False) -> alt.Chart:
    dash = alt.StrokeDash("Kind:N", legend=None,
                          scale=alt.Scale(domain=list(theme.KIND_DASH),
                                          range=list(theme.KIND_DASH.values())))
    mark = alt.Chart(long).mark_line(strokeWidth=1.8, interpolate="step-after" if area else
                                     "linear")
    return mark.encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y(f"{value}:Q", title=title, axis=alt.Axis(format=fmt),
                scale=alt.Scale(zero=area)),
        color=alt.Color("Series:N", title=None, sort=order), strokeDash=dash,
        tooltip=["Series", "Kind", alt.Tooltip("Date:T"), alt.Tooltip(f"{value}:Q", format=fmt)])


def render(ctx: DashboardContext) -> None:
    exp = ctx.exp
    page_header(ctx, "Performance & Backtesting",
                f"Monthly rebalanced, equal-weight top {exp.parameters['top_n']} per agent, "
                f"{ctx.status.periods} months, no costs or slippage yet.")
    equity, dd = ui.equity_frame(exp), ui.drawdown_frame(exp)
    series = list(equity.columns)
    default = [m.agent for m in exp.metrics if not an.is_control(m.agent)][:4] + \
        ["Random Agent"] + list(an.BENCHMARK_NAMES)
    chosen = st.multiselect("Series", series, default=default, key="perf_series")
    if not chosen:
        st.info("Select at least one series.")
        return

    t_curves, t_compare, t_bench, t_metrics = st.tabs(
        ["Equity & Drawdown", "Agent Comparison", "vs Market & Random", "Metrics"])

    with t_curves:
        st.markdown("#### Equity curve")
        st.altair_chart(theme.chart(_curve_chart(an.long_curves(equity[chosen], "Equity"),
                                                 "Equity", chosen, ".2f",
                                                 "Equity (start = 1.0)"), 360), width="stretch")
        st.markdown("#### Drawdown")
        st.altair_chart(theme.chart(_curve_chart(an.long_curves(dd[chosen], "Drawdown"),
                                                 "Drawdown", chosen, ".0%", "Below peak"), 260),
                        width="stretch")
        st.caption("Solid = strategy agents, dashed = Random Agent (placebo), dotted = "
                   "benchmarks.")

    with t_compare:
        lb = an.agent_leaderboard(exp)
        long = lb.melt(id_vars=["Agent", "Type"],
                       value_vars=["Total Return", "Alpha", "Max Drawdown"],
                       var_name="Metric", value_name="Value")
        grouped = alt.Chart(long).mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X("Value:Q", axis=alt.Axis(format="%"), title=None),
            y=alt.Y("Agent:N", sort=list(lb["Agent"]), title=None),
            color=alt.Color("Metric:N", title=None),
            yOffset="Metric:N",
            tooltip=["Agent", "Metric", alt.Tooltip("Value:Q", format="+.1%")])
        st.altair_chart(theme.chart(grouped, 460), width="stretch")
        st.markdown("#### Average monthly return by market regime")
        st.dataframe(ui.regime_frame(exp).style.format("{:+.2%}"), width="stretch")

    with t_bench:
        bench = st.selectbox("Benchmark", list(exp.benchmarks.returns), key="perf_bench")
        frame = ui.agent_vs_benchmark_frame(exp, bench)
        pct = st.column_config.NumberColumn(format="percent")
        st.dataframe(frame, hide_index=True, width="stretch", column_config={
            "Total Return": pct, "Benchmark Return": pct, "Excess CAGR": pct, "Alpha": pct,
            "Beats Random Portfolios": pct,
            "Beta": st.column_config.NumberColumn(format="%.2f"),
            "Alpha t-stat": st.column_config.NumberColumn(format="%.2f")})

        st.markdown("#### Agents against the distribution of random portfolios")
        dist = an.random_distribution(exp)
        hist = alt.Chart(dist).mark_bar(color=theme.NEUTRAL, opacity=.55).encode(
            x=alt.X("Total Return:Q", bin=alt.Bin(maxbins=40), axis=alt.Axis(format="%"),
                    title="Total return"),
            y=alt.Y("count():Q", title="Random portfolios"))
        marks = pd.DataFrame([{"Agent": m.agent, "Total Return": m.total_return,
                               "Type": "Control" if an.is_control(m.agent) else "Strategy"}
                              for m in exp.metrics])
        rules = alt.Chart(marks).mark_rule(strokeWidth=2).encode(
            x="Total Return:Q",
            color=alt.Color("Type:N", title=None,
                            scale=alt.Scale(domain=["Strategy", "Control"],
                                            range=[theme.ACCENT, theme.WARNING])),
            tooltip=["Agent", alt.Tooltip("Total Return:Q", format="+.1%")])
        st.altair_chart(theme.chart(hist + rules, 300), width="stretch")
        st.caption(f"Grey: total return of {len(dist)} seeded random top-"
                   f"{exp.parameters['top_n']} portfolios. Lines: each agent. An agent "
                   "inside the grey mass is not separable from luck.")

    with t_metrics:
        st.markdown("#### Agents")
        ui.render_leaderboard(ui.leaderboard_frame(exp))
        st.markdown("#### Benchmarks")
        pct = st.column_config.NumberColumn(format="percent")
        st.dataframe(ui.benchmark_frame(exp), hide_index=True, width="stretch", column_config={
            "Total Return": pct, "CAGR": pct, "Max Drawdown": pct, "Alpha (vs market)": pct,
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Beta": st.column_config.NumberColumn(format="%.2f")})
