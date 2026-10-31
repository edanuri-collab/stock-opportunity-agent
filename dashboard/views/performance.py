from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import analytics as an
from dashboard import components as ui
from dashboard import i18n as he
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import curve_labels, dash_scale, page_header

PCT = ("Total Return", "CAGR", "Win Rate", "Avg Win", "Avg Loss", "Expectancy", "Max Drawdown",
       "Alpha (vs market)", "Benchmark Return", "Excess CAGR", "Alpha",
       "Beats Random Portfolios")
RATIO = ("Profit Factor", "Sharpe", "Beta", "Alpha t-stat")


def _number_config(columns) -> dict:
    cfg = {}
    for name in columns:
        if name in PCT:
            cfg[he.col(name)] = st.column_config.NumberColumn(format="percent")
        elif name in RATIO:
            cfg[he.col(name)] = st.column_config.NumberColumn(format="%.2f")
    return cfg


def _curve_chart(long: pd.DataFrame, value: str, order: list, fmt: str, title: str,
                 value_title: str) -> alt.Chart:
    return alt.Chart(long).mark_line(strokeWidth=1.8).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y(f"{value}:Q", title=title, axis=alt.Axis(format=fmt),
                scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", title=None, sort=order),
        strokeDash=alt.StrokeDash("Kind:N", legend=None, scale=dash_scale()),
        tooltip=[alt.Tooltip("Series", title="סדרה"), alt.Tooltip("Kind", title="סוג"),
                 alt.Tooltip("Date:T", title="תאריך"),
                 alt.Tooltip(f"{value}:Q", title=value_title, format=fmt)])


def render(ctx: DashboardContext) -> None:
    exp = ctx.exp
    page_header(ctx, "ביצועים ובקטסט",
                f"איזון חודשי, Top {exp.parameters['top_n']} במשקל שווה לכל סוכן, "
                f"{ctx.status.periods} חודשים, עדיין בלי עמלות והחלקה.")
    equity, dd = ui.equity_frame(exp), ui.drawdown_frame(exp)
    series = list(equity.columns)
    default = [m.agent for m in exp.metrics if not an.is_control(m.agent)][:4] + \
        ["Random Agent"] + list(an.BENCHMARK_NAMES)
    chosen = st.multiselect("סדרות להצגה", series, default=default, key="perf_series",
                            format_func=he.v)
    if not chosen:
        st.info("בחר לפחות סדרה אחת.")
        return
    order = he.values(chosen)

    t_curves, t_compare, t_bench, t_metrics = st.tabs(
        ["עקומת הון וירידות", "השוואת סוכנים", "מול השוק והאקראי", "מדדי ביצוע"])

    with t_curves:
        st.markdown("#### עקומת הון")
        st.altair_chart(theme.chart(_curve_chart(
            curve_labels(an.long_curves(equity[chosen], "Equity")), "Equity", order, ".2f",
            "הון יחסי", "הון"), 360), width="stretch")
        st.markdown("#### ירידה מהשיא (Drawdown)")
        st.altair_chart(theme.chart(_curve_chart(
            curve_labels(an.long_curves(dd[chosen], "Drawdown")), "Drawdown", order, ".0%",
            "מתחת לשיא", "ירידה"), 260), width="stretch")
        st.caption("הון יחסי: ההון בתחילת התקופה הוא 1. קו רציף = סוכני אסטרטגיה, מקווקו = הסוכן האקראי (פלסבו), מנוקד = מדדי "
                   "ייחוס.")

    with t_compare:
        lb = an.agent_leaderboard(exp)
        long = lb.melt(id_vars=["Agent", "Type"],
                       value_vars=["Total Return", "Alpha", "Max Drawdown"],
                       var_name="Metric", value_name="Value")
        long["Metric"] = long["Metric"].map(he.col)
        grouped = alt.Chart(long).mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X("Value:Q", axis=alt.Axis(format="%"), title=None),
            y=alt.Y("Agent:N", sort=list(lb["Agent"]), title=None,
                    axis=alt.Axis(labelLimit=220)),
            color=alt.Color("Metric:N", title=None),
            yOffset="Metric:N",
            tooltip=[alt.Tooltip("Agent", title="סוכן"), alt.Tooltip("Metric", title="מדד"),
                     alt.Tooltip("Value:Q", title="ערך", format="+.1%")])
        st.altair_chart(theme.chart(grouped, 460), width="stretch")
        st.markdown("#### תשואה חודשית ממוצעת לפי מצב שוק")
        regimes = he.frame(ui.regime_frame(exp))
        st.dataframe(regimes.style.format("{:+.2%}"), width="stretch")

    with t_bench:
        bench = st.selectbox("מדד ייחוס", list(exp.benchmarks.returns), key="perf_bench",
                             format_func=he.v)
        frame = ui.agent_vs_benchmark_frame(exp, bench)
        st.dataframe(he.frame(frame, ["Verdict"]), hide_index=True, width="stretch",
                     column_config=_number_config(frame.columns))

        st.markdown("#### הסוכנים מול התפלגות של תיקים אקראיים")
        dist = an.random_distribution(exp)
        hist = alt.Chart(dist).mark_bar(color=theme.NEUTRAL, opacity=.55).encode(
            x=alt.X("Total Return:Q", bin=alt.Bin(maxbins=40), axis=alt.Axis(format="%"),
                    title="תשואה כוללת"),
            y=alt.Y("count():Q", title="תיקים אקראיים"))
        marks = pd.DataFrame([{"Agent": m.agent, "Total Return": m.total_return,
                               "Kind": he.v("Control" if an.is_control(m.agent)
                                            else "Strategy")}
                              for m in exp.metrics])
        rules = alt.Chart(marks).mark_rule(strokeWidth=2).encode(
            x="Total Return:Q",
            color=alt.Color("Kind:N", title=None,
                            scale=alt.Scale(domain=he.values(["Strategy", "Control"]),
                                            range=[theme.ACCENT, theme.WARNING])),
            tooltip=[alt.Tooltip("Agent", title="סוכן"),
                     alt.Tooltip("Total Return:Q", title="תשואה כוללת", format="+.1%")])
        st.altair_chart(theme.chart(hist + rules, 300), width="stretch")
        st.caption(f"באפור: התשואה הכוללת של {len(dist)} תיקי Top-"
                   f"{exp.parameters['top_n']} אקראיים. הקווים: כל אחד מהסוכנים. סוכן "
                   "שנמצא בתוך המסה האפורה לא ניתן להבדלה ממזל.")

    with t_metrics:
        st.markdown("#### סוכנים")
        lf = ui.leaderboard_frame(exp)
        st.dataframe(he.frame(lf), hide_index=True, width="stretch",
                     column_config=_number_config(lf.columns))
        st.markdown("#### מדדי ייחוס")
        bf = ui.benchmark_frame(exp)
        st.dataframe(he.frame(bf, ["Benchmark"]), hide_index=True, width="stretch",
                     column_config=_number_config(bf.columns))
