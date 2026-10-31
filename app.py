"""Stock Opportunity Agent - research & backtesting lab (synthetic data, no trading).

Layers: data/ + agents/ + engine/ produce an Experiment; dashboard/analytics.py turns it into
frames; dashboard/views/ lay those out. This file only wires settings, caching and navigation.
"""
from __future__ import annotations

import time

import streamlit as st

from config import Settings
from dashboard import theme
from dashboard.context import PAGES, DashboardContext, build_context
from dashboard.views import (
    experiments, fmt_date, leaderboard, opportunities, overview, performance, stock_detail,
)
from data.sample_data import SignalConfig
from engine.evaluation import RANK_KEYS
from engine.experiment import ExperimentParams, run_experiment

st.set_page_config(page_title="Stock Opportunity Agent", page_icon=":material/insights:",
                   layout="wide")
theme.inject_css()

VIEWS = {
    "Overview": overview, "Agents Leaderboard": leaderboard,
    "Stock Opportunities": opportunities, "Stock Detail": stock_detail,
    "Performance & Backtesting": performance, "Experiments": experiments,
}
ICONS = {"Overview": ":material/dashboard:", "Agents Leaderboard": ":material/leaderboard:",
         "Stock Opportunities": ":material/manage_search:",
         "Stock Detail": ":material/query_stats:",
         "Performance & Backtesting": ":material/show_chart:",
         "Experiments": ":material/science:"}


@st.cache_resource(show_spinner="Running experiment...")
def load(params: ExperimentParams, experiment_dir: str) -> DashboardContext:
    started = time.perf_counter()
    exp = run_experiment(params)
    try:
        exp.save(experiment_dir)  # experiment history; the UI keeps working if this fails
    except OSError:
        pass
    return build_context(exp, experiment_dir, time.perf_counter() - started)


defaults = Settings.from_env()
with st.sidebar:
    st.markdown("### :material/insights: Stock Opportunity Agent")
    st.caption("Quant research & backtesting lab")
    page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed",
                    format_func=lambda p: f"{ICONS[p]} {p}")
    st.divider()
    with st.expander("Experiment settings", icon=":material/tune:"):
        seed = st.number_input("Seed", value=defaults.seed, step=1)
        n_stocks = st.number_input("Stocks", min_value=10, max_value=500,
                                   value=max(defaults.n_stocks, 10), step=10)
        n_periods = st.number_input("Months", min_value=6, max_value=60,
                                    value=max(defaults.n_periods, 6), step=1)
        top_n = st.slider("Picks per agent (top N)", 1, 30, min(max(defaults.top_n, 1), 30))
        trials = st.slider("Random-selection trials", 20, 1000,
                           min(max(defaults.random_trials, 20), 1000))
        rank_by = st.selectbox("Rank agents by", RANK_KEYS, index=0)
        mode = st.radio("Synthetic signal", ["Weak signal (default)", "Null (no signal at all)"],
                        help="Null = features carry no information about outcomes. "
                             "Any alpha found there is pure luck.")

params = ExperimentParams(
    seed=int(seed), n_stocks=int(n_stocks), n_periods=int(n_periods), top_n=int(top_n),
    random_trials=int(trials), rank_by=rank_by,
    signal=SignalConfig.null() if mode.startswith("Null") else SignalConfig(),
)
ctx = load(params, defaults.experiment_dir)

with st.sidebar:
    status = ctx.status
    st.markdown("**Data source**")
    st.html(theme.pill(status.label, theme.WARNING) + "&nbsp;"
            + theme.pill("NO LIVE FEED", theme.NEUTRAL))
    st.caption(f"{status.source}\n\n{status.universe} stocks, {fmt_date(status.first)} to "
               f"{fmt_date(status.as_of)}, {status.signal_mode.lower()}. "
               f"Pipeline ran in {ctx.runtime_seconds:.1f}s.")
    st.caption("No broker, no external APIs, no trading.")

VIEWS[page].render(ctx)
