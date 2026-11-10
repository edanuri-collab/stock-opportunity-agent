"""Stock Opportunity Agent - research & backtesting lab (synthetic data, no trading).

Layers: data/ + agents/ + engine/ produce an Experiment; dashboard/analytics.py turns it into
frames; dashboard/views/ lay those out (Hebrew UI, dashboard/i18n.py). This file only wires
settings, caching and navigation.
"""
from __future__ import annotations

import time

import streamlit as st

from agents.registry import fingerprint, load_registry
from config import Settings
from dashboard import i18n as he
from dashboard import theme
from dashboard.context import PAGES, DashboardContext, build_context
from dashboard.views import (
    catalog, experiments, fmt_date, journal, leaderboard, opportunities, overview,
    performance, robustness, stock_detail,
)
from data.sample_data import SignalConfig
from engine.evaluation import RANK_KEYS
from engine.experiment import ExperimentParams, run_experiment

st.set_page_config(page_title="Stock Opportunity Agent | מחקר מניות",
                   page_icon=":material/insights:", layout="wide")
theme.inject_css()

VIEWS = {
    "Overview": overview, "Agents Leaderboard": leaderboard, "Agent Catalog": catalog,
    "Stock Opportunities": opportunities, "Stock Detail": stock_detail,
    "Performance & Backtesting": performance, "Fair Ranking": robustness,
    "Tracking Journal": journal,
    "Experiments": experiments,
}
ICONS = {"Overview": ":material/dashboard:", "Agents Leaderboard": ":material/leaderboard:",
         "Agent Catalog": ":material/smart_toy:",
         "Stock Opportunities": ":material/manage_search:",
         "Stock Detail": ":material/query_stats:",
         "Performance & Backtesting": ":material/show_chart:",
         "Fair Ranking": ":material/balance:",
         "Tracking Journal": ":material/menu_book:",
         "Experiments": ":material/science:"}


@st.cache_resource(show_spinner="מריץ ניסוי...")
def load(params: ExperimentParams, experiment_dir: str, agents_version: str) -> DashboardContext:
    """``agents_version`` (spec files fingerprint) is part of the cache key: adding or editing
    an agent file reruns the experiment."""
    started = time.perf_counter()
    registry = load_registry()
    exp = run_experiment(params, agents=registry.agents)
    try:
        exp.save(experiment_dir)  # experiment history; the UI keeps working if this fails
    except OSError:
        pass
    return build_context(exp, experiment_dir, registry, time.perf_counter() - started)


defaults = Settings.from_env()
with st.sidebar:
    st.markdown("### :material/insights: Stock Opportunity Agent")
    st.caption("מעבדת מחקר כמותי ובקטסט למניות")
    page = st.radio("ניווט", PAGES, key="page", label_visibility="collapsed",
                    format_func=lambda p: f"{ICONS[p]} {he.PAGES[p]}")
    st.divider()
    with st.expander("הגדרות ניסוי", icon=":material/tune:"):
        seed = st.number_input("Seed", value=defaults.seed, step=1)
        n_stocks = st.number_input("מספר מניות", min_value=10, max_value=500,
                                   value=max(defaults.n_stocks, 10), step=10)
        n_periods = st.number_input("חודשים", min_value=6, max_value=60,
                                    value=max(defaults.n_periods, 6), step=1)
        top_n = st.slider("בחירות לכל סוכן (Top N)", 1, 30, min(max(defaults.top_n, 1), 30))
        trials = st.slider("מספר תיקים אקראיים להשוואה", 20, 1000,
                           min(max(defaults.random_trials, 20), 1000))
        cost_bps = st.slider("עמלה + החלקה (נקודות בסיס לכל צד)", 0, 50,
                             min(max(defaults.cost_bps, 0), 50),
                             help="10 = 0.1% על כל קנייה ועל כל מכירה. מנוכה לפי מחזור התיק.")
        rank_by = st.selectbox("דירוג הסוכנים לפי", RANK_KEYS, index=0,
                               format_func=lambda k: he.RANK_KEYS.get(k, k))
        mode = st.radio("סיגנל סינתטי", ["weak", "null"],
                        format_func=lambda m: "סיגנל חלש (ברירת מחדל)" if m == "weak"
                        else "ללא סיגנל (ביקורת)",
                        help="ללא סיגנל = לנתונים אין שום קשר לתשואות. כל אלפא שתימצא "
                             "שם היא מזל בלבד.")

params = ExperimentParams(
    seed=int(seed), n_stocks=int(n_stocks), n_periods=int(n_periods), top_n=int(top_n),
    random_trials=int(trials), rank_by=rank_by,
    signal=SignalConfig.null() if mode == "null" else SignalConfig(),
    cost_bps=float(cost_bps),
)
agents_version = fingerprint()
ctx = load(params, defaults.experiment_dir, agents_version)
ctx.params, ctx.agents_version = params, agents_version
ctx.journal_path = defaults.journal_path

with st.sidebar:
    status = ctx.status
    st.markdown("**מקור נתונים**")
    st.html(theme.pill("סינתטי", theme.WARNING) + "&nbsp;"
            + theme.pill("אין נתונים חיים", theme.NEUTRAL))
    st.caption(f"{status.source}\n\n{status.universe} מניות, {fmt_date(status.first)} עד "
               f"{fmt_date(status.as_of)}, "
               f"{he.SIGNAL_MODES.get(status.signal_mode, status.signal_mode)}. "
               f"הניסוי רץ ב-{ctx.runtime_seconds:.1f} שניות.")
    st.caption("אין ברוקר, אין API חיצוניים ואין מסחר.")

VIEWS[page].render(ctx)
