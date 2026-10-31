"""Dashboard building blocks. Frame builders are pure pandas (testable without Streamlit)."""
from __future__ import annotations

import statistics
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from agents.base_agent import AgentResult
from engine.benchmarks import MARKET
from engine.evaluation import regime_breakdown
from engine.experiment import Experiment

ALPHA_SIGNIFICANT_T = 2.0


def verdict(t_stat: float) -> str:
    if t_stat >= ALPHA_SIGNIFICANT_T:
        return "Alpha likely real"
    if t_stat <= -ALPHA_SIGNIFICANT_T:
        return "Significantly worse than benchmark"
    return "Not distinguishable from luck"


def leaderboard_frame(exp: Experiment) -> pd.DataFrame:
    """One row per agent, ranked by the experiment's ``rank_by`` metric."""
    rows = []
    for rank, m in enumerate(exp.metrics, start=1):
        a = m.alpha[MARKET]
        rows.append({
            "Rank": rank, "Agent": m.agent,
            "Total Return": m.total_return, "CAGR": m.cagr,
            "Win Rate": m.win_rate, "Avg Win": m.average_win, "Avg Loss": m.average_loss,
            "Expectancy": m.expectancy, "Profit Factor": m.profit_factor,
            "Sharpe": m.sharpe, "Max Drawdown": m.max_drawdown, "Trades": m.num_trades,
            "Alpha (vs market)": a.alpha, "Beta": a.beta, "Alpha t-stat": a.t_stat,
        })
    return pd.DataFrame(rows)


def benchmark_frame(exp: Experiment) -> pd.DataFrame:
    rows = [{"Benchmark": name, "Total Return": v["total_return"], "CAGR": v["cagr"],
             "Sharpe": v["sharpe"], "Max Drawdown": v["max_drawdown"],
             "Alpha (vs market)": v["alpha_vs_market"], "Beta": v["beta_vs_market"]}
            for name, v in exp.results["benchmarks"].items()]
    return pd.DataFrame(rows)


def agent_vs_benchmark_frame(exp: Experiment, benchmark: str) -> pd.DataFrame:
    """Does each agent add return beyond what the chosen benchmark explains?"""
    if benchmark not in exp.benchmarks.returns:
        raise KeyError(benchmark)
    bench_total = exp.results["benchmarks"][benchmark]["total_return"]
    rows = []
    for m in exp.metrics:
        a = m.alpha[benchmark]
        rows.append({
            "Agent": m.agent, "Total Return": m.total_return,
            "Benchmark Return": bench_total, "Excess CAGR": a.excess_cagr,
            "Alpha": a.alpha, "Beta": a.beta, "Alpha t-stat": a.t_stat,
            "Beats Random Portfolios": m.percentile_vs_random,
            "Verdict": verdict(a.t_stat),
        })
    return pd.DataFrame(rows)


def ranking_frame(
    scores: Sequence[AgentResult],
    timestamp: str,
    agent: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    rows = [
        {
            "Ticker": r["ticker"],
            "Agent": r["agent"],
            "Opportunity Score": r["opportunity_score"],
            "Value": r["value_score"],
            "Catalyst": r["catalyst_score"],
            "Momentum": r["momentum_score"],
            "Risk": r["risk_score"],
        }
        for r in scores
        if r["timestamp"] == timestamp and (agent is None or r["agent"] == agent)
    ]
    df = pd.DataFrame(rows, columns=[
        "Ticker", "Agent", "Opportunity Score", "Value", "Catalyst", "Momentum", "Risk"])
    df = df.sort_values(["Opportunity Score", "Ticker", "Agent"], ascending=[False, True, True])
    df = df.reset_index(drop=True)
    return df.head(limit) if limit else df


def stock_detail(scores: Sequence[AgentResult], ticker: str, timestamp: str) -> Dict[str, Any]:
    """All agent views of one stock at one timestamp, with favourite/least/consensus."""
    rows = [r for r in scores if r["ticker"] == ticker and r["timestamp"] == timestamp]
    if not rows:
        raise ValueError(f"No scores for {ticker} at {timestamp}")
    rows = sorted(rows, key=lambda r: (-r["opportunity_score"], r["agent"]))
    values = [r["opportunity_score"] for r in rows]
    return {
        "rows": rows,
        "favorite": rows[0],
        "least_favorite": rows[-1],
        "consensus_score": sum(values) / len(values),
        "dispersion": statistics.pstdev(values),
    }


def equity_frame(exp: Experiment) -> pd.DataFrame:
    """Equity (start = 1.0) of every agent and benchmark, indexed by period end."""
    bt = exp.backtest
    index = pd.Index([bt.timestamps[0]] + list(bt.period_ends), name="Date")
    return pd.DataFrame(exp.results["equity_curves"], index=index)


def drawdown_frame(exp: Experiment) -> pd.DataFrame:
    bt = exp.backtest
    index = pd.Index([bt.timestamps[0]] + list(bt.period_ends), name="Date")
    return pd.DataFrame(exp.results["drawdown_curves"], index=index)


def regime_frame(exp: Experiment) -> pd.DataFrame:
    """Average period return, agents x regimes."""
    return pd.DataFrame(regime_breakdown(exp.backtest)).T.rename_axis("Agent")


def experiment_summary_frame(exp: Experiment) -> pd.DataFrame:
    rows = [("experiment_id", exp.experiment_id), ("dataset_version", exp.dataset_version),
            ("code_version", exp.code_version), ("timestamp", exp.timestamp)]
    return pd.DataFrame(rows, columns=["Field", "Value"])


# ---- rendering (Streamlit imported lazily so tests don't need it) ----------------------------

def render_leaderboard(df: pd.DataFrame) -> None:
    import streamlit as st

    pct = st.column_config.NumberColumn(format="percent")
    st.dataframe(
        df, hide_index=True, width="stretch",
        column_config={
            "Total Return": pct, "CAGR": pct, "Win Rate": pct, "Avg Win": pct, "Avg Loss": pct,
            "Expectancy": pct, "Max Drawdown": pct, "Alpha (vs market)": pct,
            "Profit Factor": st.column_config.NumberColumn(format="%.2f"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Beta": st.column_config.NumberColumn(format="%.2f"),
            "Alpha t-stat": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def render_stock_detail(detail: Dict[str, Any]) -> None:
    import streamlit as st

    fav, least = detail["favorite"], detail["least_favorite"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Consensus Score", f"{detail['consensus_score']:.1f}",
              help=f"Std dev across agents: {detail['dispersion']:.1f}")
    c2.metric("Most likes", fav["agent"], f"{fav['opportunity_score']:.1f}")
    c3.metric("Least likes", least["agent"], f"{least['opportunity_score']:.1f}")
    table = pd.DataFrame(detail["rows"]).rename(columns={
        "agent": "Agent", "opportunity_score": "Opportunity", "value_score": "Value",
        "catalyst_score": "Catalyst", "momentum_score": "Momentum", "risk_score": "Risk",
    })
    st.dataframe(table[["Agent", "Opportunity", "Value", "Catalyst", "Momentum", "Risk"]],
                 hide_index=True, width="stretch")
    st.subheader("Why each agent scored it this way")
    for row in detail["rows"]:
        st.markdown(f"**{row['agent']}** ({row['opportunity_score']:.1f}) - {row['reasoning']}")
