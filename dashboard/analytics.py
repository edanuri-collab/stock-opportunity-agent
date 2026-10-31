"""View-model layer for the dashboard: pure functions from an Experiment to frames/dicts.

Nothing here imports Streamlit, so every screen's numbers are unit-testable. Views only
format and lay out what these functions return.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from data.models import StockSnapshot
from engine.benchmarks import BUY_HOLD, MARKET, RANDOM
from engine.experiment import Experiment
from engine.scoring import compute_components, fcf_yield

CONTROL_AGENTS = frozenset({"Random Agent"})  # placebo: never counted as a recommendation
BENCHMARK_NAMES = (MARKET, BUY_HOLD, RANDOM)
ALPHA_SIGNIFICANT_T = 2.0
FACTOR_LABELS = ("value", "quality", "momentum", "catalyst", "breakout", "squeeze",
                 "contrarian", "risk")


def is_control(agent: str) -> bool:
    return agent in CONTROL_AGENTS


def strategy_agents(exp: Experiment) -> List[str]:
    return [a for a in exp.backtest.period_returns if not is_control(a)]


def performance_status(t_stat: float) -> str:
    """Short status label for an agent's alpha vs the market."""
    if t_stat >= ALPHA_SIGNIFICANT_T:
        return "Outperforming"
    if t_stat <= -ALPHA_SIGNIFICANT_T:
        return "Underperforming"
    return "Inconclusive"


# ---- data source ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataSourceStatus:
    source: str
    is_synthetic: bool
    live_feed: bool
    dataset_version: str
    signal_mode: str
    as_of: datetime
    first: datetime
    periods: int
    universe: int

    @property
    def label(self) -> str:
        return "SYNTHETIC" if self.is_synthetic else "LIVE"


def data_source_status(exp: Experiment) -> DataSourceStatus:
    ds = exp.dataset
    signal = exp.parameters.get("signal", {})
    null = bool(signal) and all(
        signal.get(k, 0) == 0 for k in ("value", "quality", "catalyst", "momentum", "breakout",
                                         "squeeze", "contrarian", "hidden"))
    return DataSourceStatus(
        source="Synthetic generator (data/sample_data.py)",
        is_synthetic=True,
        live_feed=False,
        dataset_version=exp.dataset_version,
        signal_mode="Null (no signal)" if null else "Weak signal",
        as_of=ds.timestamps[-1],
        first=ds.timestamps[0],
        periods=len(ds.timestamps),
        universe=len(ds.tickers),
    )


# ---- overview ------------------------------------------------------------------------------

def picks_at(exp: Experiment, timestamp: datetime) -> Dict[str, List[str]]:
    """ticker -> strategy agents that hold it in their top-N at ``timestamp``."""
    out: Dict[str, List[str]] = {}
    for t in exp.backtest.trades:
        if t.timestamp == timestamp and not is_control(t.agent):
            out.setdefault(t.ticker, []).append(t.agent)
    return out


def overview_stats(exp: Experiment) -> Dict[str, Any]:
    ds, latest = exp.dataset, exp.dataset.timestamps[-1]
    picks = picks_at(exp, latest)
    strategies = [m for m in exp.metrics if not is_control(m.agent)]
    best = max(strategies, key=lambda m: m.alpha[MARKET].alpha)
    control = next((m for m in exp.metrics if is_control(m.agent)), None)
    return {
        "stocks_scanned": len(ds.tickers),
        "snapshots_analyzed": len(ds.tickers) * len(ds.timestamps),
        "opportunities": len(picks),
        "high_conviction": sum(1 for agents in picks.values() if len(agents) >= 3),
        "active_agents": len(exp.metrics),
        "strategy_agents": len(strategies),
        "best_agent": best.agent,
        "best_alpha": best.alpha[MARKET].alpha,
        "best_t": best.alpha[MARKET].t_stat,
        "beat_market": sum(1 for m in strategies if m.alpha[MARKET].alpha > 0),
        "significant": sum(1 for m in strategies
                           if m.alpha[MARKET].t_stat >= ALPHA_SIGNIFICANT_T),
        "beat_random_agent": (sum(1 for m in strategies if m.total_return > control.total_return)
                              if control else None),
        "market_return": exp.results["benchmarks"][MARKET]["total_return"],
        "avg_sharpe": sum(m.sharpe for m in strategies) / len(strategies),
        "latest_timestamp": latest,
        "run_timestamp": exp.timestamp,
    }


# ---- leaderboard ---------------------------------------------------------------------------

def agent_leaderboard(exp: Experiment) -> pd.DataFrame:
    rows = []
    for rank, m in enumerate(exp.metrics, start=1):
        a = m.alpha[MARKET]
        control = is_control(m.agent)
        rows.append({
            "Rank": rank, "Agent": m.agent, "Type": "Control" if control else "Strategy",
            "Status": performance_status(a.t_stat),
            # a placebo's "score" is hash noise, so it is not shown as an opportunity score
            "Opportunity Score": None if control else m.average_score,
            "Alpha": a.alpha, "Alpha t-stat": a.t_stat,
            "Win Rate": m.win_rate, "Sharpe": m.sharpe, "Max Drawdown": m.max_drawdown,
            "Trades": m.num_trades, "Total Return": m.total_return,
            "Beats Random": m.percentile_vs_random,
        })
    return pd.DataFrame(rows)


# ---- opportunities -------------------------------------------------------------------------

OPPORTUNITY_COLUMNS = [
    "Ticker", "Company", "Sector", "Price", "Market Cap", "Value Score", "Catalyst Score",
    "Momentum Score", "Risk Score", "Opportunity Score", "Top Agent", "Picked By", "Agents",
]


def scores_frame(exp: Experiment) -> pd.DataFrame:
    return pd.DataFrame(exp.backtest.scores)


def opportunities_frame(exp: Experiment, timestamp: datetime,
                        scores: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per stock at ``timestamp``: factor scores, best strategy-agent score, picks."""
    scores = scores if scores is not None else scores_frame(exp)
    at = scores[(scores["timestamp"] == timestamp.isoformat())
                & ~scores["agent"].isin(CONTROL_AGENTS)]
    best = (at.sort_values(["opportunity_score", "agent"], ascending=[False, True])
              .drop_duplicates("ticker").set_index("ticker"))
    picks = picks_at(exp, timestamp)
    rows = []
    for s in exp.dataset.snapshots_at(timestamp):
        b = best.loc[s.ticker]
        agents = sorted(picks.get(s.ticker, []))
        rows.append({
            "Ticker": s.ticker, "Company": s.company_name, "Sector": s.sector,
            "Price": s.price, "Market Cap": s.market_cap,
            "Value Score": b["value_score"], "Catalyst Score": b["catalyst_score"],
            "Momentum Score": b["momentum_score"], "Risk Score": b["risk_score"],
            "Opportunity Score": b["opportunity_score"], "Top Agent": b["agent"],
            "Picked By": len(agents), "Agents": ", ".join(agents),
        })
    df = pd.DataFrame(rows, columns=OPPORTUNITY_COLUMNS)
    return df.sort_values(["Opportunity Score", "Ticker"], ascending=[False, True]) \
             .reset_index(drop=True)


def filter_opportunities(df: pd.DataFrame, query: str = "", sectors: Sequence[str] = (),
                         agent: Optional[str] = None, min_score: float = 0.0,
                         max_risk: float = 100.0, picked_only: bool = False) -> pd.DataFrame:
    out = df
    q = query.strip().lower()
    if q:
        out = out[out["Ticker"].str.lower().str.contains(q, regex=False)
                  | out["Company"].str.lower().str.contains(q, regex=False)]
    if sectors:
        out = out[out["Sector"].isin(sectors)]
    if agent:
        out = out[out["Agents"].str.split(", ").apply(lambda xs: agent in xs)]
    out = out[(out["Opportunity Score"] >= min_score) & (out["Risk Score"] <= max_risk)]
    if picked_only:
        out = out[out["Picked By"] > 0]
    return out.reset_index(drop=True)


# ---- stock detail --------------------------------------------------------------------------

def snapshot_of(exp: Experiment, ticker: str, timestamp: datetime) -> StockSnapshot:
    for s in exp.dataset.snapshots_at(timestamp):
        if s.ticker == ticker:
            return s
    raise KeyError(f"{ticker} not in dataset at {timestamp}")


def financial_metrics(s: StockSnapshot) -> Dict[str, str]:
    pe = f"{s.pe_ratio:.1f}" if s.pe_ratio > 0 else "n/m (loss)"
    return {
        "Price": f"${s.price:,.2f}", "Market Cap": format_money(s.market_cap),
        "Shares Out": f"{s.shares_outstanding / 1e6:,.0f}M", "P/E": pe,
        "P/S": f"{s.price_to_sales:.1f}", "FCF Yield": f"{fcf_yield(s):.1%}",
        "Revenue Growth": f"{s.revenue_growth:+.1%}", "Earnings Growth": f"{s.earnings_growth:+.1%}",
        "Debt / Equity": f"{s.debt_to_equity:.2f}", "Volatility": f"{s.volatility:.0%}",
        "6m Momentum": f"{s.momentum:+.1%}", "Rel. Strength": f"{s.relative_strength:.0f}",
        "Short Interest": f"{s.short_interest:.1%}", "Days to Cover": f"{s.days_to_cover:.1f}",
        "From 52w High": f"-{s.distance_from_high:.1%}", "Volume Ratio": f"{s.volume_ratio:.2f}x",
    }


def factor_profile(s: StockSnapshot) -> pd.DataFrame:
    comps = compute_components(s).as_dict()
    return pd.DataFrame({"Factor": [f.title() for f in FACTOR_LABELS],
                         "Score": [comps[f] for f in FACTOR_LABELS]})


def agent_views(exp: Experiment, ticker: str, timestamp: datetime,
                scores: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Every agent's score of one stock, its rank in that agent's universe and whether picked."""
    scores = scores if scores is not None else scores_frame(exp)
    at = scores[scores["timestamp"] == timestamp.isoformat()].copy()
    at["rank"] = at.groupby("agent")["opportunity_score"].rank(ascending=False, method="min")
    mine = at[at["ticker"] == ticker]
    top_n, universe = exp.parameters["top_n"], at["ticker"].nunique()
    rows = [{"Agent": r.agent, "Type": "Control" if is_control(r.agent) else "Strategy",
             "Opportunity Score": r.opportunity_score, "Rank": f"{int(r.rank)} / {universe}",
             "Picked": int(r.rank) <= top_n, "Reasoning": r.reasoning}
            for r in mine.itertuples()]
    return pd.DataFrame(rows).sort_values(["Opportunity Score", "Agent"],
                                          ascending=[False, True]).reset_index(drop=True)


def risk_flags(s: StockSnapshot) -> List[str]:
    """Rule-based risk notes derived from the snapshot (no outcome data)."""
    flags = []
    if s.volatility >= 0.5:
        flags.append(f"High volatility ({s.volatility:.0%} annualised)")
    if s.debt_to_equity >= 1.5:
        flags.append(f"High leverage (debt/equity {s.debt_to_equity:.2f})")
    if s.free_cash_flow < 0:
        flags.append("Negative free cash flow")
    if s.pe_ratio <= 0:
        flags.append("Loss-making (negative earnings)")
    if s.market_cap < 2e9:
        flags.append(f"Small cap ({format_money(s.market_cap)})")
    if s.short_interest >= 0.2:
        flags.append(f"Heavily shorted ({s.short_interest:.0%} of float)")
    if s.distance_from_high >= 0.3:
        flags.append(f"Deep drawdown ({s.distance_from_high:.0%} below 52w high)")
    if s.revenue_growth < 0:
        flags.append(f"Shrinking revenue ({s.revenue_growth:+.1%})")
    return flags


def score_history(exp: Experiment, ticker: str, until: datetime,
                  scores: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Long frame (Date, Agent, Score) up to and including ``until`` - never later."""
    scores = scores if scores is not None else scores_frame(exp)
    sel = scores[(scores["ticker"] == ticker) & (scores["timestamp"] <= until.isoformat())]
    return pd.DataFrame({"Date": pd.to_datetime(sel["timestamp"]), "Agent": sel["agent"],
                         "Score": sel["opportunity_score"]}).reset_index(drop=True)


def price_history(exp: Experiment, ticker: str, until: datetime) -> pd.DataFrame:
    rows = [{"Date": ts, "Price": snapshot_of(exp, ticker, ts).price}
            for ts in exp.dataset.timestamps if ts <= until]
    return pd.DataFrame(rows)


# ---- performance ---------------------------------------------------------------------------

def series_kind(name: str) -> str:
    if name in BENCHMARK_NAMES:
        return "Benchmark"
    return "Control" if is_control(name) else "Agent"


def long_curves(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Wide (Date x series) -> long (Date, Series, Kind, value) for Altair."""
    long = frame.reset_index().melt(id_vars="Date", var_name="Series", value_name=value_name)
    long["Kind"] = long["Series"].map(series_kind)
    return long


def random_distribution(exp: Experiment) -> pd.DataFrame:
    return pd.DataFrame({"Total Return": exp.benchmarks.random_trial_totals})


# ---- experiments ---------------------------------------------------------------------------

def history_frame(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        p, res = r.get("parameters", {}), r.get("results", {})
        signal = p.get("signal", {})
        ranking = res.get("ranking", [])
        best = next((a for a in ranking if not is_control(a)), None)
        agents = res.get("agents", {})
        best_alpha = (agents.get(best, {}).get("alpha", {}).get(MARKET, {}).get("alpha")
                      if best else None)
        null = bool(signal) and all(signal.get(k, 0) == 0 for k in signal
                                    if k not in ("regime_dispersion", "noise_scale"))
        rows.append({
            "Experiment": r.get("experiment_id"), "Run At": r.get("timestamp", "")[:19],
            "Seed": p.get("seed"), "Dataset Version": r.get("dataset_version"),
            "Code Version": r.get("code_version"),
            "Agent Version": _combined_agent_version(r.get("agent_versions", {})),
            "Stocks": p.get("n_stocks"), "Months": p.get("n_periods"), "Top N": p.get("top_n"),
            "Signal": "Null" if null else "Weak", "Rank By": p.get("rank_by"),
            "Best Agent": best, "Best Alpha": best_alpha,
        })
    return pd.DataFrame(rows, columns=[
        "Experiment", "Run At", "Seed", "Dataset Version", "Code Version", "Agent Version",
        "Stocks", "Months", "Top N", "Signal", "Rank By", "Best Agent", "Best Alpha"])


def _combined_agent_version(versions: Dict[str, str]) -> str:
    if not versions:
        return ""
    payload = "|".join(f"{k}={versions[k]}" for k in sorted(versions))
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def record_results_frame(record: Dict[str, Any]) -> pd.DataFrame:
    agents = record.get("results", {}).get("agents", {})
    rows = []
    for name in record.get("results", {}).get("ranking", []):
        m = agents.get(name, {})
        alpha = m.get("alpha", {}).get(MARKET, {})
        rows.append({"Agent": name, "Total Return": m.get("total_return"),
                     "Alpha": alpha.get("alpha"), "t-stat": alpha.get("t_stat"),
                     "Sharpe": m.get("sharpe"), "Win Rate": m.get("win_rate"),
                     "Max Drawdown": m.get("max_drawdown"), "Trades": m.get("num_trades"),
                     "Version": record.get("agent_versions", {}).get(name)})
    return pd.DataFrame(rows)


# ---- formatting ----------------------------------------------------------------------------

def format_money(x: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= div:
            return f"${x / div:,.1f}{unit}"
    return f"${x:,.0f}"
