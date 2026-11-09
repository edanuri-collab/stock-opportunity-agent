"""View-model layer for the dashboard: pure functions from an Experiment to frames/dicts.

Nothing here imports Streamlit, so every screen's numbers are unit-testable. Views only
format and lay out what these functions return.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from agents.base_agent import BaseAgent
from agents.registry import Registry
from agents.spec import FIELDS, AgentSpec, Filter, field_value
from dashboard import i18n as he
from dashboard.i18n import FACTORS as FACTOR_HE
from engine.backtest import decide, run_backtest
from engine.evaluation import compute_agent_metrics
from engine.stats import SIGNIFICANCE, holm
from data.models import StockSnapshot
from data.sample_data import SIGNAL_KEYS
from engine.benchmarks import BUY_HOLD, MARKET, RANDOM
from engine.experiment import Experiment
from engine.scoring import compute_components, fcf_yield, top_factors

CONTROL_AGENTS = frozenset({"Random Agent"})  # placebo: never counted as a recommendation
BENCHMARK_NAMES = (MARKET, BUY_HOLD, RANDOM)
ALPHA_SIGNIFICANT_T = 2.0
FACTOR_LABELS = ("value", "quality", "momentum", "catalyst", "breakout", "squeeze",
                 "contrarian", "risk")


def is_control(agent: str) -> bool:
    return agent in CONTROL_AGENTS


def is_meta(agent: Optional[BaseAgent]) -> bool:
    return bool(getattr(agent, "is_meta", False))


def _not_voting(exp: Experiment) -> set:
    """Agents that are not counted as recommenders: the control and the meta-agents."""
    return set(CONTROL_AGENTS) | {a.name for a in exp.agents if is_meta(a)}


def strategy_agents(exp: Experiment) -> List[str]:
    return [a for a in exp.backtest.period_returns if not is_control(a)]


def adjusted_pvalues(exp: Experiment) -> Dict[str, float]:
    """Holm-corrected p-values of alpha vs the market, across every non-control agent:
    with dozens of agents, some raw |t| > 2 results are expected by luck alone."""
    return holm({m.agent: m.alpha[MARKET].p_value for m in exp.metrics
                 if not is_control(m.agent)})


def corrected_status(t_stat: float, p_adjusted: Optional[float]) -> str:
    if p_adjusted is not None and p_adjusted < SIGNIFICANCE:
        return "Outperforming" if t_stat > 0 else "Underperforming"
    return "Inconclusive"


def performance_status(t_stat: float) -> str:
    """Short status label for an agent's alpha vs the market (uncorrected |t| >= 2)."""
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
    null = bool(signal) and all(signal.get(k, 0) == 0 for k in SIGNAL_KEYS)
    return DataSourceStatus(
        source="מחולל סינתטי (data/sample_data.py)",
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
    skip = _not_voting(exp)
    for t in exp.backtest.trades:
        if t.timestamp == timestamp and t.agent not in skip:
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
        "significant_raw": sum(1 for m in strategies
                               if m.alpha[MARKET].t_stat >= ALPHA_SIGNIFICANT_T),
        "significant": sum(1 for m in strategies if m.alpha[MARKET].t_stat > 0
                           and adjusted_pvalues(exp).get(m.agent, 1.0) < SIGNIFICANCE),
        "cost_bps": exp.parameters.get("cost_bps", 0.0),
        "beat_random_agent": (sum(1 for m in strategies if m.total_return > control.total_return)
                              if control else None),
        "market_return": exp.results["benchmarks"][MARKET]["total_return"],
        "avg_sharpe": sum(m.sharpe for m in strategies) / len(strategies),
        "latest_timestamp": latest,
        "run_timestamp": exp.timestamp,
    }


# ---- leaderboard ---------------------------------------------------------------------------

def agent_leaderboard(exp: Experiment) -> pd.DataFrame:
    agents = _agent_map(exp)
    adjusted = adjusted_pvalues(exp)
    rows = []
    for rank, m in enumerate(exp.metrics, start=1):
        a = m.alpha[MARKET]
        control = is_control(m.agent)
        rows.append({
            "Rank": rank, "Agent": m.agent, "Type": "Control" if control else "Strategy",
            "Family": getattr(agents.get(m.agent), "family", ""),
            "Status": corrected_status(a.t_stat, adjusted.get(m.agent)),
            "Adjusted p": adjusted.get(m.agent),
            # a placebo's "score" is hash noise, so it is not shown as an opportunity score
            "Opportunity Score": None if control else m.average_score,
            "Alpha": a.alpha, "Alpha t-stat": a.t_stat,
            "Win Rate": m.win_rate, "Sharpe": m.sharpe, "Max Drawdown": m.max_drawdown,
            "Trades": m.num_trades, "Total Return": m.total_return,
            "Beats Random": m.percentile_vs_random,
            "Turnover": m.turnover, "Cost": m.total_cost,
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
    """One row per stock at ``timestamp``.

    Raw scores of different agents are not comparable (a ranking agent always gives its best
    stock 100), so each agent's scores are first turned into percentiles within that agent
    (filtered-out stocks count as 0). Opportunity Score = average percentile across strategy
    agents; Top Agent = the agent that ranks the stock highest.
    """
    scores = scores if scores is not None else scores_frame(exp)
    at = scores[(scores["timestamp"] == timestamp.isoformat())
                & ~scores["agent"].isin(_not_voting(exp))].copy()
    at["pct"] = at.groupby("agent")["opportunity_score"].rank(pct=True) * 100
    at.loc[~at["eligible"].astype(bool), "pct"] = 0.0
    mean_pct = at.groupby("ticker")["pct"].mean()
    best = (at.sort_values(["pct", "opportunity_score", "agent"], ascending=[False, False, True])
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
            "Opportunity Score": round(float(mean_pct[s.ticker]), 2), "Top Agent": b["agent"],
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
    """Display label (Hebrew) -> formatted value."""
    pe = f"{s.pe_ratio:.1f}" if s.pe_ratio > 0 else "לא רלוונטי (הפסד)"
    return {
        "מחיר": f"${s.price:,.2f}", "שווי שוק": format_money(s.market_cap),
        "מניות במחזור": f"{s.shares_outstanding / 1e6:,.0f}M", "מכפיל רווח (P/E)": pe,
        "מכפיל מכירות (P/S)": f"{s.price_to_sales:.1f}",
        "תשואת תזרים חופשי": f"{fcf_yield(s):.1%}",
        "צמיחה בהכנסות": f"{s.revenue_growth:+.1%}", "צמיחה ברווח": f"{s.earnings_growth:+.1%}",
        "חוב להון": f"{s.debt_to_equity:.2f}", "תנודתיות שנתית": f"{s.volatility:.0%}",
        "מומנטום 6 חודשים": f"{s.momentum:+.1%}", "חוזק יחסי": f"{s.relative_strength:.0f}",
        "שורט מהמניות הצפות": f"{s.short_interest:.1%}",
        "ימים לכיסוי שורט": f"{s.days_to_cover:.1f}",
        "מרחק משיא 52 שבועות": f"-{s.distance_from_high:.1%}",
        "יחס מחזור מסחר": f"{s.volume_ratio:.2f}x",
        "מכפיל הון (P/B)": f"{s.price_to_book:.2f}", "תשואה על ההון (ROE)": f"{s.roe:.1%}",
        "תשואה על ההון המושקע (ROIC)": f"{s.roic:.1%}",
        "שולי רווח גולמי": f"{s.gross_margin:.1%}",
        "תשואת דיבידנד": f"{s.dividend_yield:.1%}",
        "הפתעת רווחים": f"{s.earnings_surprise:+.1%}",
        "מומנטום 12-1 חודשים": f"{s.momentum_12_1:+.1%}", "בטא": f"{s.beta:.2f}",
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
    universe = at["ticker"].nunique()
    snap = snapshot_of(exp, ticker, timestamp)
    held = {t.agent for t in exp.backtest.trades
            if t.timestamp == timestamp and t.ticker == ticker}
    agents = _agent_map(exp)
    rows = [{"Agent": r.agent, "Type": "Control" if is_control(r.agent) else "Strategy",
             "Opportunity Score": r.opportunity_score, "Rank": f"{int(r.rank)} / {universe}",
             "Picked": r.agent in held,
             "Reasoning": reasoning_he(r.agent, snap, r.opportunity_score,
                                       agents.get(r.agent), r.eligible, r.reasoning)}
            for r in mine.itertuples()]
    return pd.DataFrame(rows).sort_values(["Opportunity Score", "Agent"],
                                          ascending=[False, True]).reset_index(drop=True)


def risk_flags(s: StockSnapshot) -> List[str]:
    """Rule-based risk notes (Hebrew) derived from the snapshot (no outcome data)."""
    flags = []
    if s.volatility >= 0.5:
        flags.append(f"תנודתיות גבוהה ({s.volatility:.0%} בשנה)")
    if s.debt_to_equity >= 1.5:
        flags.append(f"מינוף גבוה (חוב להון {s.debt_to_equity:.2f})")
    if s.free_cash_flow < 0:
        flags.append("תזרים מזומנים חופשי שלילי")
    if s.pe_ratio <= 0:
        flags.append("החברה מפסידה (רווח שלילי)")
    if s.market_cap < 2e9:
        flags.append(f"חברה קטנה (שווי שוק {ltr(format_money(s.market_cap))})")
    if s.short_interest >= 0.2:
        flags.append(f"שורט גבוה ({s.short_interest:.0%} מהמניות הצפות)")
    if s.distance_from_high >= 0.3:
        flags.append(f"ירידה עמוקה ({s.distance_from_high:.0%} מתחת לשיא 52 שבועות)")
    if s.revenue_growth < 0:
        flags.append(f"הכנסות מתכווצות ({ltr(f'{s.revenue_growth:+.1%}')})")
    return flags


def _agent_map(exp: Experiment) -> Dict[str, BaseAgent]:
    return {a.name: a for a in exp.agents}

def ltr(text: object) -> str:
    """Isolate a Latin/numeric fragment so it reads left-to-right inside Hebrew text."""
    return f"⁦{text}⁩"


_FACTOR_FACTS = {
    "value": lambda s: (ltr(f"P/E {s.pe_ratio:.1f}") if s.pe_ratio > 0 else "P/E שלילי")
    + f", {ltr(f'P/S {s.price_to_sales:.1f}')}, תשואת תזרים {ltr(f'{fcf_yield(s):.1%}')}",
    "quality": lambda s: f"צמיחה בהכנסות {ltr(f'{s.revenue_growth:+.1%}')}, "
                         f"חוב להון {ltr(f'{s.debt_to_equity:.2f}')}",
    "momentum": lambda s: f"מומנטום 6 חודשים {ltr(f'{s.momentum:+.1%}')}, "
                          f"חוזק יחסי {s.relative_strength:.0f}",
    "catalyst": lambda s: f"ציון קטליזטור {s.catalyst_score:.0f}, סנטימנט {s.sentiment_score:.0f}",
    "breakout": lambda s: f"{ltr(f'{s.distance_from_high:.1%}')} מתחת לשיא, "
                          f"מחזור {ltr(f'x{s.volume_ratio:.1f}')}",
    "squeeze": lambda s: f"שורט {ltr(f'{s.short_interest:.1%}')}, "
                         f"{s.days_to_cover:.1f} ימים לכיסוי",
    "contrarian": lambda s: f"{ltr(f'{s.distance_from_high:.0%}')} מתחת לשיא, צמיחה ברווח "
                            f"{ltr(f'{s.earnings_growth:+.1%}')}",
}


def reasoning_he(agent: str, s: StockSnapshot, score: float,
                 cfg: Optional[BaseAgent] = None, eligible: bool = True,
                 english: str = "") -> str:
    """Hebrew explanation built from the same components and weights the agent used."""
    if is_meta(cfg):
        votes = re.search(r"picked by (\d+) of (\d+)", english)
        k, n = (votes.group(1), votes.group(2)) if votes else ("?", "?")
        verdict = "" if eligible else " פחות משני סוכנים בחרו בה, ולכן היא לא נבחרת."
        return (f"{ltr(agent)}: {ltr(k)} מתוך {ltr(n)} סוכני אסטרטגיה מחזיקים את "
                f"{ltr(s.ticker)} ב-Top שלהם.{verdict}")
    if getattr(cfg, "spec", None) is not None:
        return reasoning_spec_he(cfg.spec, s, score, eligible)
    if is_control(agent):
        return (f"בחירת פלסבו: הציון ({score:.1f}) נקבע מ-{ltr('hash')} של הטיקר והתאריך, "
                "ומתעלם מכל נתוני המניה.")
    if cfg is None:
        return f"{ltr(agent)} נותן ל-{ltr(s.ticker)} ציון {ltr(f'{score:.1f}/100')}."
    comps = compute_components(s)
    cd = comps.as_dict()
    drivers = top_factors(comps, cfg.weights, 2)
    driver_txt = ", ".join(f"{FACTOR_HE[f]} {cd[f]:.0f}" for f in drivers)
    facts = "; ".join(_FACTOR_FACTS[f](s) for f in drivers if f in _FACTOR_FACTS)
    penalty = cfg.risk_penalty * comps.risk
    return (f"{ltr(agent)} נותן ל-{ltr(s.ticker)} ציון {ltr(f'{score:.1f}/100')}. "
            f"גורמים מובילים: {driver_txt}. "
            f"נתונים: {facts}. ציון הסיכון ({comps.risk:.0f}) מוריד {penalty:.1f} נקודות.")


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


# ---- agent catalog & builder ---------------------------------------------------------------

def format_field(name: str, value: float) -> str:
    kind = FIELDS[name][0]
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "money":
        return format_money(value)
    if kind == "score":
        return f"{value:.0f}"
    return f"{value:.2f}"


def filter_he(f: Filter) -> str:
    return f"{he.FIELDS[f.field]} {ltr(f'{he.OPS[f.op]} {format_field(f.field, f.value)}')}"


def _scoring_he(weights, rank) -> List[str]:
    if rank:
        total = sum(r.weight for r in rank)
        return [f"דירוג: {he.FIELDS[r.field]}, {he.PREFER[r.prefer]} "
                f"(משקל {ltr(f'{r.weight / total:.0%}')})" for r in rank]
    ordered = sorted(weights, key=lambda kv: -kv[1])
    return ["שקלול גורמים: " + ", ".join(f"{FACTOR_HE[k]} {ltr(f'{v:.0%}')}"
                                       for k, v in ordered)]


def rules_he(spec: AgentSpec) -> List[str]:
    """The agent's rules as short Hebrew lines (for the catalog)."""
    lines = [f"תנאי סינון: {filter_he(f)}" for f in spec.filters]
    if spec.switch is None:
        lines += _scoring_he(spec.weights, spec.rank)
    else:
        lines += [f"כאשר {filter_he(spec.switch)}: " + line
                  for line in _scoring_he(spec.weights, spec.rank)]
        lines += ["אחרת: " + line for line in _scoring_he(spec.alt_weights, spec.alt_rank)]
    if spec.risk_penalty:
        lines.append(f"קנס סיכון: {ltr(f'{spec.risk_penalty:.2f}')} מציון הסיכון")
    return lines


def reasoning_spec_he(spec: AgentSpec, s: StockSnapshot, score: float, eligible: bool) -> str:
    who = ltr(spec.name)
    if not eligible:
        failed = [filter_he(f) for f in spec.filters if not f.passes(s)]
        return f"{who} מסנן את {ltr(s.ticker)}: לא עומדת בתנאי " + "; ".join(failed) + "."
    text = f"{who} נותן ל-{ltr(s.ticker)} ציון {ltr(f'{score:.1f}/100')}."
    if spec.filters:
        text += f" עוברת את כל {len(spec.filters)} תנאי הסינון."
    weights, rank = spec.weights_map, spec.rank
    if spec.switch is not None:
        if spec.switch.passes(s):
            text += f" מצב שוק: {filter_he(spec.switch)} מתקיים, ולכן פועלת השיטה הראשית."
        else:
            text += f" מצב שוק: {filter_he(spec.switch)} לא מתקיים, ולכן פועלת השיטה החלופית."
            weights, rank = spec.alt_weights_map, spec.alt_rank
    if rank:
        text += " דירוג לפי: " + ", ".join(
            f"{he.FIELDS[r.field]} {ltr(format_field(r.field, field_value(s, r.field)))} "
            f"({he.PREFER[r.prefer]})" for r in rank) + "."
    else:
        comps = compute_components(s)
        cd = comps.as_dict()
        drivers = top_factors(comps, weights, 2)
        text += " גורמים מובילים: " + ", ".join(f"{FACTOR_HE[f]} {cd[f]:.0f}"
                                              for f in drivers) + "."
    return text


def method_of(agent: BaseAgent) -> str:
    if is_meta(agent):
        return "meta"
    spec = getattr(agent, "spec", None)
    return spec.method if spec is not None else "weights"


def catalog_frame(exp: Experiment, registry: Registry) -> pd.DataFrame:
    metrics = {m.agent: (rank, m) for rank, m in enumerate(exp.metrics, start=1)}
    rows = []
    for a in registry.agents:
        rank, m = metrics.get(a.name, (None, None))
        alpha = m.alpha[MARKET] if m else None
        rows.append({
            "Agent": a.name, "Family": a.family, "Origin": a.origin,
            "Method": method_of(a), "Source": registry.sources.get(a.name, ""),
            "Rank": rank, "Alpha": alpha.alpha if alpha else None,
            "Alpha t-stat": alpha.t_stat if alpha else None,
            "Trades": m.num_trades if m else None, "Description": a.description,
        })
    return pd.DataFrame(rows)


def preview_agent(exp: Experiment, agent: BaseAgent) -> Dict[str, Any]:
    """Backtest one (unsaved) agent on the experiment's data, against its benchmarks."""
    top_n = exp.parameters["top_n"]
    bt = run_backtest(exp.dataset, [agent], top_n)
    m = compute_agent_metrics(bt, exp.benchmarks)[0]
    latest = exp.dataset.timestamps[-1]
    ranked = decide([agent], exp.dataset.snapshots_at(latest), top_n)[agent.name]
    eligible = [r for r in ranked if r["eligible"]]
    names = {s.ticker: s.company_name for s in exp.dataset.snapshots_at(latest)}
    picks = pd.DataFrame([{"Ticker": r["ticker"], "Company": names[r["ticker"]],
                           "Opportunity Score": r["opportunity_score"]}
                          for r in eligible[:top_n]],
                         columns=["Ticker", "Company", "Opportunity Score"])
    control = next((x for x in exp.metrics if is_control(x.agent)), None)
    return {
        "metrics": m, "alpha": m.alpha[MARKET], "picks": picks,
        "eligible_latest": len(eligible), "universe": len(ranked),
        "avg_picks": m.num_trades / max(len(exp.dataset.timestamps), 1),
        "random_agent_return": control.total_return if control else None,
    }
