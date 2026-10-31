"""Point-in-time backtest: decide from snapshots first, then realise outcomes."""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Sequence

from agents.base_agent import AgentResult, BaseAgent
from data.models import MarketDataset, StockSnapshot


@dataclass(frozen=True)
class Trade:
    timestamp: datetime
    agent: str
    ticker: str
    opportunity_score: float
    forward_return: float


@dataclass
class BacktestResult:
    top_n: int
    timestamps: List[datetime] = field(default_factory=list)
    period_ends: List[datetime] = field(default_factory=list)  # when each trade is realised
    regimes: List[str] = field(default_factory=list)
    scores: List[AgentResult] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    period_returns: Dict[str, List[float]] = field(default_factory=dict)

    def equity_curve(self, agent: str) -> List[float]:
        equity, curve = 1.0, []
        for r in self.period_returns[agent]:
            equity *= 1 + r
            curve.append(equity)
        return curve


def _next_month_end(ts: datetime) -> datetime:
    year, month = (ts.year + 1, 1) if ts.month == 12 else (ts.year, ts.month + 1)
    return datetime(year, month, calendar.monthrange(year, month)[1])


def decide(
    agents: Sequence[BaseAgent], snapshots: Sequence[StockSnapshot], top_n: int
) -> Dict[str, List[AgentResult]]:
    """Every agent analyses the SAME snapshots. Receives no outcome data by construction."""
    decisions: Dict[str, List[AgentResult]] = {}
    for agent in agents:
        results = [agent.analyze(s) for s in snapshots]
        results.sort(key=lambda r: (-r["opportunity_score"], r["ticker"]))
        decisions[agent.name] = results
    return decisions


def run_backtest(
    dataset: MarketDataset, agents: Sequence[BaseAgent], top_n: int = 10
) -> BacktestResult:
    """Each period: snapshots -> agents -> top-N equal-weight picks -> realised return."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    names = [a.name for a in agents]
    if len(set(names)) != len(names):
        raise ValueError("Agent names must be unique")

    result = BacktestResult(top_n=top_n, period_returns={n: [] for n in names})
    for ts in dataset.timestamps:
        snapshots = dataset.snapshots_at(ts)
        decisions = decide(agents, snapshots, top_n)  # 1) decisions, outcomes not touched
        result.timestamps.append(ts)
        result.regimes.append(dataset.regime_at(ts))
        for name, ranked in decisions.items():  # 2) only now realise outcomes
            result.scores.extend(ranked)
            picks = ranked[:top_n]
            returns = [dataset.outcome(r["ticker"], ts) for r in picks]
            for r, ret in zip(picks, returns):
                result.trades.append(Trade(ts, name, r["ticker"], r["opportunity_score"], ret))
            result.period_returns[name].append(sum(returns) / len(returns))
    result.period_ends = [
        result.timestamps[i + 1] if i + 1 < len(result.timestamps) else _next_month_end(ts)
        for i, ts in enumerate(result.timestamps)
    ]
    return result
