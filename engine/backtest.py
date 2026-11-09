"""Point-in-time backtest: decide from snapshots first, then realise outcomes."""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

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
    period_returns: Dict[str, List[float]] = field(default_factory=dict)  # after costs
    turnover: Dict[str, List[float]] = field(default_factory=dict)  # share of portfolio traded
    costs: Dict[str, List[float]] = field(default_factory=dict)     # cost deducted per period
    cost_bps: float = 0.0

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
    """Every agent analyses the SAME snapshots. Receives no outcome data by construction.

    Meta-agents run afterwards, on the decisions of the non-control agents of this timestamp.
    """
    decisions: Dict[str, List[AgentResult]] = {}
    metas = [a for a in agents if getattr(a, "is_meta", False)]
    for agent in agents:
        if getattr(agent, "is_meta", False):
            continue
        universe = getattr(agent, "analyze_universe", None)  # cross-sectional agents
        results = list(universe(snapshots)) if universe else [agent.analyze(s) for s in snapshots]
        results.sort(key=lambda r: (-r["opportunity_score"], r["ticker"]))
        decisions[agent.name] = results
    if metas:
        voters = {a.name: decisions[a.name] for a in agents
                  if not getattr(a, "is_meta", False) and getattr(a, "family", "") != "control"}
        for meta in metas:
            results = meta.combine(voters, top_n)
            results.sort(key=lambda r: (-r["opportunity_score"], r["ticker"]))
            decisions[meta.name] = results
    return {a.name: decisions[a.name] for a in agents}  # keep the agents' order


def rebalance_cost(old: Dict[str, float], new: Dict[str, float], cost_bps: float
                   ) -> Tuple[float, float]:
    """(one-way turnover, cost as a fraction of the portfolio) of moving from old to new
    weights. ``cost_bps`` is charged on every unit bought or sold (commission + slippage)."""
    names = set(old) | set(new)
    buys = sum(max(new.get(t, 0.0) - old.get(t, 0.0), 0.0) for t in names)
    sells = sum(max(old.get(t, 0.0) - new.get(t, 0.0), 0.0) for t in names)
    return max(buys, sells), (buys + sells) * cost_bps / 10_000


def run_backtest(
    dataset: MarketDataset, agents: Sequence[BaseAgent], top_n: int = 10,
    cost_bps: float = 0.0,
) -> BacktestResult:
    """Each period: snapshots -> agents -> top-N equal-weight picks -> realised return,
    minus trading costs for the change of holdings (``cost_bps`` per side)."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    names = [a.name for a in agents]
    if len(set(names)) != len(names):
        raise ValueError("Agent names must be unique")
    if cost_bps < 0:
        raise ValueError("cost_bps must be >= 0")

    result = BacktestResult(top_n=top_n, period_returns={n: [] for n in names},
                            turnover={n: [] for n in names}, costs={n: [] for n in names},
                            cost_bps=cost_bps)
    holdings: Dict[str, Dict[str, float]] = {n: {} for n in names}
    for ts in dataset.timestamps:
        snapshots = dataset.snapshots_at(ts)
        decisions = decide(agents, snapshots, top_n)  # 1) decisions, outcomes not touched
        result.timestamps.append(ts)
        result.regimes.append(dataset.regime_at(ts))
        for name, ranked in decisions.items():  # 2) only now realise outcomes
            result.scores.extend(ranked)
            picks = [r for r in ranked if r.get("eligible", True)][:top_n]
            returns = [dataset.outcome(r["ticker"], ts) for r in picks]
            for r, ret in zip(picks, returns):
                result.trades.append(Trade(ts, name, r["ticker"], r["opportunity_score"], ret))
            # nothing passed the agent's filters -> stays in cash for the period
            gross = sum(returns) / len(returns) if returns else 0.0
            weights = {r["ticker"]: 1 / len(picks) for r in picks}
            turnover, cost = rebalance_cost(holdings[name], weights, cost_bps)
            holdings[name] = weights
            result.turnover[name].append(turnover)
            result.costs[name].append(cost)
            result.period_returns[name].append(gross - cost)
    result.period_ends = [
        result.timestamps[i + 1] if i + 1 < len(result.timestamps) else _next_month_end(ts)
        for i, ts in enumerate(result.timestamps)
    ]
    return result
