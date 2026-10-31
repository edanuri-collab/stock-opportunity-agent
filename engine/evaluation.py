"""Performance metrics. Pure functions over returns, easy to unit-test.

Conventions: monthly periods (PERIODS_PER_YEAR = 12), risk-free rate 0, trade = one pick held
for one period, portfolio = equal-weight top-N of that period.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from engine.backtest import BacktestResult
from engine.benchmarks import MARKET, BenchmarkSet

PERIODS_PER_YEAR = 12


# ---- trade-level -----------------------------------------------------------------------------

def win_rate(returns: Sequence[float]) -> float:
    return sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0


def expectancy(returns: Sequence[float]) -> float:
    """Average return per trade."""
    return sum(returns) / len(returns) if returns else 0.0


def average_win(returns: Sequence[float]) -> float:
    wins = [r for r in returns if r > 0]
    return sum(wins) / len(wins) if wins else 0.0


def average_loss(returns: Sequence[float]) -> float:
    """Mean of losing trades, a negative number (0 if none)."""
    losses = [r for r in returns if r < 0]
    return sum(losses) / len(losses) if losses else 0.0


def profit_factor(returns: Sequence[float]) -> float:
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = -sum(r for r in returns if r < 0)
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


# ---- portfolio-level -------------------------------------------------------------------------

def equity_curve(period_returns: Sequence[float]) -> List[float]:
    """Equity including the starting value 1.0."""
    equity, curve = 1.0, [1.0]
    for r in period_returns:
        equity *= 1 + r
        curve.append(equity)
    return curve


def total_return(period_returns: Sequence[float]) -> float:
    return equity_curve(period_returns)[-1] - 1


def cagr(period_returns: Sequence[float], periods_per_year: int = PERIODS_PER_YEAR) -> float:
    if not period_returns:
        return 0.0
    final = 1 + total_return(period_returns)
    if final <= 0:
        return -1.0
    return final ** (periods_per_year / len(period_returns)) - 1


def sharpe(period_returns: Sequence[float], periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualised Sharpe ratio (risk-free 0, sample std)."""
    n = len(period_returns)
    if n < 2:
        return 0.0
    mean = sum(period_returns) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in period_returns) / (n - 1))
    return mean / sd * math.sqrt(periods_per_year) if sd > 0 else 0.0


def drawdown_curve(period_returns: Sequence[float]) -> List[float]:
    """Drawdown at each equity point, <= 0 (e.g. -0.12 = 12% below the peak)."""
    out, peak = [], 0.0
    for equity in equity_curve(period_returns):
        peak = max(peak, equity)
        out.append(equity / peak - 1)
    return out


def max_drawdown(period_returns: Sequence[float]) -> float:
    """Largest peak-to-trough fall of compounded equity, as a positive fraction."""
    return -min(drawdown_curve(period_returns))


@dataclass(frozen=True)
class AlphaStats:
    alpha: float        # annualised Jensen alpha (regression intercept x periods per year)
    beta: float
    t_stat: float       # t-statistic of the intercept; |t| < 2 is hard to tell from luck
    excess_cagr: float  # CAGR(agent) - CAGR(benchmark)


def alpha_stats(
    agent: Sequence[float], bench: Sequence[float], periods_per_year: int = PERIODS_PER_YEAR
) -> AlphaStats:
    """OLS of agent returns on benchmark returns: agent = a + b * bench + e."""
    if len(agent) != len(bench):
        raise ValueError("series must have equal length")
    n = len(agent)
    excess = cagr(agent, periods_per_year) - cagr(bench, periods_per_year)
    if n < 3:
        return AlphaStats(0.0, 0.0, 0.0, excess)
    mx, my = sum(bench) / n, sum(agent) / n
    sxx = sum((x - mx) ** 2 for x in bench)
    if sxx == 0:
        return AlphaStats(my * periods_per_year, 0.0, 0.0, excess)
    beta = sum((x - mx) * (y - my) for x, y in zip(bench, agent)) / sxx
    a = my - beta * mx
    sse = sum((y - a - beta * x) ** 2 for x, y in zip(bench, agent))
    s2 = sse / (n - 2)
    se_a = math.sqrt(s2 * (1 / n + mx ** 2 / sxx))
    return AlphaStats(a * periods_per_year, beta, a / se_a if se_a > 0 else 0.0, excess)


# ---- aggregation -----------------------------------------------------------------------------

@dataclass
class AgentMetrics:
    agent: str
    total_return: float
    cagr: float
    win_rate: float
    average_win: float
    average_loss: float
    expectancy: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    num_trades: int
    average_score: float
    percentile_vs_random: float          # share of random portfolios beaten (0-1)
    alpha: Dict[str, AlphaStats] = field(default_factory=dict)  # per benchmark name

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        return data


RANK_KEYS = ("alpha", "sharpe", "total_return", "cagr", "expectancy")


def _rank_value(m: AgentMetrics, rank_by: str) -> float:
    if rank_by == "alpha":
        return m.alpha[MARKET].alpha
    return float(getattr(m, rank_by))


def compute_agent_metrics(
    result: BacktestResult, benchmarks: BenchmarkSet, rank_by: str = "alpha"
) -> List[AgentMetrics]:
    """One AgentMetrics per agent, ranked best first by ``rank_by``."""
    if rank_by not in RANK_KEYS:
        raise ValueError(f"rank_by must be one of {RANK_KEYS}")
    out: List[AgentMetrics] = []
    for agent, series in result.period_returns.items():
        trades = [t for t in result.trades if t.agent == agent]
        returns = [t.forward_return for t in trades]
        scores = [t.opportunity_score for t in trades]
        total = total_return(series)
        out.append(AgentMetrics(
            agent=agent,
            total_return=total,
            cagr=cagr(series),
            win_rate=win_rate(returns),
            average_win=average_win(returns),
            average_loss=average_loss(returns),
            expectancy=expectancy(returns),
            profit_factor=profit_factor(returns),
            sharpe=sharpe(series),
            max_drawdown=max_drawdown(series),
            num_trades=len(trades),
            average_score=sum(scores) / len(scores) if scores else 0.0,
            percentile_vs_random=benchmarks.percentile_of(total),
            alpha={name: alpha_stats(series, bench) for name, bench in benchmarks.returns.items()},
        ))
    out.sort(key=lambda m: (-_rank_value(m, rank_by), m.agent))
    return out


def benchmark_metrics(benchmarks: BenchmarkSet) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    market = benchmarks.returns[MARKET]
    for name, series in benchmarks.returns.items():
        stats = alpha_stats(series, market)
        out[name] = {
            "total_return": total_return(series),
            "cagr": cagr(series),
            "sharpe": sharpe(series),
            "max_drawdown": max_drawdown(series),
            "alpha_vs_market": stats.alpha,
            "beta_vs_market": stats.beta,
        }
    return out


def regime_breakdown(result: BacktestResult) -> Dict[str, Dict[str, float]]:
    """Average period return per agent per market regime."""
    out: Dict[str, Dict[str, float]] = {}
    for agent, returns in result.period_returns.items():
        buckets: Dict[str, List[float]] = defaultdict(list)
        for regime, r in zip(result.regimes, returns):
            buckets[regime].append(r)
        out[agent] = {regime: sum(v) / len(v) for regime, v in sorted(buckets.items())}
    return out
