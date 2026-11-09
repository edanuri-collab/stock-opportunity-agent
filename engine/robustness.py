"""Fair ranking: is an agent good in *many* worlds, and does picking past winners work?

* Monte Carlo: the same agents run on many independent datasets (seeds). An agent's alpha
  across seeds is tested against zero, then Holm-corrected across all agents, so testing dozens
  of strategies does not manufacture a "winner".
* In-sample / out-of-sample: every run is split in half. If agents that led in the first half
  keep leading in the second (positive rank correlation), the ranking carries information.
* Walk-forward selection: each month, follow the agent with the best trailing record and take
  its *next* month. If that does not beat simply holding all strategies, chasing the leader
  adds nothing.

Everything runs on the synthetic generator: it tests the method, not the real market.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence

from agents.base_agent import BaseAgent
from data.sample_data import generate_dataset
from engine.backtest import run_backtest
from engine.benchmarks import MARKET, compute_benchmarks
from engine.evaluation import alpha_stats, total_return
from engine.experiment import ExperimentParams
from engine.stats import SIGNIFICANCE, holm, spearman, t_pvalue

LOOKBACK = 12  # months of history used by the walk-forward selector


@dataclass
class SeedResult:
    seed: int
    alpha: Dict[str, float]
    total_return: Dict[str, float]
    alpha_in: Dict[str, float]         # first half of the months
    alpha_out: Dict[str, float]        # second half
    rank_correlation: float            # in-sample vs out-of-sample alpha ranks (strategies)
    winner_out_percentile: float       # 0-1: where the in-sample #1 lands out of sample
    walk_forward: float                # total return of following the trailing leader
    walk_forward_average: float        # holding all strategies equally, same months
    walk_forward_random_agent: float   # the control, same months


@dataclass
class AgentSummary:
    agent: str
    mean_alpha: float
    sd_alpha: float
    t_stat: float
    p_value: float
    p_adjusted: float
    share_positive: float           # seeds with alpha > 0
    share_beats_random_agent: float  # seeds with total return above the control
    mean_rank: float                 # 1 = best alpha in a seed
    mean_alpha_in: float
    mean_alpha_out: float
    verdict: str                     # Robust edge | Robustly worse | Inconclusive


@dataclass
class MonteCarloResult:
    seeds: List[int]
    params: Dict[str, object]
    runs: List[SeedResult] = field(default_factory=list)
    summaries: List[AgentSummary] = field(default_factory=list)  # best mean alpha first

    def mean(self, attr: str) -> float:
        values = [getattr(r, attr) for r in self.runs]
        return sum(values) / len(values) if values else 0.0

    def share(self, predicate: Callable[[SeedResult], bool]) -> float:
        return sum(1 for r in self.runs if predicate(r)) / len(self.runs) if self.runs else 0.0


def _is_strategy(agent: BaseAgent) -> bool:
    return getattr(agent, "family", "") != "control"


def _compound(returns: Sequence[float]) -> float:
    equity = 1.0
    for r in returns:
        equity *= 1 + r
    return equity - 1


def walk_forward(period_returns: Dict[str, List[float]], strategies: Sequence[str],
                 control: Optional[str], lookback: int = LOOKBACK) -> Dict[str, float]:
    """Follow, each month, the strategy with the best trailing ``lookback``-month return.
    Only returns that were already realised are used to choose."""
    n = len(next(iter(period_returns.values())))
    start = min(lookback, n // 2)
    chosen, average, ctrl = [], [], []
    for t in range(start, n):
        best = max(strategies, key=lambda a: (_compound(period_returns[a][t - start:t]), a))
        chosen.append(period_returns[best][t])
        average.append(sum(period_returns[a][t] for a in strategies) / len(strategies))
        if control:
            ctrl.append(period_returns[control][t])
    return {"walk_forward": _compound(chosen), "average": _compound(average),
            "control": _compound(ctrl) if control else 0.0}


def _run_seed(params: ExperimentParams, agents: Sequence[BaseAgent], seed: int) -> SeedResult:
    ds = generate_dataset(params.n_stocks, params.n_periods, seed, signal=params.signal)
    bt = run_backtest(ds, agents, params.top_n, cost_bps=params.cost_bps)
    bench = compute_benchmarks(ds, params.top_n, seed, params.random_trials, params.cost_bps)
    market = bench.returns[MARKET]
    half = len(market) // 2
    names = [a.name for a in agents]
    strategies = [a.name for a in agents if _is_strategy(a)]
    control = next((a.name for a in agents if not _is_strategy(a)), None)
    alpha, alpha_in, alpha_out, tot = {}, {}, {}, {}
    for name in names:
        series = bt.period_returns[name]
        alpha[name] = alpha_stats(series, market).alpha
        alpha_in[name] = alpha_stats(series[:half], market[:half]).alpha
        alpha_out[name] = alpha_stats(series[half:], market[half:]).alpha
        tot[name] = total_return(series)
    ins = [alpha_in[a] for a in strategies]
    outs = [alpha_out[a] for a in strategies]
    winner = max(strategies, key=lambda a: (alpha_in[a], a))
    below = sum(1 for a in strategies if alpha_out[a] < alpha_out[winner])
    wf = walk_forward(bt.period_returns, strategies, control)
    return SeedResult(
        seed=seed, alpha=alpha, total_return=tot, alpha_in=alpha_in, alpha_out=alpha_out,
        rank_correlation=spearman(ins, outs),
        winner_out_percentile=below / max(len(strategies) - 1, 1),
        walk_forward=wf["walk_forward"], walk_forward_average=wf["average"],
        walk_forward_random_agent=wf["control"],
    )


def summarise(runs: Sequence[SeedResult], agents: Sequence[BaseAgent]) -> List[AgentSummary]:
    names = [a.name for a in agents]
    control = next((a.name for a in agents if not _is_strategy(a)), None)
    n = len(runs)
    stats: Dict[str, Dict[str, float]] = {}
    for name in names:
        alphas = [r.alpha[name] for r in runs]
        mean = sum(alphas) / n
        sd = statistics.stdev(alphas) if n > 1 else 0.0
        t = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
        ranks = []
        for r in runs:
            ordered = sorted(names, key=lambda a: (-r.alpha[a], a))
            ranks.append(ordered.index(name) + 1)
        stats[name] = {
            "mean": mean, "sd": sd, "t": t, "p": t_pvalue(t, n - 1) if sd > 0 else 1.0,
            "pos": sum(1 for a in alphas if a > 0) / n,
            "beats": (sum(1 for r in runs if r.total_return[name] > r.total_return[control])
                      / n) if control and name != control else 0.0,
            "rank": sum(ranks) / n,
            "in": sum(r.alpha_in[name] for r in runs) / n,
            "out": sum(r.alpha_out[name] for r in runs) / n,
        }
    adjusted = holm({a: s["p"] for a, s in stats.items()})
    out = []
    for name, s in stats.items():
        significant = adjusted[name] < SIGNIFICANCE
        verdict = ("Robust edge" if significant and s["t"] > 0 else
                   "Robustly worse" if significant and s["t"] < 0 else "Inconclusive")
        out.append(AgentSummary(name, s["mean"], s["sd"], s["t"], s["p"], adjusted[name],
                                s["pos"], s["beats"], s["rank"], s["in"], s["out"], verdict))
    out.sort(key=lambda x: (-x.mean_alpha, x.agent))
    return out


def run_monte_carlo(
    params: ExperimentParams, agents: Sequence[BaseAgent], seeds: Sequence[int],
    progress: Optional[Callable[[int, int], None]] = None,
) -> MonteCarloResult:
    """Run every agent on each seed's dataset. ``progress(done, total)`` is called per seed."""
    if len(seeds) < 2:
        raise ValueError("Monte Carlo needs at least 2 seeds")
    result = MonteCarloResult(seeds=list(seeds), params=params.to_dict())
    for i, seed in enumerate(seeds, start=1):
        result.runs.append(_run_seed(replace(params, seed=seed), agents, seed))
        if progress:
            progress(i, len(seeds))
    result.summaries = summarise(result.runs, agents)
    return result
