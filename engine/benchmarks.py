"""Benchmarks to separate real alpha from market beta and luck.

* Market (S&P proxy): cap-weighted return of the whole universe, rebalanced every period.
* Buy & Hold: equal-weight universe bought at the first timestamp and never rebalanced.
* Random Selection: top_n random stocks per period, repeated over many seeded trials; the
  series is the average trial, and every trial's total return is kept so an agent can be
  ranked against the distribution of "monkeys".
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from data.models import MarketDataset

MARKET = "Market (S&P proxy)"
BUY_HOLD = "Buy & Hold"
RANDOM = "Random Selection"


@dataclass
class BenchmarkSet:
    returns: Dict[str, List[float]] = field(default_factory=dict)  # name -> period returns
    random_trial_totals: List[float] = field(default_factory=list)  # total return per trial

    def percentile_of(self, total_return: float) -> float:
        """Share of random portfolios (0-1) that the given total return beats."""
        if not self.random_trial_totals:
            return 0.0
        return sum(1 for t in self.random_trial_totals if t < total_return) / len(
            self.random_trial_totals)


def _compound(returns: List[float]) -> float:
    equity = 1.0
    for r in returns:
        equity *= 1 + r
    return equity - 1


def market_returns(dataset: MarketDataset) -> List[float]:
    out = []
    for ts in dataset.timestamps:
        snaps = dataset.snapshots_at(ts)
        total_cap = sum(s.market_cap for s in snaps)
        out.append(sum(s.market_cap * dataset.outcome(s.ticker, ts) for s in snaps) / total_cap)
    return out


def buy_and_hold_returns(dataset: MarketDataset) -> List[float]:
    values = {t: 1.0 for t in dataset.tickers}
    previous_total, out = float(len(values)), []
    for ts in dataset.timestamps:
        for t in values:
            values[t] *= 1 + dataset.outcome(t, ts)
        total = sum(values.values())
        out.append(total / previous_total - 1)
        previous_total = total
    return out


def random_selection(
    dataset: MarketDataset, top_n: int, seed: int, trials: int, cost_bps: float = 0.0
) -> tuple[List[float], List[float]]:
    """Random portfolios pay the same trading costs as the agents (a fair comparison)."""
    if trials < 1:
        raise ValueError("trials must be >= 1")
    rng = random.Random(seed * 1_000_003 + 17)
    tickers = list(dataset.tickers)
    k = min(top_n, len(tickers))
    per_trial: List[List[float]] = [[] for _ in range(trials)]
    held: List[set] = [set() for _ in range(trials)]
    for ts in dataset.timestamps:
        for i, trial in enumerate(per_trial):
            picks = rng.sample(tickers, k)
            # equal weights: traded share = 2 x (share of names replaced)
            changed = k - len(held[i] & set(picks)) if held[i] else k
            traded = (2 * changed / k) if held[i] else 1.0
            held[i] = set(picks)
            trial.append(sum(dataset.outcome(t, ts) for t in picks) / k
                         - traded * cost_bps / 10_000)
    mean_series = [sum(trial[i] for trial in per_trial) / trials
                   for i in range(len(dataset.timestamps))]
    return mean_series, [_compound(trial) for trial in per_trial]


def compute_benchmarks(
    dataset: MarketDataset, top_n: int, seed: int, random_trials: int = 200,
    cost_bps: float = 0.0,
) -> BenchmarkSet:
    """Market and Buy & Hold barely trade, so they are modelled without costs."""
    random_series, totals = random_selection(dataset, top_n, seed, random_trials, cost_bps)
    return BenchmarkSet(
        returns={
            MARKET: market_returns(dataset),
            BUY_HOLD: buy_and_hold_returns(dataset),
            RANDOM: random_series,
        },
        random_trial_totals=totals,
    )
