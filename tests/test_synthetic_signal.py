"""The synthetic world must not hand any agent a guaranteed win."""
import statistics
from dataclasses import replace

import pandas as pd
import pytest

from agents import default_agents
from data.sample_data import SIGNAL_KEYS, SignalConfig, generate_dataset
from engine.backtest import run_backtest
from engine.benchmarks import MARKET, compute_benchmarks
from engine.evaluation import compute_agent_metrics
from engine.scoring import compute_components

FACTOR_FIELDS = ("value", "quality", "momentum", "catalyst", "breakout", "squeeze", "contrarian")


def _pooled(dataset):
    rows = []
    for ts in dataset.timestamps:
        for s in dataset.snapshots_at(ts):
            row = compute_components(s).as_dict()
            row["outcome"] = dataset.outcome(s.ticker, ts)
            rows.append(row)
    return pd.DataFrame(rows)


def test_features_only_weakly_related_to_outcome(dataset):
    df = _pooled(dataset)
    for factor in FACTOR_FIELDS:
        # Spearman = Pearson on ranks (pandas' method="spearman" requires scipy)
        ic = df[factor].rank().corr(df["outcome"].rank())
        assert abs(ic) < 0.12, (factor, ic)


def test_outcome_noise_is_large(dataset):
    outcomes = [dataset.outcome(s.ticker, ts) for ts in dataset.timestamps
                for s in dataset.snapshots_at(ts)]
    assert statistics.pstdev(outcomes) > 0.05


def test_signal_loadings_are_equal_by_default():
    cfg = SignalConfig()
    loadings = {getattr(cfg, k) for k in SIGNAL_KEYS}
    assert len(loadings) == 1  # no agent concept is favoured by construction


def test_null_config_has_zero_loadings():
    cfg = SignalConfig.null()
    assert all(getattr(cfg, k) == 0.0 for k in SIGNAL_KEYS)


def test_outcome_parameters_do_not_leak_into_observed_features():
    """Fundamental features are identical whatever drives the outcome (no peeking)."""
    a = generate_dataset(40, 6, seed=5, signal=SignalConfig())
    b = generate_dataset(40, 6, seed=5, signal=SignalConfig.null())
    c = generate_dataset(40, 6, seed=5, signal=replace(SignalConfig(), noise_scale=3.0))
    fields = ("revenue_growth", "earnings_growth", "debt_to_equity", "catalyst_score",
              "short_interest", "days_to_cover", "sector", "shares_outstanding")
    for ts in a.timestamps:
        for sa, sb, sc in zip(a.snapshots_at(ts), b.snapshots_at(ts), c.snapshots_at(ts)):
            for f in fields:
                assert getattr(sa, f) == getattr(sb, f) == getattr(sc, f), f
    assert a.version != b.version  # ...while the outcomes do differ


def test_growing_n_periods_never_changes_earlier_periods():
    short = generate_dataset(30, 5, seed=3)
    long = generate_dataset(30, 9, seed=3)
    for ts in short.timestamps:
        assert short.snapshots_at(ts) == long.snapshots_at(ts)
        assert short.regime_at(ts) == long.regime_at(ts)
        for s in short.snapshots_at(ts):
            assert short.outcome(s.ticker, ts) == long.outcome(s.ticker, ts)


def test_null_world_produces_no_systematic_alpha():
    """With no signal, agents' alpha is luck: centred on zero and rarely 'significant'."""
    alphas, tstats = [], []
    for seed in range(1, 7):
        ds = generate_dataset(100, 18, seed=seed, signal=SignalConfig.null())
        res = run_backtest(ds, default_agents(), top_n=10)
        bench = compute_benchmarks(ds, 10, seed, 30)
        for m in compute_agent_metrics(res, bench):
            alphas.append(m.alpha[MARKET].alpha)
            tstats.append(m.alpha[MARKET].t_stat)
    assert abs(statistics.mean(alphas)) < 0.05
    assert sum(1 for t in tstats if abs(t) > 2) / len(tstats) < 0.25


def test_no_agent_wins_in_every_world():
    """Across seeds the best agent changes: nobody is predetermined to win."""
    winners = set()
    for seed in range(1, 7):
        ds = generate_dataset(100, 18, seed=seed)
        res = run_backtest(ds, default_agents(), top_n=10)
        bench = compute_benchmarks(ds, 10, seed, 30)
        winners.add(compute_agent_metrics(res, bench)[0].agent)
    assert len(winners) > 1
