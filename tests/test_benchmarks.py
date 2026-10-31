import math

import pytest

from engine.benchmarks import (
    BUY_HOLD, MARKET, RANDOM, BenchmarkSet, buy_and_hold_returns, compute_benchmarks,
    market_returns, random_selection,
)


def test_all_three_benchmarks_present(dataset, benchmarks):
    assert set(benchmarks.returns) == {MARKET, BUY_HOLD, RANDOM}
    for series in benchmarks.returns.values():
        assert len(series) == len(dataset.timestamps)


def test_market_is_cap_weighted(dataset):
    ts = dataset.timestamps[0]
    snaps = dataset.snapshots_at(ts)
    expected = (sum(s.market_cap * dataset.outcome(s.ticker, ts) for s in snaps)
                / sum(s.market_cap for s in snaps))
    assert market_returns(dataset)[0] == pytest.approx(expected)


def test_buy_and_hold_never_rebalances(dataset):
    final = {t: 1.0 for t in dataset.tickers}
    for ts in dataset.timestamps:
        for t in final:
            final[t] *= 1 + dataset.outcome(t, ts)
    expected_total = sum(final.values()) / len(final) - 1
    series = buy_and_hold_returns(dataset)
    equity = 1.0
    for r in series:
        equity *= 1 + r
    assert equity - 1 == pytest.approx(expected_total)


def test_buy_and_hold_first_period_is_equal_weight_mean(dataset):
    ts = dataset.timestamps[0]
    mean = sum(dataset.outcome(t, ts) for t in dataset.tickers) / len(dataset.tickers)
    assert buy_and_hold_returns(dataset)[0] == pytest.approx(mean)


def test_random_selection_is_seeded(dataset):
    a = random_selection(dataset, 10, seed=1, trials=20)
    b = random_selection(dataset, 10, seed=1, trials=20)
    c = random_selection(dataset, 10, seed=2, trials=20)
    assert a == b and a != c


def test_random_selection_tracks_universe_average(dataset):
    series, totals = random_selection(dataset, 10, seed=1, trials=200)
    assert len(totals) == 200
    for ts, r in zip(dataset.timestamps, series):
        universe = sum(dataset.outcome(t, ts) for t in dataset.tickers) / len(dataset.tickers)
        assert abs(r - universe) < 0.01


def test_percentile_of():
    b = BenchmarkSet(random_trial_totals=[0.0, 0.1, 0.2, 0.3])
    assert b.percentile_of(0.25) == 0.75
    assert b.percentile_of(-1) == 0.0 and b.percentile_of(9) == 1.0
    assert BenchmarkSet().percentile_of(1.0) == 0.0


def test_random_trials_must_be_positive(dataset):
    with pytest.raises(ValueError):
        compute_benchmarks(dataset, 10, seed=1, random_trials=0)


def test_benchmarks_deterministic(dataset):
    a = compute_benchmarks(dataset, 10, 5, 30)
    b = compute_benchmarks(dataset, 10, 5, 30)
    assert a.returns == b.returns and a.random_trial_totals == b.random_trial_totals
    assert all(math.isfinite(x) for s in a.returns.values() for x in s)
