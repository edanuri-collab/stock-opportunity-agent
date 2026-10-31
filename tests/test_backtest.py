import copy

import pytest

from agents import default_agents
from engine.backtest import decide, run_backtest


def test_trade_counts(result, dataset):
    assert len(result.timestamps) == len(dataset.timestamps)
    assert len(result.trades) == 9 * 10 * len(dataset.timestamps)
    assert len(result.scores) == 9 * len(dataset.tickers) * len(dataset.timestamps)
    for returns in result.period_returns.values():
        assert len(returns) == len(dataset.timestamps)


def test_deterministic(dataset, result):
    again = run_backtest(dataset, default_agents(), top_n=10)
    assert again.period_returns == result.period_returns
    assert again.scores == result.scores


def test_no_look_ahead_scores_ignore_outcomes(dataset, result):
    """Rewriting every realised outcome must not change a single agent decision."""
    tampered = copy.copy(dataset)
    tampered._outcomes = {k: 5.0 for k in dataset._outcomes}
    again = run_backtest(tampered, default_agents(), top_n=10)
    assert again.scores == result.scores
    assert [(t.agent, t.ticker, t.timestamp) for t in again.trades] == \
           [(t.agent, t.ticker, t.timestamp) for t in result.trades]
    assert again.period_returns != result.period_returns  # outcomes only affect results


def test_decide_only_uses_given_snapshots(dataset, agents):
    first = dataset.snapshots_at(dataset.timestamps[0])
    out = decide(agents, first, 5)
    assert {r["timestamp"] for rs in out.values() for r in rs} == {first[0].timestamp.isoformat()}


def test_period_ends_follow_timestamps(result):
    assert result.period_ends[:-1] == result.timestamps[1:]
    assert result.period_ends[-1] > result.timestamps[-1]


def test_equity_curve_compounds(result):
    name = next(iter(result.period_returns))
    curve = result.equity_curve(name)
    expected = 1.0
    for r in result.period_returns[name]:
        expected *= 1 + r
    assert curve[-1] == pytest.approx(expected)


def test_picks_are_top_ranked(result):
    ts = result.timestamps[0].isoformat()
    agent = "Value Hunter"
    all_scores = sorted((r["opportunity_score"] for r in result.scores
                         if r["agent"] == agent and r["timestamp"] == ts), reverse=True)
    picked = sorted((t.opportunity_score for t in result.trades
                     if t.agent == agent and t.timestamp.isoformat() == ts), reverse=True)
    assert picked == all_scores[:10]


def test_invalid_inputs(dataset, agents):
    with pytest.raises(ValueError):
        run_backtest(dataset, agents, top_n=0)
    with pytest.raises(ValueError):
        run_backtest(dataset, [agents[0], agents[0]], top_n=3)
