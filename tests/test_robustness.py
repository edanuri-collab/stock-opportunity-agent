"""Fair ranking: Monte Carlo, multiple-testing correction, walk-forward and costs."""
import pytest

from agents import default_agents
from agents.meta import ConsensusAgent
from data.sample_data import SignalConfig
from engine.experiment import ExperimentParams, run_experiment
from engine.robustness import run_monte_carlo, walk_forward
from engine.stats import holm, spearman, t_pvalue

SMALL = ExperimentParams(n_stocks=40, n_periods=12, top_n=5, random_trials=10)


@pytest.fixture(scope="module")
def mc():
    return run_monte_carlo(SMALL, default_agents() + [ConsensusAgent()], [1, 2, 3, 4])


def test_stats_reference_values():
    assert t_pvalue(2.0739, 22) == pytest.approx(0.05, abs=1e-4)
    assert t_pvalue(2.2281, 10) == pytest.approx(0.05, abs=1e-4)
    assert t_pvalue(0.0, 10) == pytest.approx(1.0)
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_monte_carlo_summary_shape(mc):
    names = {a.name for a in default_agents()} | {"Consensus"}
    assert [r.seed for r in mc.runs] == [1, 2, 3, 4]
    assert {s.agent for s in mc.summaries} == names
    means = [s.mean_alpha for s in mc.summaries]
    assert means == sorted(means, reverse=True)
    for s in mc.summaries:
        assert s.p_adjusted >= s.p_value - 1e-12
        assert 0 <= s.share_positive <= 1 and 1 <= s.mean_rank <= len(names)
        assert s.verdict in {"Robust edge", "Robustly worse", "Inconclusive"}
    for r in mc.runs:
        assert -1 <= r.rank_correlation <= 1 and 0 <= r.winner_out_percentile <= 1


def test_monte_carlo_is_deterministic(mc):
    again = run_monte_carlo(SMALL, default_agents() + [ConsensusAgent()], [1, 2, 3, 4])
    assert [s.mean_alpha for s in again.summaries] == [s.mean_alpha for s in mc.summaries]


def test_monte_carlo_needs_two_seeds():
    with pytest.raises(ValueError):
        run_monte_carlo(SMALL, default_agents(), [1])


def test_null_world_produces_no_robust_edge():
    """False-positive control: with no signal at all, nothing may pass the corrected test."""
    null = ExperimentParams(n_stocks=60, n_periods=18, top_n=10, random_trials=10,
                            signal=SignalConfig.null())
    result = run_monte_carlo(null, default_agents(), list(range(1, 9)))
    assert not [s.agent for s in result.summaries if s.verdict == "Robust edge"]


def test_walk_forward_uses_only_past_returns():
    # 26 months, so the full 12-month window is used (it shrinks to half on short runs)
    returns = {"A": [0.10] * 12 + [-0.50] + [0.0] * 13, "B": [0.0] * 12 + [0.40] + [0.0] * 13,
               "R": [0.0] * 26}
    wf = walk_forward(returns, ["A", "B"], "R", lookback=12)
    # at month 12 the trailing leader is A, so the selector eats A's -50% (no peeking at B)
    assert wf["walk_forward"] == pytest.approx(-0.5)
    assert wf["average"] == pytest.approx((1 - 0.05) - 1)
    assert wf["control"] == pytest.approx(0.0)


def test_costs_reduce_returns_and_are_reported():
    free = run_experiment(ExperimentParams(n_stocks=40, n_periods=8, random_trials=10,
                                           cost_bps=0))
    paid = run_experiment(ExperimentParams(n_stocks=40, n_periods=8, random_trials=10,
                                           cost_bps=30))
    f = {m.agent: m for m in free.metrics}
    for m in paid.metrics:
        assert m.total_return <= f[m.agent].total_return + 1e-12
        assert m.total_cost > 0 and 0 < m.turnover <= 1
        assert f[m.agent].total_cost == 0
    assert paid.experiment_id != free.experiment_id
    with pytest.raises(ValueError):
        ExperimentParams(cost_bps=-5)
