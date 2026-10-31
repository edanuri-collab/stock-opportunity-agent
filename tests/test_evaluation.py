import math

import pytest

from engine.benchmarks import MARKET
from engine.evaluation import (
    AlphaStats, alpha_stats, average_loss, average_win, cagr, compute_agent_metrics,
    drawdown_curve, equity_curve, expectancy, max_drawdown, profit_factor, regime_breakdown,
    sharpe, total_return, win_rate, benchmark_metrics,
)


def test_trade_level_metrics():
    r = [0.10, -0.05, 0.02, -0.01]
    assert win_rate(r) == 0.5
    assert expectancy(r) == pytest.approx(0.015)
    assert average_win(r) == pytest.approx(0.06)
    assert average_loss(r) == pytest.approx(-0.03)
    for fn in (win_rate, expectancy, average_win, average_loss):
        assert fn([]) == 0.0
    assert average_loss([0.1]) == 0.0 and average_win([-0.1]) == 0.0


def test_profit_factor():
    assert profit_factor([0.10, -0.05, 0.05, -0.05]) == pytest.approx(1.5)
    assert profit_factor([0.1, 0.2]) == math.inf
    assert profit_factor([]) == 0.0 and profit_factor([-0.1]) == 0.0


def test_equity_total_return_cagr():
    assert equity_curve([0.1, -0.1]) == pytest.approx([1.0, 1.1, 0.99])
    assert total_return([0.1, 0.1]) == pytest.approx(0.21)
    assert cagr([0.01] * 12) == pytest.approx(1.01 ** 12 - 1)
    assert cagr([0.1] * 6) == pytest.approx(1.1 ** 12 - 1)  # 6 months compounded to a year
    assert cagr([]) == 0.0 and cagr([-1.0]) == -1.0


def test_sharpe():
    assert sharpe([0.01, 0.03]) == pytest.approx(0.02 / math.sqrt(0.0002) * math.sqrt(12))
    assert sharpe([0.01, 0.01, 0.01]) == 0.0 and sharpe([0.05]) == 0.0


def test_drawdown():
    assert drawdown_curve([0.10, -0.20]) == pytest.approx([0.0, 0.0, -0.2])
    assert max_drawdown([0.10, 0.10]) == 0.0
    assert max_drawdown([0.10, -0.20]) == pytest.approx(0.2)
    assert max_drawdown([]) == 0.0


def test_alpha_recovers_known_relationship():
    bench = [0.02, -0.01, 0.03, 0.00, -0.02, 0.04]
    agent = [0.01 + 1.5 * b for b in bench]
    s = alpha_stats(agent, bench)
    assert s.alpha == pytest.approx(0.12)
    assert s.beta == pytest.approx(1.5)


def test_alpha_of_benchmark_against_itself_is_zero():
    bench = [0.02, -0.01, 0.03, 0.00, -0.02, 0.04]
    s = alpha_stats(bench, bench)
    assert s.alpha == pytest.approx(0.0, abs=1e-12) and s.beta == pytest.approx(1.0)
    assert s.excess_cagr == pytest.approx(0.0)


def test_alpha_t_stat_with_noise():
    bench = [0.02, -0.01, 0.03, 0.00, -0.02, 0.04, 0.01, -0.03]
    noise = [0.004, -0.003, 0.001, -0.002, 0.003, -0.004, 0.002, -0.001]
    agent = [0.01 + b + n for b, n in zip(bench, noise)]
    s = alpha_stats(agent, bench)
    assert s.alpha > 0 and s.t_stat > 2
    assert alpha_stats([-x for x in agent], bench).t_stat < -2


def test_alpha_edge_cases():
    assert alpha_stats([0.1], [0.1]) == AlphaStats(0.0, 0.0, 0.0, pytest.approx(0.0))
    flat = alpha_stats([0.01, 0.02, 0.03], [0.0, 0.0, 0.0])
    assert flat.beta == 0.0 and flat.alpha == pytest.approx(0.24)
    with pytest.raises(ValueError):
        alpha_stats([0.1], [0.1, 0.2])


def test_agent_metrics(result, benchmarks):
    metrics = compute_agent_metrics(result, benchmarks)
    assert len(metrics) == 9
    n = len(result.timestamps)
    for m in metrics:
        series = result.period_returns[m.agent]
        assert m.num_trades == result.top_n * n
        assert m.total_return == pytest.approx(total_return(series))
        assert m.cagr == pytest.approx(cagr(series))
        assert 0 <= m.win_rate <= 1 and 0 <= m.max_drawdown <= 1
        assert m.average_win >= 0 >= m.average_loss
        assert 0 <= m.percentile_vs_random <= 1
        assert set(m.alpha) == set(benchmarks.returns)
    alphas = [m.alpha[MARKET].alpha for m in metrics]
    assert alphas == sorted(alphas, reverse=True)  # ranked by alpha by default


@pytest.mark.parametrize("rank_by", ["sharpe", "total_return", "cagr", "expectancy"])
def test_ranking_options(result, benchmarks, rank_by):
    values = [getattr(m, rank_by) for m in compute_agent_metrics(result, benchmarks, rank_by)]
    assert values == sorted(values, reverse=True)


def test_invalid_rank_by(result, benchmarks):
    with pytest.raises(ValueError):
        compute_agent_metrics(result, benchmarks, "luck")


def test_benchmark_metrics(benchmarks):
    out = benchmark_metrics(benchmarks)
    assert out[MARKET]["alpha_vs_market"] == pytest.approx(0.0, abs=1e-9)
    assert out[MARKET]["beta_vs_market"] == pytest.approx(1.0)


def test_regime_breakdown(result):
    out = regime_breakdown(result)
    assert set(out) == set(result.period_returns)
    assert all(set(v) <= {"bull", "bear", "sideways"} for v in out.values())
