import pytest

from dashboard.components import (
    agent_vs_benchmark_frame, benchmark_frame, drawdown_frame, equity_frame,
    experiment_summary_frame, leaderboard_frame, ranking_frame, regime_frame, stock_detail,
    verdict,
)
from engine.benchmarks import MARKET


def test_leaderboard_columns_and_order(experiment):
    df = leaderboard_frame(experiment)
    assert list(df.columns) == [
        "Rank", "Agent", "Total Return", "CAGR", "Win Rate", "Avg Win", "Avg Loss",
        "Expectancy", "Profit Factor", "Sharpe", "Max Drawdown", "Trades",
        "Alpha (vs market)", "Beta", "Alpha t-stat"]
    assert len(df) == 9 and list(df["Rank"]) == list(range(1, 10))
    assert df["Alpha (vs market)"].is_monotonic_decreasing
    assert (df["Trades"] == 120).all()


def test_benchmark_frame(experiment):
    df = benchmark_frame(experiment)
    assert set(df["Benchmark"]) == {MARKET, "Buy & Hold", "Random Selection"}


@pytest.mark.parametrize("bench", [MARKET, "Buy & Hold", "Random Selection"])
def test_agent_vs_benchmark(experiment, bench):
    df = agent_vs_benchmark_frame(experiment, bench)
    assert len(df) == 9
    assert set(df["Verdict"]) <= {"Alpha likely real", "Not distinguishable from luck",
                                  "Significantly worse than benchmark"}
    assert df["Beats Random Portfolios"].between(0, 1).all()
    with pytest.raises(KeyError):
        agent_vs_benchmark_frame(experiment, "nope")


def test_verdict_thresholds():
    assert verdict(2.0) == "Alpha likely real"
    assert verdict(1.99) == "Not distinguishable from luck"
    assert verdict(-2.5) == "Significantly worse than benchmark"


def test_equity_and_drawdown_frames(experiment):
    eq, dd = equity_frame(experiment), drawdown_frame(experiment)
    n = len(experiment.backtest.timestamps)
    assert eq.shape == (n + 1, 12) and dd.shape == eq.shape  # 9 agents + 3 benchmarks
    assert (eq.iloc[0] == 1.0).all() and (dd <= 0).all().all()
    assert eq.index.is_monotonic_increasing


def test_ranking_frame_sorted_and_filtered(result):
    ts = result.timestamps[-1].isoformat()
    df = ranking_frame(result.scores, ts)
    assert df["Opportunity Score"].is_monotonic_decreasing
    one = ranking_frame(result.scores, ts, agent="Balanced", limit=5)
    assert len(one) == 5 and set(one["Agent"]) == {"Balanced"}


def test_stock_detail(result, dataset):
    ts = result.timestamps[0].isoformat()
    d = stock_detail(result.scores, dataset.tickers[0], ts)
    assert len(d["rows"]) == 9
    assert d["favorite"]["opportunity_score"] >= d["least_favorite"]["opportunity_score"]
    mean = sum(r["opportunity_score"] for r in d["rows"]) / 9
    assert d["consensus_score"] == pytest.approx(mean)
    with pytest.raises(ValueError):
        stock_detail(result.scores, "NOPE", ts)


def test_regime_and_summary_frames(experiment):
    assert len(regime_frame(experiment)) == 9
    fields = set(experiment_summary_frame(experiment)["Field"])
    assert {"experiment_id", "dataset_version", "code_version", "timestamp"} <= fields
