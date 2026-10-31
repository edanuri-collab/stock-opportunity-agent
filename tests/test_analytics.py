"""Dashboard view-model layer: numbers each screen shows."""
from dataclasses import replace

import pytest

from dashboard import analytics as an
from engine.benchmarks import MARKET


def test_data_source_is_flagged_synthetic(experiment):
    s = an.data_source_status(experiment)
    assert s.is_synthetic and not s.live_feed and s.label == "SYNTHETIC"
    assert s.dataset_version == experiment.dataset_version
    assert s.as_of == experiment.dataset.timestamps[-1]
    assert s.signal_mode == "Weak signal"


def test_overview_stats_consistent(experiment):
    s = an.overview_stats(experiment)
    ds = experiment.dataset
    assert s["stocks_scanned"] == len(ds.tickers)
    assert s["snapshots_analyzed"] == len(ds.tickers) * len(ds.timestamps)
    assert s["active_agents"] == 9 and s["strategy_agents"] == 8
    assert 10 <= s["opportunities"] <= 8 * 10  # top-10 per strategy, overlap allowed
    assert s["best_agent"] != "Random Agent"
    assert 0 <= s["beat_market"] <= 8 and 0 <= s["beat_random_agent"] <= 8


def test_picks_exclude_control(experiment):
    picks = an.picks_at(experiment, experiment.dataset.timestamps[-1])
    assert all("Random Agent" not in agents for agents in picks.values())


def test_leaderboard_has_every_agent_and_status(experiment):
    df = an.agent_leaderboard(experiment)
    assert len(df) == 9 and set(df["Status"]) <= {"Outperforming", "Underperforming",
                                                   "Inconclusive"}
    assert df.loc[df["Agent"] == "Random Agent", "Type"].item() == "Control"
    assert df.loc[df["Type"] == "Strategy", "Opportunity Score"].between(0, 100).all()
    assert df.loc[df["Type"] == "Control", "Opportunity Score"].isna().all()


def test_performance_status_thresholds():
    assert an.performance_status(2.0) == "Outperforming"
    assert an.performance_status(-2.0) == "Underperforming"
    assert an.performance_status(1.5) == "Inconclusive"


def test_opportunities_frame(experiment):
    ts = experiment.dataset.timestamps[-1]
    df = an.opportunities_frame(experiment, ts)
    assert list(df.columns) == an.OPPORTUNITY_COLUMNS
    assert len(df) == len(experiment.dataset.tickers)
    assert df["Opportunity Score"].is_monotonic_decreasing
    assert "Random Agent" not in set(df["Top Agent"])
    picked = df[df["Picked By"] > 0]
    assert len(picked) == an.overview_stats(experiment)["opportunities"]


def test_filter_opportunities(experiment):
    df = an.opportunities_frame(experiment, experiment.dataset.timestamps[-1])
    t = df["Ticker"].iloc[0]
    assert list(an.filter_opportunities(df, query=t.lower())["Ticker"]).count(t) == 1
    sector = df["Sector"].iloc[0]
    assert set(an.filter_opportunities(df, sectors=[sector])["Sector"]) == {sector}
    assert (an.filter_opportunities(df, min_score=50)["Opportunity Score"] >= 50).all()
    assert (an.filter_opportunities(df, max_risk=30)["Risk Score"] <= 30).all()
    assert (an.filter_opportunities(df, picked_only=True)["Picked By"] > 0).all()
    agent = df.loc[df["Picked By"] > 0, "Agents"].iloc[0].split(", ")[0]
    assert an.filter_opportunities(df, agent=agent)["Agents"].str.contains(agent).all()
    assert an.filter_opportunities(df, query="zzzz-none").empty


def test_agent_views_and_ranks(experiment):
    ts = experiment.dataset.timestamps[2]
    ticker = experiment.dataset.tickers[0]
    v = an.agent_views(experiment, ticker, ts)
    assert len(v) == 9 and v["Opportunity Score"].is_monotonic_decreasing
    picked_agents = {t.agent for t in experiment.backtest.trades
                     if t.timestamp == ts and t.ticker == ticker}
    assert set(v.loc[v["Picked"], "Agent"]) == picked_agents


def test_score_and_price_history_never_go_past_as_of(experiment):
    ts = experiment.dataset.timestamps[4]
    ticker = experiment.dataset.tickers[1]
    h = an.score_history(experiment, ticker, ts)
    assert h["Date"].max() == ts and h["Date"].nunique() == 5
    assert an.price_history(experiment, ticker, ts)["Date"].max() == ts


def test_risk_flags_rules(dataset):
    s = dataset.snapshots_at(dataset.timestamps[0])[0]
    calm = replace(s, volatility=0.2, debt_to_equity=0.3, free_cash_flow=1e8, pe_ratio=15.0,
                   market_cap=5e10, short_interest=0.02, distance_from_high=0.05,
                   revenue_growth=0.1)
    assert an.risk_flags(calm) == []
    wild = replace(calm, volatility=0.9, debt_to_equity=2.0, free_cash_flow=-1e7,
                   pe_ratio=-3.0, market_cap=5e8, short_interest=0.3, distance_from_high=0.5,
                   revenue_growth=-0.1)
    assert len(an.risk_flags(wild)) == 8


def test_financial_metrics_and_factor_profile(dataset):
    s = dataset.snapshots_at(dataset.timestamps[0])[0]
    m = an.financial_metrics(s)
    assert {"P/E", "Market Cap", "Short Interest", "Days to Cover"} <= set(m)
    prof = an.factor_profile(s)
    assert "Noise" not in set(prof["Factor"]) and prof["Score"].between(0, 100).all()


def test_long_curves_kinds(experiment):
    from dashboard.components import equity_frame

    long = an.long_curves(equity_frame(experiment), "Equity")
    kinds = dict(zip(long["Series"], long["Kind"]))
    assert kinds[MARKET] == "Benchmark" and kinds["Random Agent"] == "Control"
    assert kinds["Value Hunter"] == "Agent"


def test_history_frame(experiment):
    record = experiment.to_dict()
    df = an.history_frame([record])
    row = df.iloc[0]
    assert row["Experiment"] == experiment.experiment_id and row["Seed"] == 7
    assert row["Signal"] == "Weak" and row["Best Agent"] != "Random Agent"
    assert len(row["Agent Version"]) == 8
    res = an.record_results_frame(record)
    assert len(res) == 9 and list(res["Agent"]) == record["results"]["ranking"]


@pytest.mark.parametrize("x,expected", [(2.5e12, "$2.5T"), (3.2e9, "$3.2B"), (4.5e6, "$4.5M"),
                                        (900, "$900")])
def test_format_money(x, expected):
    assert an.format_money(x) == expected
