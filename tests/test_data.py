import dataclasses
from datetime import datetime

import pytest

from data.models import StockSnapshot
from data.sample_data import generate_dataset

REQUIRED = {
    "ticker", "company_name", "sector", "industry", "price", "market_cap", "shares_outstanding",
    "revenue_growth", "earnings_growth", "free_cash_flow", "debt_to_equity", "pe_ratio",
    "price_to_sales", "relative_strength", "momentum", "volume_ratio", "volatility",
    "catalyst_score", "sentiment_score", "short_interest", "days_to_cover", "timestamp",
}


def test_at_least_100_unique_stocks(dataset):
    assert len(dataset.tickers) >= 100
    assert len(set(dataset.tickers)) == len(dataset.tickers)


def test_required_fields_exist():
    assert REQUIRED <= {f.name for f in dataclasses.fields(StockSnapshot)}


def test_new_fields_are_populated_and_sane(dataset):
    snaps = dataset.snapshots_at(dataset.timestamps[0])
    assert len({s.sector for s in snaps}) > 3
    for s in snaps:
        assert s.sector and s.industry
        assert s.shares_outstanding > 0
        assert s.market_cap == pytest.approx(s.price * s.shares_outstanding, rel=1e-3)
        assert 0.005 <= s.short_interest <= 0.5
        assert 0.1 <= s.days_to_cover <= 30


def test_sector_and_industry_stable_per_ticker(dataset):
    seen = {}
    for ts in dataset.timestamps:
        for s in dataset.snapshots_at(ts):
            assert seen.setdefault(s.ticker, (s.sector, s.industry)) == (s.sector, s.industry)


def test_every_timestamp_has_every_ticker(dataset):
    for ts in dataset.timestamps:
        assert [s.ticker for s in dataset.snapshots_at(ts)] == list(dataset.tickers)


def test_timestamps_sorted_and_unique(dataset):
    assert list(dataset.timestamps) == sorted(set(dataset.timestamps))


def test_deterministic_for_same_seed():
    a = generate_dataset(n_stocks=20, n_periods=3, seed=1)
    b = generate_dataset(n_stocks=20, n_periods=3, seed=1)
    assert a.version == b.version
    assert [s.to_dict() for t in a.timestamps for s in a.snapshots_at(t)] == \
           [s.to_dict() for t in b.timestamps for s in b.snapshots_at(t)]


def test_different_seed_differs():
    a = generate_dataset(n_stocks=20, n_periods=3, seed=1)
    b = generate_dataset(n_stocks=20, n_periods=3, seed=2)
    assert a.version != b.version


def test_value_ranges(dataset):
    for ts in dataset.timestamps:
        for s in dataset.snapshots_at(ts):
            assert 0 <= s.relative_strength <= 100
            assert 0 <= s.catalyst_score <= 100 and 0 <= s.sentiment_score <= 100
            assert s.market_cap > 0 and s.price > 0 and s.volatility > 0
            assert 0 <= s.distance_from_high <= 0.9


def test_snapshot_has_no_outcome_fields_and_is_frozen(dataset):
    names = {f.name for f in dataclasses.fields(StockSnapshot)}
    assert not any("forward" in n or "outcome" in n or "future" in n for n in names)
    snap = dataset.snapshots_at(dataset.timestamps[0])[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.price = 1.0


def test_snapshots_up_to_never_returns_future(dataset):
    cutoff = dataset.timestamps[3]
    assert max(s.timestamp for s in dataset.snapshots_up_to(cutoff)) == cutoff


def test_unknown_timestamp_returns_nothing(dataset):
    assert dataset.snapshots_at(datetime(1999, 1, 1)) == ()


def test_invalid_arguments():
    with pytest.raises(ValueError):
        generate_dataset(n_stocks=0)
