"""Extended fundamentals: point-in-time, sane, weakly informative and exposed to specs."""
import math
import random
from dataclasses import fields, replace

import pandas as pd
import pytest

from agents.configurable import build_agent
from agents.registry import load_registry
from agents.spec import FIELDS, AgentSpec, f_score, field_value
from dashboard import i18n as he
from data.models import MarketDataset, StockSnapshot
from data.sample_data import SIGNAL_KEYS, SignalConfig, generate_dataset
from engine.backtest import decide

NEW_FIELDS = ("price_to_book", "roe", "roic", "gross_margin", "dividend_yield",
              "earnings_surprise", "momentum_12_1", "return_1m", "beta", "market_trend",
              "roe_change", "gross_margin_change", "debt_change")


def _all(dataset):
    return [s for ts in dataset.timestamps for s in dataset.snapshots_at(ts)]


def test_new_fields_exist_and_are_finite_floats(dataset):
    names = {f.name for f in fields(StockSnapshot)}
    assert set(NEW_FIELDS) <= names
    for s in _all(dataset):
        for f in NEW_FIELDS:
            value = getattr(s, f)
            assert isinstance(value, float) and math.isfinite(value), (s.ticker, f)


def test_new_fields_have_sane_ranges(dataset):
    for s in _all(dataset):
        assert 0.2 <= s.price_to_book <= 25
        assert 0.05 <= s.gross_margin <= 0.9
        assert 0 <= s.dividend_yield <= 0.12
        assert -1 <= s.beta <= 4
        if s.pe_ratio <= 0:
            assert s.roe <= 0 and s.roic <= 0


def test_dividend_payers_are_stable_per_stock(dataset):
    payers = {}
    for s in _all(dataset):
        assert payers.setdefault(s.ticker, s.dividend_yield > 0) == (s.dividend_yield > 0)
    assert 0.3 < sum(payers.values()) / len(payers) < 0.9


def test_price_fields_match_the_price_path():
    ds = generate_dataset(30, 16, seed=9)
    ticker = ds.tickers[0]
    prices = [next(s for s in ds.snapshots_at(ts) if s.ticker == ticker)
              for ts in ds.timestamps]
    for prev, cur in zip(prices[:-1], prices[1:]):
        assert cur.return_1m == pytest.approx(cur.price / prev.price - 1, abs=0.002)
    last = prices[-1]
    assert last.momentum_12_1 == pytest.approx(prices[-2].price / prices[-13].price - 1, abs=0.002)


def test_market_trend_is_the_same_for_every_stock(dataset):
    for ts in dataset.timestamps:
        assert len({s.market_trend for s in dataset.snapshots_at(ts)}) == 1


def test_beta_estimate_tracks_market_exposure():
    """Estimated beta is on average near 1 but varies: it is a noisy measurement."""
    ds = generate_dataset(120, 12, seed=4)
    betas = [s.beta for s in ds.snapshots_at(ds.timestamps[-1])]
    assert 0.5 < sum(betas) / len(betas) < 1.5 and len(set(betas)) > 100


def test_growing_n_periods_keeps_new_fields_identical():
    short = generate_dataset(30, 5, seed=3)
    long = generate_dataset(30, 9, seed=3)
    for ts in short.timestamps:
        assert short.snapshots_at(ts) == long.snapshots_at(ts)


def test_new_fields_only_weakly_predict_outcomes(dataset):
    ds = generate_dataset(120, 24, seed=11)
    rows = [{**{f: getattr(s, f) for f in NEW_FIELDS}, "outcome": ds.outcome(s.ticker, ts)}
            for ts in ds.timestamps for s in ds.snapshots_at(ts)]
    df = pd.DataFrame(rows)
    for f in NEW_FIELDS:
        if f == "market_trend":
            continue  # constant across stocks at T, measured over time rather than per stock
        ic = df[f].rank().corr(df["outcome"].rank())
        assert abs(ic) < 0.12, (f, ic)


def test_null_world_removes_every_loading_including_new_ones():
    null = SignalConfig.null()
    assert all(getattr(null, k) == 0 for k in SIGNAL_KEYS)
    assert {"reversal", "low_vol", "size"} <= set(SIGNAL_KEYS)
    default = SignalConfig()
    assert len({getattr(default, k) for k in SIGNAL_KEYS}) == 1  # same strength for all concepts


def test_every_new_field_is_usable_in_specs_and_labelled(dataset):
    s = dataset.snapshots_at(dataset.timestamps[0])[0]
    for f in NEW_FIELDS + ("f_score",):
        assert f in FIELDS and f in he.FIELDS
        assert math.isfinite(field_value(s, f))
    assert set(FIELDS) <= set(he.FIELDS)


def test_f_score_counts_passed_tests(dataset):
    s = dataset.snapshots_at(dataset.timestamps[0])[0]
    strong = replace(s, roe=0.2, free_cash_flow=1e9, roe_change=0.01, debt_change=-0.1,
                     gross_margin_change=0.01, revenue_growth=0.1, market_cap=1e10,
                     pe_ratio=20.0)  # FCF yield 10% > earnings yield 5%
    weak = replace(strong, roe=-0.1, free_cash_flow=-1e9, roe_change=-0.01, debt_change=0.1,
                   gross_margin_change=-0.01, revenue_growth=-0.1)
    assert f_score(strong) == 7 and f_score(weak) == 0


def test_new_field_rank_agent_ignores_future_tampering(dataset):
    agent = build_agent(AgentSpec.from_dict({
        "name": "Future Check", "family": "quality", "description": "x",
        "filters": [{"field": "roe", "op": ">", "value": 0}],
        "rank": [{"field": "f_score"}, {"field": "momentum_12_1"},
                 {"field": "price_to_book", "prefer": "low"}]}))
    cutoff = dataset.timestamps[4]
    rng = random.Random(0)
    snaps = [s if s.timestamp <= cutoff else replace(s, roe=-1.0, price_to_book=0.2, beta=3.0,
                                                     momentum_12_1=9.0, market_trend=-0.9)
             for s in _all(dataset)]
    tampered = MarketDataset(snaps, {(s.ticker, s.timestamp): rng.uniform(-1, 5) for s in snaps},
                             {ts: "bear" for ts in dataset.timestamps})
    assert decide([agent], tampered.snapshots_at(cutoff), 10) == \
           decide([agent], dataset.snapshots_at(cutoff), 10)


def test_upgraded_library_strategies_use_the_new_fields():
    agents = {a.name: a for a in load_registry().agents}
    graham = {f.field for f in agents["Graham Defensive"].spec.filters}
    assert {"price_to_book", "dividend_yield"} <= graham
    assert "roic" in {r.field for r in agents["Magic Formula"].spec.rank}
