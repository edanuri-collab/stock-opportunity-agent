import dataclasses

import pytest

from engine.scoring import (
    FACTORS, clamp, compute_components, scale, scale_inverse, top_factors,
    validate_weights, weighted_opportunity,
)


def test_clamp_scale():
    assert clamp(150) == 100 and clamp(-5) == 0
    assert scale(5, 0, 10) == 50 and scale(-1, 0, 10) == 0 and scale(99, 0, 10) == 100
    assert scale_inverse(2, 0, 10) == 80
    with pytest.raises(ValueError):
        scale(1, 3, 3)


def test_components_in_range(dataset):
    for s in dataset.snapshots_at(dataset.timestamps[0]):
        for name, value in compute_components(s).as_dict().items():
            assert 0 <= value <= 100, (s.ticker, name, value)


def test_components_deterministic(dataset):
    s = dataset.snapshots_at(dataset.timestamps[0])[0]
    assert compute_components(s) == compute_components(s)


def test_cheaper_stock_has_higher_value_score(dataset):
    base = dataset.snapshots_at(dataset.timestamps[0])[0]
    cheap = dataclasses.replace(base, pe_ratio=10.0, price_to_sales=1.0)
    rich = dataclasses.replace(base, pe_ratio=40.0, price_to_sales=10.0)
    assert compute_components(cheap).value > compute_components(rich).value


def test_negative_earnings_gets_no_pe_credit(dataset):
    base = dataset.snapshots_at(dataset.timestamps[0])[0]
    loss = dataclasses.replace(base, pe_ratio=-1.0)
    profit = dataclasses.replace(base, pe_ratio=8.0)
    assert compute_components(loss).value < compute_components(profit).value


def test_cash_burn_raises_risk(dataset):
    base = dataset.snapshots_at(dataset.timestamps[0])[0]
    burning = dataclasses.replace(base, free_cash_flow=-1e6)
    positive = dataclasses.replace(base, free_cash_flow=1e6)
    assert compute_components(burning).risk > compute_components(positive).risk


def test_validate_weights():
    validate_weights({"value": 0.5, "quality": 0.5})
    with pytest.raises(ValueError):
        validate_weights({"value": 0.5})
    with pytest.raises(ValueError):
        validate_weights({"bogus": 1.0})
    with pytest.raises(ValueError):
        validate_weights({"value": 1.5, "quality": -0.5})


def test_weighted_opportunity_and_top_factors(dataset):
    c = compute_components(dataset.snapshots_at(dataset.timestamps[0])[0])
    weights = {"value": 0.5, "momentum": 0.5}
    expected = clamp(0.5 * c.value + 0.5 * c.momentum - 0.2 * c.risk)
    assert weighted_opportunity(c, weights, 0.2) == pytest.approx(expected)
    assert set(top_factors(c, weights, 2)) == {"value", "momentum"}
    assert set(weights) <= set(FACTORS)
