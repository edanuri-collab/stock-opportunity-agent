import dataclasses
import json

import pytest

from agents import default_agents
from engine.scoring import validate_weights

KEYS = {"ticker", "agent", "value_score", "catalyst_score", "momentum_score", "risk_score",
        "opportunity_score", "reasoning", "timestamp"}


def test_nine_agents_with_unique_names(agents):
    assert len(agents) == 9  # 8 strategies + Random Agent control
    assert len({a.name for a in agents}) == 9


def test_weights_valid(agents):
    for a in agents:
        validate_weights(a.weights)


def test_output_schema_and_ranges(agents, dataset):
    snap = dataset.snapshots_at(dataset.timestamps[0])[0]
    for a in agents:
        out = a.analyze(snap)
        assert set(out) == KEYS
        assert out["ticker"] == snap.ticker and out["agent"] == a.name
        assert out["timestamp"] == snap.timestamp.isoformat()
        for k in ("value_score", "catalyst_score", "momentum_score", "risk_score", "opportunity_score"):
            assert 0 <= out[k] <= 100
        assert out["reasoning"]
        json.dumps(out)  # JSON-serialisable


def test_deterministic_across_instances(dataset):
    snaps = dataset.snapshots_at(dataset.timestamps[0])
    first = [a.analyze(s) for a in default_agents() for s in snaps]
    second = [a.analyze(s) for a in default_agents() for s in snaps]
    assert first == second


def test_agents_actually_disagree(agents, dataset):
    snaps = dataset.snapshots_at(dataset.timestamps[0])
    top = {a.name: max(snaps, key=lambda s: (a.analyze(s)["opportunity_score"], s.ticker)).ticker
           for a in agents}
    assert len(set(top.values())) > 1


def test_value_agent_prefers_cheap(agents, dataset):
    value = next(a for a in agents if a.name == "Value Hunter")
    base = dataset.snapshots_at(dataset.timestamps[0])[0]
    cheap = dataclasses.replace(base, pe_ratio=9.0, price_to_sales=0.8)
    rich = dataclasses.replace(base, pe_ratio=60.0, price_to_sales=15.0)
    assert value.analyze(cheap)["opportunity_score"] > value.analyze(rich)["opportunity_score"]


def test_squeeze_agent_prefers_high_short_interest(agents, dataset):
    squeeze = next(a for a in agents if a.name == "Short Squeeze Hunter")
    base = dataset.snapshots_at(dataset.timestamps[0])[0]
    high = dataclasses.replace(base, short_interest=0.30, volume_ratio=3.0, momentum=0.3)
    low = dataclasses.replace(base, short_interest=0.01, volume_ratio=1.0, momentum=0.0)
    assert squeeze.analyze(high)["opportunity_score"] > squeeze.analyze(low)["opportunity_score"]


def test_contrarian_needs_drawdown_and_quality(agents, dataset):
    contrarian = next(a for a in agents if a.name == "Contrarian")
    base = dataset.snapshots_at(dataset.timestamps[0])[0]
    strong = dict(revenue_growth=0.3, earnings_growth=0.35, debt_to_equity=0.2)
    fallen = dataclasses.replace(base, distance_from_high=0.5, **strong)
    at_high = dataclasses.replace(base, distance_from_high=0.02, **strong)
    assert contrarian.analyze(fallen)["opportunity_score"] > contrarian.analyze(at_high)["opportunity_score"]


def test_weights_are_read_only(agents):
    with pytest.raises(TypeError):
        agents[0].weights["value"] = 0.9


def test_agent_validates_config():
    from agents.base_agent import BaseAgent

    class Bad(BaseAgent):
        name = "bad"
        description = "bad"
        weights = {"value": 0.4}
        risk_penalty = 0.1

    with pytest.raises(ValueError):
        Bad()


def test_random_agent_ignores_every_feature(agents, dataset):
    rnd = next(a for a in agents if a.name == "Random Agent")
    base = dataset.snapshots_at(dataset.timestamps[0])[0]
    changed = dataclasses.replace(
        base, price=base.price * 3, pe_ratio=5.0, momentum=0.9, catalyst_score=99.0,
        volatility=1.5, short_interest=0.4, free_cash_flow=-1e9, distance_from_high=0.6)
    assert rnd.analyze(changed)["opportunity_score"] == rnd.analyze(base)["opportunity_score"]


def test_random_agent_scores_spread_across_range(agents, dataset):
    rnd = next(a for a in agents if a.name == "Random Agent")
    scores = [rnd.analyze(s)["opportunity_score"]
              for ts in dataset.timestamps for s in dataset.snapshots_at(ts)]
    assert min(scores) < 10 and max(scores) > 90
    assert 40 < sum(scores) / len(scores) < 60
