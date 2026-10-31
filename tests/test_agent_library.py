"""Spec-defined agents: format, validation, registry and the same look-ahead guarantees."""
import ast
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from agents.configurable import SpecAgent, build_agent
from agents.registry import (
    LIBRARY_DIR, fingerprint, load_registry, read_spec, save_spec, spec_of,
)
from agents.spec import AgentSpec, SpecError, percentile_ranks, to_toml
from data.models import MarketDataset
from engine.backtest import decide, run_backtest

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = {"MarketDataset", "outcome", "outcomes", "sample_data", "backtest", "benchmarks",
             "regime_at", "snapshots_at", "snapshots_up_to", "experiment", "evaluation"}

BASE = {"name": "Test Agent", "family": "value", "description": "test", "weights": {"value": 1}}


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def library(registry):
    return [a for a in registry.agents if isinstance(a, SpecAgent)]


# ---- registry & library --------------------------------------------------------------------

def test_library_loads_without_errors(registry, library):
    assert registry.errors == {}
    assert len(library) == len(list(LIBRARY_DIR.glob("*.toml"))) >= 7
    names = [a.name for a in registry.agents]
    assert len(names) == len(set(names))
    assert {a.origin for a in library} <= {"known", "original"}
    assert all(a.description and a.family for a in library)


def test_every_known_strategy_cites_a_source(library):
    assert all(a.reference for a in library if a.origin == "known")


def test_broken_and_duplicate_files_are_reported_not_fatal(tmp_path):
    (tmp_path / "bad.toml").write_text("name = ", encoding="utf-8")
    (tmp_path / "dup.toml").write_text(to_toml(AgentSpec.from_dict(
        {**BASE, "name": "Value Hunter"})), encoding="utf-8")
    save_spec(AgentSpec.from_dict(BASE), tmp_path)
    reg = load_registry([tmp_path])
    assert set(reg.errors) == {"bad.toml", "dup.toml"}
    assert "Test Agent" in [a.name for a in reg.agents]
    assert reg.sources["Test Agent"] == tmp_path.name


def test_save_and_fingerprint(tmp_path):
    before = fingerprint([tmp_path])
    path = save_spec(AgentSpec.from_dict(BASE), tmp_path)
    assert path.name == "test-agent.toml" and fingerprint([tmp_path]) != before
    with pytest.raises(SpecError):
        save_spec(AgentSpec.from_dict(BASE), tmp_path)
    assert read_spec(path) == AgentSpec.from_dict(BASE)


def test_toml_roundtrip_for_every_library_spec(library):
    for agent in library:
        assert AgentSpec.from_dict(tomllib.loads(to_toml(agent.spec))) == agent.spec


def test_core_agents_expressed_as_specs(registry):
    value = next(a for a in registry.agents if a.name == "Value Hunter")
    spec, exact = spec_of(value)
    assert not exact and spec.weights_map == pytest.approx(dict(value.weights))


# ---- validation ----------------------------------------------------------------------------

@pytest.mark.parametrize("change,message", [
    ({"name": ""}, "name"),
    ({"family": "astrology"}, "family"),
    ({"description": ""}, "description"),
    ({"risk_penalty": 2}, "risk_penalty"),
    ({"weights": {}}, "exactly one"),
    ({"rank": [{"field": "volatility"}]}, "exactly one"),
    ({"weights": {"magic": 1}}, "unknown weight"),
    ({"weights": {"value": -1, "quality": 2}}, "non-negative"),
    ({"weights": {"noise": 1}}, "unknown weight"),
    ({"filters": [{"field": "future_return", "op": ">", "value": 0}]}, "unknown field"),
    ({"filters": [{"field": "pe_ratio", "op": "==", "value": 0}]}, "op"),
    ({"filters": [{"field": "pe_ratio", "op": ">", "value": "x"}]}, "number"),
    ({"surprise": 1}, "unknown keys"),
])
def test_invalid_specs_rejected(change, message):
    with pytest.raises(SpecError, match=message):
        AgentSpec.from_dict({**BASE, **change})


def test_rank_spec_validation():
    raw = {**BASE, "weights": {}, "rank": [{"field": "volatility", "prefer": "sideways"}]}
    with pytest.raises(SpecError, match="prefer"):
        AgentSpec.from_dict(raw)
    raw["rank"] = [{"field": "volatility", "weight": 0}]
    with pytest.raises(SpecError, match="weight"):
        AgentSpec.from_dict(raw)


def test_weights_are_normalised():
    spec = AgentSpec.from_dict({**BASE, "weights": {"value": 3, "quality": 1}})
    assert spec.weights_map == {"quality": 0.25, "value": 0.75}


# ---- behaviour -----------------------------------------------------------------------------

def test_percentile_ranks():
    assert percentile_ranks([5.0]) == [50.0]
    assert percentile_ranks([3.0, 1.0, 2.0]) == [100.0, 0.0, 50.0]
    assert percentile_ranks([1.0, 1.0, 2.0]) == [25.0, 25.0, 100.0]


def test_filters_mark_ineligible_and_score_zero(dataset):
    agent = build_agent(AgentSpec.from_dict({
        **BASE, "filters": [{"field": "pe_ratio", "op": ">", "value": 0}]}))
    snaps = dataset.snapshots_at(dataset.timestamps[0])
    for s, r in zip(snaps, agent.analyze_universe(snaps)):
        assert r["eligible"] == (s.pe_ratio > 0)
        if not r["eligible"]:
            assert r["opportunity_score"] == 0 and "pe_ratio > 0" in r["reasoning"]


def test_rank_agent_prefers_low_volatility(dataset):
    agent = build_agent(AgentSpec.from_dict({**BASE, "weights": {},
                                             "rank": [{"field": "volatility",
                                                       "prefer": "low"}]}))
    snaps = dataset.snapshots_at(dataset.timestamps[0])
    res = {r["ticker"]: r["opportunity_score"] for r in agent.analyze_universe(snaps)}
    calmest = min(snaps, key=lambda s: s.volatility).ticker
    wildest = max(snaps, key=lambda s: s.volatility).ticker
    assert res[calmest] == 100 and res[wildest] == 0
    with pytest.raises(TypeError):
        agent.analyze(snaps[0])  # ranking needs the whole universe


def test_agent_that_filters_everything_stays_in_cash(dataset):
    agent = build_agent(AgentSpec.from_dict({
        **BASE, "filters": [{"field": "market_cap", "op": ">", "value": 1e15}]}))
    res = run_backtest(dataset, [agent], top_n=10)
    assert res.trades == [] and set(res.period_returns["Test Agent"]) == {0.0}


def test_backtest_only_picks_eligible_stocks(dataset, library):
    graham = next(a for a in library if a.name == "Graham Defensive")
    res = run_backtest(dataset, [graham], top_n=10)
    by_ts = {(r["timestamp"], r["ticker"]): r for r in res.scores}
    assert res.trades
    assert all(by_ts[(t.timestamp.isoformat(), t.ticker)]["eligible"] for t in res.trades)


# ---- the same look-ahead guarantees as the core agents ---------------------------------------

class UniverseSpy:
    def __init__(self, inner):
        self.inner, self.name, self.seen = inner, inner.name, []

    def analyze_universe(self, stocks):
        self.seen.append(tuple(stocks))
        return self.inner.analyze_universe(stocks)


def test_spec_code_never_references_outcome_machinery():
    for name in ("spec.py", "configurable.py", "registry.py"):
        tree = ast.parse((ROOT / "agents" / name).read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names |= {p for a in node.names for p in a.name.split(".")}
            if isinstance(node, ast.ImportFrom) and node.module:
                names |= set(node.module.split("."))
        assert not names & FORBIDDEN, (name, names & FORBIDDEN)


def test_library_agents_receive_identical_data(dataset, library):
    spies = [UniverseSpy(a) for a in library]
    run_backtest(dataset, spies, top_n=5)
    expected = [dataset.snapshots_at(ts) for ts in dataset.timestamps]
    for spy in spies:
        assert spy.seen == expected


@pytest.mark.parametrize("index", [0, 5])
def test_library_decisions_unchanged_by_future(dataset, library, index):
    import random

    cutoff = dataset.timestamps[index]
    rng = random.Random(0)
    snaps = [s if s.timestamp <= cutoff else replace(s, price=s.price * 9, pe_ratio=1.0,
                                                     momentum=5.0, volatility=0.01)
             for ts in dataset.timestamps for s in dataset.snapshots_at(ts)]
    tampered = MarketDataset(snaps, {(s.ticker, s.timestamp): rng.uniform(-1, 5) for s in snaps},
                             {ts: "bear" for ts in dataset.timestamps})
    assert decide(library, tampered.snapshots_at(cutoff), 10) == \
           decide(library, dataset.snapshots_at(cutoff), 10)


def test_library_agents_stateless_and_deterministic(dataset, library):
    snaps = dataset.snapshots_at(dataset.timestamps[3])
    first = decide(library, snaps, 10)
    for a in library:
        assert vars(a) == {}
        with pytest.raises(TypeError):
            a.weights["value"] = 1.0
    assert decide(load_registry().agents[9:], snaps, 10) == first
