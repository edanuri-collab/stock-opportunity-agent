"""Look-ahead protection: what an agent sees, and what it must never be able to see."""
import ast
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from agents import default_agents
from data.models import MarketDataset, StockSnapshot
from data.sample_data import generate_dataset
from engine.backtest import decide, run_backtest

ROOT = Path(__file__).resolve().parent.parent
AGENT_FILES = sorted(p for p in (ROOT / "agents").glob("*_agent.py"))
FORBIDDEN_NAMES = {"MarketDataset", "outcome", "outcomes", "sample_data", "backtest", "benchmarks",
                   "regime_at", "snapshots_at", "snapshots_up_to", "experiment", "evaluation"}


class Spy:
    """Wraps an agent and records exactly what it is handed."""

    def __init__(self, inner):
        self.inner, self.name, self.seen = inner, inner.name, []

    def analyze(self, stock):
        self.seen.append(stock)
        return self.inner.analyze(stock)


def _names_in(path: Path) -> set:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update((node.module or "").split("."))
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.update(a.name.split("."))
    return names


# 1. every agent gets exactly the same dataset ------------------------------------------------

def test_all_agents_receive_identical_data(dataset):
    spies = [Spy(a) for a in default_agents()]
    assert len(spies) == 9
    run_backtest(dataset, spies, top_n=5)
    expected = [s for ts in dataset.timestamps for s in dataset.snapshots_at(ts)]
    first = spies[0].seen
    assert first == expected
    for spy in spies[1:]:
        assert spy.seen == first
        assert all(a is b for a, b in zip(spy.seen, first))  # literally the same objects


def test_decide_hands_same_snapshots_to_every_agent(dataset):
    spies = [Spy(a) for a in default_agents()]
    decide(spies, dataset.snapshots_at(dataset.timestamps[2]), 5)
    assert all(spy.seen == spies[0].seen for spy in spies)


# 2. an agent cannot reach future outcomes ----------------------------------------------------

def test_snapshot_holds_only_plain_values(dataset):
    for s in dataset.snapshots_at(dataset.timestamps[0]):
        assert all(isinstance(v, (str, float, datetime)) for v in vars(s).values())


def test_agent_code_never_references_outcome_machinery():
    assert AGENT_FILES
    for path in AGENT_FILES + [ROOT / "agents" / "base_agent.py", ROOT / "engine" / "scoring.py"]:
        assert not (_names_in(path) & FORBIDDEN_NAMES), path.name


def test_analyze_works_with_outcome_access_disabled(dataset, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("agent touched realised outcomes")

    monkeypatch.setattr(MarketDataset, "outcome", boom)
    for agent in default_agents():
        for s in dataset.snapshots_at(dataset.timestamps[0]):
            agent.analyze(s)


# 3. changing the future does not change the decision -----------------------------------------

def _tamper_future(dataset, cutoff):
    rng = random.Random(0)
    snaps = [s if s.timestamp <= cutoff else
             replace(s, price=s.price * 9, pe_ratio=1.0, momentum=5.0, catalyst_score=100.0,
                     short_interest=0.5, volume_ratio=4.0, distance_from_high=0.0)
             for ts in dataset.timestamps for s in dataset.snapshots_at(ts)]
    outcomes = {(s.ticker, s.timestamp): rng.uniform(-0.5, 5.0) for s in snaps}
    return MarketDataset(snaps, outcomes, {ts: "bear" for ts in dataset.timestamps})


@pytest.mark.parametrize("index", [0, 3, 7])
def test_decision_unchanged_by_future_data_and_outcomes(dataset, agents, index):
    cutoff = dataset.timestamps[index]
    tampered = _tamper_future(dataset, cutoff)
    assert tampered.version != dataset.version
    assert decide(agents, tampered.snapshots_at(cutoff), 10) == \
           decide(agents, dataset.snapshots_at(cutoff), 10)


def test_backtest_decisions_on_prefix_match_full_run(agents):
    full = generate_dataset(60, 10, seed=11)
    prefix = generate_dataset(60, 6, seed=11)
    r_full, r_prefix = run_backtest(full, agents, 5), run_backtest(prefix, agents, 5)
    cutoff = prefix.timestamps[-1].isoformat()
    assert [r for r in r_full.scores if r["timestamp"] <= cutoff] == r_prefix.scores


# 4. an agent cannot read another agent -------------------------------------------------------

def test_agents_are_stateless_and_config_read_only(agents):
    for agent in agents:
        assert vars(agent) == {}
        with pytest.raises(TypeError):
            agent.weights["value"] = 1.0


def test_agents_do_not_share_config_objects(agents):
    weights = [id(a.weights) for a in agents]
    assert len(set(weights)) == len(weights)


def test_agent_modules_do_not_import_each_other():
    for path in AGENT_FILES:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("agents"):
                assert node.module == "agents.base_agent", (path.name, node.module)


def test_result_independent_of_what_other_agents_did(dataset):
    snaps = dataset.snapshots_at(dataset.timestamps[0])
    extreme = [replace(s, pe_ratio=1.0, momentum=9.0) for s in snaps]
    alone = {a.name: [a.analyze(s) for s in snaps] for a in default_agents()}
    for target in default_agents():
        others = [a for a in default_agents() if a.name != target.name]
        for other in reversed(others):          # other agents run first, on very different data
            for s in extreme:
                other.analyze(s)
        assert [target.analyze(s) for s in snaps] == alone[target.name]


# 5. same dataset + timestamp + code = identical result ---------------------------------------

def _digest(results) -> str:
    return hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()


def test_same_inputs_give_identical_results(dataset):
    ts = dataset.timestamps[4]
    first = decide(default_agents(), dataset.snapshots_at(ts), 10)
    second = decide(default_agents(), dataset.snapshots_at(ts), 10)
    assert first == second and _digest(first) == _digest(second)


def test_identical_across_processes_and_hash_seeds():
    script = (
        "import json, hashlib;"
        "from agents import default_agents;"
        "from data.sample_data import generate_dataset;"
        "from engine.backtest import run_backtest;"
        "from engine.versioning import code_version;"
        "d = generate_dataset(30, 6, seed=3);"
        "r = run_backtest(d, default_agents(), 5);"
        "payload = json.dumps([d.version, code_version(), r.scores, r.period_returns], sort_keys=True);"
        "print(hashlib.sha256(payload.encode()).hexdigest())"
    )
    digests = set()
    for hash_seed in ("0", "123"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(ROOT)}
        out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True)
        digests.add(out.stdout.strip())
    assert len(digests) == 1
