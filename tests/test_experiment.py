import json
from datetime import datetime, timezone

import pytest

from agents import default_agents
from agents.value_agent import ValueAgent
from engine.experiment import ExperimentParams, run_experiment
from engine.versioning import agent_version, code_version

SMALL = ExperimentParams(seed=3, n_stocks=30, n_periods=8, top_n=5, random_trials=20)


def _strict_json(text: str):
    def reject(constant):
        raise ValueError(f"non-standard JSON constant {constant}")
    return json.loads(text, parse_constant=reject)


def test_experiment_record_has_all_required_fields(experiment):
    d = experiment.to_dict()
    for key in ("experiment_id", "dataset_version", "timestamp", "agent_versions",
                "parameters", "results"):
        assert d[key]
    assert experiment.experiment_id.startswith("exp-")
    assert len(experiment.agent_versions) == 9
    assert experiment.parameters["top_n"] == 10
    assert experiment.dataset_version == experiment.dataset.version


def test_results_contain_ranking_metrics_benchmarks_and_curves(experiment):
    r = experiment.results
    assert len(r["ranking"]) == 9 and r["ranking"][0] == experiment.metrics[0].agent
    assert set(r["agents"]) == set(r["ranking"])
    assert set(r["benchmarks"]) == {"Market (S&P proxy)", "Buy & Hold", "Random Selection"}
    assert set(r["equity_curves"]) == set(r["drawdown_curves"]) == set(r["ranking"]) | set(r["benchmarks"])
    n = len(experiment.backtest.timestamps)
    assert all(len(c) == n + 1 for c in r["equity_curves"].values())
    assert all(min(c) <= 0 for c in r["drawdown_curves"].values())


def test_json_is_strict_and_roundtrips(experiment):
    data = _strict_json(experiment.to_json())
    assert data["experiment_id"] == experiment.experiment_id
    assert data["results"]["ranking"] == experiment.results["ranking"]


def test_same_experiment_same_id_and_results():
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    a, b = run_experiment(SMALL, timestamp=t1), run_experiment(SMALL, timestamp=t2)
    assert a.experiment_id == b.experiment_id
    assert a.timestamp != b.timestamp and a.timestamp == t1.isoformat()
    assert a.to_dict()["results"] == b.to_dict()["results"]
    assert a.dataset_version == b.dataset_version


def test_different_parameters_change_id():
    other = ExperimentParams(seed=4, n_stocks=30, n_periods=8, top_n=5, random_trials=20)
    assert run_experiment(SMALL).experiment_id != run_experiment(other).experiment_id
    assert run_experiment(SMALL).dataset_version != run_experiment(other).dataset_version


def test_agent_version_tracks_configuration():
    class TweakedValue(ValueAgent):
        weights = {"value": 0.60, "quality": 0.20, "catalyst": 0.10, "momentum": 0.10}

    assert agent_version(ValueAgent()) == agent_version(ValueAgent())
    assert agent_version(ValueAgent()) != agent_version(TweakedValue())
    assert len({agent_version(a) for a in default_agents()}) == 9
    base = run_experiment(SMALL)
    agents = [TweakedValue() if a.name == "Value Hunter" else a for a in default_agents()]
    changed = run_experiment(SMALL, agents=agents)
    assert base.experiment_id != changed.experiment_id
    assert base.agent_versions["Balanced"] == changed.agent_versions["Balanced"]


def test_code_version_is_stable():
    assert code_version() == code_version()


def test_save_writes_json_named_after_id(experiment, tmp_path):
    path = experiment.save(tmp_path / "runs")
    assert path.name == f"{experiment.experiment_id}.json"
    assert _strict_json(path.read_text())["dataset_version"] == experiment.dataset_version


def test_invalid_rank_by_rejected():
    with pytest.raises(ValueError):
        ExperimentParams(rank_by="luck")


def test_pipeline_artifacts_available(experiment):
    assert experiment.dataset and experiment.backtest and experiment.benchmarks
    assert len(experiment.backtest.trades) == 9 * 10 * 12


def test_experiment_history_roundtrip(tmp_path):
    from engine.experiment import load_experiment_history

    assert load_experiment_history(tmp_path / "missing") == []
    a = run_experiment(SMALL, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
    b = run_experiment(ExperimentParams(seed=4, n_stocks=30, n_periods=8, top_n=5,
                                        random_trials=20),
                       timestamp=datetime(2026, 2, 1, tzinfo=timezone.utc))
    a.save(tmp_path), b.save(tmp_path)
    (tmp_path / "exp-broken.json").write_text("{not json", encoding="utf-8")
    history = load_experiment_history(tmp_path)
    assert [r["experiment_id"] for r in history] == [b.experiment_id, a.experiment_id]
    assert history[1]["parameters"]["seed"] == 3
