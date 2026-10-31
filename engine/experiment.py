"""Experiment: Dataset -> Agents -> Decisions -> Simulated Trades -> Returns -> Metrics -> Ranking.

An experiment is fully described by its parameters, the dataset version and the code/agent
versions. ``experiment_id`` is derived from exactly those, so re-running the same experiment
yields the same id and the same results; only ``timestamp`` (when it was run) differs.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agents import default_agents
from agents.base_agent import BaseAgent
from data.models import MarketDataset
from data.sample_data import SignalConfig, generate_dataset
from engine.backtest import BacktestResult, run_backtest
from engine.benchmarks import MARKET, BenchmarkSet, compute_benchmarks
from engine.evaluation import (
    RANK_KEYS, AgentMetrics, benchmark_metrics, compute_agent_metrics, drawdown_curve,
    equity_curve,
)
from engine.versioning import agent_version, code_version


@dataclass(frozen=True)
class ExperimentParams:
    seed: int = 42
    n_stocks: int = 120
    n_periods: int = 24
    top_n: int = 10
    random_trials: int = 200
    rank_by: str = "alpha"
    signal: SignalConfig = field(default_factory=SignalConfig)

    def __post_init__(self) -> None:
        if self.rank_by not in RANK_KEYS:
            raise ValueError(f"rank_by must be one of {RANK_KEYS}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _json_safe(value: Any) -> Any:
    """Make results strict-JSON: infinity / NaN become None."""
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class Experiment:
    experiment_id: str
    dataset_version: str
    timestamp: str                       # when the experiment was run (UTC, ISO 8601)
    code_version: str
    agent_versions: Dict[str, str]
    parameters: Dict[str, Any]
    results: Dict[str, Any]              # JSON-safe: ranking, metrics, benchmarks, curves
    # In-memory artefacts for the dashboard; not serialised.
    dataset: Optional[MarketDataset] = field(default=None, repr=False, compare=False)
    backtest: Optional[BacktestResult] = field(default=None, repr=False, compare=False)
    benchmarks: Optional[BenchmarkSet] = field(default=None, repr=False, compare=False)
    metrics: List[AgentMetrics] = field(default_factory=list, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "timestamp": self.timestamp,
            "code_version": self.code_version,
            "agent_versions": self.agent_versions,
            "parameters": self.parameters,
            "results": self.results,
        }

    def to_json(self) -> str:
        return json.dumps(_json_safe(self.to_dict()), indent=2, sort_keys=True)

    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{self.experiment_id}.json"
        target.write_text(self.to_json(), encoding="utf-8")
        return target


def _experiment_id(params: Dict[str, Any], dataset_version: str, code_ver: str,
                   agent_versions: Dict[str, str]) -> str:
    payload = json.dumps({"p": params, "d": dataset_version, "c": code_ver,
                          "a": agent_versions}, sort_keys=True)
    return "exp-" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def run_experiment(
    params: ExperimentParams = ExperimentParams(),
    agents: Optional[Sequence[BaseAgent]] = None,
    dataset: Optional[MarketDataset] = None,
    timestamp: Optional[datetime] = None,
) -> Experiment:
    """Run the whole pipeline once. Pass ``dataset`` to reuse prepared data."""
    agents = list(agents) if agents is not None else default_agents()
    if dataset is None:
        dataset = generate_dataset(params.n_stocks, params.n_periods, params.seed,
                                   signal=params.signal)
    backtest = run_backtest(dataset, agents, top_n=params.top_n)             # decisions + trades
    benchmarks = compute_benchmarks(dataset, params.top_n, params.seed, params.random_trials)
    metrics = compute_agent_metrics(backtest, benchmarks, params.rank_by)   # metrics + ranking

    versions = {a.name: agent_version(a) for a in agents}
    code_ver = code_version()
    param_dict = params.to_dict()
    results = {
        "ranking": [m.agent for m in metrics],
        "rank_by": params.rank_by,
        "agents": {m.agent: m.to_dict() for m in metrics},
        "benchmarks": benchmark_metrics(benchmarks),
        "equity_curves": {
            **{a: equity_curve(r) for a, r in backtest.period_returns.items()},
            **{n: equity_curve(r) for n, r in benchmarks.returns.items()},
        },
        "drawdown_curves": {
            **{a: drawdown_curve(r) for a, r in backtest.period_returns.items()},
            **{n: drawdown_curve(r) for n, r in benchmarks.returns.items()},
        },
        "primary_benchmark": MARKET,
    }
    stamp = (timestamp or datetime.now(timezone.utc)).isoformat()
    return Experiment(
        experiment_id=_experiment_id(param_dict, dataset.version, code_ver, versions),
        dataset_version=dataset.version,
        timestamp=stamp,
        code_version=code_ver,
        agent_versions=versions,
        parameters=param_dict,
        results=results,
        dataset=dataset, backtest=backtest, benchmarks=benchmarks, metrics=metrics,
    )


def load_experiment_history(directory: str | Path) -> List[Dict[str, Any]]:
    """Saved experiment records (newest run first). Unreadable files are skipped."""
    path = Path(directory)
    if not path.is_dir():
        return []
    records = []
    for file in path.glob("exp-*.json"):
        try:
            record = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and "experiment_id" in record:
            records.append(record)
    records.sort(key=lambda r: (r.get("timestamp", ""), r["experiment_id"]), reverse=True)
    return records
