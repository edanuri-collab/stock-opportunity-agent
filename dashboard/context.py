"""Everything a screen needs, built once per experiment and passed to every view."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

from agents.registry import Registry
from dashboard import analytics as an
from engine.experiment import Experiment, ExperimentParams

PAGES = ("Overview", "Agents Leaderboard", "Agent Catalog", "Stock Opportunities",
         "Stock Detail", "Performance & Backtesting", "Fair Ranking", "Experiments")


@dataclass
class DashboardContext:
    exp: Experiment
    scores: pd.DataFrame          # every agent's score of every stock at every timestamp
    status: an.DataSourceStatus
    experiment_dir: str
    registry: Registry
    runtime_seconds: float = 0.0
    params: Optional[ExperimentParams] = None
    agents_version: str = ""

    @property
    def timestamps(self) -> List[datetime]:
        return list(self.exp.dataset.timestamps)

    @property
    def latest(self) -> datetime:
        return self.exp.dataset.timestamps[-1]


def build_context(exp: Experiment, experiment_dir: str, registry: Registry,
                  runtime_seconds: float = 0.0) -> DashboardContext:
    return DashboardContext(exp=exp, scores=an.scores_frame(exp),
                            status=an.data_source_status(exp), experiment_dir=experiment_dir,
                            registry=registry, runtime_seconds=runtime_seconds)
