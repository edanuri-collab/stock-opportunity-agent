"""Everything a screen needs, built once per experiment and passed to every view."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd

from dashboard import analytics as an
from engine.experiment import Experiment

PAGES = ("Overview", "Agents Leaderboard", "Stock Opportunities", "Stock Detail",
         "Performance & Backtesting", "Experiments")


@dataclass
class DashboardContext:
    exp: Experiment
    scores: pd.DataFrame          # every agent's score of every stock at every timestamp
    status: an.DataSourceStatus
    experiment_dir: str
    runtime_seconds: float = 0.0

    @property
    def timestamps(self) -> List[datetime]:
        return list(self.exp.dataset.timestamps)

    @property
    def latest(self) -> datetime:
        return self.exp.dataset.timestamps[-1]


def build_context(exp: Experiment, experiment_dir: str, runtime_seconds: float = 0.0
                  ) -> DashboardContext:
    return DashboardContext(exp=exp, scores=an.scores_frame(exp),
                            status=an.data_source_status(exp), experiment_dir=experiment_dir,
                            runtime_seconds=runtime_seconds)
