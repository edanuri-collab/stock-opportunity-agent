"""Market data providers. The journal and the agents only talk to this interface, so a real
data source can replace the synthetic one without touching them (see ``interfaces.py``)."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence, Tuple

from data.models import MarketDataset, StockSnapshot


class SyntheticProvider:
    """Point-in-time view of a generated ``MarketDataset``. Realised outcomes are not exposed:
    later results are measured from later snapshots, exactly as with a real feed."""

    is_synthetic = True

    def __init__(self, dataset: MarketDataset, label: str = "synthetic") -> None:
        self._dataset = dataset
        self.name = label
        self.version = dataset.version

    def timestamps(self) -> Sequence[datetime]:
        return self._dataset.timestamps

    def snapshots_at(self, timestamp: datetime) -> Tuple[StockSnapshot, ...]:
        return self._dataset.snapshots_at(timestamp)
