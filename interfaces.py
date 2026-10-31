"""Extension points for future phases. Interfaces only - nothing here is implemented yet.

Every provider must be point-in-time: it may only return information that was
publicly known at or before the requested timestamp (no look-ahead).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence

from data.models import StockSnapshot


class MarketDataProvider(Protocol):
    """Market data, SEC filings, news, social media, fundamentals (one adapter each)."""

    def timestamps(self) -> Sequence[datetime]: ...

    def snapshots_at(self, timestamp: datetime) -> Sequence[StockSnapshot]: ...


class TextAnalyzer(Protocol):
    """AI layer: LLM gateway, sentiment, event extraction, company analysis.

    Output must be materialised into the dataset *before* agents run, so agents
    stay deterministic and replayable.
    """

    def analyze(self, text: str, as_of: datetime) -> Mapping[str, Any]: ...


class SignalEngine(Protocol):
    """Quant layer: features, signals, market regime, position sizing, risk."""

    def signals(self, snapshots: Sequence[StockSnapshot]) -> Mapping[str, float]: ...


class ExecutionVenue(Protocol):
    """Execution: paper trading first; broker APIs / live trading much later."""

    def submit(self, ticker: str, quantity: float, as_of: datetime) -> Mapping[str, Any]: ...
