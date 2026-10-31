"""Point-in-time data model.

Agents only ever receive a ``StockSnapshot``. Realised forward returns live in
``MarketDataset`` and are read exclusively by the backtest *after* decisions are made.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class StockSnapshot:
    """Everything that was knowable about one stock at ``timestamp``."""

    timestamp: datetime
    ticker: str
    company_name: str
    sector: str
    industry: str
    price: float
    shares_outstanding: float
    market_cap: float
    revenue_growth: float       # YoY, 0.10 = +10%
    earnings_growth: float      # YoY
    free_cash_flow: float       # trailing 12m, USD
    debt_to_equity: float
    pe_ratio: float             # <= 0 means negative earnings
    price_to_sales: float
    relative_strength: float    # cross-sectional percentile, 0-100
    momentum: float             # trailing 6m return, 0.10 = +10%
    volume_ratio: float         # recent volume / average volume
    volatility: float           # annualised
    catalyst_score: float       # 0-100, upstream event-driven signal
    sentiment_score: float      # 0-100
    short_interest: float       # fraction of float sold short
    days_to_cover: float        # short interest / average daily volume (in shares)
    distance_from_high: float   # fraction below 52-week high (0 = at the high)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class MarketDataset:
    """Snapshots indexed by timestamp, plus realised outcomes for backtesting only."""

    def __init__(
        self,
        snapshots: Sequence[StockSnapshot],
        outcomes: Mapping[Tuple[str, datetime], float],
        regimes: Mapping[datetime, str],
    ) -> None:
        ordered = sorted(snapshots, key=lambda s: (s.timestamp, s.ticker))
        by_ts: Dict[datetime, List[StockSnapshot]] = {}
        for snap in ordered:
            by_ts.setdefault(snap.timestamp, []).append(snap)
        self._by_ts = {ts: tuple(v) for ts, v in by_ts.items()}
        self._timestamps = tuple(sorted(self._by_ts))
        self._outcomes = dict(outcomes)
        self._regimes = dict(regimes)
        self._version = None

    @property
    def version(self) -> str:
        """Content hash of snapshots, outcomes and regimes: identical data -> identical version."""
        if self._version is None:
            digest = hashlib.sha256()
            for ts in self._timestamps:
                for snap in self._by_ts[ts]:
                    digest.update(json.dumps(snap.to_dict(), sort_keys=True).encode())
            for key in sorted(self._outcomes, key=lambda k: (k[1], k[0])):
                digest.update(f"{key[0]}|{key[1].isoformat()}|{self._outcomes[key]!r}".encode())
            for ts in sorted(self._regimes):
                digest.update(f"{ts.isoformat()}|{self._regimes[ts]}".encode())
            self._version = digest.hexdigest()[:12]
        return self._version

    @property
    def timestamps(self) -> Tuple[datetime, ...]:
        return self._timestamps

    @property
    def tickers(self) -> Tuple[str, ...]:
        first = self._by_ts[self._timestamps[0]] if self._timestamps else ()
        return tuple(s.ticker for s in first)

    def snapshots_at(self, timestamp: datetime) -> Tuple[StockSnapshot, ...]:
        """Snapshots as of exactly ``timestamp``."""
        return self._by_ts.get(timestamp, ())

    def snapshots_up_to(self, timestamp: datetime) -> Tuple[StockSnapshot, ...]:
        """All snapshots with timestamp <= ``timestamp`` (never the future)."""
        return tuple(s for ts in self._timestamps if ts <= timestamp for s in self._by_ts[ts])

    def regime_at(self, timestamp: datetime) -> str:
        return self._regimes.get(timestamp, "unknown")

    def outcome(self, ticker: str, timestamp: datetime) -> float:
        """Realised forward return after ``timestamp``. Backtest use only, never agents."""
        return self._outcomes[(ticker, timestamp)]
