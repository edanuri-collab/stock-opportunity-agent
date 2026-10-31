"""Shared, deterministic factor scoring. Every agent starts from the same components."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping

from data.models import StockSnapshot

FACTORS = ("value", "quality", "momentum", "catalyst", "breakout", "squeeze", "contrarian",
           "noise")


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def scale(value: float, lo: float, hi: float) -> float:
    """Linear map: ``lo`` -> 0, ``hi`` -> 100, clamped."""
    if hi == lo:
        raise ValueError("hi and lo must differ")
    return clamp((value - lo) / (hi - lo) * 100)


def scale_inverse(value: float, lo: float, hi: float) -> float:
    """Like ``scale`` but lower values score higher."""
    return 100.0 - scale(value, lo, hi)


def fcf_yield(stock: StockSnapshot) -> float:
    return stock.free_cash_flow / stock.market_cap if stock.market_cap > 0 else 0.0


def placebo_noise(stock: StockSnapshot) -> float:
    """Uniform 0-100 from a hash of ticker + timestamp. Carries no information (control factor)."""
    key = f"{stock.ticker}|{stock.timestamp.isoformat()}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2 ** 64 * 100


@dataclass(frozen=True)
class Components:
    """Factor scores in 0-100. ``risk``: higher = riskier (applied as a penalty)."""

    value: float
    quality: float
    momentum: float
    catalyst: float
    breakout: float
    squeeze: float
    contrarian: float
    noise: float        # placebo: deterministic but uninformative, used by the Random Agent
    risk: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_components(stock: StockSnapshot) -> Components:
    pe_score = 0.0 if stock.pe_ratio <= 0 else scale_inverse(stock.pe_ratio, 8, 45)
    value = (
        0.35 * pe_score
        + 0.25 * scale_inverse(stock.price_to_sales, 0.5, 12)
        + 0.40 * scale(fcf_yield(stock), -0.02, 0.10)
    )
    quality = (
        0.25 * scale(stock.revenue_growth, -0.05, 0.35)
        + 0.25 * scale(stock.earnings_growth, -0.10, 0.40)
        + 0.25 * scale(fcf_yield(stock), 0.0, 0.08)
        + 0.25 * scale_inverse(stock.debt_to_equity, 0.0, 2.5)
    )
    momentum = 0.5 * stock.relative_strength + 0.5 * scale(stock.momentum, -0.3, 0.6)
    catalyst = 0.7 * stock.catalyst_score + 0.3 * stock.sentiment_score
    volume = scale(stock.volume_ratio, 0.8, 3.0)
    breakout = (
        0.4 * scale_inverse(stock.distance_from_high, 0.0, 0.15) + 0.3 * volume + 0.3 * momentum
    )
    squeeze = (
        0.30 * scale(stock.short_interest, 0.03, 0.30)
        + 0.20 * scale(stock.days_to_cover, 1.0, 10.0)
        + 0.25 * volume
        + 0.25 * scale(stock.momentum, -0.1, 0.4)
    )
    # Geometric mean: needs BOTH a deep drawdown and strong fundamentals.
    contrarian = math.sqrt(scale(stock.distance_from_high, 0.15, 0.55) * quality)
    size_risk = scale_inverse(math.log10(max(stock.market_cap, 1.0)), 8.5, 11.5)
    risk = (
        0.40 * scale(stock.volatility, 0.15, 0.90)
        + 0.25 * scale(stock.debt_to_equity, 0.5, 3.0)
        + 0.20 * (100.0 if stock.free_cash_flow < 0 else 0.0)
        + 0.15 * size_risk
    )
    return Components(value, quality, momentum, catalyst, breakout, squeeze, contrarian,
                      placebo_noise(stock), risk)


def validate_weights(weights: Mapping[str, float]) -> None:
    unknown = set(weights) - set(FACTORS)
    if unknown:
        raise ValueError(f"Unknown factors in weights: {sorted(unknown)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Weights must be non-negative")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"Weights must sum to 1, got {sum(weights.values())}")


def weighted_opportunity(c: Components, weights: Mapping[str, float], risk_penalty: float) -> float:
    cd = c.as_dict()
    raw = sum(w * cd[name] for name, w in weights.items())
    return clamp(raw - risk_penalty * c.risk)


def top_factors(c: Components, weights: Mapping[str, float], n: int = 2) -> List[str]:
    """Factors contributing most to the score (ties broken by name for determinism)."""
    cd = c.as_dict()
    ranked = sorted(weights, key=lambda name: (-weights[name] * cd[name], name))
    return ranked[:n]
