"""Common agent interface. Agents differ only in their configured weights."""
from __future__ import annotations

from abc import ABC
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, TypedDict

from data.models import StockSnapshot
from engine.scoring import (
    Components,
    compute_components,
    fcf_yield,
    top_factors,
    validate_weights,
    weighted_opportunity,
)


class AgentResult(TypedDict):
    ticker: str
    agent: str
    value_score: float
    catalyst_score: float
    momentum_score: float
    risk_score: float
    opportunity_score: float
    reasoning: str
    timestamp: str


class BaseAgent(ABC):
    """Deterministic agent: opportunity = sum(weight * factor) - risk_penalty * risk."""

    name: ClassVar[str]
    description: ClassVar[str]
    weights: ClassVar[Mapping[str, float]]
    risk_penalty: ClassVar[float]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        weights = cls.__dict__.get("weights")
        if weights is not None and not isinstance(weights, MappingProxyType):
            cls.weights = MappingProxyType(dict(weights))  # read-only: no agent can mutate config

    def __init__(self) -> None:
        validate_weights(self.weights)
        if not 0 <= self.risk_penalty <= 1:
            raise ValueError("risk_penalty must be within [0, 1]")

    def key_facts(self, stock: StockSnapshot) -> str:
        """Agent-specific data points quoted in the reasoning. Override per agent."""
        return f"P/E {stock.pe_ratio:.1f}, momentum {stock.momentum:+.1%}"

    def analyze(self, stock: StockSnapshot) -> AgentResult:
        comps = compute_components(stock)
        opportunity = weighted_opportunity(comps, self.weights, self.risk_penalty)
        return {
            "ticker": stock.ticker,
            "agent": self.name,
            "value_score": round(comps.value, 2),
            "catalyst_score": round(comps.catalyst, 2),
            "momentum_score": round(comps.momentum, 2),
            "risk_score": round(comps.risk, 2),
            "opportunity_score": round(opportunity, 2),
            "reasoning": self._reasoning(stock, comps, opportunity),
            "timestamp": stock.timestamp.isoformat(),
        }

    def _reasoning(self, stock: StockSnapshot, comps: Components, opportunity: float) -> str:
        cd = comps.as_dict()
        drivers = ", ".join(f"{f} {cd[f]:.0f}" for f in top_factors(comps, self.weights, 2))
        penalty = self.risk_penalty * comps.risk
        return (
            f"{self.name} scores {stock.ticker} {opportunity:.1f}/100. "
            f"Main drivers: {drivers}. Key data: {self.key_facts(stock)}. "
            f"Risk {comps.risk:.0f} subtracts {penalty:.1f} points."
        )


def fcf_yield_text(stock: StockSnapshot) -> str:
    return f"FCF yield {fcf_yield(stock):.1%}"
