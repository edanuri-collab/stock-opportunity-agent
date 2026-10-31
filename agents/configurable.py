"""Agents built from an ``AgentSpec`` (no Python needed to add a strategy).

``build_agent(spec)`` creates a dedicated subclass whose configuration lives in class
attributes, so instances stay stateless exactly like the core agents.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar, List, Sequence

from agents.base_agent import AgentResult, BaseAgent
from agents.spec import AgentSpec, field_value, percentile_ranks, slug
from data.models import StockSnapshot
from engine.scoring import clamp, compute_components, top_factors, weighted_opportunity


class SpecAgent(BaseAgent):
    """Base for spec-defined agents. Use ``build_agent``; do not instantiate directly."""

    spec: ClassVar[AgentSpec]

    def __init__(self) -> None:  # the spec was validated when it was parsed
        if not hasattr(type(self), "spec"):
            raise TypeError("use build_agent(spec) to create a spec agent")

    def spec_dict(self) -> dict:
        return self.spec.to_dict()

    def failed_filters(self, stock: StockSnapshot) -> List[str]:
        return [f"{f.field} {f.op} {f.value:g}" for f in self.spec.filters if not f.passes(stock)]

    # ---- scoring ---------------------------------------------------------------------

    def analyze(self, stock: StockSnapshot) -> AgentResult:
        if self.spec.rank:
            raise TypeError(f"{self.name} ranks across stocks: use analyze_universe()")
        return self.analyze_universe([stock])[0]

    def analyze_universe(self, stocks: Sequence[StockSnapshot]) -> List[AgentResult]:
        comps = [compute_components(s) for s in stocks]
        failed = [self.failed_filters(s) for s in stocks]
        eligible = [i for i, f in enumerate(failed) if not f]
        scores = [0.0] * len(stocks)
        if self.spec.rank:
            if eligible:
                total = sum(r.weight for r in self.spec.rank)
                combined = [0.0] * len(eligible)
                for r in self.spec.rank:
                    pct = percentile_ranks([field_value(stocks[i], r.field) for i in eligible])
                    if r.prefer == "low":
                        pct = [100.0 - p for p in pct]
                    combined = [c + r.weight / total * p for c, p in zip(combined, pct)]
                for i, c in zip(eligible, combined):
                    scores[i] = clamp(c - self.risk_penalty * comps[i].risk)
        else:
            for i in eligible:
                scores[i] = weighted_opportunity(comps[i], self.weights, self.risk_penalty)
        return [self._result(s, c, score, not f, f)
                for s, c, score, f in zip(stocks, comps, scores, failed)]

    def _result(self, stock, comps, score, eligible, failed) -> AgentResult:
        if not eligible:
            reasoning = (f"{self.name} filters out {stock.ticker}: fails "
                         f"{', '.join(failed)}.")
        elif self.spec.rank:
            facts = ", ".join(f"{r.field} {field_value(stock, r.field):.3g} ({r.prefer} preferred)"
                              for r in self.spec.rank)
            reasoning = f"{self.name} scores {stock.ticker} {score:.1f}/100 by rank. {facts}."
        else:
            cd = comps.as_dict()
            drivers = ", ".join(f"{f} {cd[f]:.0f}" for f in top_factors(comps, self.weights, 2))
            reasoning = (f"{self.name} scores {stock.ticker} {score:.1f}/100. "
                         f"Main drivers: {drivers}.")
        return {
            "ticker": stock.ticker, "agent": self.name,
            "value_score": round(comps.value, 2), "catalyst_score": round(comps.catalyst, 2),
            "momentum_score": round(comps.momentum, 2), "risk_score": round(comps.risk, 2),
            "opportunity_score": round(score, 2), "reasoning": reasoning,
            "timestamp": stock.timestamp.isoformat(), "eligible": eligible,
        }


def build_agent(spec: AgentSpec) -> SpecAgent:
    cls = type(f"SpecAgent_{slug(spec.name).replace('-', '_')}", (SpecAgent,), {
        "spec": spec, "name": spec.name, "description": spec.description,
        "weights": MappingProxyType(spec.weights_map), "risk_penalty": spec.risk_penalty,
        "family": spec.family, "origin": spec.origin, "reference": spec.reference,
    })
    return cls()
