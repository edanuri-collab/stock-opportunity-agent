"""Meta-agents: a separate layer that combines the decisions the other agents made at T.

Regular agents never see each other. A meta-agent runs *after* them, on the same timestamp,
and receives only their ranked decisions for that timestamp (no snapshots of other dates, no
outcomes). Control agents are excluded from the votes.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar, Dict, List, Mapping, Sequence

from agents.base_agent import AgentResult, BaseAgent
from data.models import StockSnapshot


class MetaAgent(BaseAgent):
    is_meta: ClassVar[bool] = True
    weights = MappingProxyType({})
    risk_penalty = 0.0
    family = "multi_factor"
    origin = "original"

    def __init__(self) -> None:  # no weights to validate
        pass

    def analyze(self, stock: StockSnapshot) -> AgentResult:
        raise TypeError(f"{self.name} is a meta-agent: it combines other agents' decisions")

    def combine(self, decisions: Mapping[str, Sequence[AgentResult]], top_n: int
                ) -> List[AgentResult]:
        raise NotImplementedError


class ConsensusAgent(MetaAgent):
    """Picks the stocks that the largest share of strategy agents hold in their top-N."""

    name = "Consensus"
    description = ("Meta-agent. Score = share of strategy agents holding the stock in their "
                   "top-N at the same date; needs at least two votes. Ties: average score.")
    min_votes: ClassVar[int] = 2

    def combine(self, decisions: Mapping[str, Sequence[AgentResult]], top_n: int
                ) -> List[AgentResult]:
        voters = sorted(decisions)
        votes: Dict[str, int] = {}
        score_sum: Dict[str, float] = {}
        base: Dict[str, AgentResult] = {}
        for name in voters:
            ranked = decisions[name]
            for r in ranked:
                base.setdefault(r["ticker"], r)
                score_sum[r["ticker"]] = score_sum.get(r["ticker"], 0.0) + r["opportunity_score"]
            for r in [r for r in ranked if r.get("eligible", True)][:top_n]:
                votes[r["ticker"]] = votes.get(r["ticker"], 0) + 1
        n = max(len(voters), 1)
        out: List[AgentResult] = []
        for ticker in sorted(base):
            b, k = base[ticker], votes.get(ticker, 0)
            avg = score_sum[ticker] / n
            out.append({
                "ticker": ticker, "agent": self.name,
                "value_score": b["value_score"], "catalyst_score": b["catalyst_score"],
                "momentum_score": b["momentum_score"], "risk_score": b["risk_score"],
                # vote share first; the average score (<= 0.1 point) only breaks ties
                "opportunity_score": round(100 * k / n + avg / 1000, 3),
                "reasoning": (f"{self.name}: picked by {k} of {n} strategy agents "
                              f"(average score {avg:.1f})."),
                "timestamp": b["timestamp"], "eligible": k >= self.min_votes,
            })
        return out


def meta_agents() -> List[MetaAgent]:
    return [ConsensusAgent()]
