from __future__ import annotations

from agents.base_agent import BaseAgent
from data.models import StockSnapshot


class RandomAgent(BaseAgent):
    """Placebo control: ranks stocks by a hash of ticker + date, ignoring every feature.

    Same mechanics as the other agents (top-N, equal weight), so any edge an agent shows
    over this one is not an artefact of the backtest itself.
    """

    name = "Random Agent"
    description = "Control. Picks pseudo-randomly (deterministic hash), ignores all data."
    weights = {"noise": 1.0}
    risk_penalty = 0.0

    def key_facts(self, stock: StockSnapshot) -> str:
        return "none - placebo pick, features are ignored"
