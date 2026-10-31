from __future__ import annotations

from agents.base_agent import BaseAgent


class BalancedAgent(BaseAgent):
    name = "Balanced"
    description = "Equal weight across value, quality, momentum and catalyst."
    weights = {"value": 0.25, "quality": 0.25, "momentum": 0.25, "catalyst": 0.25}
    risk_penalty = 0.30
