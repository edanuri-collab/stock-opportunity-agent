from __future__ import annotations

from agents.base_agent import BaseAgent
from data.models import StockSnapshot


class MomentumAgent(BaseAgent):
    name = "Momentum Hunter"
    description = "Strong price momentum and relative strength."
    weights = {"momentum": 0.60, "breakout": 0.15, "catalyst": 0.15, "quality": 0.10}
    risk_penalty = 0.20

    def key_facts(self, stock: StockSnapshot) -> str:
        return f"momentum {stock.momentum:+.1%}, relative strength {stock.relative_strength:.0f}"
