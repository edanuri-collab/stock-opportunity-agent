from __future__ import annotations

from agents.base_agent import BaseAgent
from data.models import StockSnapshot


class BreakoutAgent(BaseAgent):
    name = "Breakout Hunter"
    description = "Momentum plus volume near a 52-week high."
    weights = {"breakout": 0.50, "momentum": 0.25, "catalyst": 0.15, "quality": 0.10}
    risk_penalty = 0.20

    def key_facts(self, stock: StockSnapshot) -> str:
        return (
            f"{stock.distance_from_high:.0%} below 52w high, "
            f"volume {stock.volume_ratio:.1f}x average"
        )
