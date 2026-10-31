from __future__ import annotations

from agents.base_agent import BaseAgent
from data.models import StockSnapshot


class ShortSqueezeAgent(BaseAgent):
    name = "Short Squeeze Hunter"
    description = "High short interest combined with volume and momentum."
    weights = {"squeeze": 0.55, "momentum": 0.20, "catalyst": 0.20, "value": 0.05}
    risk_penalty = 0.10

    def key_facts(self, stock: StockSnapshot) -> str:
        return (
            f"short interest {stock.short_interest:.1%} ({stock.days_to_cover:.1f} days to cover), "
            f"volume {stock.volume_ratio:.1f}x, momentum {stock.momentum:+.1%}"
        )
