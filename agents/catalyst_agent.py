from __future__ import annotations

from agents.base_agent import BaseAgent
from data.models import StockSnapshot


class CatalystAgent(BaseAgent):
    name = "Catalyst Hunter"
    description = "Events that may force the market to reprice the company."
    weights = {"catalyst": 0.55, "momentum": 0.20, "value": 0.15, "quality": 0.10}
    risk_penalty = 0.20

    def key_facts(self, stock: StockSnapshot) -> str:
        return f"catalyst {stock.catalyst_score:.0f}, sentiment {stock.sentiment_score:.0f}"
