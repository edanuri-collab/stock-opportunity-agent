from __future__ import annotations

from agents.base_agent import BaseAgent, fcf_yield_text
from data.models import StockSnapshot


class QualityAgent(BaseAgent):
    name = "Quality Hunter"
    description = "Growth, free cash flow and a strong balance sheet."
    weights = {"quality": 0.55, "value": 0.20, "catalyst": 0.15, "momentum": 0.10}
    risk_penalty = 0.35

    def key_facts(self, stock: StockSnapshot) -> str:
        return (
            f"revenue growth {stock.revenue_growth:+.1%}, "
            f"debt/equity {stock.debt_to_equity:.2f}, {fcf_yield_text(stock)}"
        )
