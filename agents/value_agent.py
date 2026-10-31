from __future__ import annotations

from agents.base_agent import BaseAgent, fcf_yield_text
from data.models import StockSnapshot


class ValueAgent(BaseAgent):
    name = "Value Hunter"
    description = "Low valuation relative to earnings, sales and free cash flow."
    weights = {"value": 0.55, "quality": 0.20, "catalyst": 0.15, "momentum": 0.10}
    risk_penalty = 0.25

    def key_facts(self, stock: StockSnapshot) -> str:
        return f"P/E {stock.pe_ratio:.1f}, P/S {stock.price_to_sales:.1f}, {fcf_yield_text(stock)}"
