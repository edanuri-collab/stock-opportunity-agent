from __future__ import annotations

from agents.base_agent import BaseAgent
from data.models import StockSnapshot


class ContrarianAgent(BaseAgent):
    name = "Contrarian"
    description = "Sharp drawdown while fundamentals remain strong."
    weights = {"contrarian": 0.65, "value": 0.20, "catalyst": 0.15}
    risk_penalty = 0.15

    def key_facts(self, stock: StockSnapshot) -> str:
        return (
            f"{stock.distance_from_high:.0%} below 52w high, "
            f"earnings growth {stock.earnings_growth:+.1%}"
        )
