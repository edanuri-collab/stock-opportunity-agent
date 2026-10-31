from __future__ import annotations

from typing import List

from agents.balanced_agent import BalancedAgent
from agents.base_agent import AgentResult, BaseAgent
from agents.breakout_agent import BreakoutAgent
from agents.catalyst_agent import CatalystAgent
from agents.contrarian_agent import ContrarianAgent
from agents.momentum_agent import MomentumAgent
from agents.quality_agent import QualityAgent
from agents.random_agent import RandomAgent
from agents.short_squeeze_agent import ShortSqueezeAgent
from agents.value_agent import ValueAgent

__all__ = ["AgentResult", "BaseAgent", "default_agents"]


def default_agents() -> List[BaseAgent]:
    """The eight strategy agents plus the Random Agent control, in a fixed order."""
    return [
        ValueAgent(),
        MomentumAgent(),
        CatalystAgent(),
        QualityAgent(),
        ContrarianAgent(),
        BreakoutAgent(),
        ShortSqueezeAgent(),
        BalancedAgent(),
        RandomAgent(),
    ]
