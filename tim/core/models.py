from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from tim.core.constants import DEFAULT_MULTIPLIER


@dataclass(frozen=True)
class OptionLeg:
    kind: str  # "call" or "put"
    position: float  # > 0 long, < 0 short
    strike: float
    premium: float  # always positive
    multiplier: int = DEFAULT_MULTIPLIER


@dataclass(frozen=True)
class StrategyInput:
    spot: float
    stock_position: float = 0.0
    avg_cost: float = 0.0
    legs: List[OptionLeg] = field(default_factory=list)
