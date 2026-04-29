from __future__ import annotations

from typing import Dict, List

from tim.core.constants import BE_TOLERANCE, EPSILON, GRID_RANGE, GRID_SIZE
from tim.core.models import OptionLeg, StrategyInput


def _build_price_grid(spot: float, steps: int = GRID_SIZE) -> List[float]:
    max_price = GRID_RANGE * spot
    if steps < 2:
        return [0.0, max_price]
    step = max_price / (steps - 1)
    return [i * step for i in range(steps)]


def _inject_strikes(grid: List[float], legs: List[OptionLeg]) -> List[float]:
    strikes = {leg.strike for leg in legs}
    combined = list(grid) + list(strikes)
    combined.sort()
    deduped: List[float] = []
    last = None
    for value in combined:
        if last is None or value != last:
            deduped.append(value)
            last = value
    return deduped


def _option_intrinsic(leg: OptionLeg, price: float) -> float:
    if leg.kind.lower() == "call":
        return max(price - leg.strike, 0.0)
    return max(leg.strike - price, 0.0)


def _compute_pnl_for_price(strategy: StrategyInput, price: float) -> float:
    option_total = 0.0
    for leg in strategy.legs:
        intrinsic = _option_intrinsic(leg, price)
        option_total += leg.position * (intrinsic - leg.premium) * leg.multiplier
    stock_total = strategy.stock_position * (price - strategy.avg_cost)
    result = option_total + stock_total
    # Eliminate IEEE -0.0 (negative zero) at the source
    return result + 0.0


def _detect_breakevens(grid: List[float], pnl: List[float]) -> List[float]:
    if not grid or not pnl:
        return []

    pnl_range = max(pnl) - min(pnl)
    if pnl_range < 1.0:
        return []

    breakevens: List[float] = []
    for i in range(len(grid) - 1):
        y0 = pnl[i]
        y1 = pnl[i + 1]
        if y0 == 0.0:
            breakevens.append(grid[i])
            continue
        if y0 * y1 < 0.0:
            x0 = grid[i]
            x1 = grid[i + 1]
            breakeven = x0 - y0 * (x1 - x0) / (y1 - y0)
            breakevens.append(breakeven)
    breakevens = sorted(set(round(b, 8) for b in breakevens))

    if len(breakevens) <= 1:
        return breakevens

    clusters: List[List[float]] = [[breakevens[0]]]
    for be in breakevens[1:]:
        prev_be = clusters[-1][-1]
        pnl_between = [
            pnl[i] for i in range(len(grid))
            if prev_be - EPSILON <= grid[i] <= be + EPSILON
        ]
        if pnl_between:
            range_val = max(pnl_between) - min(pnl_between)
            mid_val = (max(pnl_between) + min(pnl_between)) / 2.0
            if range_val < BE_TOLERANCE * 2 and abs(mid_val) <= BE_TOLERANCE:
                clusters[-1].append(be)
                continue
        clusters.append([be])

    consolidated: List[float] = []
    for cluster in clusters:
        if len(cluster) == 1:
            consolidated.append(cluster[0])
            continue
        cluster_start = cluster[0]
        cluster_end = cluster[-1]
        pnl_in_range = [
            pnl[i] for i in range(len(grid))
            if cluster_start - EPSILON <= grid[i] <= cluster_end + EPSILON
        ]
        if not pnl_in_range:
            consolidated.append(cluster[0])
            consolidated.append(cluster[-1])
            continue
        range_max = max(pnl_in_range)
        range_min = min(pnl_in_range)
        if (range_max - range_min) < BE_TOLERANCE * 2:
            flat_mid = (range_max + range_min) / 2.0
            if abs(flat_mid) <= BE_TOLERANCE:
                consolidated.append(round((cluster_start + cluster_end) / 2.0, 8))
        else:
            consolidated.append(cluster[0])
            consolidated.append(cluster[-1])

    return consolidated


def _detect_unlimited(grid: List[float], pnl: List[float]) -> Dict[str, bool]:
    if len(grid) < 2:
        return {
            "unlimited_upside": False, "unlimited_downside": False,
            "unlimited_loss_upside": False,
            "unlimited_profit_upside": False,
            "unlimited_profit_downside": False,
            "unlimited_loss_downside": False,
        }
    slope_high = (pnl[-1] - pnl[-2]) / (grid[-1] - grid[-2])
    slope_low = (pnl[1] - pnl[0]) / (grid[1] - grid[0])
    has_slope_low = slope_low < -EPSILON
    return {
        "unlimited_upside": slope_high > EPSILON,
        "unlimited_downside": has_slope_low,
        "unlimited_loss_upside": slope_high < -EPSILON,
        "unlimited_profit_upside": slope_high > EPSILON,
        "unlimited_profit_downside": has_slope_low and pnl[0] > 0,
        "unlimited_loss_downside": has_slope_low and pnl[0] < 0,
    }


def compute_payoff(strategy: StrategyInput) -> Dict[str, object]:
    grid = _build_price_grid(strategy.spot, steps=GRID_SIZE)
    grid = _inject_strikes(grid, strategy.legs)
    pnl = [_compute_pnl_for_price(strategy, price) for price in grid]
    breakevens = _detect_breakevens(grid, pnl)
    unlimited = _detect_unlimited(grid, pnl)
    return {
        "price_grid": grid,
        "pnl": pnl,
        "breakevens": breakevens,
        **unlimited,
    }
