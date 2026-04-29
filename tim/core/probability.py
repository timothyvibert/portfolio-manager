from __future__ import annotations

import math
from typing import Iterable, List, Optional

import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm

from tim.core.constants import EPSILON
from tim.core.models import OptionLeg, StrategyInput


def norm_cdf(x: float) -> float:
    return float(norm.cdf(x))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _lognormal_density(S_T: float, S_0: float, mu: float, sigma: float, T: float) -> float:
    """Risk-neutral lognormal density at terminal price S_T."""
    if S_T <= 0 or S_0 <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    drift = (mu - 0.5 * sigma ** 2) * T
    vol_sqrt_T = sigma * math.sqrt(T)
    z = (math.log(S_T / S_0) - drift) / vol_sqrt_T
    return norm.pdf(z) / (S_T * vol_sqrt_T)


def bs_d2(S: float, K: float, r: float, q: float, sigma: float, t: float) -> float:
    if t <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        if S > K:
            return float("inf")
        if S < K:
            return float("-inf")
        return 0.0
    denom = sigma * math.sqrt(t)
    return (math.log(S / K) + (r - q - 0.5 * sigma * sigma) * t) / denom


def _bs_d1(S: float, K: float, r: float, q: float, sigma: float, t: float) -> Optional[float]:
    if t <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return None
    denom = sigma * math.sqrt(t)
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * t) / denom


def prob_finish_above(
    S: float, K_star: float, r: float, q: float, sigma: float, t: float
) -> float:
    d2 = bs_d2(S, K_star, r, q, sigma, t)
    return norm_cdf(d2)


def prob_finish_below(
    S: float, K_star: float, r: float, q: float, sigma: float, t: float
) -> float:
    return 1.0 - prob_finish_above(S, K_star, r, q, sigma, t)


def leg_pop_at_expiry(
    leg: OptionLeg, S: float, r: float, q: float, sigma: float, t: float
) -> float:
    if leg.kind.lower() == "call":
        threshold = leg.strike + leg.premium
        long_prob = prob_finish_above(S, threshold, r, q, sigma, t)
    else:
        threshold = leg.strike - leg.premium
        long_prob = prob_finish_below(S, threshold, r, q, sigma, t)
    if leg.position < 0:
        return 1.0 - long_prob
    return long_prob


def leg_itm_prob(
    leg: OptionLeg, S: float, r: float, q: float, sigma: float, t: float
) -> float:
    if leg.kind.lower() == "call":
        return prob_finish_above(S, leg.strike, r, q, sigma, t)
    return prob_finish_below(S, leg.strike, r, q, sigma, t)


def _bs_vega(S: float, K: float, r: float, q: float, sigma: float, t: float) -> float:
    d1 = _bs_d1(S, K, r, q, sigma, t)
    if d1 is None:
        return 0.0
    return S * math.exp(-q * t) * math.sqrt(t) * _norm_pdf(d1)


def _safe_sigma(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        sigma = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(sigma):
        return None
    return sigma


def _fallback_atm(per_leg_iv: Iterable[Optional[float]]) -> Optional[float]:
    values = [v for v in (_safe_sigma(v) for v in per_leg_iv) if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def effective_sigma(
    input: StrategyInput,
    per_leg_iv: List[Optional[float]],
    mode: str,
    r: float,
    q: float,
    t: float,
    atm_iv: Optional[float] = None,
) -> float:
    mode_key = mode.upper()
    atm_value = _safe_sigma(atm_iv)
    if atm_value is None:
        atm_value = _fallback_atm(per_leg_iv)

    if mode_key in ("ATM", "SURFACE_ATM"):
        if atm_value is None:
            raise ValueError("ATM volatility is required.")
        return max(atm_value, EPSILON)

    if mode_key == "VEGA_WEIGHTED":
        weights = []
        sigmas = []
        for leg, sigma in zip(input.legs, per_leg_iv):
            sigma_value = _safe_sigma(sigma)
            if sigma_value is None:
                continue
            if not isinstance(leg.strike, (int, float)):
                continue
            vega = _bs_vega(
                S=input.spot,
                K=float(leg.strike),
                r=r,
                q=q,
                sigma=sigma_value,
                t=t,
            )
            if vega <= 0.0:
                continue
            weight = vega * abs(leg.position) * leg.multiplier
            weights.append(weight)
            sigmas.append(sigma_value)
        if weights:
            total_weight = sum(weights)
            weighted_sigma = sum(w * s for w, s in zip(weights, sigmas)) / total_weight
            return max(weighted_sigma, EPSILON)
        if atm_value is None:
            raise ValueError("ATM volatility is required for fallback.")
        return max(atm_value, EPSILON)

    raise ValueError(f"Unknown sigma mode: {mode}")


def strategy_pop(
    input: StrategyInput,
    payoff_fn,
    S0: float,
    r: float,
    q: float,
    sigma_mode: str = "ATM",
    atm_iv: float = 0.0,
    per_leg_iv: Optional[List[Optional[float]]] = None,
    t: float = 1.0,
    z_min: float = -5.0,
    z_max: float = 5.0,
    steps: int = 400,
    *,
    sigma: Optional[float] = None,
) -> float:
    """Strategy-level PoP via adaptive numerical integration over
    the risk-neutral lognormal distribution.

    Uses scipy.integrate.quad for machine-precision results.

    If ``sigma`` is provided directly it is used as-is; otherwise the
    effective sigma is resolved from *sigma_mode / atm_iv / per_leg_iv*.
    """
    if sigma is not None and sigma > 0:
        eff_sigma = sigma
    else:
        eff_sigma = effective_sigma(
            input, per_leg_iv or [], sigma_mode, r, q, t, atm_iv=atm_iv
        )

    if t <= 0.0 or eff_sigma <= 0.0:
        pnl = payoff_fn(input, S0)
        return 1.0 if pnl > 0.0 else 0.0

    if S0 <= 0:
        return 0.0

    mu = r - q
    log_std = eff_sigma * math.sqrt(t)
    log_mean = math.log(S0) + (mu - 0.5 * eff_sigma ** 2) * t
    S_low = max(0.01, math.exp(log_mean - 6 * log_std))
    S_high = math.exp(log_mean + 6 * log_std)

    def integrand(S_T):
        pnl = payoff_fn(input, S_T)
        if pnl > 0:
            return _lognormal_density(S_T, S0, mu, eff_sigma, t)
        return 0.0

    prob, _ = quad(integrand, S_low, S_high, limit=200, epsabs=EPSILON, epsrel=EPSILON)
    return max(0.0, min(1.0, prob))


def build_probability_details(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    breakevens: List[float],
    z_min: float = -5.0,
    z_max: float = 5.0,
    steps: int = 200,
) -> pd.DataFrame:
    if t <= 0.0 or sigma <= 0.0:
        df = pd.DataFrame(
            {"terminal_price": [S0], "cum_prob": [1.0], "breakeven": [False]}
        )
        return df

    step = (z_max - z_min) / (steps - 1)
    drift = (r - q - 0.5 * sigma * sigma) * t
    vol_term = sigma * math.sqrt(t)

    rows = []
    cum_prob = 0.0
    for i in range(steps):
        z = z_min + step * i
        prob = _norm_pdf(z) * step
        terminal = S0 * math.exp(drift + vol_term * z)
        cum_prob += prob
        rows.append({"terminal_price": terminal, "cum_prob": cum_prob})

    df = pd.DataFrame(rows)
    df["breakeven"] = False
    if breakevens:
        prices = df["terminal_price"].to_numpy()
        for breakeven in breakevens:
            idx = (abs(prices - breakeven)).argmin()
            df.at[idx, "breakeven"] = True
    return df


# ═══════════════════════════════════════════════════════════
# Assignment & Target Probability
# ═══════════════════════════════════════════════════════════

# Canonical Leg Dict Format (v2):
# {
#     "qty": 1,              # +1 = long, -1 = short
#     "type": "PUT",         # "CALL" or "PUT"
#     "strike": 380.0,
#     "premium": 7.175,      # per-share mid price
#     "iv": 0.28,            # decimal implied vol from Bloomberg
#     "dte": 40,             # days to expiry
#     "expiry": "2026-03-20",# ISO date string
# }


def compute_assignment_prob(
    legs: List[dict],
    spot: float,
    r: float,
    q: float,
    t: float,
) -> float:
    """Probability that ANY short option finishes ITM (assignment risk).

    Uses union probability with per-leg IV (not weighted average).
    P(A ∪ B) for non-overlapping ITM regions.
    """
    short_calls = []
    short_puts = []

    for leg in legs:
        qty = leg.get("qty", 0)
        try:
            qty = float(qty)
        except (TypeError, ValueError):
            continue
        if qty >= 0:
            continue

        opt_type = (leg.get("type") or "").upper()
        strike = leg.get("strike", 0.0)
        iv = leg.get("iv", 0.0)
        try:
            strike = float(strike)
            iv = float(iv) if iv else 0.0
        except (TypeError, ValueError):
            continue

        if iv <= 0:
            iv = 0.25  # fallback

        if opt_type == "CALL" and strike > 0:
            short_calls.append((strike, iv))
        elif opt_type == "PUT" and strike > 0:
            short_puts.append((strike, iv))

    if not short_calls and not short_puts:
        return 0.0

    prob_call_itm = 0.0
    if short_calls:
        min_k, min_iv = min(short_calls, key=lambda x: x[0])
        prob_call_itm = prob_finish_above(spot, min_k, r, q, min_iv, t)

    prob_put_itm = 0.0
    if short_puts:
        max_k, max_iv = max(short_puts, key=lambda x: x[0])
        prob_put_itm = prob_finish_below(spot, max_k, r, q, max_iv, t)

    return min(1.0, prob_call_itm + prob_put_itm)


def compute_target_prob(
    input_obj: StrategyInput,
    payoff_fn,
    S0: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    target_pnl: float,
) -> float:
    """Probability of strategy P&L exceeding a specific dollar threshold.

    Same integration as strategy_pop but with a non-zero threshold.
    """
    if t <= 0 or sigma <= 0 or S0 <= 0:
        return 0.0

    mu = r - q
    log_std = sigma * math.sqrt(t)
    log_mean = math.log(S0) + (mu - 0.5 * sigma ** 2) * t
    S_low = max(0.01, math.exp(log_mean - 6 * log_std))
    S_high = math.exp(log_mean + 6 * log_std)

    def integrand(S_T):
        pnl = payoff_fn(input_obj, S_T)
        if pnl > target_pnl:
            return _lognormal_density(S_T, S0, mu, sigma, t)
        return 0.0

    prob, _ = quad(integrand, S_low, S_high, limit=200, epsabs=EPSILON, epsrel=EPSILON)
    return max(0.0, min(1.0, prob))


# ═══════════════════════════════════════════════════════════
# Greeks (Black-Scholes)
# ═══════════════════════════════════════════════════════════

# --- Phase 4: Multi-expiry Greeks (not yet integrated) ---
# These Black-Scholes Greek functions are implemented but not currently called.
# They will be needed for Phase 4 (multi-expiry engine) to value surviving legs
# after near-term expiry using theoretical pricing.

def bs_price(S: float, K: float, r: float, q: float, sigma: float, t: float,
             option_type: str = "CALL") -> float:
    """European option price via Black-Scholes."""
    d1 = _bs_d1(S, K, r, q, sigma, t)
    d2 = bs_d2(S, K, r, q, sigma, t)
    if d1 is None:
        return 0.0
    if option_type.upper() == "CALL":
        return S * math.exp(-q * t) * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    else:
        return K * math.exp(-r * t) * norm.cdf(-d2) - S * math.exp(-q * t) * norm.cdf(-d1)


def bs_delta(S: float, K: float, r: float, q: float, sigma: float, t: float,
             option_type: str = "CALL") -> float:
    """Delta: dV/dS."""
    d1 = _bs_d1(S, K, r, q, sigma, t)
    if d1 is None:
        return 0.0
    if option_type.upper() == "CALL":
        return math.exp(-q * t) * norm.cdf(d1)
    else:
        return math.exp(-q * t) * (norm.cdf(d1) - 1)


def bs_gamma(S: float, K: float, r: float, q: float, sigma: float, t: float) -> float:
    """Gamma: d²V/dS² (same for calls and puts)."""
    if S <= 0 or t <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, r, q, sigma, t)
    if d1 is None:
        return 0.0
    return math.exp(-q * t) * norm.pdf(d1) / (S * sigma * math.sqrt(t))


def bs_theta(S: float, K: float, r: float, q: float, sigma: float, t: float,
             option_type: str = "CALL") -> float:
    """Theta: dV/dt (per year). Divide by 365 for daily."""
    if S <= 0 or t <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, r, q, sigma, t)
    d2 = bs_d2(S, K, r, q, sigma, t)
    if d1 is None:
        return 0.0
    sqrt_t = math.sqrt(t)
    term1 = -(S * math.exp(-q * t) * norm.pdf(d1) * sigma) / (2 * sqrt_t)
    if option_type.upper() == "CALL":
        return term1 - r * K * math.exp(-r * t) * norm.cdf(d2) + q * S * math.exp(-q * t) * norm.cdf(d1)
    else:
        return term1 + r * K * math.exp(-r * t) * norm.cdf(-d2) - q * S * math.exp(-q * t) * norm.cdf(-d1)


def bs_rho(S: float, K: float, r: float, q: float, sigma: float, t: float,
           option_type: str = "CALL") -> float:
    """Rho: dV/dr."""
    d2 = bs_d2(S, K, r, q, sigma, t)
    if option_type.upper() == "CALL":
        return K * t * math.exp(-r * t) * norm.cdf(d2)
    else:
        return -K * t * math.exp(-r * t) * norm.cdf(-d2)


# ═══════════════════════════════════════════════════════════
# Convenience: compute_all_probabilities
# ═══════════════════════════════════════════════════════════

def compute_all_probabilities(
    input_obj: StrategyInput,
    payoff_fn,
    legs: List[dict],
    spot: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    max_profit: Optional[float] = None,
) -> dict:
    """Compute all probability metrics for display.

    Args:
        input_obj: StrategyInput for payoff computation.
        payoff_fn: callable(StrategyInput, price) -> total strategy PnL.
        legs: list of leg dicts (each with qty, type, strike, premium, iv).
        spot: current spot price.
        r: risk-free rate (decimal).
        q: dividend yield (decimal).
        sigma: pre-computed effective sigma (vega-weighted or ATM).
        t: time to expiry (years).
        max_profit: pre-computed max profit.  If None, estimated from grid.

    Returns:
        dict with all probability metrics.
    """
    result: dict = {}

    # PoP
    pop = strategy_pop(input_obj, payoff_fn, spot, r, q, t=t, sigma=sigma)
    result["pop"] = pop
    result["pop_pct"] = f"{pop * 100:.1f}%"

    # Assignment probability
    assign = compute_assignment_prob(legs, spot, r, q, t)
    result["assignment_prob"] = assign
    result["assignment_prob_pct"] = f"{assign * 100:.1f}%"

    # Estimate max_profit from price grid when not supplied
    if max_profit is None or max_profit <= 0:
        import numpy as np
        prices = np.linspace(max(0.01, spot * 0.01), spot * 3.0, 500)
        pnls = [payoff_fn(input_obj, p) for p in prices]
        max_profit = max(pnls) if pnls else 0.0

    # Target profit probabilities
    if max_profit > 0:
        result["prob_25_profit"] = compute_target_prob(
            input_obj, payoff_fn, spot, r, q, sigma, t, max_profit * 0.25,
        )
        result["prob_50_profit"] = compute_target_prob(
            input_obj, payoff_fn, spot, r, q, sigma, t, max_profit * 0.50,
        )
        result["prob_100_profit"] = compute_target_prob(
            input_obj, payoff_fn, spot, r, q, sigma, t, max_profit * 0.99,
        )
    else:
        result["prob_25_profit"] = None
        result["prob_50_profit"] = None
        result["prob_100_profit"] = None

    def fmt(v):
        return f"{v * 100:.1f}%" if v is not None else "--"

    result["prob_25_pct"] = fmt(result["prob_25_profit"])
    result["prob_50_pct"] = fmt(result["prob_50_profit"])
    result["prob_100_pct"] = fmt(result["prob_100_profit"])
    result["iv_used"] = sigma
    result["iv_used_pct"] = f"{sigma * 100:.1f}%"
    result["max_profit_ref"] = max_profit

    return result
