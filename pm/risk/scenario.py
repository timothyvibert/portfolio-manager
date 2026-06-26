"""Deterministic stress / scenario engine (risk rung 2, part 2b).

Pre-computed in the load path (``run_account_scenario``) -> ``AccountState.scenario``;
the UI renders, never recomputes. Built on the 2a pricing adapter
(``pm.risk.pricing_adapter``) -- the first engine-consuming risk view.

Pricer tiers (enforced):
  * the scenario TABLE prices each shocked state at **truth-CRR** (discrete-div,
    scalar, tree-node gamma) -- accurate points;
  * the portfolio P&L CURVE sweeps spot at **fast vectorized BS2002**
    (continuous-q) -- illustrative shape.
A sweep is NEVER priced at truth (~29 s on the book -- infeasible). Greeks / prices
come from the engine (the scenario boundary); the exposure rung's outputs are untouched.

Shock library (blueprint Section 5.2) -- co-moving axis shocks applied *together*:
market +-5/+-10/+-20 % (beta-mapped via the rung-1 SPX ``EQY_BETA``), vol +-5/+-10
points, crash (-20 % spot + 10 vol), melt-up (+15 % spot - 5 vol), rates +-50 bps
(parallel curve shift before ``pick_rate_for_dte``), time +1w / +1m (a full reprice
at the shorter tenor with the discrete-div schedule re-anchored to the shifted date --
NOT a theta extrapolation, so the divergent engine theta never touches the headline
P&L). A custom scenario (2c) plugs in as one more ``ShockSpec``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from pm.core.bloomberg_client import pick_rate_for_dte
from pm.pricing import american_crr, european, strategy
from pm.pricing.conventions import year_frac
from pm.risk.pricing_adapter import EngineLeg, build_engine_legs

logger = logging.getLogger(__name__)

BETA_FIELD = "EQY_BETA"          # rung-1 SPX-adjusted beta (snapshot.underlyings)
DEFAULT_BETA = 1.0               # full market participation when a name has no beta
SIGMA_FLOOR = 1e-4
_T_FLOOR = 1e-6
CURVE_SPAN_PCT = 25.0            # +- SPX move spanned by the P&L curve
CURVE_POINTS = 101


# --------------------------------------------------------------------------
# Shock library
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ShockSpec:
    """One co-moving scenario. ``spot_pct`` is an SPX move (beta-mapped per name);
    ``vol_pts`` an absolute vol-point shift; ``rate_bps`` a parallel curve shift;
    ``time_days`` a forward calendar-day shift."""
    name: str
    label: str
    spot_pct: float = 0.0
    vol_pts: float = 0.0
    rate_bps: float = 0.0
    time_days: int = 0

    def axes(self) -> dict:
        d = {}
        if self.spot_pct:
            d["spot_pct"] = self.spot_pct
        if self.vol_pts:
            d["vol_pts"] = self.vol_pts
        if self.rate_bps:
            d["rate_bps"] = self.rate_bps
        if self.time_days:
            d["time_days"] = self.time_days
        return d


SHOCK_LIBRARY: list[ShockSpec] = [
    ShockSpec("mkt_dn_20", "SPX -20%", spot_pct=-20.0),
    ShockSpec("mkt_dn_10", "SPX -10%", spot_pct=-10.0),
    ShockSpec("mkt_dn_5", "SPX -5%", spot_pct=-5.0),
    ShockSpec("mkt_up_5", "SPX +5%", spot_pct=5.0),
    ShockSpec("mkt_up_10", "SPX +10%", spot_pct=10.0),
    ShockSpec("mkt_up_20", "SPX +20%", spot_pct=20.0),
    ShockSpec("vol_up_10", "Vol +10 pts", vol_pts=10.0),
    ShockSpec("vol_up_5", "Vol +5 pts", vol_pts=5.0),
    ShockSpec("vol_dn_5", "Vol -5 pts", vol_pts=-5.0),
    ShockSpec("vol_dn_10", "Vol -10 pts", vol_pts=-10.0),
    ShockSpec("crash", "Crash (-20% spot, +10 vol)", spot_pct=-20.0, vol_pts=10.0),
    ShockSpec("meltup", "Melt-up (+15% spot, -5 vol)", spot_pct=15.0, vol_pts=-5.0),
    ShockSpec("rates_up_50", "Rates +50 bps", rate_bps=50.0),
    ShockSpec("rates_dn_50", "Rates -50 bps", rate_bps=-50.0),
    ShockSpec("time_1w", "Time +1 week", time_days=7),
    ShockSpec("time_1m", "Time +1 month", time_days=30),
]

# The pure-spot market shocks whose truth points double as the curve's CRR
# confidence-band anchors (co-moving crash/melt-up are excluded -- not pure spot).
_MARKET_BAND_SHOCKS = ("mkt_dn_20", "mkt_dn_10", "mkt_dn_5", "mkt_up_5", "mkt_up_10", "mkt_up_20")


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------
@dataclass
class ScenarioPoint:
    name: str
    label: str
    axes: dict
    pnl: float                       # account P&L under the shock (truth-CRR)
    pnl_pct: Optional[float]         # vs NAV
    attribution: dict                # {delta, gamma, vega, theta, residual}; theta caveated
    trace: dict


@dataclass
class AccountScenario:
    account: str
    as_of: date
    nav: Optional[float]
    scenarios: list[ScenarioPoint]   # ranked worst-loss first
    curve: dict                      # {x_pct, pnl, band_lo, band_hi, truth_x, truth_pnl, breakevens}
    n_priceable: int
    n_unpriceable: int
    div_modes: dict
    warnings: list
    trace: dict


# --------------------------------------------------------------------------
# Load-path entry point
# --------------------------------------------------------------------------
def run_account_scenario(state, today=None) -> None:
    """Attach a pre-computed ``AccountScenario`` to every account. Mirrors
    ``run_account_exposure``: pure, read-only over already-loaded state."""
    for acc in getattr(state, "accounts", {}).values():
        try:
            acc.scenario = compute_account_scenario(state, acc, today=today)
        except Exception as exc:  # noqa: BLE001 -- one bad account never blocks the load
            logger.warning("scenario compute failed for %s: %s",
                           getattr(acc, "account", "?"), exc)
            acc.scenario = None


def compute_account_scenario(state, account_state, today=None) -> AccountScenario:
    today_ts = _normalize_today(today)
    legs = [lg for lg in build_engine_legs(state, account_state, today=today_ts) if lg.priceable]
    n_all = len(build_engine_legs(state, account_state, today=today_ts))
    beta_map = _beta_map(account_state)
    equities = _equity_legs(account_state, beta_map)
    curve_pts = getattr(state, "risk_free_curve", None) or []

    # Base truth prices (per leg) + base truth greeks (for attribution, computed once).
    base_price = {lg.position_id: _truth_price(lg, lg.spot, lg.sigma, lg.r, lg.T, lg.today)
                  for lg in legs}
    base_greeks = {lg.position_id: (lg.greeks(mode="truth") or {}) for lg in legs}
    nav = _num(getattr(account_state, "nav", None))

    scenarios: list[ScenarioPoint] = []
    for shock in SHOCK_LIBRARY:
        pnl, attrib = _scenario_pnl(legs, equities, beta_map, base_price, base_greeks,
                                    curve_pts, today_ts, shock)
        scenarios.append(ScenarioPoint(
            name=shock.name, label=shock.label, axes=shock.axes(),
            pnl=pnl, pnl_pct=(pnl / nav if nav else None),
            attribution=attrib,
            trace={"pricer": "truth-CRR (per point)", "shock": shock.axes(),
                   "beta_source": f"BBG {BETA_FIELD} (SPX adjusted, rung 1)"},
        ))
    scenarios.sort(key=lambda s: s.pnl)   # worst loss first

    curve = _portfolio_curve(legs, equities, beta_map, scenarios, today_ts)

    div_modes: dict = {}
    for lg in legs:
        div_modes[lg.div_mode] = div_modes.get(lg.div_mode, 0) + 1

    return AccountScenario(
        account=getattr(account_state, "account", ""), as_of=today_ts.date(), nav=nav,
        scenarios=scenarios, curve=curve,
        n_priceable=len(legs), n_unpriceable=n_all - len(legs), div_modes=div_modes,
        warnings=[w for lg in legs for w in lg.warnings],
        trace={"pricer_table": "truth-CRR", "pricer_curve": "fast vectorized BS2002",
               "shock_count": len(SHOCK_LIBRARY), "as_of": today_ts.date().isoformat()},
    )


# --------------------------------------------------------------------------
# Scenario P&L (truth) + attribution
# --------------------------------------------------------------------------
def _scenario_pnl(legs, equities, beta_map, base_price, base_greeks, curve_pts,
                  today_ts, shock: ShockSpec):
    total = 0.0
    a_delta = a_gamma = a_vega = a_theta = 0.0
    # time shift (shared)
    if shock.time_days:
        today2 = today_ts + pd.Timedelta(days=shock.time_days)
        dt_bd = float(np.busday_count(today_ts.date(), today2.date()))
    else:
        today2, dt_bd = today_ts, 0.0
    shifted_curve = _shifted_curve(curve_pts, shock.rate_bps)

    for lg in legs:
        beta = beta_map.get(lg.underlying_bbg, DEFAULT_BETA)
        d_spot = lg.spot * beta * shock.spot_pct / 100.0
        s_shocked = lg.spot + d_spot
        sigma_shocked = max(lg.sigma + shock.vol_pts / 100.0, SIGMA_FLOOR)
        if shock.time_days:
            T = max(year_frac(today2.date(), lg.expiry), _T_FLOOR)
        else:
            T = lg.T
        r = _shocked_rate(lg, shifted_curve, today2, shock.rate_bps)
        px = _truth_price(lg, s_shocked, sigma_shocked, r, T, today2)
        mult = lg.qty * lg.multiplier
        total += mult * (px - base_price[lg.position_id])
        # attribution from base greeks (engine-native units)
        g = base_greeks.get(lg.position_id) or {}
        a_delta += mult * g.get("delta", 0.0) * d_spot
        a_gamma += mult * 0.5 * g.get("gamma", 0.0) * d_spot ** 2
        a_vega += mult * g.get("vega", 0.0) * shock.vol_pts
        a_theta += mult * g.get("theta", 0.0) * dt_bd

    for eq in equities:                                   # linear, market only
        d_spot = eq["spot"] * eq["beta"] * shock.spot_pct / 100.0
        total += eq["qty"] * d_spot
        a_delta += eq["qty"] * d_spot

    attrib = {"delta": a_delta, "gamma": a_gamma, "vega": a_vega, "theta": a_theta,
              "residual": total - (a_delta + a_gamma + a_vega + a_theta)}
    return total, attrib


# --------------------------------------------------------------------------
# Portfolio P&L curve (fast vectorized BS2002) + CRR confidence band
# --------------------------------------------------------------------------
def _portfolio_curve(legs, equities, beta_map, scenarios, today_ts) -> dict:
    grid = np.linspace(-CURVE_SPAN_PCT, CURVE_SPAN_PCT, CURVE_POINTS)
    pnl = np.zeros_like(grid)
    for lg in legs:
        beta = beta_map.get(lg.underlying_bbg, DEFAULT_BETA)
        s_arr = lg.spot * (1.0 + beta * grid / 100.0)
        p = np.asarray(strategy.price_leg(s_arr, lg.K, lg.T, lg.r, lg.q, lg.sigma,
                                          lg.opt_type, style=lg.style, mode="fast"),
                       dtype=float)
        p0 = float(strategy.price_leg(lg.spot, lg.K, lg.T, lg.r, lg.q, lg.sigma,
                                      lg.opt_type, style=lg.style, mode="fast"))
        pnl += lg.qty * lg.multiplier * (p - p0)
    for eq in equities:
        pnl += eq["qty"] * eq["spot"] * eq["beta"] * grid / 100.0

    # CRR confidence band: the pure-spot market scenarios are truth points at known
    # SPX moves; the gap to the fast curve at those moves is the BS2002-vs-CRR band.
    truth_x, truth_pnl = [], []
    by_name = {s.name: s for s in scenarios}
    for nm in _MARKET_BAND_SHOCKS:
        s = by_name.get(nm)
        if s is not None:
            truth_x.append(s.axes.get("spot_pct", 0.0))
            truth_pnl.append(s.pnl)
    band = _band_halfwidth(grid, pnl, truth_x, truth_pnl)

    return {
        "x_pct": grid.tolist(),
        "pnl": pnl.tolist(),
        "band_lo": (pnl - band).tolist(),
        "band_hi": (pnl + band).tolist(),
        "truth_x": truth_x,
        "truth_pnl": truth_pnl,
        "breakevens": _zero_crossings(grid, pnl),
        "x_label": "SPX move %", "pricer": "fast vectorized BS2002",
    }


def _band_halfwidth(grid, pnl_fast, truth_x, truth_pnl) -> np.ndarray:
    """A spot-varying band: |truth - fast| at the market points, interpolated onto
    the grid (0 where there is no anchor)."""
    if not truth_x:
        return np.zeros_like(grid)
    fast_at = np.interp(truth_x, grid, pnl_fast)
    gaps = np.abs(np.asarray(truth_pnl) - fast_at)
    order = np.argsort(truth_x)
    return np.interp(grid, np.asarray(truth_x)[order], gaps[order])


def _zero_crossings(grid, pnl) -> list:
    out: list = []
    for i in range(1, len(pnl)):
        if (pnl[i - 1] <= 0 <= pnl[i]) or (pnl[i - 1] >= 0 >= pnl[i]):
            if pnl[i] != pnl[i - 1]:
                t = -pnl[i - 1] / (pnl[i] - pnl[i - 1])
                x = float(grid[i - 1] + t * (grid[i] - grid[i - 1]))
                if not out or abs(x - out[-1]) > 1e-6:   # dedupe a point sitting on zero
                    out.append(x)
    return out


# --------------------------------------------------------------------------
# Truth pricing router (today-aware -- price_leg/price() lack a `today` param, so
# the time shock's discrete-div re-anchoring routes through the CRR core directly).
# --------------------------------------------------------------------------
def _truth_price(leg: EngineLeg, S, sigma, r, T, today) -> float:
    if leg.style == "American":
        if leg.divs_df is not None and len(leg.divs_df) > 0:
            return float(american_crr.crr_price(S, leg.K, T, r, sigma, leg.divs_df,
                                                leg.opt_type, today=pd.Timestamp(today)))
        return float(american_crr.crr_price_continuous_q(S, leg.K, T, r, leg.q, sigma,
                                                         leg.opt_type))
    return float(european.price(S, leg.K, T, r, leg.q, sigma, leg.opt_type))


def _shocked_rate(leg, shifted_curve, today2, rate_bps) -> float:
    dte = (pd.Timestamp(leg.expiry) - today2).days
    pick = pick_rate_for_dte(shifted_curve, dte)
    if pick and pick.get("rate") is not None:
        return pick["rate"]
    return leg.r + rate_bps / 10000.0          # fallback: shift the leg's base rate


def _shifted_curve(curve, rate_bps):
    if not rate_bps or not curve:
        return curve
    d = rate_bps / 10000.0
    return [{**pt, "rate": (pt["rate"] + d) if pt.get("rate") is not None else None}
            for pt in curve]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def _beta_map(account_state) -> dict:
    snap = getattr(getattr(account_state, "snapshot", None), "underlyings", None)
    out: dict = {}
    if snap is None or BETA_FIELD not in getattr(snap, "columns", []):
        return out
    for idx, val in snap[BETA_FIELD].items():
        b = _num(val)
        if b is not None:
            out[idx] = b
    return out


def _equity_legs(account_state, beta_map) -> list:
    snap = getattr(getattr(account_state, "snapshot", None), "underlyings", None)
    out = []
    for p in getattr(account_state, "positions", []) or []:
        if getattr(p, "asset_class", None) not in ("equity", "fund_etf"):
            continue
        bbg = getattr(p, "bbg_ticker", None)
        spot = _num(_snap_val(snap, bbg, "PX_LAST"))
        qty = _num(getattr(p, "quantity", None))
        if spot is None or qty is None:
            continue
        out.append({"bbg": bbg, "spot": spot, "qty": qty,
                    "beta": beta_map.get(bbg, DEFAULT_BETA)})
    return out


def _snap_val(snap, idx, col):
    if snap is None or idx is None:
        return None
    try:
        if idx not in snap.index or col not in getattr(snap, "columns", []):
            return None
        v = snap.loc[idx, col]
    except Exception:  # noqa: BLE001
        return None
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v) else None
    return v


def _normalize_today(today) -> pd.Timestamp:
    if today is None:
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(today).normalize()


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None
