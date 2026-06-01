"""Read/own the UI layer's runtime state.

The UI never recomputes — it reads what ``run_insight_engine`` already
produced and attached to ``PortfolioState``. The singleton PortfolioState is
OWNED here (``_RUNTIME``), because this module is only ever imported as
``pm.ui.state_access`` — never executed as ``__main__``. ``pm/app.py`` is the
entry point and, under ``python -m pm.app``, runs as ``__main__``; a global
stored there would be a *different* object from the ``pm.app`` callbacks
import, so the state would be invisible to them. Owning it here gives one
canonical instance for both the entry point and every callback.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from pm.ingest.position_builder import Position
from pm.insight.patterns import Fire
from pm.insight.signal_library import SignalDict, SignalValue
from pm.store.portfolio_state import AccountState, PortfolioState


# ---------------------------------------------------------------------------
# Signal-sheet group catalog (display order + display names per Part 1/3.9).
# Groups A–D, F come from AccountState.signals[underlying]; group E is
# per-position and read from AccountState.position_signals[position_id].
# ---------------------------------------------------------------------------

GROUP_A = ("A — Trend & Momentum", [
    ("spot_vs_50d_ma", "Spot vs 50d MA"),
    ("spot_vs_200d_ma", "Spot vs 200d MA"),
    ("ma_stack_regime", "MA stack regime"),
    ("return_horizons", "Returns (1D / 5D / 3M / YTD / 1Y)"),
    ("rsi_14d_regime", "RSI 14d + regime"),
    ("distance_from_52w_high", "Distance from 52w high"),
    ("distance_from_52w_low", "Distance from 52w low"),
    ("vol_adjusted_move", "Vol-adjusted move (today)"),
])
GROUP_B = ("B — Volatility", [
    ("rv_30d", "Realized vol (30d)"),
    ("iv_1m_atm", "IV 1M ATM"),
    ("iv_3m_atm", "IV 3M ATM"),
    ("iv_6m_atm", "IV 6M ATM"),
    ("iv_3m_percentile_1y", "IV 3M percentile (1Y range)"),
    ("iv_term_structure", "IV term structure (3M − 6M)"),
    ("vrp_30d", "Vol risk premium (1M IV − 30d RV)"),
])
GROUP_C = ("C — Catalysts", [
    ("days_to_earnings", "Days to earnings"),
    ("earnings_implied_move", "Earnings implied move"),
    ("days_to_ex_div", "Days to ex-dividend"),
    ("dte_nearest_expiry_in_account", "DTE to nearest expiry (account)"),
])
GROUP_D = ("D — Sentiment & Ratings", [
    ("ubs_rating_and_target", "UBS rating / target / upside"),
    ("street_consensus_rating_and_target", "Street rating / target / upside"),
    ("ubs_analyst_note_recent", "UBS analyst note (recent)"),
])
GROUP_E = ("E — Position-specific", [
    ("position_size_pct_of_nav", "Position size (% of NAV)"),
    ("position_unrealized_pnl_pct", "P&L %"),
    ("option_captured_pct", "Premium captured (%)"),
    ("option_dte", "DTE"),
    ("option_moneyness", "Moneyness"),
])
GROUP_F = ("F — Composite", [
    ("composite_score", "Composite score (0–100)"),
])

# A–D, F come from the per-underlying SignalDict; E is per-position.
UNDERLYING_GROUPS = [GROUP_A, GROUP_B, GROUP_C, GROUP_D, GROUP_F]
POSITION_GROUP = GROUP_E


# ---------------------------------------------------------------------------
# Global runtime state — OWNED HERE.
#
# This must live in a module that is only ever imported as ``pm.ui.state_access``
# (never executed as ``__main__``). If the global lived in ``pm/app.py``,
# ``python -m pm.app`` would run that file as ``__main__`` — a *separate* module
# object from the ``pm.app`` that ``get_state`` imports — so state set at startup
# would be invisible to callbacks (get_state() → None → dead drawers). Keeping
# the singleton here guarantees one instance for both the entry point and every
# callback.
# ---------------------------------------------------------------------------

_RUNTIME: dict = {"state": None, "active_account": None}


def get_state() -> Optional[PortfolioState]:
    """Return the current global PortfolioState, or None if not loaded."""
    return _RUNTIME.get("state")


def set_state(state: Optional[PortfolioState],
              active_account: Optional[str] = None) -> Optional[PortfolioState]:
    """Install the global PortfolioState (called once at app build)."""
    _RUNTIME["state"] = state
    if active_account is not None:
        _RUNTIME["active_account"] = active_account
    return state


def reload_state() -> Optional[PortfolioState]:
    """Refresh the global PortfolioState in place. Returns the new state."""
    from pm.config import ADW_DATA_DIR
    from pm.store.portfolio_state import refresh_portfolio_state
    prev = _RUNTIME.get("state")
    new_state = refresh_portfolio_state(prev, ADW_DATA_DIR)
    _RUNTIME["state"] = new_state
    return new_state


# ---------------------------------------------------------------------------
# Fire / signal / position lookups
# ---------------------------------------------------------------------------

def all_fires(state: PortfolioState) -> list[Fire]:
    """Flat list of every fire across all accounts."""
    out: list[Fire] = []
    for acc in state.accounts.values():
        out.extend(acc.fires)
    return out


def fires_for_account(state: PortfolioState, account: str) -> list[Fire]:
    acc = state.accounts.get(account)
    return list(acc.fires) if acc else []


def fires_for_underlying(state: PortfolioState, account: str, underlying: str) -> list[Fire]:
    """All fires on a given underlying within one account (for the signal sheet)."""
    acc = state.accounts.get(account)
    if acc is None:
        return []
    return [f for f in acc.fires if f.underlying == underlying]


def fires_for_position(state: PortfolioState, account: str, position_id: str) -> list[Fire]:
    """All fires (alerts) on one position, most-severe first — for the modal's
    Alert view, which stacks every alert on a consolidated position row."""
    acc = state.accounts.get(account)
    if acc is None:
        return []
    fires = [f for f in acc.fires if f.position_id == position_id]
    return sorted(fires, key=lambda f: f.tier)


def signals_for_underlying(
    state: PortfolioState, account: str, underlying: str,
) -> Optional[SignalDict]:
    acc = state.accounts.get(account)
    if acc is None:
        return None
    return acc.signals.get(underlying)


def position_signals_for(
    state: PortfolioState, account: str, position_id: str,
) -> Optional[SignalDict]:
    """The merged per-position SignalDict (carries Group E), or None."""
    acc = state.accounts.get(account)
    if acc is None:
        return None
    return acc.position_signals.get(position_id)


def fire_by_id(
    state: PortfolioState, account: str, position_id: str, pattern_id: str,
) -> Optional[Fire]:
    """Locate a single fire for drawer rendering."""
    acc = state.accounts.get(account)
    if acc is None:
        return None
    for f in acc.fires:
        if f.position_id == position_id and f.pattern_id == pattern_id:
            return f
    return None


def position_by_id(
    state: PortfolioState, account: str, position_id: str,
) -> Optional[Position]:
    acc = state.accounts.get(account)
    if acc is None:
        return None
    for p in acc.positions:
        if p.position_id == position_id:
            return p
    return None


def positions_for_underlying(
    state: PortfolioState, account: str, underlying: str,
) -> list[Position]:
    """Held positions whose (underlying_symbol or symbol) == underlying."""
    acc = state.accounts.get(account)
    if acc is None:
        return []
    return [p for p in acc.positions
            if (p.underlying_symbol or p.symbol) == underlying]


# ---------------------------------------------------------------------------
# Snapshot access (for the signal-sheet header)
# ---------------------------------------------------------------------------

def bbg_ticker_for_underlying(
    state: PortfolioState, account: str, underlying: str,
) -> Optional[str]:
    """First BBG ticker we find for this bare-symbol underlying in the account."""
    acc = state.accounts.get(account)
    if acc is None:
        return None
    for p in acc.positions:
        if p.asset_class in ("equity", "fund_etf") and p.symbol == underlying:
            return p.bbg_ticker or None
        if p.asset_class == "option" and p.underlying_symbol == underlying:
            return p.underlying_bbg_ticker or None
    return None


def snapshot_row_for_underlying(
    state: PortfolioState, account: str, underlying: str,
) -> Optional[dict]:
    """The snapshot row (dict of BBG fields) for an underlying, or None.
    Read-only — pulls the row already fetched onto AccountState.snapshot."""
    acc = state.accounts.get(account)
    if acc is None:
        return None
    bbg = bbg_ticker_for_underlying(state, account, underlying)
    if not bbg:
        return None
    df = acc.snapshot.underlyings
    if df is None or df.empty or bbg not in df.index:
        return None
    series = df.loc[bbg]
    return {col: series[col] for col in df.columns}


# ---------------------------------------------------------------------------
# Small shared coercion helpers (display layer)
# ---------------------------------------------------------------------------

def is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def coerce_float(v: Any) -> Optional[float]:
    if is_missing(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
