"""Per-position derived metrics for the recommendation engine.

Builds a PositionContext from (position row, snapshot row, option snapshot,
portfolio totals) so rule functions don't recompute the same things.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from tim.core.holdings_parser import ParsedPortfolio
from tim.core.portfolio_signals import FIELDS


@dataclass
class PositionContext:
    """All the derived facts a recommendation rule might need."""
    # Identification
    bbg_ticker: str
    underlying_symbol: str
    underlying_bbg_ticker: str
    instrument_type: str            # 'equity' | 'option' | 'warrant' | 'other'

    # For options
    right: Optional[str]            # 'PUT' | 'CALL' | None for equity
    strike: Optional[float]
    expiry: Optional[date]
    dte: Optional[int]

    # Position economics
    quantity: float                  # signed (negative = short)
    multiplier: int                  # 1 for equity, 100 for option
    cost_basis: Optional[float]      # signed
    market_value: float              # signed
    pct_pnl: Optional[float]         # +0.50 = 50% profit, regardless of side

    # Live market
    spot: Optional[float]
    delta: Optional[float]           # raw greek (per-share); not multiplied by qty
    vega: Optional[float]
    theta: Optional[float]
    gamma: Optional[float]
    iv_1m: Optional[float]
    iv_3m: Optional[float]

    # Derived
    moneyness: Optional[float]       # signed: positive = ITM
    pct_captured: Optional[float]    # = pct_pnl
    portfolio_pct_of_nav: float
    has_long_stock: bool
    has_short_calls: bool
    has_short_puts: bool
    has_long_protection: bool

    # Underlying-level convenience copies
    rsi_14d: Optional[float]
    earnings_date: Optional[date]
    days_to_earnings: Optional[int]
    beta: Optional[float]
    iv_percentile: Optional[float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signed_moneyness(spot, strike, right) -> Optional[float]:
    """ITM-positive moneyness. For calls: (S-K)/K. For puts: -(S-K)/K = (K-S)/K."""
    if spot is None or strike is None or strike == 0:
        return None
    try:
        if pd.isna(spot) or pd.isna(strike):
            return None
    except (TypeError, ValueError):
        return None
    raw = (float(spot) - float(strike)) / float(strike)
    if right == "CALL":
        return raw
    if right == "PUT":
        return -raw
    return None


def _dte(expiry, today: Optional[date] = None) -> Optional[int]:
    if expiry is None:
        return None
    today = today or date.today()
    try:
        return (expiry - today).days
    except Exception:
        return None


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _spot_for(ticker, snapshot_underlyings) -> Optional[float]:
    if (snapshot_underlyings is None or snapshot_underlyings.empty
            or ticker not in snapshot_underlyings.index
            or "PX_LAST" not in snapshot_underlyings.columns):
        return None
    return _safe_float(snapshot_underlyings.loc[ticker, "PX_LAST"])


def _greek_for(opt_ticker, snapshot_options, col) -> Optional[float]:
    if (snapshot_options is None or snapshot_options.empty
            or opt_ticker not in snapshot_options.index
            or col not in snapshot_options.columns):
        return None
    return _safe_float(snapshot_options.loc[opt_ticker, col])


def _earnings_date(ticker, snapshot_underlyings) -> Optional[date]:
    col = FIELDS["earn_dt"]
    if (snapshot_underlyings is None or snapshot_underlyings.empty
            or ticker not in snapshot_underlyings.index
            or col not in snapshot_underlyings.columns):
        return None
    val = snapshot_underlyings.loc[ticker, col]
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None


def _build_overlay_map(portfolio: ParsedPortfolio) -> dict[str, dict[str, bool]]:
    """For each underlying symbol, compute booleans:
    has_long_stock, has_short_calls, has_short_puts, has_long_protection.
    Keyed by underlying_symbol (matches the parser's Symbol column).
    """
    out: dict[str, dict[str, bool]] = {}
    for sym in pd.concat([
        portfolio.equity_positions["symbol"]
            if not portfolio.equity_positions.empty else pd.Series(dtype=object),
        portfolio.option_positions["underlying_symbol"]
            if not portfolio.option_positions.empty else pd.Series(dtype=object),
    ]).dropna().unique():
        sym = str(sym)
        eq = (portfolio.equity_positions[
            portfolio.equity_positions["symbol"] == sym
        ] if not portfolio.equity_positions.empty
            else portfolio.equity_positions)
        op = (portfolio.option_positions[
            portfolio.option_positions["underlying_symbol"] == sym
        ] if not portfolio.option_positions.empty
            else portfolio.option_positions)
        out[sym] = {
            "has_long_stock":      bool(((eq.get("quantity", pd.Series(dtype=float)) > 0)).any())
                                   if not eq.empty else False,
            "has_short_calls":     bool((((op.get("right", pd.Series(dtype=object)) == "CALL")
                                          & (op.get("quantity", pd.Series(dtype=float)) < 0))).any())
                                   if not op.empty else False,
            "has_short_puts":      bool((((op.get("right", pd.Series(dtype=object)) == "PUT")
                                          & (op.get("quantity", pd.Series(dtype=float)) < 0))).any())
                                   if not op.empty else False,
            "has_long_protection": bool((((op.get("right", pd.Series(dtype=object)) == "PUT")
                                          & (op.get("quantity", pd.Series(dtype=float)) > 0))).any())
                                   if not op.empty else False,
        }
    return out


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_position_contexts(
    portfolio: ParsedPortfolio,
    snapshot,                 # PortfolioSnapshot
    iv_pctl_by_ticker: Optional[dict[str, float]] = None,
) -> list[PositionContext]:
    """Build a PositionContext for every equity + option position.

    ``snapshot`` is a PortfolioSnapshot with .underlyings (indexed by
    bbg_ticker) and .options (indexed by option bbg_ticker).
    """
    iv_pctl_by_ticker = iv_pctl_by_ticker or {}
    underlyings = snapshot.underlyings if hasattr(snapshot, "underlyings") else None
    options = snapshot.options if hasattr(snapshot, "options") else None

    overlay_map = _build_overlay_map(portfolio)

    nav = portfolio.portfolio_total.get("total_market_value") or 0.0
    nav = float(nav) if nav else 0.0

    contexts: list[PositionContext] = []

    # ---- Equity rows ------------------------------------------------------
    if not portfolio.equity_positions.empty:
        for _, p in portfolio.equity_positions.iterrows():
            sym = str(p.get("symbol") or "")
            bbg = str(p.get("bbg_ticker") or "")
            mv = _safe_float(p.get("market_value")) or 0.0
            spot = _spot_for(bbg, underlyings)
            ovl = overlay_map.get(sym, {})
            beta = _safe_float(
                underlyings.loc[bbg, FIELDS["beta"]]
                if (underlyings is not None and not underlyings.empty
                    and bbg in underlyings.index
                    and FIELDS["beta"] in underlyings.columns) else None
            )
            rsi = _safe_float(
                underlyings.loc[bbg, FIELDS["rsi_14d"]]
                if (underlyings is not None and not underlyings.empty
                    and bbg in underlyings.index
                    and FIELDS["rsi_14d"] in underlyings.columns) else None
            )
            iv_3m = _safe_float(
                underlyings.loc[bbg, FIELDS["iv_3m"]]
                if (underlyings is not None and not underlyings.empty
                    and bbg in underlyings.index
                    and FIELDS["iv_3m"] in underlyings.columns) else None
            )
            iv_1m = _safe_float(
                underlyings.loc[bbg, FIELDS["iv_1m"]]
                if (underlyings is not None and not underlyings.empty
                    and bbg in underlyings.index
                    and FIELDS["iv_1m"] in underlyings.columns) else None
            )
            ed = _earnings_date(bbg, underlyings)
            d2e = _dte(ed)

            contexts.append(PositionContext(
                bbg_ticker=bbg,
                underlying_symbol=sym,
                underlying_bbg_ticker=bbg,
                instrument_type="equity",
                right=None,
                strike=None,
                expiry=None,
                dte=None,
                quantity=_safe_float(p.get("quantity")) or 0.0,
                multiplier=int(p.get("multiplier") or 1),
                cost_basis=_safe_float(p.get("cost_basis")),
                market_value=mv,
                pct_pnl=_safe_float(p.get("pct_pnl")),
                spot=spot,
                delta=1.0,  # equity has unit delta
                vega=0.0, theta=0.0, gamma=0.0,
                iv_1m=iv_1m, iv_3m=iv_3m,
                moneyness=None,
                pct_captured=_safe_float(p.get("pct_pnl")),
                portfolio_pct_of_nav=(abs(mv) / nav) if nav else 0.0,
                has_long_stock=ovl.get("has_long_stock", False),
                has_short_calls=ovl.get("has_short_calls", False),
                has_short_puts=ovl.get("has_short_puts", False),
                has_long_protection=ovl.get("has_long_protection", False),
                rsi_14d=rsi,
                earnings_date=ed,
                days_to_earnings=d2e,
                beta=beta,
                iv_percentile=iv_pctl_by_ticker.get(bbg),
            ))

    # ---- Option rows ------------------------------------------------------
    if not portfolio.option_positions.empty:
        for _, p in portfolio.option_positions.iterrows():
            sym = str(p.get("underlying_symbol") or "")
            opt_ticker = str(p.get("bbg_ticker") or "")
            under_ticker = str(p.get("underlying_bbg_ticker") or "")
            mv = _safe_float(p.get("market_value")) or 0.0
            spot = _spot_for(under_ticker, underlyings)
            strike = _safe_float(p.get("strike"))
            right = p.get("right") if isinstance(p.get("right"), str) else None
            expiry = p.get("expiry")
            try:
                if isinstance(expiry, pd.Timestamp):
                    expiry = expiry.date()
            except Exception:
                pass
            dte = _dte(expiry) if isinstance(expiry, date) else None

            ovl = overlay_map.get(sym, {})
            beta = _safe_float(
                underlyings.loc[under_ticker, FIELDS["beta"]]
                if (underlyings is not None and not underlyings.empty
                    and under_ticker in underlyings.index
                    and FIELDS["beta"] in underlyings.columns) else None
            )
            rsi = _safe_float(
                underlyings.loc[under_ticker, FIELDS["rsi_14d"]]
                if (underlyings is not None and not underlyings.empty
                    and under_ticker in underlyings.index
                    and FIELDS["rsi_14d"] in underlyings.columns) else None
            )
            iv_3m = _safe_float(
                underlyings.loc[under_ticker, FIELDS["iv_3m"]]
                if (underlyings is not None and not underlyings.empty
                    and under_ticker in underlyings.index
                    and FIELDS["iv_3m"] in underlyings.columns) else None
            )
            iv_1m = _safe_float(
                underlyings.loc[under_ticker, FIELDS["iv_1m"]]
                if (underlyings is not None and not underlyings.empty
                    and under_ticker in underlyings.index
                    and FIELDS["iv_1m"] in underlyings.columns) else None
            )
            ed = _earnings_date(under_ticker, underlyings)
            d2e = _dte(ed)

            contexts.append(PositionContext(
                bbg_ticker=opt_ticker,
                underlying_symbol=sym,
                underlying_bbg_ticker=under_ticker,
                instrument_type="option",
                right=right,
                strike=strike,
                expiry=expiry if isinstance(expiry, date) else None,
                dte=dte,
                quantity=_safe_float(p.get("quantity")) or 0.0,
                multiplier=int(p.get("multiplier") or 100),
                cost_basis=_safe_float(p.get("cost_basis")),
                market_value=mv,
                pct_pnl=_safe_float(p.get("pct_pnl")),
                spot=spot,
                delta=_greek_for(opt_ticker, options, "delta_mid"),
                vega=_greek_for(opt_ticker, options, "vega"),
                theta=_greek_for(opt_ticker, options, "theta"),
                gamma=_greek_for(opt_ticker, options, "gamma"),
                iv_1m=iv_1m, iv_3m=iv_3m,
                moneyness=_signed_moneyness(spot, strike, right),
                pct_captured=_safe_float(p.get("pct_pnl")),
                portfolio_pct_of_nav=(abs(mv) / nav) if nav else 0.0,
                has_long_stock=ovl.get("has_long_stock", False),
                has_short_calls=ovl.get("has_short_calls", False),
                has_short_puts=ovl.get("has_short_puts", False),
                has_long_protection=ovl.get("has_long_protection", False),
                rsi_14d=rsi,
                earnings_date=ed,
                days_to_earnings=d2e,
                beta=beta,
                iv_percentile=iv_pctl_by_ticker.get(under_ticker),
            ))

    return contexts
