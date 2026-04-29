"""Phase 1 portfolio-level diagnostics: sector exposure, weighted beta,
style mix, earnings calendar.

These are the 'macro view' aggregations a salesperson uses to set up the
'why this client needs to act' part of the conversation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from tim.core.holdings_parser import ParsedPortfolio
from tim.core.portfolio_signals import FIELDS


@dataclass
class PortfolioDiagnostics:
    sector_exposure: dict[str, float]    # GICS sector → fraction of equity NAV
    style_mix: dict[str, float]          # Value/Core/Growth/Eurozone → fraction
    weighted_beta: float | None          # NAV-weighted β across equity book
    earnings_calendar: list[dict]        # [{ticker, symbol, name, date,
                                         #   days_to_earnings}]
    warnings: list[str] = field(default_factory=list)


def compute_portfolio_diagnostics(
    portfolio: ParsedPortfolio,
    underlying_snapshot: pd.DataFrame,
) -> PortfolioDiagnostics:
    eq = portfolio.equity_positions.copy()
    if eq.empty:
        return PortfolioDiagnostics({}, {}, None, [], ["No equity positions."])

    # Pull only the fields we need from the snapshot (these may be missing
    # when BBG is unavailable).
    wanted = ["GICS_SECTOR_NAME", "security_name",
              FIELDS["beta"], FIELDS["earn_dt"]]
    cols = [c for c in wanted if c in underlying_snapshot.columns]
    snap_subset = (
        underlying_snapshot[cols].copy()
        if cols else pd.DataFrame(index=underlying_snapshot.index)
    )
    snap_subset = snap_subset.rename_axis("bbg_ticker").reset_index()
    eq = eq.merge(snap_subset, on="bbg_ticker", how="left")

    eq_nav = eq["market_value"].abs().sum()
    if eq_nav == 0:
        return PortfolioDiagnostics({}, {}, None, [], ["Zero equity NAV."])

    # ---- Sector exposure --------------------------------------------------
    sector_exp: dict[str, float] = {}
    if "GICS_SECTOR_NAME" in eq.columns:
        grouped = (
            eq.groupby("GICS_SECTOR_NAME", dropna=False)["market_value"].sum()
            / eq_nav
        ).fillna(0)
        for k, v in grouped.items():
            key = "Unclassified" if (k is None or pd.isna(k)) else str(k)
            sector_exp[key] = sector_exp.get(key, 0.0) + float(v)
    else:
        sector_exp = {"Unclassified": 1.0}

    # ---- Style mix (from holdings file `style` column) -------------------
    style_grouped = (
        eq.groupby("style", dropna=False)["market_value"].sum() / eq_nav
    ).fillna(0)
    style_mix: dict[str, float] = {}
    for k, v in style_grouped.items():
        key = "Unclassified" if (k is None or pd.isna(k)) else str(k)
        style_mix[key] = style_mix.get(key, 0.0) + float(v)

    # ---- Weighted beta ----------------------------------------------------
    beta_col = FIELDS["beta"]
    weighted_beta: float | None = None
    if beta_col in eq.columns:
        eligible = eq[eq[beta_col].notna() & (eq["market_value"].abs() > 0)]
        if not eligible.empty:
            num = (eligible[beta_col].astype(float) * eligible["market_value"]).sum()
            den = eligible["market_value"].sum()
            if den != 0:
                weighted_beta = float(num / den)

    # ---- Earnings calendar (next 30 days) ---------------------------------
    today = date.today()
    horizon = today + timedelta(days=30)
    earnings: list[dict] = []
    earn_col = FIELDS["earn_dt"]
    if earn_col in eq.columns:
        for _, r in eq.iterrows():
            ed = r.get(earn_col)
            if ed is None:
                continue
            try:
                if pd.isna(ed):
                    continue
            except (TypeError, ValueError):
                continue
            try:
                ed_date = pd.to_datetime(ed).date()
            except Exception:
                continue
            if today <= ed_date <= horizon:
                earnings.append({
                    "ticker": r["bbg_ticker"],
                    "symbol": r["symbol"],
                    "name": (r.get("security_name", "")
                             if isinstance(r.get("security_name", ""), str)
                             else ""),
                    "date": ed_date.isoformat(),
                    "days_to_earnings": (ed_date - today).days,
                })
    earnings.sort(key=lambda x: x["days_to_earnings"])

    warnings: list[str] = []
    if weighted_beta is None:
        warnings.append("Beta missing for all equities \u2014 weighted \u03b2 skipped.")

    return PortfolioDiagnostics(
        sector_exposure=sector_exp,
        style_mix=style_mix,
        weighted_beta=weighted_beta,
        earnings_calendar=earnings,
        warnings=warnings,
    )
