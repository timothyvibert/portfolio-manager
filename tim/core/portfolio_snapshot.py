"""Portfolio-level snapshot orchestration. Combines the parsed holdings with
live Bloomberg data into a single ``PortfolioSnapshot`` ready for display.

In prompt 3: underlying snapshot + option snapshot. Vol surface and BQL-driven
historical IV come in later prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tim.core.bloomberg_client import (
    OPTION_SNAPSHOT_FIELDS,
    UNDERLYING_FIELDS,
    fetch_option_snapshots,
    fetch_underlying_snapshots,
)
from tim.core.holdings_parser import ParsedPortfolio, get_unique_underlyings


@dataclass
class PortfolioSnapshot:
    """Snapshot bundle. v0.3 has underlying + option data."""
    underlyings: pd.DataFrame
    # index = underlying bbg_ticker, cols = UNDERLYING_FIELDS + 'security_name'
    options: pd.DataFrame
    # index = option bbg_ticker, cols = OPTION_SNAPSHOT_FIELDS + canonical greek cols
    fetch_warnings: list[str] = field(default_factory=list)
    bloomberg_available: bool = False


def fetch_portfolio_snapshot(
    portfolio: ParsedPortfolio,
    bloomberg_available: bool,
) -> PortfolioSnapshot:
    """Fetch live snapshots for every unique underlying and every option in
    the portfolio. Routes to a no-op result when bloomberg_available is
    False so the rest of the app can render gracefully without a Terminal.
    """
    if not bloomberg_available:
        return PortfolioSnapshot(
            underlyings=_empty_underlyings_df(),
            options=_empty_options_df(),
            fetch_warnings=["Bloomberg unavailable — snapshot skipped."],
            bloomberg_available=False,
        )

    warnings: list[str] = []

    # ---- Underlyings ------------------------------------------------------
    tickers = get_unique_underlyings(portfolio)
    if tickers:
        under_df = fetch_underlying_snapshots(tickers)
        missing_underlyings: list[str] = []
        if not under_df.empty:
            for t in tickers:
                if t not in under_df.index:
                    missing_underlyings.append(t)
                    continue
                if under_df.loc[t].isna().all():
                    missing_underlyings.append(t)
        else:
            missing_underlyings = list(tickers)
        if missing_underlyings:
            warnings.append(
                f"BBG returned no data for {len(missing_underlyings)} "
                f"underlying ticker(s): {missing_underlyings[:5]}"
                + ("..." if len(missing_underlyings) > 5 else "")
            )
    else:
        under_df = _empty_underlyings_df()
        warnings.append("No underlyings to fetch.")

    # ---- Options ----------------------------------------------------------
    if not portfolio.option_positions.empty:
        option_tickers = (
            portfolio.option_positions["bbg_ticker"].dropna().tolist()
        )
    else:
        option_tickers = []

    if option_tickers:
        opts_df = fetch_option_snapshots(option_tickers)
        missing_options: list[str] = []
        if not opts_df.empty:
            for t in option_tickers:
                if t not in opts_df.index:
                    missing_options.append(t)
                    continue
                if opts_df.loc[t].isna().all():
                    missing_options.append(t)
        else:
            missing_options = list(option_tickers)
        if missing_options:
            warnings.append(
                f"BBG returned no data for {len(missing_options)} "
                f"option ticker(s): {missing_options[:5]}"
                + ("..." if len(missing_options) > 5 else "")
            )
    else:
        opts_df = _empty_options_df()

    return PortfolioSnapshot(
        underlyings=under_df,
        options=opts_df,
        fetch_warnings=warnings,
        bloomberg_available=True,
    )


def _empty_underlyings_df() -> pd.DataFrame:
    cols = ["security_name"] + list(UNDERLYING_FIELDS)
    return pd.DataFrame(columns=cols)


def _empty_options_df() -> pd.DataFrame:
    """Columns mirror what fetch_option_snapshots produces (sans 'security'
    since it's the index).
    """
    cols = [
        "BID", "ASK", "PX_MID", "PX_LAST", "IVOL_MID", "IVOL",
        "DAYS_TO_EXPIRATION", "DAYS_EXPIRE", "OPT_STRIKE_PX", "OPT_PUT_CALL",
        "DELTA_MID_RT", "THETA", "THETA_MID", "GAMMA", "VEGA", "RHO",
        "dte", "delta_mid", "theta", "gamma", "vega", "rho", "iv_mid",
    ]
    return pd.DataFrame(columns=cols)
