"""Top-level in-memory state container for the Portfolio Manager app.

The state is keyed by account. All accounts load in one pass: one BBG
roundtrip across the union of unique underlyings, then per-account
slices feed the per-account context / recommendation / greeks /
diagnostics pipelines.

V1 is in-memory only. V2 introduces SQLite persistence for
day-over-day diffs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from pm.config import DEFAULT_RISK_FREE_RATE
from pm.core.bloomberg_client import is_bloomberg_available
from pm.core.composite_score import compute_all_composite_scores
from pm.core.pitch_synthesizer import synthesize_pitch
from pm.core.portfolio_diagnostics import (
    PortfolioDiagnostics,
    compute_portfolio_diagnostics,
)
from pm.core.portfolio_greeks import PortfolioGreeks, compute_portfolio_greeks
from pm.core.portfolio_signals import compute_per_underlying_signals
from pm.core.portfolio_snapshot import (
    PortfolioSnapshot,
    fetch_portfolio_snapshot,
)
from pm.core.position_context import PositionContext, build_position_contexts
from pm.core.recommender import Recommendation, compute_recommendations
from pm.ingest.adw_loader import ADWExtract, find_latest_adw_extract, load_adw_extract
from pm.ingest.position_builder import Position, build_positions


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-account state
# ---------------------------------------------------------------------------

@dataclass
class AccountState:
    """Everything Tab 2 needs for a single account.

    Note: two signal containers coexist by design.
      - ``signals_by_ticker``: legacy ``pm.core.portfolio_signals.Signal``
        objects keyed by BBG-format ticker (e.g. ``"AAPL US Equity"``).
        Produced by ``compute_per_underlying_signals`` and consumed by
        the legacy ``recommender.py``.
      - ``signals``: V1 insight ``SignalValue`` objects keyed by *bare
        symbol* (e.g. ``"AAPL"``).
        Produced by ``pm.insight.engine.run_insight_engine`` and
        consumed by the new pattern detectors / signal-sheet UI.

    The new ``fires`` list is populated by the insight engine in the
    same pass.
    """
    account: str
    nav: float
    positions: list[Position]
    snapshot: PortfolioSnapshot
    contexts: list[PositionContext]
    recommendations: list[Recommendation]
    diagnostics: PortfolioDiagnostics
    greeks: PortfolioGreeks
    composite_scores: dict
    signals_by_ticker: dict
    pitch_themes: list
    trades: pd.DataFrame
    trades_by_underlying: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Insight engine output (populated by run_insight_engine):
    signals: dict = field(default_factory=dict)  # underlying-symbol -> SignalDict (A–D, F)
    fires: list = field(default_factory=list)    # list[pm.insight.patterns.Fire]
    # Per-position merged SignalDict (underlying A–D/F layered with the
    # position's E group), keyed by Position.position_id. The engine already
    # builds this for the detectors; persisting it lets the signal-sheet UI
    # read Group E without recomputing.
    position_signals: dict = field(default_factory=dict)  # position_id -> SignalDict
    # Multi-leg structure groupings (covered call, collar, vertical, covered /
    # cash-secured put, straddle / strangle), detected in the load path after the
    # engine. The UI reads these and never recomputes. See pm.insight.structures.
    structures: list = field(default_factory=list)  # list[pm.insight.structures.Structure]


@dataclass
class PortfolioState:
    """Top-level state container. The single source of truth for the
    running app."""
    extract: ADWExtract
    loaded_at: datetime
    bloomberg_ok: bool
    accounts: dict[str, AccountState]
    all_warnings: list[str]
    # Pre-fetched BBG data the insight engine consumes (engine itself
    # does no I/O — all fetches happen here at load time).
    iv_histories: dict = field(default_factory=dict)        # BBG-ticker -> pd.Series
    ubs_data_by_ticker: dict = field(default_factory=dict)  # BBG-ticker -> dict
    ubs_note_dates_by_ticker: dict = field(default_factory=dict)  # BBG-ticker -> pd.Timestamp | None
    # Assumed short rate for the deep-ITM short-put carry check (estimate; see config).
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def load_portfolio_state(
    data_dir: Path,
    bbg_ok: Optional[bool] = None,
) -> PortfolioState:
    """End-to-end load: find the latest holdings extract, build positions, fetch
    a single BBG snapshot across the union of underlyings, then slice
    per account.

    ``bbg_ok``:
      - ``None``: probe Bloomberg via ``is_bloomberg_available()``.
      - ``True`` / ``False``: trust the caller (used by tests).
    """
    latest = find_latest_adw_extract(data_dir)
    if latest is None:
        raise FileNotFoundError(
            f"No `adw_extract_YYYYMMDD_HHMMSS.xlsx` found in {data_dir}."
        )

    extract = load_adw_extract(latest)
    positions = build_positions(extract)

    if bbg_ok is None:
        bbg_ok = is_bloomberg_available()

    global_snapshot = fetch_portfolio_snapshot(positions, bbg_ok)
    global_signals = compute_per_underlying_signals(
        global_snapshot.underlyings, bloomberg_available=bbg_ok,
    )

    # Pre-fetch BBG inputs the insight engine consumes (V1: IV history
    # for B5; UBS analyst override for D1; UBS analyst-note dates for D3).
    iv_histories, ubs_data, ubs_note_dates = _prefetch_insight_inputs(positions, bbg_ok)

    accounts: dict[str, AccountState] = {}
    for account_id in extract.accounts:
        accounts[account_id] = _build_account_state(
            account_id=account_id,
            all_positions=positions,
            all_trades=extract.trades,
            global_snapshot=global_snapshot,
            global_signals=global_signals,
        )

    all_warnings = list(extract.parse_warnings) + list(global_snapshot.fetch_warnings)
    for acc in accounts.values():
        if acc.greeks.warnings:
            sample = acc.greeks.warnings[:3]
            rest = len(acc.greeks.warnings) - len(sample)
            all_warnings.extend(f"[{acc.account}] {w}" for w in sample)
            if rest > 0:
                all_warnings.append(
                    f"[{acc.account}] … and {rest} more greeks warning(s)."
                )

    state = PortfolioState(
        extract=extract,
        loaded_at=datetime.now(),
        bloomberg_ok=bbg_ok,
        accounts=accounts,
        all_warnings=all_warnings,
        iv_histories=iv_histories,
        ubs_data_by_ticker=ubs_data,
        ubs_note_dates_by_ticker=ubs_note_dates,
    )

    # Run the insight engine, which writes signals + fires onto each
    # AccountState and may append further warnings.
    from pm.insight.engine import run_insight_engine
    run_insight_engine(state)

    # Detect multi-leg structures from the grouped holdings (pure; reads the
    # built positions, writes Structure proposals onto each AccountState).
    from pm.insight.structures import run_structure_detection
    run_structure_detection(state)

    # Structure-aware management fires (coverage breach, ex-div context, carry,
    # at-cap, pin, collar-monetize). Appends Fires to each AccountState and
    # annotates leg fires that belong to a confirmed structure.
    from pm.insight.structure_fires import run_structure_fires
    run_structure_fires(state)

    return state


def refresh_portfolio_state(
    current: PortfolioState | None,
    data_dir: Path,
) -> PortfolioState:
    """Re-runs ``load_portfolio_state``. Re-probes Bloomberg every time —
    one BDP call is cheap and avoids carrying stale connection state if
    the Terminal comes online mid-session. Logs whether the extract
    timestamp changed."""
    new_state = load_portfolio_state(data_dir, bbg_ok=None)
    if current is not None:
        if new_state.extract.extract_ts != current.extract.extract_ts:
            logger.info(
                "Refresh: extract timestamp changed %s -> %s",
                current.extract.extract_ts, new_state.extract.extract_ts,
            )
        else:
            logger.info(
                "Refresh: extract timestamp unchanged (%s).",
                new_state.extract.extract_ts,
            )
    return new_state


# ---------------------------------------------------------------------------
# Insight-input pre-fetch (BBG I/O happens here, not in the engine)
# ---------------------------------------------------------------------------

def _prefetch_insight_inputs(
    positions: list[Position], bbg_ok: bool,
) -> tuple[dict, dict, dict]:
    """Fetch the BBG-derived inputs the insight engine needs:
      - 1Y IV history per unique underlying BBG ticker (B5)
      - UBS-overridden analyst data per unique underlying (D1)
      - UBS analyst-note dates per unique underlying (D3)
    Returns (iv_histories, ubs_data_by_ticker, ubs_note_dates_by_ticker).
    All three are empty when ``bbg_ok`` is False or no underlyings are present.
    """
    iv_histories: dict = {}
    ubs_data: dict = {}
    ubs_note_dates: dict = {}
    if not bbg_ok:
        return iv_histories, ubs_data, ubs_note_dates

    tickers: set[str] = set()
    for p in positions:
        if p.asset_class in ("equity", "fund_etf") and p.bbg_ticker:
            tickers.add(p.bbg_ticker)
        elif p.asset_class == "option" and p.underlying_bbg_ticker:
            tickers.add(p.underlying_bbg_ticker)

    if not tickers:
        return iv_histories, ubs_data, ubs_note_dates

    # IV histories — one BDH call across all tickers
    try:
        from pm.core.bloomberg_client import fetch_iv_history
        iv_histories = fetch_iv_history(sorted(tickers), lookback_days=365) or {}
    except Exception as exc:
        logger.warning("Insight prefetch: IV history failed: %s", exc)

    # UBS analyst data — one BDP per ticker (helper does its own batching)
    try:
        from pm.core.bloomberg_client import fetch_ubs_analyst_data
        for t in sorted(tickers):
            try:
                ubs_data[t] = fetch_ubs_analyst_data(t) or {}
            except Exception as exc:
                logger.warning("Insight prefetch: UBS data for %s failed: %s", t, exc)
                ubs_data[t] = {}
    except Exception as exc:
        logger.warning("Insight prefetch: UBS data fetch wrapper failed: %s", exc)

    # UBS analyst-note dates (D3) — one batched BDP with the override pair
    try:
        from pm.core.bloomberg_client import fetch_ubs_analyst_note_dates, with_session
        with with_session() as query:
            note_df = fetch_ubs_analyst_note_dates(query, sorted(tickers))
        for t in sorted(tickers):
            ts = None
            if note_df is not None and not note_df.empty and t in note_df.index:
                val = note_df.loc[t, "analyst_note_date"]
                if not pd.isna(val):
                    ts = val
            ubs_note_dates[t] = ts
    except Exception as exc:
        logger.warning("Insight prefetch: UBS analyst-note dates failed: %s", exc)

    return iv_histories, ubs_data, ubs_note_dates


# ---------------------------------------------------------------------------
# Per-account assembly
# ---------------------------------------------------------------------------

def _build_account_state(
    *,
    account_id: str,
    all_positions: list[Position],
    all_trades: pd.DataFrame,
    global_snapshot: PortfolioSnapshot,
    global_signals: dict,
) -> AccountState:
    positions = [p for p in all_positions if p.account == account_id]
    # NAV = net asset value (signed sum of market values across all
    # positions in the account). Short option market values are negative
    # and subtract correctly. portfolio_pct_of_nav uses abs(mv)/nav.
    nav = sum(p.market_value for p in positions if p.market_value is not None)

    account_snapshot = _slice_snapshot(global_snapshot, positions)
    iv_pctl = _iv_pctl_dict(global_signals)

    contexts = build_position_contexts(
        positions, account_snapshot, account_nav=nav, iv_pctl_by_ticker=iv_pctl,
    )
    recommendations = compute_recommendations(contexts, global_signals)
    greeks = compute_portfolio_greeks(
        positions, account_snapshot.underlyings, account_snapshot.options,
    )
    diagnostics = compute_portfolio_diagnostics(
        positions, account_snapshot.underlyings,
    )

    account_signals = {
        ticker: sigs for ticker, sigs in global_signals.items()
        if ticker in {
            p.bbg_ticker for p in positions if p.asset_class in ("equity", "fund_etf")
        } | {
            p.underlying_bbg_ticker for p in positions if p.asset_class == "option"
        }
    }
    composite_scores = compute_all_composite_scores(account_signals)
    pitch_themes = synthesize_pitch(recommendations)

    trades = _filter_trades(all_trades, account_id)
    trades_by_underlying = _group_trades_by_underlying(trades, positions)

    return AccountState(
        account=account_id,
        nav=float(nav),
        positions=positions,
        snapshot=account_snapshot,
        contexts=contexts,
        recommendations=recommendations,
        diagnostics=diagnostics,
        greeks=greeks,
        composite_scores=composite_scores,
        signals_by_ticker=account_signals,
        pitch_themes=pitch_themes,
        trades=trades,
        trades_by_underlying=trades_by_underlying,
    )


def _slice_snapshot(
    global_snapshot: PortfolioSnapshot,
    positions: list[Position],
) -> PortfolioSnapshot:
    """Filter the global snapshot to only the tickers this account holds."""
    under_tickers: set[str] = set()
    opt_tickers: set[str] = set()
    for p in positions:
        if p.asset_class in ("equity", "fund_etf") and p.bbg_ticker:
            under_tickers.add(p.bbg_ticker)
        elif p.asset_class == "option":
            if p.underlying_bbg_ticker:
                under_tickers.add(p.underlying_bbg_ticker)
            if p.bbg_ticker:
                opt_tickers.add(p.bbg_ticker)

    under_df = global_snapshot.underlyings
    if not under_df.empty:
        keep = [t for t in under_tickers if t in under_df.index]
        under_df = under_df.loc[keep] if keep else under_df.iloc[0:0]

    opt_df = global_snapshot.options
    if not opt_df.empty:
        keep_opts = [t for t in opt_tickers if t in opt_df.index]
        opt_df = opt_df.loc[keep_opts] if keep_opts else opt_df.iloc[0:0]

    return PortfolioSnapshot(
        underlyings=under_df,
        options=opt_df,
        fetch_warnings=list(global_snapshot.fetch_warnings),
        bloomberg_available=global_snapshot.bloomberg_available,
    )


def _filter_trades(trades: pd.DataFrame, account_id: str) -> pd.DataFrame:
    if trades is None or trades.empty or "account" not in trades.columns:
        return trades.iloc[0:0] if trades is not None else pd.DataFrame()
    df = trades[trades["account"] == account_id].copy()
    if "trade_date" in df.columns:
        df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)
    return df


def _group_trades_by_underlying(
    trades: pd.DataFrame,
    positions: list[Position],
) -> dict[str, pd.DataFrame]:
    """Key by underlying_ticker for options and ticker_final for
    equities/funds. Each value is a Trade-sheet slice sorted by trade_date asc.

    The set of keys is the union of:
      - option underlying_tickers seen in the trades themselves, and
      - equity/fund tickers seen in the account's positions
    so that every position with potential trade activity gets a slot
    (possibly empty).
    """
    out: dict[str, pd.DataFrame] = {}
    if trades is None or trades.empty:
        for p in positions:
            if p.asset_class in ("equity", "fund_etf") and p.symbol:
                out.setdefault(p.symbol, pd.DataFrame(columns=trades.columns if trades is not None else []))
        return out

    sorted_trades = trades.sort_values("trade_date") if "trade_date" in trades.columns else trades

    # Options: key by underlying_ticker
    if "underlying_ticker" in sorted_trades.columns and "asset_class" in sorted_trades.columns:
        opt_trades = sorted_trades[sorted_trades["asset_class"] == "option"]
        for ticker, sub in opt_trades.groupby("underlying_ticker"):
            if isinstance(ticker, str) and ticker:
                out[ticker] = sub.reset_index(drop=True)

    # Equity / fund trades: key by ticker_final
    if "ticker_final" in sorted_trades.columns and "asset_class" in sorted_trades.columns:
        eq_trades = sorted_trades[sorted_trades["asset_class"].isin(("equity", "fund_etf"))]
        for ticker, sub in eq_trades.groupby("ticker_final"):
            if isinstance(ticker, str) and ticker:
                out[ticker] = sub.reset_index(drop=True)

    # Make sure every held equity/fund has a slot (possibly empty)
    empty_template = sorted_trades.iloc[0:0]
    for p in positions:
        if p.asset_class in ("equity", "fund_etf") and p.symbol:
            out.setdefault(p.symbol, empty_template)

    return out


def _iv_pctl_dict(signals_by_ticker: dict) -> dict[str, float]:
    """Extract iv_percentile.metric_value into a {ticker: pctl} dict."""
    out: dict[str, float] = {}
    for ticker, sigs in signals_by_ticker.items():
        for s in sigs:
            if s.signal_type == "iv_percentile":
                out[ticker] = s.metric_value
                break
    return out
