"""Ticker construction utilities — pure-Python subset of
adapters/bloomberg.py from Options-Tool-Python3.

Includes the option-ticker construction primitives. The live BBG
validation (`validate_tickers`, `resolve_option_ticker_from_strike`)
calls Bloomberg in production; here `validate_tickers` is left as a
stub that raises if invoked without a monkeypatch (deferred to
tim/core/bloomberg_client.py in prompt 2). The resolver itself is
ported intact so it can be tested via monkey-patched validator.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Sequence

import pandas as pd


# Sector keywords that may appear at the end of a Bloomberg ticker
MARKET_SECTOR_KEYWORDS = {"EQUITY", "INDEX", "CURNCY", "COMDTY"}

# Strike offset ladder used by resolve_option_ticker_from_strike to walk
# nearby strikes when the exact strike isn't tradeable.
DEFAULT_STRIKE_OFFSETS = [-0.5, 0.5, -1.0, 1.0, -2.5, 2.5, -5.0, 5.0]


def _normalize_put_call(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip().upper()
    if text in {"C", "CALL"}:
        return "CALL"
    if text in {"P", "PUT"}:
        return "PUT"
    return None


def format_bbg_expiry(expiry: object) -> str:
    if isinstance(expiry, (date, datetime, pd.Timestamp)):
        expiry_date = expiry.date() if isinstance(expiry, datetime) else expiry
    else:
        parsed = pd.to_datetime(expiry, errors="coerce")
        if pd.isna(parsed):
            raise ValueError("Invalid expiry provided for Bloomberg ticker.")
        expiry_date = parsed.date()
    month = expiry_date.month
    day = expiry_date.day
    year = expiry_date.year % 100
    return f"{month}/{day}/{year:02d}"


def _strip_sector_suffix(underlying: str) -> str:
    parts = underlying.strip().split()
    if not parts:
        return ""
    if parts[-1].upper() in MARKET_SECTOR_KEYWORDS:
        parts = parts[:-1]
    return " ".join(parts).strip()


def _infer_sector_suffix(underlying: str, sector_hint: Optional[str]) -> str:
    hint = (sector_hint or "").strip()
    if hint:
        hint_key = hint.upper()
        if hint_key in MARKET_SECTOR_KEYWORDS:
            return hint_key.title()
        return hint
    if "INDEX" in underlying.upper():
        return "Index"
    return "Equity"


def _format_strike_for_ticker(strike: float) -> str:
    try:
        value = float(strike)
    except (TypeError, ValueError):
        raise ValueError("Strike must be numeric for ticker construction.")
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text


def construct_option_ticker(
    underlying: str,
    expiry: str,
    put_call: str,
    strike: float,
    sector_hint: Optional[str] = None,
) -> str:
    base = _strip_sector_suffix(underlying)
    if not base:
        raise ValueError("Underlying is required for ticker construction.")
    side = _normalize_put_call(put_call)
    if side is None:
        raise ValueError("put_call must be CALL or PUT.")
    prefix = "C" if side == "CALL" else "P"
    expiry_text = format_bbg_expiry(expiry)
    strike_text = _format_strike_for_ticker(strike)
    sector = _infer_sector_suffix(underlying, sector_hint)
    return f"{base} {expiry_text} {prefix}{strike_text} {sector}".strip()


def validate_tickers(tickers: list[str]) -> pd.DataFrame:
    """Live BDP probe — DEFERRED TO PROMPT 2 (bloomberg_client.py).

    Returns the subset of tickers that resolve to a real Bloomberg
    contract. Tests monkey-patch this; production callers will be
    pointed at tim.core.bloomberg_client.validate_tickers when prompt
    2 lands.
    """
    raise NotImplementedError(
        "validate_tickers requires a live Bloomberg session — wire up "
        "tim.core.bloomberg_client in prompt 2 or monkeypatch this in tests."
    )


def resolve_option_ticker_from_strike(
    underlying: str,
    expiry: str,
    put_call: str,
    strike: float,
    sector_hint: Optional[str] = None,
    offsets: Optional[Sequence[float]] = None,
) -> Optional[str]:
    exact = construct_option_ticker(
        underlying, expiry, put_call, strike, sector_hint
    )
    exact_df = validate_tickers([exact])
    if not exact_df.empty:
        return exact

    offset_list = list(offsets) if offsets is not None else DEFAULT_STRIKE_OFFSETS
    candidates: list[str] = []
    seen: set[str] = set()
    for offset in offset_list:
        if offset == 0:
            continue
        candidate_strike = float(strike) + float(offset)
        if candidate_strike <= 0:
            continue
        ticker = construct_option_ticker(
            underlying, expiry, put_call, candidate_strike, sector_hint
        )
        key = ticker.upper()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(ticker)

    if not candidates:
        return None
    validated = validate_tickers(candidates)
    if validated.empty:
        return None
    valid_set = {
        str(value).strip().upper()
        for value in validated["security"].dropna().tolist()
    }
    for ticker in candidates:
        if ticker.upper() in valid_set:
            return ticker
    return None


def _value_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
