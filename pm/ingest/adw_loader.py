"""Loader for the holdings-extract xlsx files.

The extract is a two-sheet workbook (`Holdings`, `Trades`) emitted by a
broker-data warehouse. Filenames are `adw_extract_YYYYMMDD_HHMMSS.xlsx`.

This module is the only piece of the codebase that reads the raw xlsx.
Downstream consumers (`pm.ingest.position_builder`, `pm.store`) take the
produced ``ADWExtract`` and never touch the file again.

Column-name normalization is hard-coded via the two mapping dicts below
for reviewability.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Column-name normalization maps (raw header in xlsx -> snake_case)
# ---------------------------------------------------------------------------

HOLDINGS_COLUMN_MAP: dict[str, str] = {
    "Account": "account",
    "Position Date": "position_date",
    "Asset Class": "asset_class",
    "Instrument Type": "instrument_type",
    "Product Name": "product_name",
    "Underlying Ticker": "underlying_ticker",
    "Underlying Name": "underlying_name",
    "Underlying ISIN": "underlying_isin",
    "Underlying Issuer Country Code": "underlying_issuer_country_code",
    "Option Type": "option_type",
    "Option Expiration": "option_expiration",
    "Option Strike": "option_strike",
    "Ticker Final": "ticker_final",
    "CUSIP Final": "cusip_final",
    "ISIN Final": "isin_final",
    "Issuer Country Code Final": "issuer_country_code_final",
    "ISIN Country Code Final": "isin_country_code_final",
    "Listing Hint Country Code": "listing_hint_country_code",
    "Quantity": "quantity",
    "Valuation Price": "valuation_price",
    "Market Value": "market_value",
    "Cost Basis": "cost_basis",
    "Unrealized P&L": "unrealized_pnl",
    "Unrealized P&L %": "unrealized_pnl_pct",
    "Tax Lot Count": "tax_lot_count",
    "Option Contract Key": "option_contract_key",
}

TRADES_COLUMN_MAP: dict[str, str] = {
    "Account": "account",
    "Trade Date": "trade_date",
    "Buy/Sell": "buy_sell",
    "Option Lifecycle Action": "option_lifecycle_action",
    "Asset Class": "asset_class",
    "Instrument Type": "instrument_type",
    "Product Name": "product_name",
    "Underlying Ticker": "underlying_ticker",
    "Underlying Name": "underlying_name",
    "Underlying ISIN": "underlying_isin",
    "Underlying Issuer Country Code": "underlying_issuer_country_code",
    "Option Type": "option_type",
    "Option Expiration": "option_expiration",
    "Option Strike": "option_strike",
    "Ticker Final": "ticker_final",
    "CUSIP Final": "cusip_final",
    "ISIN Final": "isin_final",
    "Issuer Country Code Final": "issuer_country_code_final",
    "ISIN Country Code Final": "isin_country_code_final",
    "Listing Hint Country Code": "listing_hint_country_code",
    "Quantity": "quantity",
    "Principal Amount": "principal_amount",
    "Trade Type Code": "trade_type_code",
    "Cancel Code": "cancel_code",
    "Option Contract Key": "option_contract_key",
}

# Asset Class raw value -> canonical
ASSET_CLASS_MAP: dict[str, str] = {
    "Option": "option",
    "Equity": "equity",
    "Fund / ETF": "fund_etf",
    "Cash": "cash",
    "Other": "other",
}

# Option Type raw -> canonical CALL/PUT
OPTION_TYPE_MAP: dict[str, str] = {
    "Call": "CALL",
    "Put": "PUT",
    "CALL": "CALL",
    "PUT": "PUT",
}

_HOLDINGS_NUMERIC_COLS = (
    "quantity", "valuation_price", "market_value", "cost_basis",
    "unrealized_pnl", "unrealized_pnl_pct", "tax_lot_count", "option_strike",
)
_TRADES_NUMERIC_COLS = (
    "quantity", "principal_amount", "trade_type_code", "option_strike",
)
_HOLDINGS_DATE_COLS = ("position_date", "option_expiration")
_TRADES_DATE_COLS = ("trade_date", "option_expiration")

_FILENAME_RE = re.compile(r"^adw_extract_(\d{8})_(\d{6})\.xlsx$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ADWExtract:
    """Result of loading a single extract xlsx file."""
    extract_ts: datetime
    source_path: Path
    holdings: pd.DataFrame
    trades: pd.DataFrame
    accounts: list[str]
    parse_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_latest_adw_extract(data_dir: Path) -> Path | None:
    """Return the newest `adw_extract_YYYYMMDD_HHMMSS.xlsx` in ``data_dir``
    by the filename timestamp. Returns None if no matching file exists.

    Filename timestamps (not mtimes) are used because xlsx files are often
    copied around with stale mtimes.
    """
    if not data_dir.exists():
        return None
    candidates: list[tuple[datetime, Path]] = []
    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        ts = _parse_filename_ts(path.name)
        if ts is not None:
            candidates.append((ts, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _parse_filename_ts(name: str) -> datetime | None:
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main load
# ---------------------------------------------------------------------------

def load_adw_extract(path: Path) -> ADWExtract:
    """Load the Holdings + Trades sheets, normalize columns and dtypes,
    and accumulate parse warnings.

    Raises ``ValueError`` if required columns are missing from either
    sheet. Never raises for individual bad rows — those become warnings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Extract not found: {path}")

    extract_ts = _parse_filename_ts(path.name)
    if extract_ts is None:
        extract_ts = datetime.fromtimestamp(path.stat().st_mtime)

    warnings: list[str] = []

    holdings = _load_sheet(
        path, sheet_name="Holdings",
        column_map=HOLDINGS_COLUMN_MAP,
        numeric_cols=_HOLDINGS_NUMERIC_COLS,
        date_cols=_HOLDINGS_DATE_COLS,
        warnings=warnings,
    )
    trades = _load_sheet(
        path, sheet_name="Trades",
        column_map=TRADES_COLUMN_MAP,
        numeric_cols=_TRADES_NUMERIC_COLS,
        date_cols=_TRADES_DATE_COLS,
        warnings=warnings,
    )

    holdings = _normalize_categoricals(holdings)
    trades = _normalize_categoricals(trades)

    _row_level_warnings(holdings, warnings)

    accounts: list[str]
    if "account" in holdings.columns:
        accounts = sorted(holdings["account"].dropna().astype(str).unique().tolist())
    else:
        accounts = []

    return ADWExtract(
        extract_ts=extract_ts,
        source_path=path,
        holdings=holdings,
        trades=trades,
        accounts=accounts,
        parse_warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Per-sheet helpers
# ---------------------------------------------------------------------------

def _load_sheet(
    path: Path,
    *,
    sheet_name: str,
    column_map: dict[str, str],
    numeric_cols: tuple[str, ...],
    date_cols: tuple[str, ...],
    warnings: list[str],
) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name)

    missing = [c for c in column_map if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Sheet '{sheet_name}' in {path.name} is missing required "
            f"column(s): {missing}. Found: {list(raw.columns)}"
        )

    df = raw.rename(columns=column_map).copy()

    for col in numeric_cols:
        if col == "quantity":
            mask_dash = df[col].astype(object).apply(
                lambda v: isinstance(v, str) and v.strip() == "-"
            )
            if mask_dash.any():
                warnings.append(
                    f"{sheet_name}: {int(mask_dash.sum())} row(s) had "
                    f"Quantity '-' (coerced to NaN; routed via asset_class)."
                )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = df[col].apply(lambda v: v.date() if isinstance(v, pd.Timestamp) else None)

    return df


def _normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    if "asset_class" in df.columns:
        df["asset_class"] = df["asset_class"].map(
            lambda v: ASSET_CLASS_MAP.get(v, v) if isinstance(v, str) else v
        )
    if "instrument_type" in df.columns:
        df["instrument_type"] = df["instrument_type"].map(
            lambda v: ASSET_CLASS_MAP.get(v, v) if isinstance(v, str) else v
        )
    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].map(
            lambda v: OPTION_TYPE_MAP.get(v, v) if isinstance(v, str) else v
        )
    return df


def _row_level_warnings(holdings: pd.DataFrame, warnings: list[str]) -> None:
    """Soft warnings for unexpected NaN combinations."""
    if "asset_class" not in holdings.columns:
        return

    options = holdings[holdings["asset_class"] == "option"]
    if "underlying_ticker" in options.columns:
        bad = options[options["underlying_ticker"].isna()]
        if not bad.empty:
            warnings.append(
                f"Holdings: {len(bad)} option row(s) have NaN underlying_ticker."
            )
    if "option_contract_key" in options.columns:
        bad = options[options["option_contract_key"].isna()]
        if not bad.empty:
            warnings.append(
                f"Holdings: {len(bad)} option row(s) have NaN option_contract_key."
            )

    non_cash_non_other = holdings[~holdings["asset_class"].isin(["cash", "other"])]
    if "ticker_final" in non_cash_non_other.columns:
        bad = non_cash_non_other[non_cash_non_other["ticker_final"].isna()]
        if not bad.empty:
            warnings.append(
                f"Holdings: {len(bad)} non-cash/non-other row(s) have NaN ticker_final."
            )
