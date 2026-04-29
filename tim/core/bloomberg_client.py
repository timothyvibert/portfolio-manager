"""Bloomberg adapter for Portfolio-Manager / tim.

Ported from
``C:\\Users\\timot\\Documents\\Options-Tool-Python3\\adapters\\bloomberg.py``
with imports rewritten and unused functions omitted. Function bodies must not
diverge from the source — see ``tim/tests/test_bloomberg_client.py`` for the
pinned behaviors.

Scope (prompts 2 + 3):
  * Session context manager + lazy ``polars_bloomberg`` import.
  * ``fetch_spot``, ``fetch_underlying_snapshot``, ``fetch_underlying_snapshots``.
  * ``fetch_option_snapshot``, ``fetch_option_snapshots``  (prompt 3).
  * ``fetch_risk_free_rate`` (treasury picker).
  * ``fetch_ubs_analyst_data`` (BE998=UBS override).
  * ``is_bloomberg_available`` (startup probe — NEW, not in source).

Not yet ported (deferred to later prompts):
  * ``fetch_vol_surface`` / ``extract_atm_iv_for_expiry``         → prompt 4
  * ``bql_query`` / ``bql_query_raw``                             → prompt 4
  * ``_fetch_bds_rows`` / ``_parse_bds_dividend_rows`` /
    ``fetch_projected_dividend`` (real impl) /
    ``fetch_dividend_sum_to_expiry``                              → prompt 3+
  * ``validate_tickers`` / ``resolve_option_ticker_from_strike`` /
    ``build_leg_price_updates``                                   → prompt 5

NOTE: ``fetch_underlying_snapshot`` in the source calls
``fetch_projected_dividend`` (BDS-backed). To keep the ported function body
verbatim while honoring the OMIT list, ``fetch_projected_dividend`` is shipped
as a safe-default stub here; the BDS implementation lands in prompt 3.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import date, datetime
from typing import Dict, Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (verbatim from source)
# ---------------------------------------------------------------------------

OPTION_SNAPSHOT_FIELDS = [
    "BID",
    "ASK",
    "PX_MID",
    "MID",
    "PX_LAST",
    "IVOL_MID",
    "IVOL",
    "DAYS_TO_EXPIRATION",
    "DAYS_EXPIRE",
    "OPT_STRIKE_PX",
    "OPT_PUT_CALL",
    "DELTA_MID_RT",
    "THETA",
    "THETA_MID",
    "GAMMA",
    "VEGA",
    "RHO",
]

MARKET_SECTOR_KEYWORDS = {"EQUITY", "INDEX", "CURNCY", "COMDTY"}

UNDERLYING_FIELDS = [
    "PX_LAST",
    "NAME",
    "GICS_SECTOR_NAME",
    "INDUSTRY_SECTOR",
    "52WK_HIGH",
    "52WK_LOW",
    "HIGH_52WEEK",
    "LOW_52WEEK",
    "HIGH_DT_52WEEK",
    "LOW_DT_52WEEK",
    "CHG_PCT_1YR",
    "EQY_TRR_PCT_1YR",
    "CHG_PCT_5D",
    "CHG_PCT_3M",
    "CHG_PCT_YTD",
    "VOL_PERCENTILE",
    "3MTH_IMPVOL_100.0%MNY_DF",
    "EARNINGS_RELATED_IMPLIED_MOVE",
    "CHG_PCT_1D",
    "CHG_NET_1D",
    "6MTH_IMPVOL_100.0%MNY_DF",
    "DVD_YLD",
    "EQY_DVD_YLD_IND",
    "DVD_EX_DT",
    "EQY_DVD_EX_DT",
    "DVD_EX_DATE",
    "EQY_DVD_EX_DATE",
    "EXPECTED_REPORT_DT",
    "EARNINGS_ANNOUNCEMENT_DATE",
    "MOV_AVG_200D",   # 200-day moving average (used by Cash-Secured Put template)
    "PUT_CALL_OPEN_INTEREST_RATIO",
    "PUT_CALL_VOLUME_RATIO_CUR_DAY",
    # --- Added in prompt 5 (probed against AAPL + JPM, all populated) ---
    "BETA_ADJ_OVERRIDABLE",   # 2y daily beta vs market index, adjusted
    "RSI_14D",                # 14-day Wilder RSI
    "MOV_AVG_50D",            # 50-day SMA (paired with MOV_AVG_200D for trend stack)
    "CALL_IMP_VOL_30D",       # 30D call IV — substitute for 1M ATM IV (BDP doesn't
                              # populate 1MTH_IMPVOL_100.0%MNY_DF for most US names)
    "BEST_EPS",               # Consensus EPS estimate (next reporting period)
    "VOLATILITY_30D",         # 30-day realized vol (annualized %)
]
DIVIDEND_FIELDS = [
    "DVD_EX_DT",
    "EQY_DVD_EX_DT",
    "DVD_EX_DATE",
    "EQY_DVD_EX_DATE",
]


# Treasury index tickers ordered by maturity threshold (days)
_TREASURY_MAP = [
    (30,    "USGG1M Index",   "1M UST"),
    (90,    "USGG3M Index",   "3M UST"),
    (180,   "USGG6M Index",   "6M UST"),
    (365,   "USGG12M Index",  "12M UST"),
    (730,   "USGG2YR Index",  "2Y UST"),
    (1095,  "USGG3YR Index",  "3Y UST"),
    (1825,  "USGG5YR Index",  "5Y UST"),
    (2555,  "USGG7YR Index",  "7Y UST"),
    (3650,  "USGG10YR Index", "10Y UST"),
    (7300,  "USGG20YR Index", "20Y UST"),
    (99999, "USGG30YR Index", "30Y UST"),
]

_RFR_DEFAULT: dict = {"rate": 0.0, "ticker": "", "label": ""}


# ---------------------------------------------------------------------------
# Session + DataFrame helpers (verbatim)
# ---------------------------------------------------------------------------

@contextmanager
def with_session():
    from polars_bloomberg import BQuery

    with BQuery() as query:
        yield query


def _to_pandas(data: object) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    to_pandas = getattr(data, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()
    return pd.DataFrame(data)


def _ensure_security_column(df: pd.DataFrame) -> pd.DataFrame:
    if "security" in df.columns:
        return df
    for candidate in ["Security", "SECURITY", "ticker", "Ticker"]:
        if candidate in df.columns:
            return df.rename(columns={candidate: "security"})
    if df.index.name:
        return df.reset_index().rename(columns={df.index.name: "security"})
    return df.reset_index().rename(columns={"index": "security"})


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df


# ---------------------------------------------------------------------------
# Spot / snapshot (verbatim)
# ---------------------------------------------------------------------------

def fetch_spot(ticker: str) -> float:
    with with_session() as query:
        raw = query.bdp([ticker], ["PX_LAST"])
    df = _ensure_security_column(_to_pandas(raw))
    if "PX_LAST" not in df.columns or df.empty:
        return float("nan")
    row = df.loc[df["security"] == ticker]
    if row.empty:
        row = df.iloc[[0]]
    value = row["PX_LAST"].iloc[0]
    if pd.isna(value):
        return float("nan")
    return float(value)


def fetch_underlying_snapshot(ticker: str) -> pd.Series:
    if not ticker:
        return pd.Series(dtype=object)
    with with_session() as query:
        raw = query.bdp([ticker], UNDERLYING_FIELDS)
    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, UNDERLYING_FIELDS)
    if df.empty:
        return pd.Series(dtype=object)
    row = df.loc[df["security"] == ticker]
    if row.empty:
        row = df.iloc[[0]]
    record = row.iloc[0].copy()
    def _clean_value(value: object) -> object:
        try:
            if pd.isna(value):
                return None
        except Exception:
            logger.debug("pd.isna check failed for value: %r", value)
        return value

    high_52week = record.get("HIGH_52WEEK")
    if pd.isna(high_52week):
        high_52week = record.get("52WK_HIGH")
    low_52week = record.get("LOW_52WEEK")
    if pd.isna(low_52week):
        low_52week = record.get("52WK_LOW")
    record["high_52week"] = _clean_value(high_52week)
    record["low_52week"] = _clean_value(low_52week)
    record["week_52_high"] = record.get("high_52week")
    record["week_52_low"] = record.get("low_52week")
    record["high_dt_52week"] = _clean_value(record.get("HIGH_DT_52WEEK"))
    record["low_dt_52week"] = _clean_value(record.get("LOW_DT_52WEEK"))
    record["chg_pct_1yr"] = _clean_value(record.get("CHG_PCT_1YR"))
    record["eqy_trr_pct_1yr"] = _clean_value(record.get("EQY_TRR_PCT_1YR"))
    record["chg_pct_5d"] = _clean_value(record.get("CHG_PCT_5D"))
    record["chg_pct_3m"] = _clean_value(record.get("CHG_PCT_3M"))
    record["chg_pct_ytd"] = _clean_value(record.get("CHG_PCT_YTD"))
    record["vol_percentile"] = _clean_value(record.get("VOL_PERCENTILE"))
    record["impvol_3m_atm"] = _clean_value(record.get("3MTH_IMPVOL_100.0%MNY_DF"))
    record["chg_pct_1d"] = _clean_value(record.get("CHG_PCT_1D"))
    record["chg_net_1d"] = _clean_value(record.get("CHG_NET_1D"))
    record["impvol_6m_atm"] = _clean_value(record.get("6MTH_IMPVOL_100.0%MNY_DF"))
    record["earnings_related_implied_move"] = _clean_value(
        record.get("EARNINGS_RELATED_IMPLIED_MOVE")
    )
    bds_dividend = fetch_projected_dividend(ticker)
    bdp_ex_div_date = get_next_dividend_date(ticker, snapshot=record.to_dict())
    ex_div_date = bds_dividend.get("ex_div_date") or bdp_ex_div_date
    record["ex_div_date"] = ex_div_date
    record["projected_dividend"] = bds_dividend.get("projected_dividend")
    record["dividend_status"] = bds_dividend.get("dividend_status")
    record["dividend_debug"] = bds_dividend.get("debug") or {}
    record["mov_avg_200d"] = _clean_value(record.get("MOV_AVG_200D"))
    record["put_call_oi_ratio"] = _clean_value(record.get("PUT_CALL_OPEN_INTEREST_RATIO"))
    record["put_call_vol_ratio"] = _clean_value(record.get("PUT_CALL_VOLUME_RATIO_CUR_DAY"))
    analyst = fetch_ubs_analyst_data(ticker)
    record["ubs_rating"] = analyst.get("ubs_rating")
    record["ubs_target"] = analyst.get("ubs_target")
    return record


# ---------------------------------------------------------------------------
# UBS analyst data (verbatim)
# ---------------------------------------------------------------------------

def fetch_ubs_analyst_data(ticker: str) -> Dict[str, object]:
    """Fetch UBS-specific analyst rating and target price from Bloomberg.

    Uses BDP with BE998=UBS override for firm-specific data.
    Falls back to consensus (BEST_ANALYST_REC / BEST_TARGET_PRICE) if
    the polars_bloomberg override API is not available.
    """
    if not ticker:
        return {}
    fields = ["BEST_ANALYST_REC", "BEST_TARGET_PRICE"]
    result: Dict[str, object] = {}
    try:
        with with_session() as query:
            bdp_fn = getattr(query, "bdp", None)
            if bdp_fn is None:
                return result
            # polars_bloomberg BQuery.bdp overrides: list[tuple] | None
            try:
                raw = bdp_fn([ticker], fields, overrides=[("BE998", "UBS")])
            except TypeError:
                raw = bdp_fn([ticker], fields)
            except Exception:
                logger.debug("BDP call with overrides failed, retrying without")
                raw = bdp_fn([ticker], fields)
            df = _ensure_security_column(_to_pandas(raw))
            df = _ensure_columns(df, fields)
            if df.empty:
                return result
            row = df.loc[df["security"] == ticker]
            if row.empty:
                row = df.iloc[[0]]
            rec = row.iloc[0]
            rating = rec.get("BEST_ANALYST_REC")
            target = rec.get("BEST_TARGET_PRICE")
            if rating is not None:
                try:
                    if not pd.isna(rating):
                        result["ubs_rating"] = str(rating).strip()
                except (TypeError, ValueError):
                    result["ubs_rating"] = str(rating).strip()
            if target is not None:
                try:
                    if not pd.isna(target):
                        result["ubs_target"] = float(target)
                except (TypeError, ValueError):
                    pass
    except Exception as e:
        logger.warning("UBS analyst data fetch failed for %s: %s", ticker, e)
    return result


# ---------------------------------------------------------------------------
# Risk-free rate (verbatim)
# ---------------------------------------------------------------------------

def fetch_risk_free_rate(dte: int) -> dict:
    """Fetch risk-free rate from US Treasury indices matched to option DTE.

    Selects the first treasury index whose maturity threshold >= *dte*,
    fetches PX_LAST (yield as %), and returns a dict with the rate as a
    decimal (e.g. 4.5% → 0.045), the Bloomberg ticker, and a short label.
    Returns ``_RFR_DEFAULT`` on any failure.
    """
    if dte <= 0:
        return dict(_RFR_DEFAULT)
    # Pick the first bucket where dte <= threshold
    best = next((t for t in _TREASURY_MAP if dte <= t[0]), _TREASURY_MAP[-1])
    best_ticker, best_label = best[1], best[2]
    try:
        with with_session() as query:
            raw = query.bdp([best_ticker], ["PX_LAST"])
        df = _ensure_security_column(_to_pandas(raw))
        if "PX_LAST" not in df.columns or df.empty:
            return dict(_RFR_DEFAULT)
        row = df.loc[df["security"] == best_ticker]
        if row.empty:
            row = df.iloc[[0]]
        value = row["PX_LAST"].iloc[0]
        if pd.isna(value):
            return dict(_RFR_DEFAULT)
        return {"rate": float(value) / 100.0, "ticker": best_ticker, "label": best_label}
    except Exception:
        logger.warning("Risk-free rate lookup failed, using default")
        return dict(_RFR_DEFAULT)


# ---------------------------------------------------------------------------
# Generic helpers (verbatim)
# ---------------------------------------------------------------------------

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


def _has_market_sector(ticker: str) -> bool:
    parts = ticker.strip().split()
    if not parts:
        return False
    return parts[-1].upper() in MARKET_SECTOR_KEYWORDS


def resolve_security(user_ticker: str) -> str:
    if _has_market_sector(user_ticker):
        return user_ticker

    with with_session() as query:
        raw = query.bsrch(user_ticker)
    df = _ensure_security_column(_to_pandas(raw))
    if df.empty or "security" not in df.columns:
        return user_ticker
    first = df["security"].dropna()
    if first.empty:
        return user_ticker
    return str(first.iloc[0])


def normalize_iso_date(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if text == "":
        return None
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return text
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _normalize_dividend_date(value: object) -> Optional[date]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def get_next_dividend_date(
    ticker: str, snapshot: Optional[Dict[str, object]] = None
) -> Optional[date]:
    if snapshot:
        for field in DIVIDEND_FIELDS:
            value = snapshot.get(field)
            parsed = _normalize_dividend_date(value)
            if parsed is not None:
                return parsed

    if not ticker:
        return None

    with with_session() as query:
        raw = query.bdp([ticker], DIVIDEND_FIELDS)
    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, DIVIDEND_FIELDS)
    if df.empty:
        return None
    row = df.loc[df["security"] == ticker]
    if row.empty:
        row = df.iloc[[0]]
    record = row.iloc[0]
    for field in DIVIDEND_FIELDS:
        parsed = _normalize_dividend_date(record.get(field))
        if parsed is not None:
            return parsed
    return None


# ---------------------------------------------------------------------------
# fetch_projected_dividend STUB — full BDS-backed impl deferred to prompt 3
# ---------------------------------------------------------------------------

def fetch_projected_dividend(ticker: str) -> Dict[str, object]:
    """Stub. The real BDS-backed implementation is deferred to prompt 3
    along with ``_fetch_bds_rows`` / ``_parse_bds_dividend_rows`` /
    ``fetch_dividend_sum_to_expiry``. ``fetch_underlying_snapshot`` calls
    this; returning safe defaults preserves the source body verbatim
    without pulling in BDS.
    """
    return {
        "ex_div_date": None,
        "projected_dividend": None,
        "dividend_status": None,
        "debug": {},
    }


# ---------------------------------------------------------------------------
# Batched underlying snapshot — NEW (single BDP round trip for all tickers)
# ---------------------------------------------------------------------------

def fetch_underlying_snapshots(tickers: list[str]) -> pd.DataFrame:
    """Batch BDP for the full ``UNDERLYING_FIELDS`` set across N tickers.

    Returns a pandas DataFrame indexed by Bloomberg ticker (e.g.
    ``'AAPL US Equity'``), one column per field in ``UNDERLYING_FIELDS``
    plus ``security_name`` (= the ``NAME`` field). Missing values are
    NaN, never None and never the string ``'N.A.'`` / ``'#N/A N/A'``.

    On any error (no session, request rejected, partial response), logs
    at WARNING and returns an empty DataFrame with the expected columns.
    Does NOT raise.

    ``tickers`` must already include the market sector suffix (e.g.
    ``'AAPL US Equity'``, ``'RACE IM Equity'``). The parser produces
    these correctly; do NOT re-resolve here.
    """
    cols = ["security_name"] + list(UNDERLYING_FIELDS)
    if not tickers:
        return pd.DataFrame(columns=cols)

    try:
        with with_session() as query:
            raw = query.bdp(list(tickers), UNDERLYING_FIELDS)
    except Exception as exc:
        logger.warning("Batched BDP for underlyings failed: %s", exc)
        return pd.DataFrame(columns=cols)

    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, UNDERLYING_FIELDS)

    if df.empty:
        return pd.DataFrame(columns=cols)

    # Normalize Bloomberg sentinel strings to NaN. Bloomberg occasionally
    # ships "N.A." / "#N/A N/A" / "" rather than missing values.
    sentinels = {"N.A.", "#N/A N/A", "#N/A Field Not Applicable", ""}
    for col in UNDERLYING_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: pd.NA if isinstance(v, str) and v.strip() in sentinels else v
            )

    df["security_name"] = df.get("NAME", pd.NA)

    df = df.set_index("security")
    # Reindex so callers see a row per requested ticker, even if BBG dropped some
    df = df.reindex(list(tickers))
    return df[cols]


# ---------------------------------------------------------------------------
# Option snapshot — verbatim port + batch wrapper
# ---------------------------------------------------------------------------
#
# The source ``fetch_option_snapshot`` body normalizes BBG field aliases
# (DAYS_EXPIRE → dte, MID → PX_MID, IVOL_MID/IVOL → iv_mid, etc.) inline.
# That logic is lifted into ``_normalize_option_fields`` here so both the
# per-call function and the batch wrapper share it. The lift is mechanical
# — every line is identical to the source body.
#
# ``fetch_option_snapshot`` body itself is otherwise verbatim from
# adapters/bloomberg.py:434–542.

_OPTION_SNAPSHOT_OUTPUT_COLS = [
    "security",
    "BID",
    "ASK",
    "PX_MID",
    "PX_LAST",
    "IVOL_MID",
    "IVOL",
    "DAYS_TO_EXPIRATION",
    "DAYS_EXPIRE",
    "OPT_STRIKE_PX",
    "OPT_PUT_CALL",
    "DELTA_MID_RT",
    "THETA",
    "THETA_MID",
    "GAMMA",
    "VEGA",
    "RHO",
    "dte",
    "delta_mid",
    "theta",
    "gamma",
    "vega",
    "rho",
    "iv_mid",
]


def _normalize_option_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the source adapter's option-field normalization to a BDP frame.

    Coalesces MID into PX_MID, casts numeric fields, and exposes lowercase
    canonical columns (dte, delta_mid, theta, gamma, vega, rho, iv_mid).
    Verbatim from adapters/bloomberg.py:469–514.
    """
    if "MID" in df.columns:
        df["PX_MID"] = pd.to_numeric(df["PX_MID"], errors="coerce")
        mid_values = pd.to_numeric(df["MID"], errors="coerce")
        df["PX_MID"] = df["PX_MID"].fillna(mid_values)
    if "MID" in df.columns:
        df = df.drop(columns=["MID"])

    numeric_fields = [
        "BID",
        "ASK",
        "PX_MID",
        "PX_LAST",
        "IVOL_MID",
        "IVOL",
        "DAYS_TO_EXPIRATION",
        "DAYS_EXPIRE",
        "OPT_STRIKE_PX",
        "DELTA_MID_RT",
        "THETA",
        "THETA_MID",
        "GAMMA",
        "VEGA",
        "RHO",
    ]
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")

    dte = pd.to_numeric(df.get("DAYS_TO_EXPIRATION"), errors="coerce")
    if "DAYS_EXPIRE" in df.columns:
        dte_alt = pd.to_numeric(df.get("DAYS_EXPIRE"), errors="coerce")
        dte = dte.fillna(dte_alt)
    df["dte"] = dte

    df["delta_mid"] = pd.to_numeric(df.get("DELTA_MID_RT"), errors="coerce")
    theta = pd.to_numeric(df.get("THETA"), errors="coerce")
    if "THETA_MID" in df.columns:
        theta = theta.fillna(pd.to_numeric(df.get("THETA_MID"), errors="coerce"))
    df["theta"] = theta
    df["gamma"] = pd.to_numeric(df.get("GAMMA"), errors="coerce")
    df["vega"] = pd.to_numeric(df.get("VEGA"), errors="coerce")
    df["rho"] = pd.to_numeric(df.get("RHO"), errors="coerce")
    iv_mid = pd.to_numeric(df.get("IVOL_MID"), errors="coerce")
    if "IVOL" in df.columns:
        iv_mid = iv_mid.fillna(pd.to_numeric(df.get("IVOL"), errors="coerce"))
    df["iv_mid"] = iv_mid
    return df


def fetch_option_snapshot(option_tickers: list[str]) -> pd.DataFrame:
    if not option_tickers:
        return pd.DataFrame(columns=_OPTION_SNAPSHOT_OUTPUT_COLS)

    with with_session() as query:
        raw = query.bdp(option_tickers, OPTION_SNAPSHOT_FIELDS)
    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, OPTION_SNAPSHOT_FIELDS)
    df = _normalize_option_fields(df)
    return df[_OPTION_SNAPSHOT_OUTPUT_COLS]


def fetch_option_snapshots(option_tickers: list[str]) -> pd.DataFrame:
    """Batch BDP for ``OPTION_SNAPSHOT_FIELDS`` across N option tickers in
    a single Bloomberg round trip.

    Returns a pandas DataFrame indexed by option ticker (e.g.
    ``'BX US 5/15/26 P105 Equity'``). Greek columns follow the source
    adapter's normalization (``DAYS_EXPIRE`` → ``dte``, ``MID`` / ``PX_MID``
    coalesced, ``IVOL_MID`` / ``IVOL`` coalesced, ``DELTA_MID_RT`` exposed
    as ``delta_mid``). Missing values are NaN.

    On any error (no session, request rejected, partial response), logs at
    WARNING and returns an empty DataFrame with the expected columns.
    Does NOT raise.

    ``option_tickers`` must be Bloomberg-formatted (e.g.
    ``'BX US 5/15/26 P105 Equity'``). The parser produces these correctly.
    """
    cols = [c for c in _OPTION_SNAPSHOT_OUTPUT_COLS if c != "security"]
    if not option_tickers:
        return pd.DataFrame(columns=cols)

    try:
        with with_session() as query:
            raw = query.bdp(list(option_tickers), OPTION_SNAPSHOT_FIELDS)
    except Exception as exc:
        logger.warning("Batched BDP for options failed: %s", exc)
        return pd.DataFrame(columns=cols)

    df = _ensure_security_column(_to_pandas(raw))
    df = _ensure_columns(df, OPTION_SNAPSHOT_FIELDS)
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = _normalize_option_fields(df)
    df = df.set_index("security")
    df = df.reindex(list(option_tickers))
    # Same column order as fetch_option_snapshot, minus the 'security' col
    return df[cols]


# ---------------------------------------------------------------------------
# BDH historical fetchers — prompt 5
# ---------------------------------------------------------------------------

def _bdh_to_dict(
    df_in: object,
    field: str,
) -> dict[str, "pd.Series"]:
    """Coerce a polars-bloomberg BDH response into ``{ticker: Series}``."""
    df = _to_pandas(df_in)
    if df is None or df.empty:
        return {}

    # polars-bloomberg ships BDH responses with ['security', 'date', <field>]
    # columns. Defend against alternate shapes.
    df = _ensure_security_column(df)
    if "date" not in df.columns:
        for cand in ["DATE", "Date", "AS_OF_DATE", "asof_date"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break
    if "date" not in df.columns or field not in df.columns:
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df[field] = pd.to_numeric(df[field], errors="coerce")

    out: dict[str, pd.Series] = {}
    for ticker, sub in df.groupby("security"):
        s = sub.set_index("date")[field].sort_index()
        out[str(ticker)] = s
    return out


def fetch_price_history(
    tickers: list[str],
    lookback_days: int = 365,
) -> dict[str, "pd.Series"]:
    """BDH PX_LAST for N tickers over the lookback window. Returns
    ``{ticker: pd.Series of daily close prices indexed by date}``.

    On error, logs WARNING and returns ``{}``. Does NOT raise. Tickers
    that come back with no rows are silently omitted from the result.
    """
    if not tickers:
        return {}
    end = date.today()
    start = end - pd.Timedelta(days=lookback_days).to_pytimedelta()
    try:
        with with_session() as q:
            raw = q.bdh(
                securities=list(tickers),
                fields=["PX_LAST"],
                start_date=start,
                end_date=end,
            )
    except Exception as exc:
        logger.warning("BDH price-history fetch failed: %s", exc)
        return {}
    return _bdh_to_dict(raw, "PX_LAST")


def fetch_iv_history(
    tickers: list[str],
    lookback_days: int = 365,
    iv_field: str = "3MTH_IMPVOL_100.0%MNY_DF",
) -> dict[str, "pd.Series"]:
    """BDH for the specified IV field over lookback. Returns
    ``{ticker: pd.Series of daily IV indexed by date}``.

    Default ``iv_field`` is ``3MTH_IMPVOL_100.0%MNY_DF`` (3M ATM IV) —
    BDH-historical of 3M ATM IV does populate even when the BDP point-
    in-time of the same field doesn't. On error, returns ``{}``.
    """
    if not tickers:
        return {}
    end = date.today()
    start = end - pd.Timedelta(days=lookback_days).to_pytimedelta()
    try:
        with with_session() as q:
            raw = q.bdh(
                securities=list(tickers),
                fields=[iv_field],
                start_date=start,
                end_date=end,
            )
    except Exception as exc:
        logger.warning("BDH IV-history fetch failed: %s", exc)
        return {}
    return _bdh_to_dict(raw, iv_field)


# ---------------------------------------------------------------------------
# Startup probe — NEW
# ---------------------------------------------------------------------------

def is_bloomberg_available(probe_ticker: str = "AAPL US Equity") -> bool:
    """Probe the Bloomberg connection at startup. Returns True iff a BDP
    call for PX_LAST against ``probe_ticker`` succeeds.

    Wraps every failure path: missing ``polars_bloomberg`` install, no
    Bloomberg Terminal running, network/auth errors. Logs the failure
    reason at DEBUG. Never raises.
    """
    try:
        with with_session() as q:
            df = q.bdp(securities=[probe_ticker], fields=["PX_LAST"])
        # polars-bloomberg returns a polars DataFrame; check it has a row
        return df is not None and len(df) > 0
    except Exception as exc:
        logger.debug("Bloomberg probe failed: %s", exc)
        return False
