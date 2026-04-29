"""Dash app for Portfolio-Manager / tim. v0.8 adds drill-down transparency,
composite scores, rich earnings panel, narrative banner, and a density pass.

Cosmetic + transparency-only pass. The recommendation engine, signal logic,
and BBG client are untouched.
"""
from __future__ import annotations

import math
from typing import Optional

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import (
    Input, Output, State,
    ctx, dash_table, dcc, html, no_update,
)

from tim.config import DEFAULT_HOLDINGS_FILE, HOST, PORT
from tim.core.bloomberg_client import is_bloomberg_available
from tim.core.composite_score import compute_all_composite_scores
from tim.core.holdings_parser import parse_holdings
from tim.core.pitch_synthesizer import synthesize_pitch
from tim.core.portfolio_diagnostics import compute_portfolio_diagnostics
from tim.core.portfolio_greeks import compute_portfolio_greeks
from tim.core.portfolio_signals import (
    FIELDS,
    SIGNAL_DEFINITIONS,
    compute_per_underlying_signals,
)
from tim.core.portfolio_snapshot import fetch_portfolio_snapshot
from tim.core.position_context import build_position_contexts
from tim.core.recommender import (
    RECOMMENDATION_DEFINITIONS,
    compute_recommendations,
)


# ---------------------------------------------------------------------------
# Module-level state — populated in build_app(), read by drill-down callbacks.
# ---------------------------------------------------------------------------
_DASHBOARD_STATE: dict = {}


# Curated subset of UNDERLYING_FIELDS to display in the live snapshot.
DISPLAY_UNDERLYING_COLS: list[tuple[str, str]] = [
    ("security_name",                 "Name"),
    ("GICS_SECTOR_NAME",              "Sector"),
    ("PX_LAST",                       "Last"),
    ("CHG_PCT_1D",                    "1D %"),
    ("CHG_PCT_YTD",                   "YTD %"),
    ("3MTH_IMPVOL_100.0%MNY_DF",      "3M IV ATM"),
    ("6MTH_IMPVOL_100.0%MNY_DF",      "6M IV ATM"),
    ("HIGH_52WEEK",                   "52W High"),
    ("LOW_52WEEK",                    "52W Low"),
    ("EQY_DVD_YLD_IND",               "Div Yld %"),
    ("MOV_AVG_200D",                  "200D MA"),
]

_NUMERIC_DISPLAY_DECIMALS = {
    "PX_LAST": 2, "CHG_PCT_1D": 2, "CHG_PCT_YTD": 2,
    "3MTH_IMPVOL_100.0%MNY_DF": 1, "6MTH_IMPVOL_100.0%MNY_DF": 1,
    "HIGH_52WEEK": 2, "LOW_52WEEK": 2, "EQY_DVD_YLD_IND": 2,
    "MOV_AVG_200D": 2,
}


# ---------------------------------------------------------------------------
# Number formatting helpers
# ---------------------------------------------------------------------------

def _is_nan(x) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        try:
            return math.isnan(float(x))
        except (TypeError, ValueError):
            return False


def _fmt_dollars(x, signed: bool = False, decimals: int = 0) -> str:
    if _is_nan(x):
        return "\u2014"
    x = float(x)
    if x >= 0:
        sign = "+" if signed else ""
        return f"{sign}${x:,.{decimals}f}"
    return f"-${abs(x):,.{decimals}f}"


def _fmt_pct(x, decimals: int = 1, signed: bool = True) -> str:
    if _is_nan(x):
        return "\u2014"
    x = float(x)
    if signed:
        sign = "+" if x >= 0 else "-"
        return f"{sign}{abs(x):.{decimals}f}%"
    return f"{x:.{decimals}f}%"


def _fmt_num(x, decimals: int = 2) -> str:
    if _is_nan(x):
        return "\u2014"
    return f"{float(x):,.{decimals}f}"


def _iv_pctl_dict(signals_by_ticker):
    """Extract iv_percentile.metric_value into a {ticker: pctl} dict."""
    out = {}
    for ticker, sigs in signals_by_ticker.items():
        for s in sigs:
            if s.signal_type == "iv_percentile":
                out[ticker] = s.metric_value
                break
    return out


# ---------------------------------------------------------------------------
# Hero strip
# ---------------------------------------------------------------------------

def _hero_strip(portfolio, greeks, diagnostics, snapshot_warnings):
    nav = portfolio.portfolio_total["total_market_value"] or 0.0
    unrealized = portfolio.portfolio_total.get("total_unrealized") or 0.0
    cost_base = (nav - unrealized) if (nav and unrealized is not None) else 0.0
    upnl_pct = (unrealized / cost_base * 100) if cost_base else 0.0

    delta = greeks.totals.get("dollar_delta", 0.0)
    delta_pct = (greeks.totals.get("delta_pct_of_nav") or 0.0) * 100
    beta = diagnostics.weighted_beta

    cards = [
        html.Div(className="hero-card", children=[
            html.Div("Total NAV", className="label"),
            html.Div(_fmt_dollars(nav), className="value"),
            html.Div(
                f"{len(portfolio.equity_positions)} equities \u00b7 "
                f"{len(portfolio.option_positions)} options",
                className="sub",
            ),
        ]),
        html.Div(
            className=f"hero-card {'pos' if unrealized >= 0 else 'neg'}",
            children=[
                html.Div("Unrealized P&L", className="label"),
                html.Div(_fmt_dollars(unrealized, signed=True),
                         className="value"),
                html.Div(_fmt_pct(upnl_pct, signed=True) + " on cost",
                         className="sub"),
            ],
        ),
        html.Div(
            className=f"hero-card {'pos' if delta >= 0 else 'neg'}",
            children=[
                html.Div("Net $ Delta", className="label"),
                html.Div(_fmt_dollars(delta, signed=True), className="value"),
                html.Div(_fmt_pct(delta_pct, signed=True) + " of NAV",
                         className="sub"),
            ],
        ),
        html.Div(className="hero-card", children=[
            html.Div("Weighted \u03b2", className="label"),
            html.Div(
                _fmt_num(beta, decimals=2) if beta is not None else "\u2014",
                className="value",
            ),
            html.Div("vs. SPX", className="sub"),
        ]),
    ]

    children = [html.Div(cards, className="hero-row")]
    if snapshot_warnings:
        children.append(html.Details(
            className="bbg-disclosure",
            children=[
                html.Summary(
                    f"\u24d8 Bloomberg notices ({len(snapshot_warnings)})"
                ),
                html.Ul([html.Li(w) for w in snapshot_warnings]),
            ],
        ))
    return html.Div(children)


# ---------------------------------------------------------------------------
# Narrative banner
# ---------------------------------------------------------------------------

def _narrative_banner(portfolio, greeks, diagnostics, recommendations, themes):
    nav = portfolio.portfolio_total["total_market_value"] or 0.0
    delta = greeks.totals.get("dollar_delta", 0.0)
    delta_pct = (delta / nav * 100) if nav else 0.0
    beta = diagnostics.weighted_beta or 1.0

    if diagnostics.sector_exposure:
        top_sector = max(diagnostics.sector_exposure.items(),
                          key=lambda x: x[1])
    else:
        top_sector = (None, 0)
    sector_pct = top_sector[1] * 100

    yield_count = sum(len(t.recommendations) for t in themes
                      if t.theme_name == "Yield Enhancement")
    risk_count = sum(len(t.recommendations) for t in themes
                     if t.theme_name == "Risk Mitigation")
    purge_count = sum(len(t.recommendations) for t in themes
                      if t.theme_name == "Dead-Weight Purge")
    earn_count = len(diagnostics.earnings_calendar)

    parts = []
    parts.append(
        f"<strong>{_fmt_dollars(nav)} portfolio</strong>, net long delta "
        f"{delta_pct:+.0f}% of NAV, &beta; {beta:.2f}."
    )
    if sector_pct >= 50 and top_sector[0]:
        parts.append(
            f"<strong>Concentration risk</strong>: {top_sector[0]} represents "
            f"{sector_pct:.0f}% of equity NAV."
        )

    actionables = []
    if purge_count: actionables.append(f"{purge_count} positions to close at target")
    if yield_count: actionables.append(f"{yield_count} premium-collection opportunities")
    if risk_count:  actionables.append(f"{risk_count} hedge actions")
    if actionables:
        parts.append("Action items: " + " \u00b7 ".join(actionables) + ".")

    if earn_count:
        parts.append(
            f"{earn_count} {'name reports' if earn_count == 1 else 'names report'} "
            f"earnings in the next 30 days."
        )

    return html.Div(
        className="narrative",
        children=[dcc.Markdown(
            " ".join(parts),
            dangerously_allow_html=True,
        )],
    )


# ---------------------------------------------------------------------------
# Composition row: sector donut + style mix + rich earnings panel
# ---------------------------------------------------------------------------

SECTOR_PALETTE = [
    "#EC0000", "#2B2B2B", "#6E6E6E", "#B5B5B5",
    "#D4A017", "#1E7E34", "#0C4A6E", "#8E44AD",
]


def _sector_donut(diagnostics):
    items = sorted(diagnostics.sector_exposure.items(), key=lambda x: -x[1])
    labels = [k for k, _ in items]
    values = [v * 100 for _, v in items]
    colors = SECTOR_PALETTE[: len(labels)]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(family="Frutiger, Helvetica Neue, Arial", size=11),
        sort=False, direction="clockwise",
    )])

    largest_pct = values[0] if values else 0
    largest_name = labels[0] if labels else ""
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=240,
        annotations=[dict(
            text=(f"<b>{largest_pct:.0f}%</b><br>"
                  f"<span style='font-size:11px;color:#6E6E6E'>"
                  f"{largest_name}</span>"),
            x=0.5, y=0.5,
            font=dict(size=20, color="#2B2B2B"),
            showarrow=False,
        )],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def _style_bars(diagnostics):
    style_items = sorted(diagnostics.style_mix.items(), key=lambda x: -x[1])
    bars = []
    for name, pct in style_items:
        bars.append(html.Div(
            style={"marginBottom": "8px"},
            children=[
                html.Div([
                    html.Span(name, style={"fontSize": "12px",
                                            "fontWeight": "500"}),
                    html.Span(
                        f"{pct*100:.0f}%",
                        style={"fontSize": "12px",
                               "color": "var(--ubs-grey-500)",
                               "float": "right"},
                    ),
                ]),
                html.Div(
                    style={
                        "background": "var(--ubs-grey-100)", "height": "6px",
                        "borderRadius": "3px", "marginTop": "4px",
                        "overflow": "hidden",
                    },
                    children=[html.Div(style={
                        "background": "var(--ubs-charcoal)",
                        "height": "100%", "width": f"{pct*100:.1f}%",
                    })],
                ),
            ],
        ))
    return html.Div(bars)


def _earnings_implied_move(snap_row, days_to_earnings):
    """BBG field if present, else iv_1m * sqrt(dte/252)."""
    if snap_row is None:
        return None
    field = FIELDS.get("earn_implied_move")
    if field and field in snap_row.index:
        v = snap_row[field]
        if pd.notna(v):
            return float(v)
    iv = snap_row.get(FIELDS["iv_1m"])
    if (iv is not None and pd.notna(iv)
            and days_to_earnings is not None and days_to_earnings > 0):
        return float(iv) * math.sqrt(days_to_earnings / 252)
    return None


def _earnings_avg_realized(snap_row):
    if snap_row is None:
        return None
    field = FIELDS.get("earn_avg_realized")
    if not field or field not in snap_row.index:
        return None
    v = snap_row[field]
    if pd.notna(v):
        return float(v)
    return None


def _earnings_panel(diagnostics, snapshot_df):
    earn = diagnostics.earnings_calendar
    if not earn:
        return html.Div(
            "None in next 30 days",
            style={"color": "var(--ubs-grey-500)", "fontSize": "13px"},
        )

    rows = [html.Div(className="earnings-header", children=[
        html.Div("Symbol"), html.Div("Days"),
        html.Div("Implied"), html.Div("Avg Real"), html.Div("VRP"),
        html.Div("Date"),
    ])]

    for e in earn[:10]:
        ticker = e["ticker"]
        snap_row = (snapshot_df.loc[ticker]
                    if (snapshot_df is not None
                        and not snapshot_df.empty
                        and ticker in snapshot_df.index)
                    else None)
        implied = _earnings_implied_move(snap_row, e["days_to_earnings"])
        avg_real = _earnings_avg_realized(snap_row)
        vrp = ((implied - avg_real)
               if (implied is not None and avg_real is not None)
               else None)
        vrp_class = ("positive" if (vrp is not None and vrp > 0)
                     else "negative" if (vrp is not None and vrp < 0)
                     else "")

        rows.append(html.Div(className="earnings-row", children=[
            html.Div(e["symbol"], className="symbol"),
            html.Div(f"{e['days_to_earnings']}d", className="days"),
            html.Div(
                f"\u00b1{implied:.1f}%" if implied is not None else "\u2014",
                className="imp",
            ),
            html.Div(
                f"\u00b1{avg_real:.1f}%" if avg_real is not None else "\u2014",
                className="real",
            ),
            html.Div(
                f"{vrp:+.1f}pp" if vrp is not None else "\u2014",
                className=f"vrp {vrp_class}",
            ),
            html.Div(
                e["date"][:10],
                style={"color": "var(--ubs-grey-500)", "fontSize": "11px"},
            ),
        ]))

    return html.Div(rows)


def _composition_row(diagnostics, snapshot_df):
    return html.Div(className="composition-row", children=[
        html.Div(className="comp-panel", children=[
            html.H3("Sector concentration"),
            _sector_donut(diagnostics),
        ]),
        html.Div(className="comp-panel", children=[
            html.H3("Style mix"),
            _style_bars(diagnostics),
        ]),
        html.Div(className="comp-panel", children=[
            html.H3(
                f"Earnings \u226430d  \u00b7  "
                f"{len(diagnostics.earnings_calendar)} names"
            ),
            _earnings_panel(diagnostics, snapshot_df),
        ]),
    ])


# ---------------------------------------------------------------------------
# Greeks summary
# ---------------------------------------------------------------------------

def _greeks_summary_cards(greeks, portfolio):
    t = greeks.totals
    cards = [
        ("$ Delta", _fmt_dollars(t.get("dollar_delta"), signed=True),
                   _fmt_pct((t.get("delta_pct_of_nav") or 0) * 100,
                             signed=True) + " of NAV"),
        ("$ Vega",  _fmt_dollars(t.get("dollar_vega"), signed=True),
                   "per +1 vol pt"),
        ("$ Theta", _fmt_dollars(t.get("dollar_theta"), signed=True),
                   "per day"),
        ("$ Gamma", _fmt_dollars(t.get("dollar_gamma"), signed=True),
                   "$\u0394 per $1 spot"),
    ]
    return html.Div(className="metric-row", children=[
        html.Div(className="metric-card", children=[
            html.Div(label, className="label"),
            html.Div(value, className="value"),
            html.Div(sub, className="sub"),
        ])
        for label, value, sub in cards
    ])


# ---------------------------------------------------------------------------
# Action Items panel
# ---------------------------------------------------------------------------

_THEME_DATA_KEY = {
    "Yield Enhancement":    "yield",
    "Risk Mitigation":      "risk",
    "Dead-Weight Purge":    "purge",
    "Tactical Opportunity": "tactical",
}


def _action_items_panel(themes):
    if not themes:
        return html.Div(
            "No actionable themes today \u2014 book is in a steady state.",
            style={
                "color": "var(--ubs-grey-500)", "fontStyle": "italic",
                "padding": "16px",
                "border": "1px dashed var(--ubs-grey-300)",
                "borderRadius": "4px", "marginBottom": "16px",
            },
        )
    return html.Div([_pitch_theme_card(t) for t in themes])


def _pitch_theme_card(theme):
    rec_rows = [
        html.Tr([
            html.Td(r.position_id),
            html.Td(r.action),
            html.Td(r.rationale),
        ])
        for r in theme.recommendations
    ]
    return html.Details(
        open=True,
        className="pitch-card",
        **{"data-theme": _THEME_DATA_KEY.get(theme.theme_name, "purge")},
        children=[
            html.Summary([
                html.Span(theme.theme_name),
                html.Span(
                    f"  \u00b7  {theme.summary_metric}",
                    style={
                        "color": "var(--ubs-grey-500)", "fontWeight": "400",
                        "fontSize": "12px", "marginLeft": "10px",
                    },
                ),
            ]),
            html.Div(className="body", children=[
                html.Div(theme.headline, className="headline"),
                html.Table(rec_rows),
            ]),
        ],
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

_DATATABLE_COMMON = {
    "page_size": 30,
    "sort_action": "native",
    "filter_action": "native",
    "fixed_rows": {"headers": True},
    "cell_selectable": True,
    "style_table": {"overflowX": "auto", "maxHeight": "640px",
                    "overflowY": "auto"},
    "style_cell": {
        "fontFamily": "var(--font-stack)",
        "padding": "6px 10px",
        "whiteSpace": "normal",
        "height": "auto",
        "textAlign": "left",
        "minWidth": "60px",
        "maxWidth": "400px",
        "fontSize": "13px",
    },
    "style_data": {"whiteSpace": "normal", "height": "auto"},
    "style_header": {
        "backgroundColor": "var(--ubs-grey-50)",
        "fontWeight": "600",
        "borderBottom": "2px solid var(--ubs-grey-300)",
    },
}

_RATIONALE_CELL_COND = [
    {"if": {"column_id": "rationale"},
     "minWidth": "320px", "maxWidth": "600px"},
]


def _signals_table(signals_by_ticker, snapshot_df, composite_scores):
    records = []
    for ticker, sigs in signals_by_ticker.items():
        rec = {"ticker": ticker}
        if ticker in snapshot_df.index:
            rec["name"] = snapshot_df.loc[ticker].get("security_name", "")
        cs = composite_scores.get(ticker)
        rec["score"] = cs.total if cs else 0
        rec["score_label"] = cs.label if cs else "\u2014"
        for sig in sigs:
            band = sig.detail.split(":")[0].strip()
            rec[sig.signal_type] = band
        records.append(rec)

    cols_in_order = [
        "ticker", "name", "score", "score_label",
        "move_vs_iv", "trend_200d", "momentum",
        "ytd_performance", "iv_level", "iv_percentile",
        "iv_term", "vol_risk_premium",
        "earnings_within_30d", "rsi_extreme", "breakout",
    ]
    cols_present = [c for c in cols_in_order if any(c in r for r in records)]

    score_cond = []
    for thresh, bg in [(0, "#FDECEA"), (30, "#FFF3CD"), (50, "#FFFDF5"),
                       (70, "#E6F4EA"), (85, "#D4F0DA")]:
        score_cond.append({
            "if": {"filter_query": f"{{score}} >= {thresh}",
                   "column_id": "score"},
            "backgroundColor": bg,
            "fontWeight": "600",
        })

    return dash_table.DataTable(
        id="signals-dt",
        data=records,
        columns=[{"name": c.replace("_", " ").title(), "id": c}
                 for c in cols_present],
        style_data_conditional=[
            *score_cond,
            {"if": {"filter_query": '{trend_200d} contains "Bullish"',
                    "column_id": "trend_200d"},
             "backgroundColor": "var(--pos-bg)", "color": "var(--pos)"},
            {"if": {"filter_query": '{trend_200d} contains "Bearish"',
                    "column_id": "trend_200d"},
             "backgroundColor": "var(--neg-bg)", "color": "var(--neg)"},
            {"if": {"filter_query": '{ytd_performance} contains "Leader"',
                    "column_id": "ytd_performance"},
             "backgroundColor": "var(--pos-bg)", "color": "var(--pos)"},
            {"if": {"filter_query": '{ytd_performance} contains "laggard"',
                    "column_id": "ytd_performance"},
             "backgroundColor": "var(--neg-bg)", "color": "var(--neg)"},
            {"if": {"filter_query": '{momentum} contains "bullish"',
                    "column_id": "momentum"},
             "backgroundColor": "var(--pos-bg)", "color": "var(--pos)"},
            {"if": {"filter_query": '{momentum} contains "bearish"',
                    "column_id": "momentum"},
             "backgroundColor": "var(--neg-bg)", "color": "var(--neg)"},
        ],
        **_DATATABLE_COMMON,
    )


def _options_with_greeks_table(greeks, recs_by_ticker, portfolio):
    df = greeks.by_position[
        greeks.by_position["instrument_type"] == "option"
    ].copy()
    if df.empty:
        return html.Div(
            "No option positions or greeks unavailable.",
            style={"color": "var(--ubs-grey-500)", "fontStyle": "italic"},
        )

    df["action"] = df["bbg_ticker"].map(
        lambda t: recs_by_ticker[t].action if t in recs_by_ticker else ""
    )
    df["rationale"] = df["bbg_ticker"].map(
        lambda t: recs_by_ticker[t].rationale if t in recs_by_ticker else ""
    )

    nav = portfolio.portfolio_total.get("total_market_value") or 0
    if nav:
        df["pct_nav"] = (df["bbg_ticker"]
                         .map(lambda t: _option_market_value(portfolio, t))
                         .abs() / nav * 100).round(1)
    else:
        df["pct_nav"] = 0.0

    for col in ("spot", "delta", "vega", "theta", "gamma"):
        if col in df.columns:
            df[col] = df[col].round(3 if col in ("delta", "gamma") else 2)
    for col in ("dollar_delta", "dollar_vega", "dollar_theta", "dollar_gamma"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: _fmt_dollars(x, signed=True))

    cols_in_order = [
        "bbg_ticker", "underlying_ticker", "right", "quantity", "pct_nav",
        "spot", "delta", "vega", "theta", "gamma",
        "dollar_delta", "dollar_vega", "dollar_theta", "dollar_gamma",
        "action", "rationale",
    ]
    cols_present = [c for c in cols_in_order if c in df.columns]

    return dash_table.DataTable(
        id="options-dt",
        data=df[cols_present].to_dict("records"),
        columns=[{"name": c, "id": c} for c in cols_present],
        style_data_conditional=_action_color_rules(),
        style_cell_conditional=_RATIONALE_CELL_COND,
        **_DATATABLE_COMMON,
    )


def _option_market_value(portfolio, opt_ticker):
    """Helper: pull the parser-recorded market_value for one option ticker."""
    if portfolio.option_positions.empty:
        return 0.0
    matches = portfolio.option_positions[
        portfolio.option_positions["bbg_ticker"] == opt_ticker
    ]
    if matches.empty:
        return 0.0
    val = matches["market_value"].iloc[0]
    return float(val) if pd.notna(val) else 0.0


def _equity_with_recs_table(portfolio, recs_by_ticker, snapshot_df):
    equity_df = portfolio.equity_positions
    if equity_df.empty:
        return html.Div(
            "No equity positions.",
            style={"color": "var(--ubs-grey-500)", "fontStyle": "italic"},
        )
    df = equity_df.copy()
    df["action"] = df["bbg_ticker"].map(
        lambda t: recs_by_ticker[t].action if t in recs_by_ticker else ""
    )
    df["rationale"] = df["bbg_ticker"].map(
        lambda t: recs_by_ticker[t].rationale if t in recs_by_ticker else ""
    )

    nav = portfolio.portfolio_total.get("total_market_value") or 0
    if nav:
        df["pct_nav"] = (df["market_value"].abs() / nav * 100).round(1)
    else:
        df["pct_nav"] = 0.0

    for col in ("market_value", "unrealized_pnl"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: _fmt_dollars(x, signed=(col == "unrealized_pnl"))
            )
    if "price" in df.columns:
        df["price"] = df["price"].apply(lambda x: _fmt_num(x, decimals=2))

    cols_in_order = [
        "symbol", "bbg_ticker", "region", "style",
        "quantity", "price", "market_value", "pct_nav", "unrealized_pnl",
        "action", "rationale",
    ]
    cols_present = [c for c in cols_in_order if c in df.columns]

    return dash_table.DataTable(
        id="equity-dt",
        data=df[cols_present].to_dict("records"),
        columns=[{"name": c, "id": c} for c in cols_present],
        style_data_conditional=_action_color_rules(),
        style_cell_conditional=_RATIONALE_CELL_COND,
        **_DATATABLE_COMMON,
    )


def _action_color_rules():
    color_by_action = {
        "CLOSE":             ("#fff2cc", "#7a5b00"),
        "ROLL_OUT":          ("#fde8d3", "#723a00"),
        "ROLL_DOWN":         ("#fdd5c7", "#7a1f00"),
        "ROLL_UP":           ("#dde9ff", "#0a3d8c"),
        "ROLL_OUT_AND_DOWN": ("#fdecea", "#611a15"),
        "ROLL_UP_AND_OUT":   ("#fdecea", "#611a15"),
        "HARVEST_THETA":     ("#e6f4ea", "#1e4620"),
        "ADD_OVERLAY":       ("#e6f4ea", "#1e4620"),
        "ADD_HEDGE":         ("#fdecea", "#611a15"),
        "TRIM":              ("#fff2cc", "#7a5b00"),
        "ADD":               ("#dde9ff", "#0a3d8c"),
        "MONITOR":           ("#f0f0f0", "#666"),
    }
    return [
        {
            "if": {"filter_query": f'{{action}} = "{action}"',
                   "column_id": "action"},
            "backgroundColor": bg, "color": fg, "fontWeight": "600",
        }
        for action, (bg, fg) in color_by_action.items()
    ]


def _underlyings_table(snapshot):
    df = snapshot.underlyings
    if df.empty:
        return html.Div(
            "No underlying data available.",
            style={"color": "var(--ubs-grey-500)", "fontStyle": "italic"},
        )

    df = df.reset_index().rename(columns={"index": "bbg_ticker"})
    if "security" in df.columns and "bbg_ticker" not in df.columns:
        df = df.rename(columns={"security": "bbg_ticker"})

    available = [(src, label) for src, label in DISPLAY_UNDERLYING_COLS
                 if src in df.columns]
    for src, _ in available:
        if src in df.columns and pd.api.types.is_numeric_dtype(df[src]):
            decimals = _NUMERIC_DISPLAY_DECIMALS.get(src, 2)
            df[src] = df[src].round(decimals)

    cols = [{"name": "Ticker", "id": "bbg_ticker"}] + [
        {"name": label, "id": src} for src, label in available
    ]
    data = df[["bbg_ticker"] + [src for src, _ in available]].to_dict("records")

    return dash_table.DataTable(
        data=data,
        columns=cols,
        **_DATATABLE_COMMON,
    )


def _table(df, cols):
    cols_present = [c for c in cols if c in df.columns]
    return dash_table.DataTable(
        data=df[cols_present].to_dict("records"),
        columns=[{"name": c, "id": c} for c in cols_present],
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "var(--font-stack)",
                     "fontSize": "13px", "padding": "6px 10px"},
        style_header={"backgroundColor": "var(--ubs-grey-50)",
                      "fontWeight": "600"},
    )


# ---------------------------------------------------------------------------
# Definitions panel
# ---------------------------------------------------------------------------

def _signal_def_card(sig_def):
    threshold_rows = [
        html.Tr([
            html.Td(band, style={"padding": "4px 8px", "fontWeight": "bold"}),
            html.Td(rng,  style={"padding": "4px 8px",
                                  "fontFamily": "Consolas, monospace"}),
            html.Td(direction, style={"padding": "4px 8px",
                                       "color": "var(--ubs-grey-500)"}),
        ])
        for band, rng, direction in sig_def.thresholds
    ]
    return html.Div(
        style={
            "border": "1px solid var(--ubs-grey-100)", "borderRadius": "4px",
            "padding": "12px", "marginBottom": "10px", "background": "white",
        },
        children=[
            html.H4(sig_def.display_name,
                    style={"marginTop": "0", "fontWeight": "500"}),
            html.P(sig_def.what_it_measures,
                   style={"color": "var(--ubs-grey-700)"}),
            html.Div([
                html.Strong("Formula: "),
                html.Code(sig_def.formula),
            ], style={"marginBottom": "8px"}),
            html.Strong("Bands:"),
            html.Table(threshold_rows,
                       style={"borderCollapse": "collapse",
                              "marginTop": "4px"}),
            html.P(
                [html.Strong("Rationale: "), sig_def.rationale],
                style={"color": "var(--ubs-grey-700)", "marginTop": "8px",
                        "fontSize": "13px"},
            ),
        ],
    )


def _rec_def_card(rec_def):
    return html.Div(
        style={
            "border": "1px solid var(--ubs-grey-100)", "borderRadius": "4px",
            "padding": "12px", "marginBottom": "10px", "background": "white",
        },
        children=[
            html.H4([
                html.Span(rec_def.display_name),
                html.Span(
                    f"  \u2192  {rec_def.action}",
                    style={
                        "color": "#0066cc",
                        "fontFamily": "Consolas, monospace",
                        "fontSize": "13px", "marginLeft": "8px",
                    },
                ),
            ], style={"marginTop": "0", "fontWeight": "500"}),
            html.Div([
                html.Strong("Applies to: "),
                html.Code(rec_def.applies_to),
            ], style={"marginBottom": "4px"}),
            html.Div([
                html.Strong("Triggers (all required): "),
                html.Span(" \u00b7 ".join(rec_def.triggers)),
            ], style={"marginBottom": "4px", "fontSize": "13px"}),
            html.P(
                [html.Strong("Rationale: "), rec_def.rationale],
                style={"color": "var(--ubs-grey-700)", "marginTop": "8px",
                        "fontSize": "13px"},
            ),
            html.P(
                [html.Em("Source: "), rec_def.institutional_source],
                style={"color": "var(--ubs-grey-500)", "fontSize": "12px"},
            ),
        ],
    )


def _definitions_panel():
    signal_cards = [_signal_def_card(d) for d in SIGNAL_DEFINITIONS.values()]
    rec_cards = [_rec_def_card(d) for d in RECOMMENDATION_DEFINITIONS.values()]
    return html.Details(
        className="definitions",
        open=False,
        children=[
            html.Summary(
                "\U0001f4d6  Definitions \u2014 signals & recommendation logic"
            ),
            html.Div([
                html.H3("Signals"),
                html.Div(signal_cards),
                html.H3("Recommendation rules", style={"marginTop": "20px"}),
                html.Div(rec_cards),
            ], style={"padding": "12px"}),
        ],
    )


# ---------------------------------------------------------------------------
# Position-detail drill-down
# ---------------------------------------------------------------------------

def _detail_raw_data(snap_row, ticker, kind):
    if snap_row is None:
        return html.Div(className="detail-section", children=[
            html.H4("Raw data"),
            html.Div("No live data.",
                     style={"color": "var(--ubs-grey-500)"}),
        ])

    rows = []
    def kv(k, v):
        rows.append(html.Div(className="kv-row", children=[
            html.Span(k, className="k"), html.Span(v, className="v"),
        ]))

    px = snap_row.get(FIELDS["spot"])
    chg = snap_row.get(FIELDS["chg_1d"])
    ytd = snap_row.get(FIELDS["ytd"])
    iv1 = snap_row.get(FIELDS["iv_1m"])
    iv3 = snap_row.get(FIELDS["iv_3m"])
    iv6 = snap_row.get(FIELDS["iv_6m"])
    rv  = snap_row.get(FIELDS["rv_30d"])
    ma50 = snap_row.get(FIELDS["ma_50d"])
    ma200 = snap_row.get(FIELDS["ma_200d"])
    rsi = snap_row.get(FIELDS["rsi_14d"])
    sector = snap_row.get("GICS_SECTOR_NAME")
    beta = snap_row.get(FIELDS["beta"])
    earn = snap_row.get(FIELDS["earn_dt"])

    if px is not None and pd.notna(px):     kv("Price",         f"${float(px):,.2f}")
    if chg is not None and pd.notna(chg):   kv("1D",            f"{float(chg):+.2f}%")
    if ytd is not None and pd.notna(ytd):   kv("YTD",           f"{float(ytd):+.2f}%")
    if isinstance(sector, str):             kv("Sector",        sector)
    if beta is not None and pd.notna(beta): kv("Beta",          f"{float(beta):.2f}")
    if iv1 is not None and pd.notna(iv1):   kv("1M ATM IV",     f"{float(iv1):.1f}%")
    if iv3 is not None and pd.notna(iv3):   kv("3M ATM IV",     f"{float(iv3):.1f}%")
    if iv6 is not None and pd.notna(iv6):   kv("6M ATM IV",     f"{float(iv6):.1f}%")
    if rv is not None and pd.notna(rv):     kv("30D RV",        f"{float(rv):.1f}%")
    if (iv1 is not None and pd.notna(iv1)
        and rv is not None and pd.notna(rv)):
        kv("VRP", f"{float(iv1)-float(rv):+.1f}pp")
    if ma50 is not None and pd.notna(ma50): kv("50D MA",        f"${float(ma50):,.2f}")
    if ma200 is not None and pd.notna(ma200): kv("200D MA",     f"${float(ma200):,.2f}")
    if rsi is not None and pd.notna(rsi):   kv("RSI(14)",       f"{float(rsi):.1f}")
    if earn is not None:
        try:
            if pd.notna(earn):
                kv("Next earnings", str(pd.to_datetime(earn).date()))
        except (TypeError, ValueError):
            pass

    return html.Div(className="detail-section", children=[
        html.H4("Raw data"),
        html.Div(rows),
    ])


def _detail_composite(composite):
    if composite is None:
        return html.Div(className="detail-section", children=[
            html.H4("Composite score"),
            html.Div("No composite score.",
                     style={"color": "var(--ubs-grey-500)"}),
        ])
    rows = [
        html.Div(className="composite-component", children=[
            html.Div(name.replace("_", " ").title(), className="name"),
            html.Div(f"{c['raw']}", className="raw"),
            html.Div(html.Div(className="fill",
                              style={"width": f"{c['raw']}%"}),
                     className="bar"),
            html.Div(f"{c['weighted']}", className="weighted"),
        ])
        for name, c in composite.components.items()
    ]
    return html.Div(className="detail-section", children=[
        html.H4(f"Composite score \u00b7 {composite.label}"),
        html.Div(className="composite-bar", children=[
            html.Div(f"{composite.total:.0f}", className="total"),
            html.Div(rows, className="components"),
        ]),
    ])


def _detail_signal_derivation(signals):
    if not signals:
        return html.Div(
            "No signals fired for this name \u2014 review the Definitions "
            "panel for what each signal looks for.",
            style={"color": "var(--ubs-grey-500)", "fontSize": "13px"},
        )
    headers = ["Signal", "Direction", "Metric", "Value", "Threshold",
               "Strength", "Formula"]
    header_row = html.Tr([
        html.Th(h, style={
            "textAlign": "left", "padding": "6px 10px",
            "fontSize": "11px", "textTransform": "uppercase",
            "color": "var(--ubs-grey-500)",
            "borderBottom": "1px solid var(--ubs-grey-300)",
        })
        for h in headers
    ])

    body_rows = []
    for s in signals:
        chip_class = (
            "bullish" if s.direction == "bullish"
            else "bearish" if s.direction == "bearish"
            else "event"   if s.direction in ("event", "premium-buy",
                                              "premium-sell")
            else "neutral"
        )
        body_rows.append(html.Tr([
            html.Td(html.Span(s.signal_type.replace("_", " "),
                              className=f"signal-chip {chip_class}"),
                    style={"padding": "6px 10px"}),
            html.Td(s.direction, style={"padding": "6px 10px",
                                         "color": "var(--ubs-grey-500)"}),
            html.Td(s.metric_name, style={"padding": "6px 10px",
                                           "fontFamily": "Consolas, monospace",
                                           "fontSize": "12px"}),
            html.Td(f"{s.metric_value}",
                    style={"padding": "6px 10px",
                            "fontFamily": "Consolas, monospace"}),
            html.Td(f"{s.threshold_value}",
                    style={"padding": "6px 10px",
                            "fontFamily": "Consolas, monospace",
                            "color": "var(--ubs-grey-500)"}),
            html.Td(f"{s.strength:.2f}",
                    style={"padding": "6px 10px", "fontWeight": "600"}),
            html.Td(s.strength_formula,
                    style={"padding": "6px 10px",
                            "fontFamily": "Consolas, monospace",
                            "fontSize": "11px",
                            "color": "var(--ubs-grey-700)"}),
        ]))

    return html.Table([header_row] + body_rows,
                      style={"width": "100%", "borderCollapse": "collapse"})


def _detail_recommendations(recs, kind, ticker):
    if not recs:
        all_recs = _DASHBOARD_STATE.get("recommendations", [])
        matching = []
        sym = ticker.split()[0] if ticker else ""
        for r in all_recs:
            if r.position_id == ticker:
                matching.append(r)
            elif kind == "underlying" and sym and sym == r.position_id.split()[0]:
                matching.append(r)
        recs = matching

    if not recs:
        return html.Div(
            "No actionable recommendation. The position is in the watch "
            "state \u2014 no rule's trigger conditions are met at current "
            "levels. See the Definitions panel for what each rule looks for.",
            style={"color": "var(--ubs-grey-700)", "fontSize": "13px",
                   "lineHeight": "1.5"},
        )

    blocks = []
    for r in recs:
        blocks.append(html.Div(
            style={"marginBottom": "12px", "padding": "10px 12px",
                    "border": "1px solid var(--ubs-grey-100)",
                    "borderRadius": "4px"},
            children=[
                html.Div([
                    html.Span(r.position_id,
                              style={"fontFamily": "Consolas, monospace",
                                      "fontWeight": "600",
                                      "marginRight": "10px"}),
                    html.Span(r.action, style={"fontWeight": "600",
                                                "color": "var(--ubs-red)"}),
                    html.Span(f"  \u00b7  {r.priority}",
                              style={"color": "var(--ubs-grey-500)",
                                      "fontSize": "12px"}),
                ]),
                html.Div(r.rationale, style={"fontSize": "13px",
                                              "marginTop": "6px",
                                              "lineHeight": "1.5"}),
                html.Div([
                    html.Span("Rule: ", style={"color": "var(--ubs-grey-500)",
                                                "fontSize": "11px"}),
                    html.Span(r.rule_id,
                              style={"fontFamily": "Consolas, monospace",
                                      "fontSize": "11px"}),
                ], style={"marginTop": "6px"}),
            ],
        ))
    return html.Div(blocks)


def _position_detail_panel(selected):
    state = _DASHBOARD_STATE
    ticker = selected["ticker"]
    kind = selected["kind"]

    portfolio = state["portfolio"]
    if kind == "option":
        opt_row = portfolio.option_positions[
            portfolio.option_positions["bbg_ticker"] == ticker
        ]
        underlying_ticker = (opt_row["underlying_bbg_ticker"].iloc[0]
                              if not opt_row.empty else ticker)
    else:
        underlying_ticker = ticker

    snap = state["snapshot"].underlyings
    snap_row = (snap.loc[underlying_ticker]
                if (snap is not None and not snap.empty
                    and underlying_ticker in snap.index)
                else None)
    signals = state["signals_by_ticker"].get(underlying_ticker, [])
    composite = state["composite_scores"].get(underlying_ticker)
    recs = [r for r in state["recommendations"] if r.position_id == ticker]

    title_name = (snap_row.get("security_name", ticker)
                  if snap_row is not None else ticker)
    if title_name is None or (isinstance(title_name, float)
                               and pd.isna(title_name)):
        title_name = ticker

    return html.Div(className="detail-panel", children=[
        html.Div(className="detail-panel-header", children=[
            html.Div([
                html.Span(ticker,
                          style={"fontFamily": "Consolas, monospace",
                                 "fontWeight": "600", "marginRight": "12px"}),
                html.Span(str(title_name),
                          style={"color": "var(--ubs-grey-700)"}),
            ], className="title"),
            html.Button("\u2715  Close", className="close-btn",
                        id="detail-close-btn", n_clicks=0),
        ]),
        html.Div(className="detail-panel-body", children=[
            html.Div(className="detail-grid-2", children=[
                _detail_raw_data(snap_row, ticker, kind),
                _detail_composite(composite),
            ]),
            html.Div(className="detail-section", children=[
                html.H4("Signal derivation"),
                _detail_signal_derivation(signals),
            ]),
            html.Div(className="detail-section", children=[
                html.H4("Recommendation"),
                _detail_recommendations(recs, kind, ticker),
            ]),
        ]),
    ])


# ---------------------------------------------------------------------------
# Drill-down callback registration (called once per Dash instance)
# ---------------------------------------------------------------------------

def _register_drilldown_callbacks(app: dash.Dash) -> None:
    """Attach the three drill-down callbacks to ``app``."""

    @app.callback(
        Output("selected-position-store", "data"),
        [Input("signals-dt",  "active_cell"),
         Input("options-dt",  "active_cell"),
         Input("equity-dt",   "active_cell")],
        [State("signals-dt",  "data"),
         State("options-dt",  "data"),
         State("equity-dt",   "data")],
        prevent_initial_call=True,
    )
    def _on_table_click(sig_cell, opt_cell, eq_cell,
                        sig_data, opt_data, eq_data):
        triggered = ctx.triggered_id
        if triggered == "signals-dt" and sig_cell:
            row = sig_data[sig_cell["row"]]
            return {"ticker": row.get("ticker"), "kind": "underlying"}
        if triggered == "options-dt" and opt_cell:
            row = opt_data[opt_cell["row"]]
            return {"ticker": row.get("bbg_ticker"), "kind": "option"}
        if triggered == "equity-dt" and eq_cell:
            row = eq_data[eq_cell["row"]]
            return {"ticker": row.get("bbg_ticker"), "kind": "equity"}
        return no_update

    @app.callback(
        Output("detail-panel-container", "children"),
        Input("selected-position-store", "data"),
    )
    def _render_detail_panel(selected):
        if not selected or not selected.get("ticker"):
            return None
        return _position_detail_panel(selected)

    @app.callback(
        Output("selected-position-store", "data", allow_duplicate=True),
        Input("detail-close-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _close_detail(n):
        if n:
            return None
        return no_update


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_app() -> dash.Dash:
    portfolio = parse_holdings(DEFAULT_HOLDINGS_FILE)

    bbg_ok = is_bloomberg_available()
    snapshot = fetch_portfolio_snapshot(portfolio, bbg_ok)
    greeks = compute_portfolio_greeks(
        portfolio, snapshot.underlyings, snapshot.options
    )
    signals_by_ticker = compute_per_underlying_signals(
        snapshot.underlyings, bloomberg_available=bbg_ok,
    )
    composite_scores = compute_all_composite_scores(signals_by_ticker)
    diagnostics = compute_portfolio_diagnostics(portfolio, snapshot.underlyings)

    contexts = build_position_contexts(
        portfolio, snapshot, _iv_pctl_dict(signals_by_ticker)
    )
    recommendations = compute_recommendations(contexts, signals_by_ticker)
    themes = synthesize_pitch(recommendations)
    recs_by_ticker = {r.position_id: r for r in recommendations}

    _DASHBOARD_STATE.update({
        "portfolio": portfolio,
        "snapshot": snapshot,
        "greeks": greeks,
        "diagnostics": diagnostics,
        "signals_by_ticker": signals_by_ticker,
        "composite_scores": composite_scores,
        "contexts": contexts,
        "recommendations": recommendations,
    })

    all_warnings = list(snapshot.fetch_warnings)
    if greeks.warnings:
        sample = greeks.warnings[:3]
        rest = len(greeks.warnings) - len(sample)
        all_warnings.extend(sample)
        if rest > 0:
            all_warnings.append(f"\u2026 and {rest} more greeks warning(s).")

    bbg_status = (
        "Bloomberg connected" if bbg_ok
        else "Bloomberg unavailable \u2014 parsed-data view only"
    )

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Portfolio Manager"

    _register_drilldown_callbacks(app)

    app.layout = html.Div(className="app-container", children=[
        # 1. Header
        html.Div(className="app-header", children=[
            html.H1("Portfolio Manager"),
            html.Div(
                f"v0.8  \u00b7  client portfolio review  \u00b7  {bbg_status}",
                className="meta",
            ),
        ]),

        # 2. Hero metrics
        _hero_strip(portfolio, greeks, diagnostics, all_warnings),

        # 3. 30-second narrative
        _narrative_banner(portfolio, greeks, diagnostics, recommendations,
                          themes),

        # 4. Portfolio composition (donut + style + earnings panel)
        html.Div(className="section", children=[
            html.H2("Portfolio composition"),
            _composition_row(diagnostics, snapshot.underlyings),
        ]),

        # 5. Holdings tables FIRST (per TV's reorder)
        html.Div(className="section", children=[
            html.H2("Equity positions"),
            _equity_with_recs_table(portfolio, recs_by_ticker,
                                     snapshot.underlyings),
        ]),
        html.Div(className="section", children=[
            html.H2("Option positions"),
            _options_with_greeks_table(greeks, recs_by_ticker, portfolio),
        ]),

        # 6. Signals + composite scores
        html.Div(className="section", children=[
            html.H2("Signals & composite scores"),
            _signals_table(signals_by_ticker, snapshot.underlyings,
                            composite_scores),
        ]),

        # 7. Drill-down panel container + selection store
        html.Div(id="detail-panel-container"),
        dcc.Store(id="selected-position-store", data=None),

        # 8. Action items
        html.Div(className="section", children=[
            html.H2("Action items"),
            _action_items_panel(themes),
        ]),

        # 9. Aggregate greeks
        html.Div(className="section", children=[
            html.H2("Aggregate greeks"),
            _greeks_summary_cards(greeks, portfolio),
        ]),

        # 10. Other positions
        html.Div(className="section", children=[
            html.H2("Other positions"),
            _table(
                portfolio.other_positions,
                ["symbol", "description", "quantity",
                 "market_value", "manual_review_reason"],
            ),
        ]),

        # 11. Underlyings live snapshot — technical reference
        html.Div(className="section", children=[
            html.H2("Underlyings \u2014 live data reference"),
            _underlyings_table(snapshot),
        ]),

        # 12. Definitions
        _definitions_panel(),
    ])
    return app


if __name__ == "__main__":
    build_app().run(host=HOST, port=PORT, debug=False)
