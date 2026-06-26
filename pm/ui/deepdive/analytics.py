"""Section 3 — Analytics.

Detailed positioning view. Renders already-computed diagnostics/greeks plus the
three new pure aggregations. Sector weights are drawn as utilitarian CSS bars —
no charting dependency (constraint). Everything here READS state; the
aggregations are presentation reframings, not recomputations.
"""
from __future__ import annotations

from typing import Optional

from dash import html

from pm.ui.deepdive.aggregations import (
    _fmt_money,
    expiry_ladder,
    long_short_premium_split,
    top_concentrations,
)


def _pct(v: Optional[float], dp: int = 1) -> str:
    return "—" if v is None else f"{v * 100:.{dp}f}%"


def _stat(label: str, value: str, sub: Optional[str] = None, cls: str = "") -> html.Div:
    children = [html.Div(label, className="dd-stat-label"),
                html.Div(value, className="dd-stat-value")]
    if sub:
        children.append(html.Div(sub, className="dd-stat-sub"))
    return html.Div(className=f"dd-stat {cls}".strip(), children=children)


# ---- panels ---------------------------------------------------------------

def _premium_panel(account_state) -> html.Div:
    s = long_short_premium_split(account_state)
    short_pct = s["short_share"]
    bar = html.Div(className="dd-split-bar", children=[
        html.Div(className="dd-split-collected",
                 style={"width": f"{(short_pct or 0) * 100:.1f}%"}),
        html.Div(className="dd-split-paid",
                 style={"width": f"{(1 - (short_pct or 0)) * 100:.1f}%"}),
    ]) if s["total"] else None
    return html.Div(className="dd-panel", children=[
        html.H3("Options premium — collected vs paid", className="dd-panel-title"),
        html.Div(className="dd-stat-row", children=[
            _stat("Collected (short)", _fmt_money(s["collected"]),
                  f"{s['n_short']} legs", cls="dd-stat-pos"),
            _stat("Paid (long)", _fmt_money(s["paid"]),
                  f"{s['n_long']} legs", cls="dd-stat-neg"),
            _stat("Net", _fmt_money(s["net"]), s["posture"]),
        ]),
        bar,
        html.Div(s["interpretation"], className="dd-panel-note"),
    ])


def _ladder_panel(account_state) -> html.Div:
    ladder = expiry_ladder(account_state)
    header = html.Div(className="dd-ladder-row dd-ladder-head", children=[
        html.Span("Window"),
        html.Span("Contracts"),
        html.Span("Notional (strike)",
                  title="Strike obligation = contracts × 100 × strike"),
    ])
    rows = [header]
    for b in ladder:
        rows.append(html.Div(className="dd-ladder-row", children=[
            html.Span(b["label"], className="dd-ladder-bucket"),
            html.Span(str(b["count"]), className="dd-ladder-count"),
            html.Span(_fmt_money(b["notional"]) if b["notional"] else "—",
                      className="dd-ladder-notional"),
        ]))
    return html.Div(className="dd-panel", children=[
        html.H3("Expiry ladder", className="dd-panel-title"),
        html.Div("Strike-obligation exposure by expiry window",
                 className="dd-panel-subtitle"),
        html.Div(className="dd-ladder", children=rows),
        html.Div("Notional is the strike obligation (contracts × 100 × strike), "
                 "not market value — driven by position size, shown beside the "
                 "contract count.", className="dd-panel-note"),
    ])


def _sector_panel(account_state) -> html.Div:
    diag = getattr(account_state, "diagnostics", None)
    sectors = dict(getattr(diag, "sector_exposure", {}) or {})
    items = sorted(sectors.items(), key=lambda kv: (kv[1] or 0), reverse=True)
    max_w = max((w or 0) for _, w in items) if items else 0
    bars = []
    for name, w in items:
        width = (w / max_w * 100) if max_w else 0
        bars.append(html.Div(className="dd-bar-row", children=[
            html.Span(name, className="dd-bar-label"),
            html.Div(className="dd-bar-track", children=[
                html.Div(className="dd-bar-fill", style={"width": f"{width:.1f}%"}),
            ]),
            html.Span(_pct(w), className="dd-bar-val"),
        ]))
    if not bars:
        bars = [html.Div("No equity-like sector exposure.", className="dd-empty")]
    beta = getattr(diag, "weighted_beta", None)
    beta_str = f"{beta:.2f}" if isinstance(beta, (int, float)) else "—"
    return html.Div(className="dd-panel", children=[
        html.Div(className="dd-panel-headrow", children=[
            html.H3("Sector breakdown", className="dd-panel-title"),
            html.Span(f"Weighted β {beta_str}", className="dd-beta-chip"),
        ]),
        html.Div(className="dd-bars", children=bars),
    ])


def _concentration_panel(account_state) -> html.Div:
    top = top_concentrations(account_state, n=5)
    max_w = max((r["pct_nav"] or 0) for r in top) if top else 0
    rows = []
    for r in top:
        width = ((r["pct_nav"] or 0) / max_w * 100) if max_w else 0
        label = r["underlying"] or r["symbol"] or "—"
        rows.append(html.Div(className="dd-bar-row", children=[
            html.Span(label, className="dd-bar-label"),
            html.Div(className="dd-bar-track", children=[
                html.Div(className="dd-bar-fill", style={"width": f"{width:.1f}%"}),
            ]),
            html.Span(_pct(r["pct_nav"]), className="dd-bar-val"),
        ]))
    if not rows:
        rows = [html.Div("No positions.", className="dd-empty")]
    return html.Div(className="dd-panel", children=[
        html.H3("Top-5 concentrations (% NAV)", className="dd-panel-title"),
        html.Div(className="dd-bars", children=rows),
    ])


def render_analytics_section(account_state) -> html.Div:
    return html.Div(className="dd-section", children=[
        html.Div(className="dd-section-head", children=[
            html.H2("Analytics", className="dd-section-title"),
        ]),
        html.Div(className="dd-analytics-grid", children=[
            _premium_panel(account_state),
            _ladder_panel(account_state),
            _sector_panel(account_state),
            _concentration_panel(account_state),
        ]),
    ])
