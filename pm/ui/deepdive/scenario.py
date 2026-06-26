"""Section — Scenario / stress (risk rung 2, part 2b).

A read-only render of the pre-computed ``acc.scenario`` (pm.risk.scenario): the
co-moving stress table (truth-CRR P&L per shock, ranked worst-first, with on-expand
delta/gamma/vega/theta attribution) and the app's first plotly chart — the portfolio
P&L curve vs an SPX move (fast vectorized BS2002) with the truth-CRR points and the
BS2002-vs-CRR confidence band.

plotly is imported lazily inside the chart builder, so ``import pm.ui`` (and the live
app at rest) stays plotly-free until this section actually renders. The figure is
token-styled to the design system (charcoal chrome, --pos/--neg for P&L meaning,
Frutiger) — plotly needs explicit colors, so the tokens are mirrored as hex here; no
new palette. The stress table uses native ``<details>`` for callback-free expansion
(no recompute — the data is all pre-computed on the load path).
"""
from __future__ import annotations

from typing import Optional

from dash import dcc, html

from pm.ui.deepdive.aggregations import _fmt_money

# Token hex mirroring assets/style.css (:root) — plotly can't read CSS vars.
_CHARCOAL = "#2B2B2B"
_POS = "#1E7E34"
_NEG = "#C62828"
_GRID = "#E8E8E8"
_AMBER = "#B7791F"
_MUTED = "#6E6E6E"
_FONT = '"Frutiger 45 Light","Frutiger","Helvetica Neue","Segoe UI",Arial,sans-serif'


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:+.2f}%"


def _sign_cls(v: Optional[float]) -> str:
    if v is None or v == 0:
        return ""
    return "scenario-pos" if v > 0 else "scenario-neg"


def render_scenario_section(account_state) -> html.Div:
    sc = getattr(account_state, "scenario", None)
    head = html.Div(className="dd-section-head", children=[
        html.H2("Scenario & stress", className="dd-section-title"),
        html.Span("hypothetical-state · engine-priced", className="dd-section-meta"),
    ])
    if sc is None or not getattr(sc, "scenarios", None):
        return html.Div(className="dd-section", children=[
            head,
            html.Div("Scenario views unavailable (Bloomberg off or no priceable options).",
                     className="dd-empty"),
        ])
    return html.Div(className="dd-section", children=[
        head,
        html.Div(className="dd-panel", children=[
            html.H3("Co-moving stress table", className="dd-panel-title"),
            _coverage_meta(sc),
            _stress_table(sc),
        ]),
        html.Div(className="dd-panel", children=[
            html.H3("Portfolio P&L vs SPX move", className="dd-panel-title"),
            _curve_graph(sc),
            _curve_caption(sc),
        ]),
    ])


def _coverage_meta(sc) -> html.Div:
    modes = " · ".join(f"{k}:{v}" for k, v in sorted(sc.div_modes.items()))
    bits = [f"{sc.n_priceable} priceable legs"]
    if sc.n_unpriceable:
        bits.append(f"{sc.n_unpriceable} skipped (expired/no data)")
    if modes:
        bits.append(f"dividends — {modes}")
    return html.Div(" · ".join(bits), className="scenario-meta")


def _stress_table(sc) -> html.Div:
    rows = [html.Div(className="scenario-head-row", children=[
        html.Span("Scenario", className="scenario-name"),
        html.Span("Account P&L", className="scenario-pnl"),
        html.Span("% NAV", className="scenario-pct"),
    ])]
    for s in sc.scenarios:                       # already ranked worst-first
        rows.append(html.Details(className="scenario-row", children=[
            # The grid layout lives on an inner div, NOT the <summary> itself —
            # a `display:grid/flex` summary suppresses Chrome's native click-to-toggle.
            html.Summary(className="scenario-summary", children=[
                html.Div(className="scenario-summary-grid", children=[
                    html.Span(s.label, className="scenario-name"),
                    html.Span(_fmt_money(s.pnl), className=f"scenario-pnl {_sign_cls(s.pnl)}"),
                    html.Span(_pct(s.pnl_pct), className=f"scenario-pct {_sign_cls(s.pnl_pct)}"),
                ]),
            ]),
            _attribution(s.attribution),
        ]))
    return html.Div(rows, className="scenario-table")


def _attribution(a: dict) -> html.Div:
    cells = [
        _attrib_cell("Δ delta", a.get("delta")),
        _attrib_cell("Γ gamma", a.get("gamma")),
        _attrib_cell("ν vega", a.get("vega")),
        _attrib_cell("θ theta", a.get("theta"), caveat=True),
        _attrib_cell("residual", a.get("residual")),
    ]
    return html.Div(className="scenario-attrib", children=[
        html.Div("P&L attribution (truth-CRR base greeks × shock)", className="scenario-attrib-title"),
        html.Div(cells, className="scenario-attrib-grid"),
        html.Div("θ is engine per-business-day — diverges from the BBG snapshot θ "
                 "(see 2a); shown as a directional guide, not a reconciled figure.",
                 className="scenario-attrib-note"),
    ])


def _attrib_cell(label: str, v: Optional[float], caveat: bool = False) -> html.Div:
    return html.Div(className="scenario-attrib-cell", children=[
        html.Span(label + ("*" if caveat else ""), className="scenario-attrib-label"),
        html.Span("—" if v is None else _fmt_money(v),
                  className=f"scenario-attrib-val {_sign_cls(v)}"),
    ])


def _curve_graph(sc) -> dcc.Graph:
    import plotly.graph_objects as go          # lazy: keeps the app plotly-free until render

    cur = sc.curve
    x, y = cur["x_pct"], cur["pnl"]
    lo, hi = cur["band_lo"], cur["band_hi"]
    fig = go.Figure()
    # BS2002-vs-CRR confidence band
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1], y=list(hi) + list(lo)[::-1], fill="toself",
        fillcolor="rgba(110,110,110,0.14)", line=dict(width=0), hoverinfo="skip",
        name="BS2002–CRR band"))
    # fast P&L curve (charcoal chrome)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=_CHARCOAL, width=2),
        name="P&L (fast BS2002)",
        hovertemplate="SPX %{x:.1f}%<br>P&L %{y:$,.0f}<extra></extra>"))
    # truth-CRR points, coloured by sign (--pos/--neg)
    tx, tp = cur.get("truth_x", []), cur.get("truth_pnl", [])
    if tx:
        fig.add_trace(go.Scatter(
            x=tx, y=tp, mode="markers",
            marker=dict(size=9, color=[_POS if v >= 0 else _NEG for v in tp],
                        line=dict(width=1, color=_CHARCOAL)),
            name="truth-CRR points",
            hovertemplate="SPX %{x:.0f}%<br>truth P&L %{y:$,.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=_GRID, width=1))
    fig.add_vline(x=0, line=dict(color=_AMBER, width=1, dash="dot"))   # current mark
    for b in cur.get("breakevens", []):
        if abs(b) > 1e-6:
            fig.add_vline(x=b, line=dict(color=_AMBER, width=1, dash="dash"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, color=_CHARCOAL, size=12),
        margin=dict(l=64, r=18, t=28, b=42), height=320,
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)),
        xaxis=dict(title=cur.get("x_label", "SPX move %"), gridcolor=_GRID, zeroline=False),
        yaxis=dict(title="Account P&L ($)", gridcolor=_GRID, zeroline=False, tickformat="$,.2s"),
        hovermode="x unified",
    )
    return dcc.Graph(figure=fig, className="scenario-graph",
                     config={"displayModeBar": False, "responsive": True})


def _curve_caption(sc) -> html.Div:
    be = ", ".join(f"{b:+.1f}%" for b in sc.curve.get("breakevens", []) if abs(b) > 1e-6) or "none in range"
    return html.Div(className="scenario-caption", children=[
        f"Curve: fast vectorized BS2002 over a beta-mapped SPX sweep (β from the rung-1 "
        f"SPX EQY_BETA). Markers + shaded band: truth-CRR at the ±5/10/20% points (the band "
        f"is the BS2002-vs-CRR gap). Table P&L: truth-CRR per shock. Breakevens: {be}.",
    ])
