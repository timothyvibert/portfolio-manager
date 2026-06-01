"""Alert view — every alert (Fire) on a consolidated position, stacked.

Rows are now one-per-position, so a position can carry multiple alerts. The
Alert view shows ALL of them: each alert's pattern + tier badge + client-ready
rationale + full audit trace, in sequential sections separated by dividers.
The Alert/Tearsheet toggle lives on the modal header (see shell/callbacks), so
there is no in-content "view signal sheet" button anymore.
"""
from __future__ import annotations

from dash import html

from pm.insight.patterns import Fire
from pm.store.portfolio_state import PortfolioState
from pm.ui import state_access as sa
from pm.ui.blotter.grid import format_position_descriptor
from pm.ui.drawers.trace_table import render_trace


_TIER_WORD = {1: "T1 · Act today", 2: "T2 · Worth raising", 3: "T3 · FYI"}


def _alert_section(fire: Fire, state: PortfolioState) -> html.Div:
    tier_cls = f"drawer-tier-{fire.tier}"
    return html.Div(className="alert-section", children=[
        html.Div(className="drawer-header-main", children=[
            html.Span(fire.pattern_id, className="drawer-pattern-id"),
            html.Span(fire.pattern_name, className="drawer-pattern-name"),
            html.Span(_TIER_WORD.get(fire.tier, f"T{fire.tier}"),
                      className=f"drawer-tier-badge {tier_cls}"),
        ]),
        html.Div(className="drawer-section", children=[
            html.Div("Rationale", className="drawer-section-label"),
            html.Div(fire.rationale, className="drawer-rationale"),
        ]),
        html.Div(className="drawer-section", children=[
            html.Div("Audit trace", className="drawer-section-label"),
            render_trace(fire.trace),
        ]),
    ])


def render_alerts(account: str, position_id: str, state: PortfolioState) -> html.Div:
    """Render every alert on this position, stacked. Header = position
    descriptor + alert count; then one section per fire with a divider."""
    fires = sa.fires_for_position(state, account, position_id)
    # Dedupe defensively (one section per distinct alert), preserving order.
    seen, distinct = set(), []
    for f in fires:
        if f.pattern_id not in seen:
            seen.add(f.pattern_id)
            distinct.append(f)
    fires = distinct
    position = sa.position_by_id(state, account, position_id)
    descriptor = (format_position_descriptor(position)
                  if position is not None else position_id)
    n = len(fires)

    header = html.Div(className="alerts-header", children=[
        html.Span(account, className="drawer-account"),
        html.Span(" · "),
        html.Span(descriptor, className="drawer-position"),
        html.Span(f"{n} alert{'s' if n != 1 else ''}", className="alerts-count"),
    ])

    sections = []
    for i, fire in enumerate(fires):
        if i > 0:
            sections.append(html.Hr(className="alert-divider"))
        sections.append(_alert_section(fire, state))
    if not sections:
        sections = [html.Div("No alerts on this position.", className="trace-muted")]

    return html.Div(className="drawer-content evidence-drawer", children=[header, *sections])
