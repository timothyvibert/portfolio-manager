"""Tab 2 — Structures section: inferred groupings as confirm/override proposals.

Renders each detected structure as a PROPOSED card (visually distinct from a
confirmed one), with a 3-band confidence chip, a plain-language rationale from the
canonical trace, partial cover shown as covered + residual sub-rows, and the
affordances on every proposal: Confirm · Edit (remove a leg) · Reject. A genuine
contention is rendered as one card offering its ranked alternatives, and can only be
resolved by an explicit choice — there is no one-click confirm on it. A
grouped-vs-standalone toggle switches between the structure cards and the legs that
are not part of any active grouping.

Pure presentation: reads ``AccountState.structures`` (already resolved against the
store in the load path) and never recomputes. Confirm/override is handled by the
callbacks, which write through ``state_access``.
"""
from __future__ import annotations

from dash import dcc, html

from pm.insight import structures as S

_PRIMARY = {S.COVERED_CALL, S.COLLAR, S.VERTICAL, S.COVERED_PUT,
            S.CASH_SECURED_PUT, S.STRADDLE, S.STRANGLE}
_BAND_LABEL = {S.HIGH: "High", S.MEDIUM: "Medium", S.LOW_AMBIGUOUS: "Low-Ambiguous"}
_TYPE_LABEL = {
    S.COVERED_CALL: "Covered call", S.COLLAR: "Collar", S.VERTICAL: "Vertical",
    S.COVERED_PUT: "Covered put", S.CASH_SECURED_PUT: "Cash-secured put",
    S.STRADDLE: "Straddle", S.STRANGLE: "Strangle",
    S.RESIDUAL_LONG: "Uncovered long (residual)",
    S.NAKED_EXCESS_SHORT_CALL: "Naked-excess short call",
}


def _chip(band: str) -> html.Span:
    return html.Span(_BAND_LABEL.get(band, band),
                     className=f"struct-chip struct-chip-{band}")


def _rationale(s) -> str:
    t = s.rationale_trace or {}
    return t.get("computation") or t.get("result") or ""


def _leg_rows(account, s, removable: bool):
    rows = []
    for leg in s.legs:
        children = [html.Span(leg.role.replace("_", " "), className="struct-leg-role"),
                    html.Span(f"{leg.allocated_qty:g}", className="struct-leg-qty"),
                    html.Span(leg.position_id, className="struct-leg-id")]
        if removable and len(s.legs) > 1:
            children.append(html.Button(
                "remove", className="struct-leg-remove",
                id={"type": "struct-removeleg", "account": account, "sid": s.structure_id,
                    "leg": leg.position_id}, n_clicks=0))
        rows.append(html.Div(children, className="struct-leg-row"))
    return rows


def _subrows(account, structures, underlying):
    """Residual / naked-excess slices for a covered call on this underlying."""
    out = []
    for s in structures:
        if s.underlying == underlying and s.type in (S.RESIDUAL_LONG, S.NAKED_EXCESS_SHORT_CALL):
            out.append(html.Div(className="struct-subrow", children=[
                html.Span(_TYPE_LABEL.get(s.type, s.type), className="struct-subrow-label"),
                _chip(s.confidence_band),
                html.Span(_rationale(s), className="struct-subrow-note"),
            ]))
    return out


def _render_card(account, s, structures):
    confirmed = s.status in ("confirmed", "edited")
    status_cls = "confirmed" if confirmed else "proposed"
    head = [
        html.Span(_TYPE_LABEL.get(s.type, s.type) + f" · {s.underlying}", className="struct-card-title"),
        _chip(s.confidence_band),
    ]
    if confirmed:
        when = (s.resolved_at or "")[:10]
        head.append(html.Span(f"Confirmed by you{(' on ' + when) if when else ''}",
                              className="struct-confirmed-by"))
    body = [
        html.Div(head, className="struct-card-head"),
        html.Div(_rationale(s), className="struct-card-rationale"),
        html.Div(_leg_rows(account, s, removable=not confirmed) + _subrows(account, structures, s.underlying),
                 className="struct-legs"),
    ]
    if confirmed:
        actions = [html.Button("Undo", className="struct-btn",
                               id={"type": "struct-reject", "account": account, "sid": s.structure_id}, n_clicks=0)]
    else:
        actions = [
            html.Button("Confirm", className="struct-btn struct-btn-primary",
                        id={"type": "struct-confirm", "account": account, "sid": s.structure_id}, n_clicks=0),
            html.Button("Reject", className="struct-btn",
                        id={"type": "struct-reject", "account": account, "sid": s.structure_id}, n_clicks=0),
            html.Span("Edit: remove a leg above", className="struct-edit-hint"),
        ]
    body.append(html.Div(actions, className="struct-actions"))
    return html.Div(body, className=f"struct-card struct-card-{status_cls}")


def _render_contention_card(account, alts):
    # alts: the ranked alternatives sharing a contention group
    confirmed = [a for a in alts if a.status == "confirmed"]
    group = alts[0].contention_group
    if confirmed:
        chosen = confirmed[0]
        when = (chosen.resolved_at or "")[:10]
        return html.Div(className="struct-card struct-card-confirmed", children=[
            html.Div([html.Span(f"{_TYPE_LABEL.get(chosen.type, chosen.type)} · {chosen.underlying}",
                                className="struct-card-title"),
                      html.Span(f"You chose this reading{(' on ' + when) if when else ''}",
                                className="struct-confirmed-by")], className="struct-card-head"),
            html.Div(_rationale(chosen), className="struct-card-rationale"),
            html.Button("Undo", className="struct-btn",
                        id={"type": "struct-reject", "account": account, "sid": chosen.structure_id}, n_clicks=0),
        ])
    options = [{"label": f"{_TYPE_LABEL.get(a.type, a.type)} — {_rationale(a)}", "value": a.structure_id}
               for a in alts]
    return html.Div(className="struct-card struct-card-contention", children=[
        html.Div([html.Span(f"Contention · {alts[0].underlying}", className="struct-card-title"),
                  _chip(S.LOW_AMBIGUOUS),
                  html.Span("needs your choice — these legs read two ways", className="struct-contention-note")],
                 className="struct-card-head"),
        dcc.RadioItems(id={"type": "struct-radio", "group": group}, options=options, value=None,
                       className="struct-radio"),
        html.Div([
            html.Button("Confirm choice", className="struct-btn struct-btn-primary",
                        id={"type": "struct-choose", "account": account, "group": group}, n_clicks=0),
            html.Button("Reject both", className="struct-btn",
                        id={"type": "struct-reject", "account": account, "sid": alts[0].structure_id}, n_clicks=0),
        ], className="struct-actions"),
    ])


def _render_standalone(account_state, structures):
    """Legs not part of any active (proposed/confirmed) grouping — option legs that
    fell through detection plus the legs of rejected structures."""
    claimed = set()
    for s in structures:
        if s.status != "rejected" and s.type in _PRIMARY:
            claimed.update(l.position_id for l in s.legs)
    rows = []
    for p in account_state.positions:
        if p.asset_class == "option" and p.position_id not in claimed and p.quantity:
            rows.append(html.Div(className="struct-leg-row", children=[
                html.Span(f"{p.underlying_symbol} {p.right} {p.strike:g}", className="struct-leg-role"),
                html.Span(f"{p.quantity:g}", className="struct-leg-qty"),
                html.Span(p.position_id, className="struct-leg-id")]))
    return html.Div([html.Div("Standalone legs (not in any active grouping)", className="struct-subrow-label"),
                     html.Div(rows or [html.Div("None.", className="dd-empty")], className="struct-legs")])


def _render_toggle(view_mode):
    def cls(mode):
        return "group-toggle" + (" group-toggle-active" if view_mode == mode else "")
    return html.Div(className="struct-toggle", children=[
        html.Span("View:", className="blotter-control-label"),
        html.Button("Grouped", id="struct-grouped-btn", n_clicks=0, className=cls("grouped")),
        html.Button("Standalone", id="struct-standalone-btn", n_clicks=0, className=cls("standalone")),
    ])


def render_structures_section(account_state, view_mode: str = "grouped") -> html.Div:
    if account_state is None:
        return html.Div(className="dd-section", children=[
            html.Div(className="dd-section-head", children=[html.H2("Structures", className="dd-section-title")]),
            html.Div("No account selected.", className="dd-empty")])
    structures = list(account_state.structures or [])
    groups, singles = {}, []
    for s in structures:
        (groups.setdefault(s.contention_group, []).append(s) if s.contention_group else singles.append(s))

    if view_mode == "standalone":
        body = [_render_standalone(account_state, structures)]
    else:
        body = [_render_contention_card(account_state.account, alts) for _, alts in sorted(groups.items())]
        for s in singles:
            if s.type in _PRIMARY and s.status != "rejected":
                body.append(_render_card(account_state.account, s, structures))
        if not body:
            body = [html.Div("No structures detected for this account.", className="dd-empty")]

    return html.Div(className="dd-section", children=[
        html.Div(className="dd-section-head", children=[
            html.H2("Structures", className="dd-section-title"),
            html.Span(f"{sum(1 for s in singles if s.type in _PRIMARY) + len(groups)} grouped",
                      className="dd-section-meta"),
            _render_toggle(view_mode),
        ]),
        html.Div(body, id="struct-cards"),
    ])
