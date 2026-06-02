"""Tab 2 callbacks — account-picker / refresh / tab-switch repopulation and the
two grid click-throughs into the shared modal.

These add to (never duplicate) the Tab-1 + drawer callbacks: the modal body
renderer (``_render_body``), the underlying resolver (``_underlying_for``) and
the open-class constant are imported from ``blotter.callbacks``; the view-toggle,
close, Escape and the source-aware prev/next live there and serve both tabs.
Registered once via ``register_deepdive_callbacks(app)``.
"""
from __future__ import annotations

import dash
from dash import ALL, Input, Output, State, ctx, no_update

from pm.ui import state_access as sa
from pm.ui.blotter.callbacks import _OPEN_CLS, _render_body, _underlying_for
from pm.ui.deepdive.header import account_options, default_account
from pm.ui.deepdive.layout import render_deepdive_sections
from pm.ui.deepdive.structures_panel import render_structures_section

_DD_HOST_IDS = [
    "deepdive-kpi", "deepdive-positions", "deepdive-structures",
    "deepdive-analytics", "deepdive-trades",
]


def _radio_value_for_group(group):
    """The selected alternative sid for a contention group's radio (or None)."""
    for st in (ctx.states_list[0] or []):
        if isinstance(st, dict) and st.get("id", {}).get("group") == group:
            return st.get("value")
    return None


def _resolve_account(state, picker_value):
    """The account to scope to: the picker value if valid, else the default."""
    if state is None:
        return None
    if picker_value and picker_value in state.accounts:
        return picker_value
    return default_account(state)


def _ids_from_cell(cell):
    """(account, position_id, underlying) from a grid cellClicked event. Prefers
    the row ``data`` (carries the hidden keys); falls back to parsing rowId
    (``account::position_id[::pattern_id]``)."""
    data = cell.get("data") or {}
    account = data.get("_account")
    position_id = data.get("_position_id")
    underlying = data.get("_underlying")
    if account and position_id:
        return account, position_id, underlying
    row_id = cell.get("rowId") or ""
    if "::" not in row_id:
        return None, None, None
    parts = row_id.split("::")
    pid = parts[1] if len(parts) > 1 else None
    return parts[0], pid, underlying


def register_deepdive_callbacks(app: dash.Dash) -> None:

    # ---- Picker / refresh / tab-switch → repopulate all five sections ------
    @app.callback(
        Output("deepdive-kpi", "children"),
        Output("deepdive-positions", "children"),
        Output("deepdive-structures", "children"),
        Output("deepdive-analytics", "children"),
        Output("deepdive-trades", "children"),
        Input("deepdive-account-picker", "value"),
        Input("deepdive-refresh-tick", "data"),
        Input("pm-tabs", "value"),
        State("struct-view-mode", "data"),
    )
    def _populate_deepdive(account, _tick, tab_value, struct_view):
        # On a tab switch, only rebuild when entering the deep-dive tab (this
        # also re-paints with fresh data after a refresh done while on Tab 1).
        # Picker / refresh always rebuild — the tab is active then.
        if ctx.triggered_id == "pm-tabs" and tab_value != "tab-account":
            return (no_update,) * len(_DD_HOST_IDS)
        state = sa.get_state()
        acct = _resolve_account(state, account)
        sections = render_deepdive_sections(state, acct, struct_view or "grouped")
        return tuple(sections[h] for h in _DD_HOST_IDS)

    # ---- Structure confirm / reject / edit / choose-alternative -------------
    # Writes through state_access (the single state owner) → the store interface;
    # re-applies the resolution to the in-memory structures and re-renders the
    # section. Does NOT recompute signals. Pattern-matching inputs carry the
    # account + structure id (and the leg for an edit) so the click is concrete.
    @app.callback(
        Output("deepdive-structures", "children", allow_duplicate=True),
        Input({"type": "struct-confirm", "account": ALL, "sid": ALL}, "n_clicks"),
        Input({"type": "struct-reject", "account": ALL, "sid": ALL}, "n_clicks"),
        Input({"type": "struct-removeleg", "account": ALL, "sid": ALL, "leg": ALL}, "n_clicks"),
        Input({"type": "struct-choose", "account": ALL, "group": ALL}, "n_clicks"),
        State({"type": "struct-radio", "group": ALL}, "value"),
        State("deepdive-account-picker", "value"),
        State("struct-view-mode", "data"),
        prevent_initial_call=True,
    )
    def _resolve_structure(_c, _r, _rm, _ch, _radio, picker_value, struct_view):
        trig = ctx.triggered_id
        state = sa.get_state()
        if not isinstance(trig, dict) or state is None:
            return no_update
        # ignore the spurious fire when a card (re)mounts with n_clicks None/0
        if not (ctx.triggered[0] if ctx.triggered else {}).get("value"):
            return no_update
        ttype = trig.get("type")
        if ttype == "struct-confirm":
            sa.resolve_structure(trig["account"], trig["sid"], "confirmed")
        elif ttype == "struct-reject":
            sa.resolve_structure(trig["account"], trig["sid"], "rejected")
        elif ttype == "struct-removeleg":
            acc = state.accounts.get(trig["account"])
            s = next((x for x in acc.structures if x.structure_id == trig["sid"]), None) if acc else None
            if s is None:
                return no_update
            kept = [l.position_id for l in s.legs if l.position_id != trig["leg"]]
            if not kept:
                return no_update
            sa.resolve_structure(trig["account"], trig["sid"], "edited", edited_legs=kept)
        elif ttype == "struct-choose":
            chosen_sid = _radio_value_for_group(trig.get("group"))
            if not chosen_sid:          # explicit choice required — no silent confirm
                return no_update
            sa.resolve_structure(trig["account"], chosen_sid, "confirmed")
        acct = _resolve_account(state, picker_value)
        acc_state = state.accounts.get(acct) if acct else None
        return render_structures_section(acc_state, struct_view or "grouped")

    # ---- Grouped vs standalone toggle --------------------------------------
    @app.callback(
        Output("struct-view-mode", "data"),
        Output("deepdive-structures", "children", allow_duplicate=True),
        Input("struct-grouped-btn", "n_clicks"),
        Input("struct-standalone-btn", "n_clicks"),
        State("deepdive-account-picker", "value"),
        prevent_initial_call=True,
    )
    def _toggle_struct_view(_g, _s, picker_value):
        mode = "standalone" if ctx.triggered_id == "struct-standalone-btn" else "grouped"
        state = sa.get_state()
        acct = _resolve_account(state, picker_value)
        acc_state = state.accounts.get(acct) if (state and acct) else None
        return mode, render_structures_section(acc_state, mode)

    # ---- Positions grid click → shared modal -------------------------------
    # Ticker column → Tearsheet; any other column → Alert when the position has
    # fires, else Tearsheet (a no-fire position has no alert panel to show).
    @app.callback(
        Output("drawer-body", "children", allow_duplicate=True),
        Output("drawer-root", "className", allow_duplicate=True),
        Output("drawer-state", "data", allow_duplicate=True),
        Input("deepdive-positions-grid", "cellClicked"),
        prevent_initial_call=True,
    )
    def _open_from_dd_positions(cell):
        state = sa.get_state()
        if state is None or not cell:
            return no_update, no_update, no_update
        account, position_id, underlying = _ids_from_cell(cell)
        if not account or not position_id:
            return no_update, no_update, no_update
        underlying = underlying or _underlying_for(state, account, position_id)
        has_fires = bool(sa.fires_for_position(state, account, position_id))
        view = "alert" if (cell.get("colId") != "underlying" and has_fires) else "tearsheet"
        body = _render_body(state, account, position_id, underlying, view)
        return body, _OPEN_CLS, {"view": view, "account": account,
                                 "position_id": position_id, "underlying": underlying,
                                 "source": "deepdive-positions"}
