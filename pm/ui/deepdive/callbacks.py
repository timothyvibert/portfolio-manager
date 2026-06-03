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
from pm.ui.deepdive.positions import render_positions_section
from pm.ui.deepdive.structures_panel import build_structure_rows, render_structure_detail

_DD_HOST_IDS = [
    "deepdive-kpi", "deepdive-positions", "deepdive-analytics", "deepdive-trades",
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

    # ---- Picker / refresh / tab-switch → repopulate the deep-dive sections --
    # A picker change resets the Holdings view to By Position + collapses any
    # expansion (a fresh account starts clean); refresh/tab-switch keep the
    # current view via the stores.
    @app.callback(
        Output("deepdive-kpi", "children"),
        Output("deepdive-positions", "children"),
        Output("deepdive-analytics", "children"),
        Output("deepdive-trades", "children"),
        Input("deepdive-account-picker", "value"),
        Input("deepdive-refresh-tick", "data"),
        Input("pm-tabs", "value"),
        State("pos-view-mode", "data"),
        State("struct-expanded", "data"),
    )
    def _populate_deepdive(account, _tick, tab_value, pos_view, expanded):
        # On a tab switch, only rebuild when entering the deep-dive tab (this
        # also re-paints with fresh data after a refresh done while on Tab 1).
        # Picker / refresh always rebuild — the tab is active then.
        if ctx.triggered_id == "pm-tabs" and tab_value != "tab-account":
            return (no_update,) * len(_DD_HOST_IDS)
        if ctx.triggered_id == "deepdive-account-picker":
            pos_view, expanded = "position", []
        state = sa.get_state()
        acct = _resolve_account(state, account)
        sections = render_deepdive_sections(state, acct, pos_view or "position", expanded)
        return tuple(sections[h] for h in _DD_HOST_IDS)

    # ---- Structure confirm / reject / edit / choose-alternative -------------
    # The affordances live in the modal now. Each writes through the single state
    # owner (state_access.resolve_structure → the store; re-applies the resolution
    # to the in-memory structures; no signal recompute), then re-renders the modal
    # body (new status) AND the Holdings section (the By Structure grid row).
    @app.callback(
        Output("drawer-body", "children", allow_duplicate=True),
        Output("deepdive-positions-grid", "rowData", allow_duplicate=True),
        Input({"type": "struct-confirm", "account": ALL, "sid": ALL}, "n_clicks"),
        Input({"type": "struct-reject", "account": ALL, "sid": ALL}, "n_clicks"),
        Input({"type": "struct-removeleg", "account": ALL, "sid": ALL, "leg": ALL}, "n_clicks"),
        Input({"type": "struct-choose", "account": ALL, "group": ALL}, "n_clicks"),
        State({"type": "struct-radio", "group": ALL}, "value"),
        State("deepdive-account-picker", "value"),
        State("pos-view-mode", "data"),
        State("struct-expanded", "data"),
        prevent_initial_call=True,
    )
    def _resolve_structure(_c, _r, _rm, _ch, _radio, picker_value, pos_view, expanded):
        trig = ctx.triggered_id
        state = sa.get_state()
        if not isinstance(trig, dict) or state is None:
            return no_update, no_update
        # ignore the spurious fire when a control (re)mounts with n_clicks None/0
        if not (ctx.triggered[0] if ctx.triggered else {}).get("value"):
            return no_update, no_update
        ttype = trig.get("type")
        account = trig.get("account")
        sid_for_modal = trig.get("sid")
        if ttype == "struct-confirm":
            sa.resolve_structure(account, trig["sid"], "confirmed")
        elif ttype == "struct-reject":
            sa.resolve_structure(account, trig["sid"], "rejected")
        elif ttype == "struct-removeleg":
            acc = state.accounts.get(account)
            s = next((x for x in acc.structures if x.structure_id == trig["sid"]), None) if acc else None
            if s is None:
                return no_update, no_update
            kept = [l.position_id for l in s.legs if l.position_id != trig["leg"]]
            if not kept:
                return no_update, no_update
            sa.resolve_structure(account, trig["sid"], "edited", edited_legs=kept)
        elif ttype == "struct-choose":
            chosen_sid = _radio_value_for_group(trig.get("group"))
            if not chosen_sid:          # explicit choice required — no silent confirm
                return no_update, no_update
            sa.resolve_structure(account, chosen_sid, "confirmed")
            sid_for_modal = chosen_sid
        acc_state = state.accounts.get(account) if account else None
        body = (render_structure_detail(account, sid_for_modal, state)
                if (account and sid_for_modal) else no_update)
        # Update the mounted grid's rowData (not a section re-render) so the new
        # status shows on the persistent grid — same reason as the caret toggle.
        rows = (build_structure_rows(acc_state, state, expanded) if acc_state is not None
                else no_update)
        return body, rows

    # ---- By Position | By Structure toggle (server-side re-render) ----------
    @app.callback(
        Output("pos-view-mode", "data"),
        Output("deepdive-positions", "children", allow_duplicate=True),
        Input("pos-byposition-btn", "n_clicks"),
        Input("pos-bystructure-btn", "n_clicks"),
        State("deepdive-account-picker", "value"),
        State("struct-expanded", "data"),
        prevent_initial_call=True,
    )
    def _toggle_pos_view(_p, _s, picker_value, expanded):
        mode = "structure" if ctx.triggered_id == "pos-bystructure-btn" else "position"
        state = sa.get_state()
        acct = _resolve_account(state, picker_value)
        acc_state = state.accounts.get(acct) if (state and acct) else None
        return mode, render_positions_section(acc_state, state, mode, expanded)

    # ---- Holdings grid click → modal / expand ------------------------------
    # One grid id serves both views (only one is mounted at a time), so a single
    # cellClicked handler dispatches by row kind: structure rows (By Structure)
    # caret-expand in place or open the structure modal; position rows (By
    # Position) open the shared Alert/Tearsheet modal (ticker col → Tearsheet,
    # else Alert when the position has fires).
    @app.callback(
        Output("drawer-body", "children", allow_duplicate=True),
        Output("drawer-root", "className", allow_duplicate=True),
        Output("drawer-state", "data", allow_duplicate=True),
        Output("struct-expanded", "data", allow_duplicate=True),
        Output("deepdive-positions-grid", "rowData", allow_duplicate=True),
        Input("deepdive-positions-grid", "cellClicked"),
        State("struct-expanded", "data"),
        State("pos-view-mode", "data"),
        prevent_initial_call=True,
    )
    def _on_holdings_click(cell, expanded, pos_view):
        nup = (no_update, no_update, no_update, no_update, no_update)
        state = sa.get_state()
        if state is None or not cell:
            return nup
        # By Structure rows are identified by their stable _row_id (this grid's
        # cellClicked echoes rowId but not row data). The structure_id embeds the
        # account as its first '|'-segment.
        row_id = cell.get("rowId") or ""
        prefix = row_id.split("::", 1)[0]
        if prefix in ("structure", "contention", "leg", "sub", "standalone"):
            if prefix == "standalone":
                return nup  # display-only
            sid = row_id.split("::")[1]
            account = sid.split("|")[0]
            acc_state = state.accounts.get(account)
            if cell.get("colId") == "caret" and prefix == "structure":
                # Update rowData on the SAME mounted grid (not a section re-render):
                # replacing the grid would reconcile-reuse it and leave cellClicked
                # stale, so a second caret click wouldn't re-fire.
                ex = list(expanded or [])
                ex.remove(sid) if sid in ex else ex.append(sid)
                rows = build_structure_rows(acc_state, state, ex)
                return no_update, no_update, no_update, ex, rows
            body = render_structure_detail(account, sid, state)
            return (body, _OPEN_CLS,
                    {"view": "structure", "account": account, "structure_id": sid},
                    no_update, no_update)

        # --- By Position rows ---
        account, position_id, underlying = _ids_from_cell(cell)
        if not account or not position_id:
            return nup
        underlying = underlying or _underlying_for(state, account, position_id)
        has_fires = bool(sa.fires_for_position(state, account, position_id))
        view = "alert" if (cell.get("colId") != "underlying" and has_fires) else "tearsheet"
        body = _render_body(state, account, position_id, underlying, view)
        return (body, _OPEN_CLS,
                {"view": view, "account": account, "position_id": position_id,
                 "underlying": underlying, "source": "deepdive-positions"},
                no_update, no_update)
