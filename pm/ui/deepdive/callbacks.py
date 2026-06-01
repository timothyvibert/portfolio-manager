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
from dash import Input, Output, ctx, no_update

from pm.ui import state_access as sa
from pm.ui.blotter.callbacks import _OPEN_CLS, _render_body, _underlying_for
from pm.ui.deepdive.header import account_options, default_account
from pm.ui.deepdive.layout import render_deepdive_sections

_DD_HOST_IDS = [
    "deepdive-kpi", "deepdive-positions", "deepdive-analytics", "deepdive-trades",
]


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
        Output("deepdive-analytics", "children"),
        Output("deepdive-trades", "children"),
        Input("deepdive-account-picker", "value"),
        Input("deepdive-refresh-tick", "data"),
        Input("pm-tabs", "value"),
    )
    def _populate_deepdive(account, _tick, tab_value):
        # On a tab switch, only rebuild when entering the deep-dive tab (this
        # also re-paints with fresh data after a refresh done while on Tab 1).
        # Picker / refresh always rebuild — the tab is active then.
        if ctx.triggered_id == "pm-tabs" and tab_value != "tab-account":
            return (no_update,) * len(_DD_HOST_IDS)
        state = sa.get_state()
        acct = _resolve_account(state, account)
        sections = render_deepdive_sections(state, acct)
        return tuple(sections[h] for h in _DD_HOST_IDS)

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
