"""All Tab-1 + drawer + refresh callbacks.

Registered once per Dash instance via ``register_callbacks(app)``. The UI
reads state through ``state_access`` and never recomputes.
"""
from __future__ import annotations

import dash
from dash import Input, Output, State, ctx, html, no_update

from pm.ui import state_access as sa
from pm.ui.blotter.grid import (
    build_blotter_columns,
    cell_click_target,
    consolidate_fires_to_rows,
    default_grid_options,
    nav_display,
    sort_rows,
    step_row,
)
from pm.ui.components.status_bar import render_status_bar
from pm.ui.deepdive.header import account_options
from pm.ui.drawers.evidence import render_alerts
from pm.ui.drawers.signal_sheet import render_signal_sheet


_ALL_TIERS = [1, 2, 3]
_OPEN_CLS = "drawer-root drawer-open"


def _chip_class(base: str, active: bool) -> str:
    return base + (" tier-chip-active" if active else "")


def _filter_rows(rows: list[dict], tiers: list[int]) -> list[dict]:
    tset = set(tiers or [])
    return [r for r in rows if r.get("tier") in tset]


def _underlying_for(state, account: str, position_id: str) -> str:
    """The underlying ticker for a position (for the Tearsheet view)."""
    fires = sa.fires_for_position(state, account, position_id)
    if fires:
        return fires[0].underlying or ""
    pos = sa.position_by_id(state, account, position_id)
    if pos is not None:
        return pos.underlying_symbol or pos.symbol or ""
    return ""


def _render_body(state, account: str, position_id: str, underlying: str, view: str):
    """Render the modal body for the active view (Alert = all alerts on the
    position, stacked; Tearsheet = the per-underlying signal sheet)."""
    if view == "tearsheet":
        return render_signal_sheet(account, underlying, state)
    return render_alerts(account, position_id, state)


def _visible_rows_for_source(source, blotter_rows, dd_pos_rows):
    """The ordered consolidated rows the modal's prev/next should step through,
    chosen by which grid opened the modal. Defaults to the blotter so Tab 1
    behaviour is unchanged (Tab 2 stamps an explicit ``source``)."""
    if source == "deepdive-positions":
        return dd_pos_rows
    return blotter_rows


def register_callbacks(app: dash.Dash) -> None:

    # ---- Tier filter chips → tier-filter store + chip classes -------------
    @app.callback(
        Output("tier-filter", "data"),
        Output("tier-chip-all", "className"),
        Output("tier-chip-1", "className"),
        Output("tier-chip-2", "className"),
        Output("tier-chip-3", "className"),
        Input("tier-chip-all", "n_clicks"),
        Input("tier-chip-1", "n_clicks"),
        Input("tier-chip-2", "n_clicks"),
        Input("tier-chip-3", "n_clicks"),
        State("tier-filter", "data"),
        prevent_initial_call=True,
    )
    def _on_tier_chip(_a, _1, _2, _3, current):
        tiers = set(current or _ALL_TIERS)
        trig = ctx.triggered_id
        if trig == "tier-chip-all":
            tiers = set(_ALL_TIERS)
        elif trig == "tier-chip-1":
            tiers ^= {1}
        elif trig == "tier-chip-2":
            tiers ^= {2}
        elif trig == "tier-chip-3":
            tiers ^= {3}
        ordered = sorted(tiers)
        all_on = tiers == set(_ALL_TIERS)
        return (
            ordered,
            _chip_class("tier-chip", all_on),
            _chip_class("tier-chip tier-chip-t1", 1 in tiers),
            _chip_class("tier-chip tier-chip-t2", 2 in tiers),
            _chip_class("tier-chip tier-chip-t3", 3 in tiers),
        )

    # ---- Grouping toggle → group-mode store + toggle classes --------------
    @app.callback(
        Output("group-mode", "data"),
        Output("group-account", "className"),
        Output("group-pattern", "className"),
        Input("group-account", "n_clicks"),
        Input("group-pattern", "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_grouping(_acc, _pat):
        mode = "pattern" if ctx.triggered_id == "group-pattern" else "account"
        acc_cls = "group-toggle" + (" group-toggle-active" if mode == "account" else "")
        pat_cls = "group-toggle" + (" group-toggle-active" if mode == "pattern" else "")
        return mode, acc_cls, pat_cls

    # ---- (tier filter, grouping, full rows) → grid rows (re-sorted) -------
    # Community has no row grouping, so the Account|Pattern toggle is a
    # server-side re-sort: filter by tier, then sort_rows by the active mode.
    @app.callback(
        Output("blotter-grid", "columnDefs"),
        Output("blotter-grid", "rowData"),
        Output("blotter-grid", "dashGridOptions"),
        Input("tier-filter", "data"),
        Input("group-mode", "data"),
        Input("blotter-all-rows", "data"),
    )
    def _sync_grid(tiers, group_mode, all_rows):
        group_mode = group_mode if group_mode in ("account", "pattern") else "account"
        filtered = _filter_rows(all_rows or [], tiers if tiers is not None else _ALL_TIERS)
        rows = sort_rows(filtered, group_mode)
        return build_blotter_columns(), rows, default_grid_options()

    # =======================================================================
    # Modal: open / view-toggle / prev-next. Rows are consolidated (one per
    # position); drawer-state carries {account, position_id, underlying, view}
    # where view ∈ {"alert","tearsheet"}. All write the drawer outputs via
    # allow_duplicate=True. The grid-click handler has ONLY the concrete
    # cellClicked input (a single concrete input — no zero-match wildcards).
    # =======================================================================

    # 1) GRID cell click → open Alert (any column) or Tearsheet (ticker).
    @app.callback(
        Output("drawer-body", "children", allow_duplicate=True),
        Output("drawer-root", "className", allow_duplicate=True),
        Output("drawer-state", "data", allow_duplicate=True),
        Input("blotter-grid", "cellClicked"),
        prevent_initial_call=True,
    )
    def _open_from_grid(cell):
        state = sa.get_state()
        if state is None or not cell:
            return no_update, no_update, no_update
        row_id = cell.get("rowId")
        if not row_id or "::" not in row_id:
            return no_update, no_update, no_update          # blank/odd row
        account, _, position_id = row_id.partition("::")    # position_id may contain '|'
        if not position_id:
            return no_update, no_update, no_update
        underlying = _underlying_for(state, account, position_id)
        view = cell_click_target(cell.get("colId"))          # 'tearsheet' on ticker, else 'alert'
        body = _render_body(state, account, position_id, underlying, view)
        return body, _OPEN_CLS, {"view": view, "account": account,
                                 "position_id": position_id, "underlying": underlying,
                                 "source": "blotter"}

    # 2) Alert | Tearsheet view toggle — swaps the body, keeps the position.
    @app.callback(
        Output("drawer-body", "children", allow_duplicate=True),
        Output("drawer-state", "data", allow_duplicate=True),
        Input("view-alert", "n_clicks"),
        Input("view-tearsheet", "n_clicks"),
        State("drawer-state", "data"),
        prevent_initial_call=True,
    )
    def _toggle_view(_a, _t, drawer_state):
        state = sa.get_state()
        ds = drawer_state or {}
        if state is None or ds.get("view") is None or not ds.get("position_id"):
            return no_update, no_update
        view = "alert" if ctx.triggered_id == "view-alert" else "tearsheet"
        if view == ds.get("view"):
            return no_update, no_update
        body = _render_body(state, ds["account"], ds["position_id"],
                            ds.get("underlying", ""), view)
        return body, {**ds, "view": view}

    # 3) Prev/Next: step POSITION-to-position in the visible (filter+sort)
    #    order (virtualRowData = consolidated rows). View mode persists.
    @app.callback(
        Output("drawer-body", "children", allow_duplicate=True),
        Output("drawer-root", "className", allow_duplicate=True),
        Output("drawer-state", "data", allow_duplicate=True),
        Input("drawer-prev", "n_clicks"),
        Input("drawer-next", "n_clicks"),
        State("drawer-state", "data"),
        State("blotter-grid", "virtualRowData"),
        State("deepdive-positions-grid", "virtualRowData"),
        prevent_initial_call=True,
    )
    def _prev_next(_p, _n, drawer_state, blotter_rows, dd_pos_rows):
        state = sa.get_state()
        ds = drawer_state or {}
        # Prev/Next steps positions; the structure modal view has no position nav.
        if state is None or ds.get("view") not in ("alert", "tearsheet"):
            return no_update, no_update, no_update
        # ``or []`` — the source grid may be absent if the view was toggled
        # while a position modal stayed open (the other grid isn't rendered).
        visible_rows = _visible_rows_for_source(
            ds.get("source"), blotter_rows, dd_pos_rows) or []
        direction = "prev" if ctx.triggered_id == "drawer-prev" else "next"
        target = step_row(visible_rows, ds.get("account"), ds.get("position_id"), direction)
        if target is None:
            return no_update, no_update, no_update          # boundary
        account = target.get("_account")
        position_id = target.get("_position_id")
        underlying = target.get("_underlying") or _underlying_for(state, account, position_id)
        view = ds.get("view")
        body = _render_body(state, account, position_id, underlying, view)
        return body, _OPEN_CLS, {"view": view, "account": account,
                                 "position_id": position_id, "underlying": underlying,
                                 "source": ds.get("source", "blotter")}

    # ---- Close drawer (close button or overlay click) ---------------------
    @app.callback(
        Output("drawer-root", "className", allow_duplicate=True),
        Output("drawer-state", "data", allow_duplicate=True),
        Input("drawer-close-btn", "n_clicks"),
        Input("drawer-overlay", "n_clicks"),
        prevent_initial_call=True,
    )
    def _close_drawer(_c, _o):
        return "drawer-root", {"view": None}

    # ---- Nav + toggle presentation: derived from drawer-state -------------
    # Prev/next is visible in BOTH modes; indicator/disabled come from the
    # visible order; the active view's toggle button is highlighted.
    @app.callback(
        Output("drawer-nav", "className"),
        Output("drawer-prev", "disabled"),
        Output("drawer-next", "disabled"),
        Output("drawer-pos", "children"),
        Output("view-alert", "className"),
        Output("view-tearsheet", "className"),
        Input("drawer-state", "data"),
        State("blotter-grid", "virtualRowData"),
        State("deepdive-positions-grid", "virtualRowData"),
        prevent_initial_call=True,
    )
    def _nav_and_toggle(drawer_state, blotter_rows, dd_pos_rows):
        ds = drawer_state or {}
        view = ds.get("view")
        # The structure modal view has no position prev/next or Alert|Tearsheet
        # toggle — hide both. (view is None when the modal is closed.)
        if view not in ("alert", "tearsheet"):
            return ("drawer-nav drawer-nav-hidden", True, True, "",
                    "view-toggle-btn view-toggle-btn-hidden",
                    "view-toggle-btn view-toggle-btn-hidden")
        visible_rows = _visible_rows_for_source(
            ds.get("source"), blotter_rows, dd_pos_rows) or []
        text, prev_dis, next_dis = nav_display(
            visible_rows, ds.get("account"), ds.get("position_id"))
        alert_cls = "view-toggle-btn" + (" view-toggle-btn-active" if view == "alert" else "")
        tear_cls = "view-toggle-btn" + (" view-toggle-btn-active" if view == "tearsheet" else "")
        return "drawer-nav", prev_dis, next_dis, text, alert_cls, tear_cls

    # ---- Escape-to-close: one-time clientside keydown listener -------------
    # Attaches a guarded document listener once (drawer-root.id fires once on
    # load); on Escape, if the modal is open, clicks the existing close button.
    app.clientside_callback(
        """
        function(_id) {
            if (!window._pmEscBound) {
                window._pmEscBound = true;
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Escape') {
                        var root = document.getElementById('drawer-root');
                        if (root && root.className.indexOf('drawer-open') !== -1) {
                            var btn = document.getElementById('drawer-close-btn');
                            if (btn) { btn.click(); }
                        }
                    }
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("esc-listener-dummy", "data"),
        Input("drawer-root", "id"),
    )

    # ---- Initial load OR Refresh BBG → load state + status + rows + Tab 2 --
    # One callback backs both the one-shot post-render load and the manual
    # Refresh BBG button. It owns the (re)load, so it also bumps the deep-dive
    # tick (race-free: the tick changes only after reload_state() completes and
    # has written the singleton, which the Tab-2 populate callback then reads)
    # and refreshes the account-picker options/value. The spinner is driven by
    # ``bbg-load-sentinel`` (a dcc.Loading child): it shows while this callback
    # runs and clears when it returns. Existing data stays visible throughout —
    # only the outputs swap, on completion — so refresh is non-blocking.
    @app.callback(
        Output("status-bar-host", "children"),
        Output("blotter-all-rows", "data"),
        Output("deepdive-refresh-tick", "data"),
        Output("deepdive-account-picker", "options"),
        Output("deepdive-account-picker", "value"),
        Output("bbg-load-sentinel", "children"),
        Input("initial-load", "n_intervals"),
        Input("refresh-button", "n_clicks"),
        State("deepdive-account-picker", "value"),
        State("deepdive-refresh-tick", "data"),
        prevent_initial_call=True,
    )
    def _load_or_refresh(_n_intervals, _n_clicks, picker_value, tick):
        try:
            new_state = sa.reload_state()  # handles prev=None (first load)
        except Exception as exc:  # surface failure in the status bar
            return (html.Div(f"Load failed: {exc}", className="status-left status-empty"),
                    no_update, no_update, no_update, no_update, "")
        rows = consolidate_fires_to_rows(sa.all_fires(new_state), new_state)
        opts = account_options(new_state)
        valid = [o["value"] for o in opts]
        # Preserve the user's selection on a manual refresh; default to the
        # first account on the initial load (picker_value is None then).
        value = picker_value if picker_value in valid else (valid[0] if valid else None)
        return (render_status_bar(new_state), rows, (tick or 0) + 1, opts, value, "")
