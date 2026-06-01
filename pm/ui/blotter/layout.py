"""Tab 1 layout: tier filter chips + grouping toggle + the AG Grid blotter.

The persistent dcc.Store components (tier filter, grouping mode, full row
set, drawer state) live in the shell so they survive drawer interactions;
this module owns the static structure of the tab and the grid's initial
props.
"""
from __future__ import annotations

from typing import Optional

import dash_ag_grid as dag
from dash import html

from pm.store.portfolio_state import PortfolioState
from pm.ui import state_access as sa
from pm.ui.blotter.grid import (
    build_blotter_columns,
    consolidate_fires_to_rows,
    default_grid_options,
    sort_rows,
)


def _tier_chips() -> html.Div:
    return html.Div(className="blotter-chips", children=[
        html.Span("Tiers:", className="blotter-control-label"),
        html.Button("All", id="tier-chip-all", n_clicks=0,
                    className="tier-chip tier-chip-active"),
        html.Button("T1", id="tier-chip-1", n_clicks=0,
                    className="tier-chip tier-chip-active tier-chip-t1"),
        html.Button("T2", id="tier-chip-2", n_clicks=0,
                    className="tier-chip tier-chip-active tier-chip-t2"),
        html.Button("T3", id="tier-chip-3", n_clicks=0,
                    className="tier-chip tier-chip-active tier-chip-t3"),
    ])


def _grouping_toggle() -> html.Div:
    return html.Div(className="blotter-grouping", children=[
        html.Span("Group by:", className="blotter-control-label"),
        html.Button("Account", id="group-account", n_clicks=0,
                    className="group-toggle group-toggle-active"),
        html.Button("Pattern", id="group-pattern", n_clicks=0,
                    className="group-toggle"),
    ])


def render_blotter_tab(state: Optional[PortfolioState]) -> html.Div:
    """Always renders the controls + grid, even before data loads (state is
    None → empty grid). The grid then fills via the ``blotter-all-rows`` store
    once the async load completes, so the component exists as a callback target
    from first paint."""
    rows = (sort_rows(consolidate_fires_to_rows(sa.all_fires(state), state), "account")
            if state else [])
    columns = build_blotter_columns()
    grid_options = default_grid_options()

    grid = dag.AgGrid(
        id="blotter-grid",
        columnDefs=columns,
        rowData=rows,
        dashGridOptions=grid_options,
        className="ag-theme-balham blotter-grid",
        # No columnSize sizeToFit — the 'alerts' column uses flex:2 to fill
        # remaining width while the metric columns keep fixed widths.
        # One row per position → rowId = account::position_id. '::' separator
        # because position_id itself contains '|'; account never contains '::'.
        getRowId={"function":
                  "params.data._account + '::' + params.data._position_id"},
        style={"height": "calc(100vh - 190px)", "width": "100%"},
    )

    return html.Div(className="blotter-tab", children=[
        html.Div(className="blotter-controls", children=[
            _tier_chips(),
            _grouping_toggle(),
        ]),
        grid,
    ])
