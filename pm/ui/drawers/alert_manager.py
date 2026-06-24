"""Alert Manager — the book-wide review/reverse surface for suppressed alerts (item 9).

A discrete modal (separate from the per-alert drawer) opened from the top-right
control cluster. Two tabs: **Suppressed** (the active suppressions, restorable) and
**Thresholds** (a stub for a later increment). The Suppressed tab is a dense
``html.Table`` — not a second AG-Grid — matching the signal-sheet / trace design
language. Restore here uses the *same* ``state_access.restore_alert`` as the modal's
Muted footer; there is no second mechanism.

Days-active (today − created_at) is the deliberate staleness cue: a suppression aging
past usefulness shows it and the user restores it — there is no automatic
re-surfacing in this increment (that is the deferred material-change feature).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from dash import html

from pm.insight.pattern_groups import all_pattern_meta
from pm.store import suppression_store


def _days_active(created_at: Optional[str], today: date) -> int:
    """Whole days since the suppression was set (>= 0). Defensive against a missing or
    unparseable timestamp."""
    if not created_at:
        return 0
    try:
        started = datetime.fromisoformat(created_at).date()
    except ValueError:
        return 0
    return max(0, (today - started).days)


def _state_text(record: dict) -> str:
    until = record.get("suppressed_until")
    return f"Snoozed until {until}" if until else "Suppressed"


def _restore_id(record: dict) -> dict:
    return {"type": "am-restore", "account": record["account"],
            "name": record["name"], "pat": record["pattern_id"]}


def render_suppressed_tab(today: Optional[date] = None) -> html.Div:
    """The active suppressions, sorted/grouped by account. Empty → a neutral note."""
    today = today or date.today()
    meta = all_pattern_meta()
    records = sorted(suppression_store.active_suppressions(today).values(),
                     key=lambda r: (r["account"], r["name"], r["pattern_id"]))
    if not records:
        return html.Div("No suppressed alerts", className="am-empty")

    header = html.Tr([html.Th(h, className="am-th") for h in
                      ("Account", "Name", "Alert type", "State", "Days active", "")])
    body_rows = []
    for r in records:
        pid = r["pattern_id"]
        alert_type = meta.get(pid, (pid, None))[0]
        # captured_rationale surfaced as a row tooltip — what the alert looked like
        # when it was muted, so a long-lived suppression can be eyeballed.
        rationale = (r.get("captured_rationale") or "").strip()
        body_rows.append(html.Tr(
            className="am-row",
            title=rationale or None,
            children=[
                html.Td(r["account"], className="am-acct"),
                html.Td(r["name"], className="am-name"),
                html.Td(alert_type, className="am-type"),
                html.Td(_state_text(r), className="am-state"),
                html.Td(str(_days_active(r.get("created_at"), today)), className="am-days"),
                html.Td(html.Button("Restore", id=_restore_id(r), n_clicks=0,
                                    className="alert-action-btn am-restore-btn")),
            ],
        ))
    return html.Table(className="am-table", children=[
        html.Thead(header), html.Tbody(body_rows)])


def render_thresholds_tab() -> html.Div:
    """A visible-but-stubbed frame. No threshold logic in this increment."""
    return html.Div(className="am-stub", children=[
        html.Div("Editable thresholds", className="am-stub-title"),
        html.Div("Tuning alert thresholds from here is coming in a later release.",
                 className="am-stub-note"),
    ])


def render_alert_manager_body(tab: str = "suppressed",
                              today: Optional[date] = None) -> html.Div:
    inner = render_thresholds_tab() if tab == "thresholds" else render_suppressed_tab(today)
    return html.Div(className="am-body-inner", children=[inner])
