"""Local persistence for per-alert suppress / snooze decisions (item 9).

A suppression silences a *specific* alert — keyed ``(account, name, pattern_id)``
where ``name == fire.underlying`` — either permanently (``suppressed_until`` is
NULL) or until a date (a snooze). It is **selective by design**: suppressing P12
(concentration) on XLK in one account silences only P12 on XLK there; every other
pattern on XLK, and P12 on other names, still fires.

This is the first feature that *writes* user state through the SQLite seam
(``pm/store/db.py``). It mirrors ``structure_store``'s shape — thin I/O over
``db.connection()`` plus an ``apply_suppressions`` pass — and adds no dependency.

**No-recompute contract.** ``apply_suppressions`` runs in the load path *after* the
engine has produced fires. It only *marks* a fire (sets ``Fire.suppression``); it
never removes the fire from ``acc.fires`` and never re-derives anything. So a restore
(or a snooze that has expired) plus the next apply clears the mark and the alert
returns — the fire object was there all along. The active surfaces (blotter
consolidation, deep-dive rows) filter on :func:`is_active`; the modal's Muted footer
and the Alert Manager read the *muted* fires to display and restore them.

Snooze expiry is **lazy treat-as-inactive**: a row whose ``suppressed_until`` is
before ``as_of`` is simply absent from the active set (so the alert fires again), but
the row is retained — auditable, no background cleanup job. ``created_at`` drives the
Alert Manager's "days active" staleness cue.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Optional

from pm.insight.patterns import SuppressionMark
from pm.store import db


# ---------------------------------------------------------------------------
# Row <-> record mapping
# ---------------------------------------------------------------------------
_COLUMNS = (
    "account", "name", "pattern_id", "suppressed_until",
    "created_at", "captured_trace", "captured_rationale",
)
_SELECT = "SELECT " + ", ".join(_COLUMNS) + " FROM suppressions"


def _record(row) -> dict:
    return dict(zip(_COLUMNS, row))


# ---------------------------------------------------------------------------
# Write interface
# ---------------------------------------------------------------------------
def suppress(account: str, name: str, pattern_id: str, *,
             suppressed_until: Optional[str] = None,
             trace: Optional[dict] = None,
             rationale: Optional[str] = None,
             now: Optional[datetime] = None) -> None:
    """Suppress (``suppressed_until=None``) or snooze (a ``'YYYY-MM-DD'`` string) the
    alert ``(account, name, pattern_id)``. Upserts on the composite key, so re-calling
    flips suppress<->snooze and refreshes ``created_at`` (a re-decision restarts the
    days-active clock).

    ``trace`` / ``rationale`` capture the acting fire's baseline at suppress time
    (the fire is in hand where the control fires). ``trace`` is serialized with
    ``default=str`` so a live BBG trace carrying ``date`` objects in ``as_of`` cannot
    crash the write. This is **store-only** — nothing in this increment ever compares
    the captured baseline against the current fire (that is the deferred
    material-change feature)."""
    created_at = (now or datetime.now(timezone.utc)).isoformat()
    captured_trace = json.dumps(trace, default=str) if trace is not None else None
    with db.connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO suppressions"
            "(account, name, pattern_id, suppressed_until, created_at,"
            " captured_trace, captured_rationale) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (account, name, pattern_id, suppressed_until, created_at,
             captured_trace, rationale),
        )


def restore(account: str, name: str, pattern_id: str) -> None:
    """Remove the suppression — the alert returns on the next apply. A pure no-op when
    nothing is persisted yet (never materializes the database)."""
    if not db.store_exists():
        return
    with db.connection() as conn:
        conn.execute(
            "DELETE FROM suppressions WHERE account = ? AND name = ? AND pattern_id = ?",
            (account, name, pattern_id),
        )


def clear_all() -> None:
    """Drop every suppression (test/reset helper). No-op on an empty store."""
    if not db.store_exists():
        return
    with db.connection() as conn:
        conn.execute("DELETE FROM suppressions")


# ---------------------------------------------------------------------------
# Read interface — pure, never creates the database (mirrors structure_store)
# ---------------------------------------------------------------------------
def get_suppression(account: str, name: str, pattern_id: str) -> Optional[dict]:
    if not db.store_exists():
        return None
    with db.connection() as conn:
        row = conn.execute(
            _SELECT + " WHERE account = ? AND name = ? AND pattern_id = ?",
            (account, name, pattern_id),
        ).fetchone()
    return _record(row) if row else None


def all_suppressions() -> list[dict]:
    """Every stored suppression as a full record (the Alert Manager renders these and
    computes days-active). Includes expired snoozes — they are retained, not deleted."""
    if not db.store_exists():
        return []
    with db.connection() as conn:
        rows = conn.execute(_SELECT).fetchall()
    return [_record(r) for r in rows]


def _is_active(suppressed_until: Optional[str], as_of_str: str) -> bool:
    # Permanent, or snoozed through a date >= today. ISO 'YYYY-MM-DD' strings compare
    # lexicographically the same as by date, so the boundary day is still suppressed.
    return suppressed_until is None or suppressed_until >= as_of_str


def active_suppressions(as_of: Optional[date] = None) -> dict[tuple[str, str, str], dict]:
    """The currently-active suppressions as ``{(account, name, pattern_id): record}``.
    A snooze with ``suppressed_until < as_of`` is inactive (the alert returns); its row
    persists. ``as_of`` defaults to today."""
    if not db.store_exists():
        return {}
    as_of_str = (as_of or date.today()).isoformat()
    return {
        (r["account"], r["name"], r["pattern_id"]): r
        for r in all_suppressions()
        if _is_active(r["suppressed_until"], as_of_str)
    }


# ---------------------------------------------------------------------------
# Apply onto already-computed fires (the load-path marking pass)
# ---------------------------------------------------------------------------
def _mark_for(record: dict) -> SuppressionMark:
    until = record["suppressed_until"]
    return SuppressionMark(kind="snoozed" if until else "suppressed", until=until)


def _mark_fires(account: str, fires, active: dict) -> None:
    """Set each fire's ``suppression`` from the active set — and CLEAR it where no
    active suppression matches. Clearing is what makes restore / snooze-expiry return
    the alert on a re-apply without recomputing."""
    for f in fires:
        rec = active.get((account, f.underlying, f.pattern_id))
        f.suppression = _mark_for(rec) if rec else None


def apply_suppressions(state, as_of: Optional[date] = None) -> None:
    """Mark every fire across all accounts whose ``(account, underlying, pattern_id)``
    is actively suppressed. Called in ``load_portfolio_state`` after the engine — it
    only flags already-computed fires (no recompute, no Bloomberg)."""
    active = active_suppressions(as_of)
    for acc in state.accounts.values():
        _mark_fires(acc.account, acc.fires, active)


def remark_account(account_state, as_of: Optional[date] = None) -> None:
    """Re-mark one account's fires — used by ``state_access.resolve_structure`` after it
    re-derives a structure's fires, so a just-confirmed fire matching an active
    suppression is muted without a reload. Same marking logic as the load path."""
    _mark_fires(account_state.account, account_state.fires, active_suppressions(as_of))


def is_active(fire) -> bool:
    """The shared predicate: a fire is an active alert iff it carries no suppression
    mark. Every surface that hides muted alerts (blotter consolidation, deep-dive rows)
    uses this so they can never diverge from one another."""
    return fire.suppression is None
