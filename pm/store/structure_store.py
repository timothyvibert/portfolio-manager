"""Minimal local persistence for structure confirm/override resolutions.

A small JSON file under the data dir, keyed by ``(account, sorted leg-set)``. The
detection pass reads these resolutions and applies them to the freshly-detected
proposals: when the leg-set still matches, the user's confirm / reject / choose /
edit is honoured; when the legs have changed, the stale key no longer matches and
the structure demotes to a fresh proposal.

This is a deliberately small, single-file store behind a clean interface. The
larger V2 snapshot store will replace it behind this same interface. The JSON file
is sample-state under ``pm/data`` and is gitignored — never committed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from pm.config import DATA_DIR

_STORE_PATH = DATA_DIR / "structure_resolutions.json"

CONFIRMED = "confirmed"
REJECTED = "rejected"
EDITED = "edited"


# ---------------------------------------------------------------------------
# Keying + file I/O
# ---------------------------------------------------------------------------
def _key(account: str, leg_pids) -> str:
    return account + "||" + ",".join(sorted(leg_pids))


def _load() -> dict:
    try:
        return json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public write/read interface
# ---------------------------------------------------------------------------
def save_resolution(account: str, leg_pids, resolution: str,
                    chosen_type: Optional[str] = None,
                    edited_legs: Optional[list] = None,
                    now: Optional[datetime] = None) -> None:
    data = _load()
    data[_key(account, leg_pids)] = {
        "resolution": resolution,            # confirmed | rejected | edited
        "chosen_type": chosen_type,          # for a contention choice: the picked reading
        "edited_legs": edited_legs,          # for an edit: the kept leg position_ids
        "timestamp": (now or datetime.now(timezone.utc)).isoformat(),
    }
    _save(data)


def get_resolution(account: str, leg_pids) -> Optional[dict]:
    return _load().get(_key(account, leg_pids))


def all_resolutions() -> dict:
    return _load()


def clear_resolution(account: str, leg_pids) -> None:
    data = _load()
    data.pop(_key(account, leg_pids), None)
    _save(data)


def clear_all() -> None:
    _save({})


# ---------------------------------------------------------------------------
# Apply stored resolutions onto freshly-detected structures (in place)
# ---------------------------------------------------------------------------
def decision_leg_pids(structures, s) -> list:
    """The leg-set the resolution is keyed on. For a contended structure it is the
    union of all legs across the contention group (the decision spans the
    alternatives); otherwise it is the structure's own legs."""
    if s.contention_group:
        pids = set()
        for other in structures:
            if other.contention_group == s.contention_group:
                pids.update(leg.position_id for leg in other.legs)
        return sorted(pids)
    return sorted(leg.position_id for leg in s.legs)


def apply_resolutions(account: str, structures, resolutions: Optional[dict] = None) -> None:
    """Set each structure's ``status`` from the stored resolution for its leg-set.
    Legs unchanged → honour confirm / reject / choose / edit. Legs changed → no
    stored key matches → the structure stays ``proposed`` (a fresh proposal)."""
    data = resolutions if resolutions is not None else _load()
    for s in structures:
        r = data.get(_key(account, decision_leg_pids(structures, s)))
        if not r:
            s.status = "proposed"
            continue
        resolution = r.get("resolution")
        if s.contention_group:
            # One reading is chosen; the other alternatives are rejected.
            if resolution == CONFIRMED:
                s.status = CONFIRMED if r.get("chosen_type") == s.type else REJECTED
            else:
                s.status = REJECTED if resolution == REJECTED else "proposed"
        else:
            if resolution == EDITED:
                kept = set(r.get("edited_legs") or [])
                s.legs = [leg for leg in s.legs if leg.position_id in kept] or s.legs
                s.status = EDITED
            elif resolution in (CONFIRMED, REJECTED):
                s.status = resolution
            else:
                s.status = "proposed"
        if r.get("timestamp"):
            s.resolved_at = r["timestamp"]
