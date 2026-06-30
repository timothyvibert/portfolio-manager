"""Section — Client profile (the trade-history behavioural read).

Renders the precomputed per-account ``acc.client_profile`` (from
pm.insight.client_profile): a five-second, pre-call glance at *how this client
trades* — strategy posture, tenor, direction, sector lean, sizing — over a
coverage band that keeps the read honest about thin history. A pure read of
``acc.client_profile``: no compute, no Bloomberg, no recompute.

This surface is behavioural, not P&L, so it stays in neutral chrome. Sign-colour
is reserved for the one genuine direction the desk reads (net delta lean);
magnitudes and the coverage / confidence signals stay quiet, and a suppressed or
low-confidence dimension says so rather than showing a precise-looking false
number.
"""
from __future__ import annotations

from typing import Optional

from dash import html

from pm.ui.deepdive.bars import bar_row
from pm.ui.deepdive.formatters import money_compact, pct

# Display labels (render-only — the engine keys are the source of truth).
_POSTURE_LABELS = {
    "long_call": "Long calls",
    "short_call": "Short calls",
    "long_put": "Long puts",
    "short_put": "Short puts",
    "long_stock": "Long stock",
    "unclassified_open": "Unclassified",
}
_TENOR_LABELS = {"short": "Short-dated", "swing": "Swing", "leaps": "LEAPS"}
_BAND_HELP = ("Coverage band from trade count and round-trip evidence — thin "
              "history reads low, so dimensions degrade rather than mislead.")


# ---------------------------------------------------------------------------
# Small render helpers (mirror the exposure/analytics section primitives)
# ---------------------------------------------------------------------------

def _stat(label: str, value: str, sub: Optional[str] = None, cls: str = "") -> html.Div:
    children = [html.Div(label, className="dd-stat-label"),
                html.Div(value, className="dd-stat-value")]
    if sub:
        children.append(html.Div(sub, className="dd-stat-sub"))
    return html.Div(className=f"dd-stat {cls}".strip(), children=children)


def _sign_cls(v: Optional[float]) -> str:
    """Sign colouring — used ONLY for a genuine directional ± (net delta lean)."""
    if v is None or v == 0:
        return ""
    return "dd-stat-pos" if v > 0 else "dd-stat-neg"


def _conf_suffix(confidence: Optional[str]) -> str:
    """A quiet caveat appended to a card subtitle when the read is thin."""
    return " · low confidence" if confidence == "low" else ""


def _window_text(days: Optional[int]) -> str:
    return "—" if days is None else f"{days}d"


def _skew_text(v: Optional[float], pos_word: str, neg_word: str) -> str:
    """A signed lean as a magnitude + side, e.g. +0.64 -> '64% long'."""
    if v is None:
        return "—"
    return f"{pct(abs(v), 0)} {pos_word if v >= 0 else neg_word}"


def _hhi_text(h: Optional[float]) -> str:
    return "—" if h is None else f"{h:.2f}"


def _tenor_text(median_dte: Optional[float], bucket: Optional[str]) -> str:
    label = _TENOR_LABELS.get(bucket, bucket or "—")
    if median_dte is None:
        return label
    return f"{label} · {median_dte:.0f}d median"


def _dist_text(dist: dict) -> str:
    if not dist:
        return ""
    return (f"{pct(dist.get('short', 0), 0)}/{pct(dist.get('swing', 0), 0)}/"
            f"{pct(dist.get('leaps', 0), 0)} short/swing/LEAPS")


# ---------------------------------------------------------------------------
# Pure card-builders (ClientProfile -> display rows; unit-testable, no browser)
# ---------------------------------------------------------------------------

def build_strategy_rows(profile) -> list[dict]:
    """Strategy-posture bars, largest weight first."""
    sb = profile.strategy_bias
    ranked = sorted(sb.weights.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"key": k, "label": _POSTURE_LABELS.get(k, k), "weight": w} for k, w in ranked]


def build_lean_rows(profile) -> dict:
    """Sector lean when any name resolved to a GICS sector, else the
    always-available underlying lean — flagged by ``mode``."""
    sl = profile.sector_lean
    if sl.top:
        return {"mode": "sector", "classified": sl.classified_fraction,
                "rows": [{"label": s, "weight": w} for s, w in sl.top]}
    return {"mode": "name", "classified": sl.classified_fraction,
            "rows": [{"label": n, "weight": w} for n, w in sl.by_name]}


def build_coverage_stats(profile) -> dict:
    """The history-depth numbers, with the two fractions kept distinct: the
    round-trip rate and the held-positions-with-an-opening-trade rate (which
    carries the opens-predate-window caveat)."""
    c = profile.coverage
    return {
        "band": c.band,
        "n_trades": c.n_trades,
        "window_days": c.window_days,
        "paired_fraction": c.paired_fraction,
        "positions_with_derivable_open_fraction": c.positions_with_derivable_open_fraction,
    }


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

def _fingerprint_panel(profile) -> html.Div:
    band = profile.coverage.band
    return html.Div(className="dd-panel", children=[
        html.Div(className="dd-panel-headrow", children=[
            html.H3("How this client trades", className="dd-panel-title"),
            html.Span(f"coverage · {band}", className="dd-beta-chip", title=_BAND_HELP),
        ]),
        html.Div(profile.headline or "—", className="dd-stat-value"),
        html.Div(f"{profile.coverage.n_trades} trades over "
                 f"{_window_text(profile.coverage.window_days)}",
                 className="dd-panel-subtitle"),
    ])


def _strategy_panel(profile) -> html.Div:
    rows = build_strategy_rows(profile)
    if rows:
        max_w = max(r["weight"] for r in rows)
        body = html.Div(className="dd-bars", children=[
            bar_row(r["label"], r["weight"], max_w, sign_color=False) for r in rows
        ])
    else:
        body = html.Div("Insufficient history.", className="dd-empty")
    return html.Div(className="dd-panel", children=[
        html.H3("Strategy mix", className="dd-panel-title"),
        html.Div("What the account opens, by posture — from trade flow"
                 + _conf_suffix(profile.strategy_bias.confidence) + ".",
                 className="dd-panel-subtitle"),
        body,
    ])


def _tenor_direction_panel(profile) -> html.Div:
    t = profile.tenor_pref
    d = profile.direction_bias
    if t.bucket:
        tenor_stat = _stat("At-open tenor", _tenor_text(t.median_dte_at_open, t.bucket),
                           sub=_dist_text(t.distribution))
    else:
        tenor_stat = _stat("At-open tenor", "—", sub="insufficient history")
    return html.Div(className="dd-panel", children=[
        html.H3("Tenor & direction", className="dd-panel-title"),
        html.Div("At-open days-to-expiry and directional lean"
                 + _conf_suffix(d.confidence) + ".", className="dd-panel-subtitle"),
        html.Div(className="dd-stat-row", children=[
            tenor_stat,
            # Call vs put is a composition, not a gain/loss — kept neutral.
            _stat("Call / put", _skew_text(d.call_put_skew, "calls", "puts")),
            # Net delta lean is a genuine bullish/bearish direction — sign-coloured.
            _stat("Net Δ", _skew_text(d.long_short_skew, "long", "short"),
                  cls=_sign_cls(d.long_short_skew)),
        ]),
    ])


def _lean_panel(profile) -> html.Div:
    info = build_lean_rows(profile)
    rows = info["rows"]
    if rows:
        max_w = max(r["weight"] for r in rows)
        body = html.Div(className="dd-bars", children=[
            bar_row(r["label"], r["weight"], max_w, sign_color=False) for r in rows
        ])
    else:
        body = html.Div("Insufficient history.", className="dd-empty")
    if info["mode"] == "sector":
        sub = f"By GICS sector · {pct(info['classified'], 0)} of flow classified"
    else:
        sub = "By underlying (sector unavailable for traded names)"
    return html.Div(className="dd-panel", children=[
        html.H3("Sector / name lean", className="dd-panel-title"),
        html.Div(sub, className="dd-panel-subtitle"),
        body,
    ])


def _sizing_coverage_panel(profile) -> html.Div:
    s = profile.sizing
    cad = profile.cadence
    c = profile.coverage
    size_stats = [
        _stat("Median trade",
              money_compact(s.median_principal) if s.median_principal is not None else "—"),
        _stat("Concentration", _hhi_text(s.concentration_hhi), sub="HHI · 1/N…1"),
        _stat("Cadence",
              f"{cad.trades_per_month:.1f}/mo" if cad.trades_per_month is not None else "—",
              sub=None if cad.trades_per_month is not None else "insufficient history"),
    ]
    pos_open = c.positions_with_derivable_open_fraction
    cov_stats = [
        _stat("History", f"{c.n_trades} trades · {_window_text(c.window_days)}"),
        _stat("Round-tripped", pct(c.paired_fraction, 0), sub="open + close in window"),
        _stat("Entry in window", pct(pos_open, 0) if pos_open is not None else "—",
              sub="held positions with an opening trade"),
    ]
    return html.Div(className="dd-panel", children=[
        html.H3("Sizing & coverage", className="dd-panel-title"),
        html.Div("Trade sizing, name concentration, and how much history backs this read.",
                 className="dd-panel-subtitle"),
        html.Div(className="dd-stat-row", children=size_stats),
        html.Div(className="dd-stat-row", children=cov_stats),
    ])


def _fragile_panel() -> html.Div:
    """Deferred holds/rolls — honestly stubbed; a later pass fills it in."""
    return html.Div(className="dd-panel", children=[
        html.Details(open=False, children=[
            html.Summary("Holding period & rolls", className="dd-panel-title"),
            html.Div("Holding-period, roll, and realised-income views need open↔close "
                     "trade pairing — derivable for only part of the current book today — "
                     "so they are deferred to a later pass.", className="dd-panel-note"),
        ]),
    ])


def render_trade_insights_section(account_state) -> html.Div:
    """The #5 deep-dive section — a pure read of ``acc.client_profile``."""
    profile = getattr(account_state, "client_profile", None)
    head = html.Div(className="dd-section-head", children=[
        html.H2("Client profile", className="dd-section-title"),
        html.Span("trade-history behaviour · pre-call read", className="dd-section-meta"),
    ])
    if profile is None:
        return html.Div(className="dd-section", children=[
            head,
            html.Div("Trade-history profile unavailable for this account.", className="dd-empty"),
        ])
    grid = html.Div(className="dd-analytics-grid", children=[
        _fingerprint_panel(profile),
        _strategy_panel(profile),
        _tenor_direction_panel(profile),
        _lean_panel(profile),
        _sizing_coverage_panel(profile),
    ])
    return html.Div(className="dd-section", children=[head, grid, _fragile_panel()])
