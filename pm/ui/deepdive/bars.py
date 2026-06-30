"""Shared CSS bar primitive for the deep-dive sections — one source for the
``dd-bar`` look used by the Analytics sector / concentration panels and the
client-profile cards. No charting dependency (constraint).

The exposure default sign-colours the value (green/red for +/-) and tints a
negative fill red. The behavioural client profile renders magnitudes that carry
no gain/loss meaning, so it passes ``sign_color=False`` for plain neutral bars —
colour stays reserved for the few places a genuine direction is read.
"""
from __future__ import annotations

from typing import Optional

from dash import html

from pm.ui.deepdive.formatters import pct


def bar_row(label: str, value: Optional[float], max_w: float,
            sign_color: bool = True, value_text: Optional[str] = None) -> html.Div:
    """One horizontal bar; width is |value| / max_w (magnitude). With
    ``sign_color`` a negative value tints the fill and value red and a positive
    greens the value; without it the bar stays neutral charcoal. ``value_text``
    overrides the default percent label."""
    neg = sign_color and value is not None and value < 0
    pos = sign_color and value is not None and value > 0
    width = (abs(value or 0) / max_w * 100) if max_w else 0
    fill_cls = "dd-bar-fill dd-bar-fill-neg" if neg else "dd-bar-fill"
    val_cls = "dd-bar-val"
    if neg:
        val_cls += " exposure-neg"
    elif pos:
        val_cls += " exposure-pos"
    return html.Div(className="dd-bar-row", children=[
        html.Span(label, className="dd-bar-label"),
        html.Div(className="dd-bar-track", children=[
            html.Div(className=fill_cls, style={"width": f"{width:.1f}%"}),
        ]),
        html.Span(value_text if value_text is not None else pct(value), className=val_cls),
    ])
