"""Smoke entrypoint. Parses the sample portfolio, prints summary, exits."""
import sys

# Signal detail strings include Unicode (e.g. \u03c3 for sigma). On Windows
# the default console codec is cp1252 and would crash on print. Reconfigure
# stdout to UTF-8 so the smoke output is faithful.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

from tim.config import DEFAULT_HOLDINGS_FILE
from tim.core.holdings_parser import get_unique_underlyings, parse_holdings


def main() -> None:
    portfolio = parse_holdings(DEFAULT_HOLDINGS_FILE)

    print("=" * 72)
    print("Portfolio Summary")
    print("=" * 72)
    print(f"  Total MV:           ${portfolio.portfolio_total['total_market_value']:>16,.2f}")
    tc = portfolio.portfolio_total.get("total_cost") or 0
    tu = portfolio.portfolio_total.get("total_unrealized") or 0
    print(f"  Total Cost:         ${tc:>16,.2f}")
    print(f"  Total Unrealized:   ${tu:>16,.2f}")
    print()
    print(f"  Equity positions:   {len(portfolio.equity_positions):>4}")
    print(f"  Option positions:   {len(portfolio.option_positions):>4}")
    print(f"  Other positions:    {len(portfolio.other_positions):>4}")
    print(f"  Unique underlyings: {len(get_unique_underlyings(portfolio)):>4}")

    print()
    print("Region breakdown:")
    eq = portfolio.equity_positions
    op = portfolio.option_positions
    print(f"  US equities:        {(eq['region']=='us').sum():>4}")
    print(f"  Non-US equities:    {(eq['region']=='non_us').sum():>4}")
    print(f"  US options:         {(op['region']=='us').sum():>4}")
    print(f"  Non-US options:     {(op['region']=='non_us').sum():>4}")

    print()
    print("Multiplier sanity (options, where price and qty present):")
    mask = op["price"].notna() & (op["quantity"] != 0)
    sub = op[mask]
    if len(sub):
        recomputed = sub["quantity"] * sub["price"] * sub["multiplier"]
        diff = (recomputed - sub["market_value"]).abs()
        print(f"  Max |recomputed - reported| MV: ${diff.max():.2f}")
    else:
        print("  No priced live options to check.")

    if portfolio.parse_warnings:
        print()
        print(f"Warnings ({len(portfolio.parse_warnings)}):")
        for w in portfolio.parse_warnings:
            print(f"  - {w}")
    else:
        print()
        print("Warnings: none")

    print()
    print("=" * 72)
    print("Equity positions (head)")
    print("=" * 72)
    print(portfolio.equity_positions[
        ["symbol", "bbg_ticker", "region", "quantity", "price", "market_value"]
    ].to_string(index=False))

    print()
    print("=" * 72)
    print("Option positions (head)")
    print("=" * 72)
    print(portfolio.option_positions[
        ["bbg_ticker", "underlying_symbol", "right", "strike", "expiry",
         "quantity", "price", "market_value"]
    ].to_string(index=False))

    print()
    print("=" * 72)
    print("Live underlying snapshot")
    print("=" * 72)
    from tim.core.bloomberg_client import is_bloomberg_available
    from tim.core.portfolio_snapshot import fetch_portfolio_snapshot

    bbg_ok = is_bloomberg_available()
    print(f"  Bloomberg available: {bbg_ok}")

    snap = fetch_portfolio_snapshot(portfolio, bbg_ok)
    if snap.fetch_warnings:
        for w in snap.fetch_warnings:
            print(f"  WARN: {w}")

    if not snap.underlyings.empty:
        cols_preview = [
            c for c in [
                "security_name", "PX_LAST", "CHG_PCT_1D",
                "3MTH_IMPVOL_100.0%MNY_DF", "GICS_SECTOR_NAME",
            ]
            if c in snap.underlyings.columns
        ]
        print()
        print(snap.underlyings[cols_preview].head(20).to_string())

    print()
    print("=" * 72)
    print("Portfolio Greeks")
    print("=" * 72)
    from tim.core.portfolio_greeks import compute_portfolio_greeks
    greeks = compute_portfolio_greeks(portfolio, snap.underlyings, snap.options)

    t = greeks.totals
    print(f"  Net $ Delta:   ${t['dollar_delta']:>16,.0f}    "
          f"({t['delta_pct_of_nav']*100:+.1f}% of NAV)")
    print(f"  Net $ Vega:    ${t['dollar_vega']:>16,.0f}    (per +1 vol pt)")
    print(f"  Net $ Theta:   ${t['dollar_theta']:>16,.0f}    (per day)")
    print(f"  Net $ Gamma:   ${t['dollar_gamma']:>16,.0f}    "
          "($delta per $1 spot move)")
    print(f"  Long opts:     {t['net_long_options_count']:>4}")
    print(f"  Short opts:    {t['net_short_options_count']:>4}")
    if t.get("coverage_ratio_by_underlying"):
        print()
        print("  Coverage ratios (short calls vs long stock, per underlying):")
        for sym, cov in t["coverage_ratio_by_underlying"].items():
            print(f"    {sym:8s}  {cov*100:.0f}%")

    if greeks.warnings:
        print()
        print(f"  Warnings ({len(greeks.warnings)}):")
        for w in greeks.warnings[:10]:
            print(f"    - {w}")
        if len(greeks.warnings) > 10:
            print(f"    ... and {len(greeks.warnings)-10} more")

    print()
    print("=" * 72)
    print("Per-underlying signals")
    print("=" * 72)
    from tim.core.portfolio_signals import compute_per_underlying_signals

    signals_by_ticker = compute_per_underlying_signals(
        snap.underlyings, bloomberg_available=bbg_ok,
    )

    for ticker in sorted(signals_by_ticker.keys()):
        sigs = signals_by_ticker[ticker]
        if not sigs:
            print(f"  {ticker}:   (no signals - missing data)")
            continue
        print(f"  {ticker}:")
        for sig in sigs:
            arrow = (
                "\u2191" if sig.direction == "bullish"
                else "\u2193" if sig.direction == "bearish"
                else "\u00b7"
            )
            print(f"     {arrow}  {sig.signal_type:20s}  {sig.detail}")

    # -- Per-signal fire histogram (helps verify new prompt-5 signals) ----
    from collections import Counter
    hist: Counter = Counter()
    for sg in signals_by_ticker.values():
        for s in sg:
            hist[s.signal_type] += 1
    print()
    print(f"  Signal-fire histogram (across {len(signals_by_ticker)} underlyings):")
    for k, v in hist.most_common():
        print(f"    {k:25s}  {v}")

    print()
    print("=" * 72)
    print("Portfolio Diagnostics")
    print("=" * 72)
    from tim.core.portfolio_diagnostics import compute_portfolio_diagnostics

    diag = compute_portfolio_diagnostics(portfolio, snap.underlyings)

    beta = diag.weighted_beta
    print(f"  Weighted \u03b2: {beta:.2f}" if beta is not None
          else "  Weighted \u03b2: \u2014")
    print()
    print("  Sector exposure:")
    for sec, pct in sorted(diag.sector_exposure.items(), key=lambda x: -x[1]):
        print(f"    {sec:30s}  {pct*100:>5.1f}%")
    print()
    print("  Style mix:")
    for sty, pct in sorted(diag.style_mix.items(), key=lambda x: -x[1]):
        print(f"    {sty:30s}  {pct*100:>5.1f}%")
    print()
    print(f"  Earnings \u226430d ({len(diag.earnings_calendar)} names):")
    for e in diag.earnings_calendar[:10]:
        print(
            f"    {e['symbol']:8s}  {e['date']}  "
            f"({e['days_to_earnings']}d)  {e['name'][:40]}"
        )
    if diag.warnings:
        print()
        print(f"  Warnings ({len(diag.warnings)}):")
        for w in diag.warnings:
            print(f"    - {w}")

    print()
    print("=" * 72)
    print("Recommendations + Pitch Themes")
    print("=" * 72)
    from collections import Counter

    from tim.core.pitch_synthesizer import synthesize_pitch
    from tim.core.position_context import build_position_contexts
    from tim.core.recommender import compute_recommendations

    contexts = build_position_contexts(portfolio, snap, {})
    recs = compute_recommendations(contexts, signals_by_ticker)
    themes = synthesize_pitch(recs)

    actions = Counter(r.action for r in recs)
    priorities = Counter(r.priority for r in recs)
    print(f"  Total recommendations: {len(recs)}")
    print(f"  Actions:    {dict(actions.most_common())}")
    print(f"  Priorities: {dict(priorities.most_common())}")

    for theme in themes:
        print()
        print(f"  ==== {theme.theme_name}  ({theme.summary_metric}) ====")
        print(f"  {theme.headline}")
        for r in theme.recommendations:
            print(f"     \u2022 [{r.action}]  {r.position_id}")
            print(f"        {r.rationale}")


if __name__ == "__main__":
    main()
