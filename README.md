# Portfolio Manager

A single-page Dash application for institutional portfolio review. Ingests
a client holdings spreadsheet, fetches live Bloomberg data, computes
per-underlying signals plus portfolio diagnostics, and produces grouped
position-level recommendations themed for client conversations.

## Features

- **Portfolio composition view** — sector concentration donut, style mix,
  earnings calendar with implied move and average historical move
- **Live greeks aggregation** — net dollar delta, vega, theta, gamma with
  NAV-percentage context
- **11 per-underlying signals** across vol regime (IV percentile, term
  structure, vol risk premium), trend (200D, momentum, YTD), event pressure
  (earnings, RSI extremes), and price action (move vs IV, breakout)
- **Composite score per name** — 5-component weighted score modeled on
  institutional desk conventions
- **Themed action items** — recommendations grouped into Yield Enhancement,
  Risk Mitigation, Dead-Weight Purge, and Tactical Opportunity buckets via
  19 deterministic rules
- **Per-position drill-down** — click any row to see raw data, signal
  derivation math, and the full rationale

## Stack

- Python 3.12+
- Dash + dash-mantine-components (UI)
- polars-bloomberg (BLPAPI integration)
- pandas, numpy, scipy (math)
- Plotly (visualization)

## Setup

Requires a Bloomberg Terminal with BLPAPI available on localhost:8194.

```bash
conda create -n portfolio-manager python=3.12 -y
conda activate portfolio-manager
pip install -r requirements.txt
```

Place your holdings file at `tim/data/Holdings.xlsx` (gitignored — never
committed). The expected sheet structure is documented in
`tim/core/holdings_parser.py`.

## Run

```bash
launch.bat
```

The app boots on `http://127.0.0.1:8052/`.

## Architecture

```
tim/
├── app.py                       Dash entry point, layout assembly
├── core/
│   ├── holdings_parser.py       XLSX → typed portfolio dataclass
│   ├── bloomberg_client.py      polars-bloomberg session + BDP/BDH
│   ├── portfolio_snapshot.py    Orchestrates underlying + option fetches
│   ├── portfolio_greeks.py      Net dollar greeks aggregation
│   ├── portfolio_signals.py     11 signals across 5 categories
│   ├── portfolio_diagnostics.py Sector / style / beta / earnings
│   ├── recommender.py           19 deterministic recommendation rules
│   ├── pitch_synthesizer.py     Theme-grouped action items
│   ├── composite_score.py       5-component weighted score per underlying
│   ├── probability.py           Black-Scholes greeks
│   └── vol_metrics.py           IV percentile, RV, VRP from BDH series
└── assets/
    └── style.css                Single stylesheet
```
