# 50ETF Butterfly Dashboard

Streamlit dashboard for the 50ETF butterfly wing-sizing / iron-condor model.
Analytics and suggestions only — it never places or moves trades.

## Run locally

```bash
pip install -r requirements.txt
cd analysis/macro_model
streamlit run app.py
```

First load fits the model and pulls the SSE option IV chain (a few minutes, needs
internet). Results are cached until you hit a refresh button.

## Deploy on Streamlit Community Cloud

- **Repository:** this repo
- **Branch:** `main`
- **Main file path:** `analysis/macro_model/app.py`

The app reads `Data/akshare/panel_monthly.csv` and the cached model outputs in
`analysis/macro_model/out/`, both of which are committed so the app can render
without a cold rebuild.

> **Note on live data:** the live panels fetch from Chinese market feeds
> (sina, SSE). US-hosted clouds may not be able to reach these, so the live
> spec / tracker may fall back to the cached month or error there even though
> the app deploys fine. The historical/backtest views work from the committed
> caches regardless.

## Layout

```
analysis/macro_model/
  app.py                       # Streamlit entry point
  pipeline.py                  # freshness + refresh runner
  tracking.py                  # position store + valuation
  butterfly_*.py, iron_*.py    # model / backtest / optimizer modules
  out/                         # cached model outputs (committed)
Data/akshare/panel_monthly.csv # factor panel read at startup
```
