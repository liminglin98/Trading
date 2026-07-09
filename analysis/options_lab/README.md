# Options Lab

Streamlit dashboard over `Data/options/option_daily_{510050,510300}.csv`
(full daily chains 2015 → present, with exchange Greeks/IV).

## Run

```bash
pip install -r ../../requirements.txt
cd analysis/options_lab
streamlit run app.py
```

## Tabs

1. **Data** — shows the last recorded chain date per underlying and a T-style
   snapshot. "Update now" (or `python updater.py` from a scheduler, after
   15:30 on trading days) fetches the EOD chain from sina, appends it to the
   CSV in the original schema, and caches ETF daily OHLC from tencent for
   the candlestick charts. Idempotent — re-running on the same date appends
   nothing. First live run: use `python updater.py --dry-run` and eyeball
   the rows (esp. 成交额 units) before writing.

2. **Positions** — enter any multi-leg combo (contract picker from the
   latest chain, direction, qty, fill price), then the daily P&L is
   reconstructed from the chain history automatically — no daily logging
   needed. Close at latest marks (or edit `positions.csv` for exact fills);
   closed positions freeze their realized P&L.

3. **Backtest** — template + rules engine: straddle / strangle / iron
   condor / butterflies / verticals / calendar / fully custom legs; entry by
   DTE window + ATM-IV percentile gate; strike selection by delta target or
   ATM offset; exits by profit target, stop loss, DTE, or expiry
   (settled at intrinsic). Output: candlestick chart with entry/exit
   markers, equity curve, drawdown, P&L histogram, trades table + CSV
   download. P&L in yuan per 1x combo (multiplier 10000).

## Data caveats

* New rows store close price in the 结算价 column (sina has no settlement
  feed) and computed Black-Scholes Rho — fine for marking/backtests.
* Adjusted contracts (names ending "A") are excluded from backtests by
  default.
* Fills are assumed at daily close; no fees/slippage are modeled — treat
  results as upper bounds, especially for high-turnover exits.

## Files

```
app.py          Streamlit UI (Data / Positions / Backtest)
data_store.py   schema constants, CSV load/append
updater.py      daily EOD fetch (sina options + tencent ETF OHLC), CLI
positions.py    combo store + mark-to-market + P&L history
backtest.py     template+rules engine
charts.py       plotly figures
positions.csv   created at first use
```
