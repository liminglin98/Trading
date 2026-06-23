# Butterfly Forecast — Factor & Data Availability Checklist

**Purpose.** The macro-only model (`butterfly_macro_forecast.py`) had **no directional
skill** (out-of-sample direction hit-rate 37.5%, worse than a coin flip) but a
**well-calibrated dispersion band** (50%/80% interval coverage of 75%/88%). The
takeaway: surveyed real-economy macro (CPI, IP, GDP) is the *wrong* input for
monthly index *direction*. The factors that actually forecast equity indices are
mostly **market / financial / flow** data, available in real time.

This checklist is for going through QMT (and other sources you have) to see which
of the canonical index-forecasting factors you can actually build for the
**50ETF / 300ETF** next-month model. Mark each ✅ available / ⚠️ partial / ❌ not
available, and note the field name + frequency + history start.

> Target reminder: predict next-month **return** (→ center strike via
> `spot × exp(return)`), and use the model's predictive **band** to size wings.
> Everything below must be **known at forecast time** (real-time, no look-ahead).

---

## 1. Momentum / Trend  *(highest priority — robust, price-only, free)*

The most reliable short-to-medium-horizon signal that survives out-of-sample.
All derivable from the index/ETF price series you almost certainly already have.

| Factor | Definition | QMT field / source | Avail | Notes |
|---|---|---|---|---|
| 3-month trailing return | `P_t / P_{t-63} - 1` | | | price only |
| 6-month trailing return | `P_t / P_{t-126} - 1` | | | price only |
| 12-month trailing return | `P_t / P_{t-252} - 1` | | | classic TS-momentum |
| Price vs 200-day MA | `P_t / MA200 - 1` | | | trend filter |
| Price vs 50-day MA | `P_t / MA50 - 1` | | | faster trend |
| Realized vol (20d, 60d) | rolling std of daily returns | | | also a regime/wing input |

**Question for QMT:** do you have clean daily OHLC history for **50ETF, 300ETF,
and the underlying SSE50 / CSI300 indices** back to ~2015 or earlier? (The ETF
file we have starts 2020 — index history would extend the sample a lot.)

---

## 2. Valuation / Mean-Reversion  *(medium horizon, 6–12m)*

Works better at longer horizons than 1 month, but useful as a slow tilt /
regime context. China-relevant versions:

| Factor | Definition | QMT field / source | Avail | Notes |
|---|---|---|---|---|
| Index P/E (TTM) percentile | P/E vs own 5–10y history | | | CSI300 / SSE50 index P/E |
| Index P/B percentile | P/B vs own history | | | |
| Earnings yield − bond yield | `1/PE − 10Y CGB yield` | | | the "equity risk premium" / Fed-model spread |
| Dividend yield | index dividend yield | | | |

**Question for QMT:** does it expose **index-level P/E, P/B, dividend yield**
time series (not just per-stock)? And the **10-year China Government Bond
yield**?

---

## 3. Liquidity / Monetary  *(China-specific — A-shares are flow-driven)*

This is where A-share prediction is most distinctive. These are the sharp,
market-relevant cousins of the slow M2 figure we used.

| Factor | Definition | QMT field / source | Avail | Notes |
|---|---|---|---|---|
| Margin financing balance | 融资融券余额, level & MoM growth | | | strongest A-share liquidity gauge |
| Northbound flows | Stock Connect net buy (沪股通/深股通) | | | ⚠️ disclosure curtailed since Aug-2024 |
| Aggregate turnover / volume | market or index turnover, vs trailing avg | | | |
| New investor accounts | monthly account openings | | | sentiment/retail flow |
| Mutual fund issuance | new equity-fund launches | | | |
| Credit impulse | Δ in social financing (社融) growth | | | sharper than M2; we only have M2 |
| Repo / SHIBOR levels | FR007 / DR007 / 7d SHIBOR | | | we have FR007 already ✅ |

**Question for QMT:** which of margin balance, Northbound flow, turnover, and
social-financing (社融) does it provide, and at what frequency/history?

---

## 4. Sentiment / Positioning  *(contrarian at extremes)*

| Factor | Definition | QMT field / source | Avail | Notes |
|---|---|---|---|---|
| Implied vol level | QVIX level & percentile | | | we have QVIX ✅ (50/300) |
| Put/call ratio | option volume or OI ratio | | | from option data |
| IV term structure slope | near vs next-term IV | | | needs option chain |
| IPO / issuance pace | new-share supply | | | heavy supply caps index |

**Question for QMT:** does the option export include **volume & open interest
per contract** (for put/call ratio), and a clean **near/next-term IV**? (The
`option_daily.csv` we have looked unreliable — confirm what's actually trustworthy.)

---

## 5. Cross-Asset / Global  *(risk appetite, EM flows)*

| Factor | Definition | QMT field / source | Avail | Notes |
|---|---|---|---|---|
| CNY exchange rate | USDCNY level & MoM change | | | we have CNYUSDSpot ✅ |
| US 10Y Treasury yield | level & change | | | global risk-free / EM flows |
| US equity / VIX | S&P or VIX level | | | global risk proxy |
| Commodity index | CRB / oil / copper | | | China commodity sensitivity |
| Gold | USD gold | | | risk-off hedge |

---

## What we already have in the folder (no QMT needed)

- ✅ **QVIX** (50ETF, 300ETF + index variants), daily OHLC, 2015→present — sentiment + realized/implied vol factors.
- ✅ **50ETF daily OHLC** (2020→present) — momentum/trend factors, but short history.
- ✅ **FR007** daily repo rate (2000→present) — rates factor.
- ✅ **CNYUSD spot + NEER** monthly — FX factor.
- ⚠️ **Macro panel** (CPI, IP, GDP, M2, FAI, trade) monthly to ~2025 — keep only as *regime/wing context*, not direction.
- ❌ Margin balance, Northbound flows, social financing, index P/E, 10Y CGB yield, put/call — **not present, source from QMT.**

---

## Recommended first model (once data is confirmed)

Drop the real-economy macro for *direction*; build the return forecast on:

1. **Trailing momentum** (3/6/12-month) — price only, highest priority.
2. **QVIX level** (sentiment) — already have it.
3. **Margin-balance growth** + **turnover** (liquidity) — if QMT provides.
4. **Earnings-yield − 10Y CGB spread** (valuation) — if QMT provides index P/E + bond yield.
5. **USDCNY change** + **US 10Y** (cross-asset) — partly have.

Keep the same **Bayesian walk-forward** machinery; just swap the predictor panel.
The dispersion band (wing-sizing) already works and carries over unchanged.

> **Caveat:** none of this is investment advice. Every one of these factors has
> regimes where it fails (momentum crashes, valuation stays cheap for years).
> The honest prior from the macro test is that monthly *direction* is hard;
> the model's real value may remain in **wing-sizing**, not the directional tilt.

---

## Notes to fill in from QMT (scratch)

- Index OHLC history start: ______
- Margin balance field / freq: ______
- Northbound flow field / freq: ______
- Index P/E, P/B, div yield available? ______
- 10Y CGB yield field: ______
- Social financing (社融) available? ______
- Option volume/OI per contract reliable? ______
