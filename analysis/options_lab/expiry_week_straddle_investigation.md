# 到期周平值双买 (expiry-week ATM long straddle): Investigation Summary & Verdict

**Bottom line: buying an ATM straddle on the Monday / Wednesday / Friday of expiry week, on the 当月 contract, has no edge on any of IO, MO or HO — REJECTED.** Across 1,434 trades and 2.48M yuan of premium deployed the combined result is **−32,423 yuan (−1.31% of premium)** before any costs, and that headline is *flattered* by three lucky months in MO; strip those and it is −163,481 yuan (−6.6%). One tick of slippage per leg pushes almost every cell further negative. This is a theta-bleed structure, not a gamma opportunity.

Reproduce: `python expiry_week_atm_straddle_backtest.py` (add `--entry-basis tradingday` for the robustness basis, `--slippage 0.5` for a costed run). Charts: `python expiry_week_atm_straddle_chart.py`.

---

## 1. What this is testing

股指期权每月第三个周五到期. The proposition: on the 周一 / 周三 / 周五 of that expiry week, buy an at-the-money 双买 (long call + long put, same strike) on the contract expiring that same Friday.

It is a pure gamma-vs-theta question with no signal in it. Expiry-week ATM options carry the most gamma and bleed the most theta, and which side wins is empirical. It is also the null hypothesis every signal-gated long-straddle script in this lab (`iv_low_long_straddle_backtest.py`, `long_straddle_rv_contraction_backtest.py`) implicitly claims to beat, so it is worth having measured on the record.

Three exit rules were run side by side, per the request:

| rule | Mon leg | Wed leg | Fri leg |
|---|---|---|---|
| `to_expiry` | Fri delivery | Fri delivery | same-day |
| `next_obs` | Wed close | Fri close | same-day |
| `one_day` | Tue close | Thu close | same-day |

## 2. Data and methodology

`Data/买断日更/consolidated/{000300,000852,000016}.XSHG.parquet` via `consolidated_store.load_chain()`. IO 2019-12-23→2026-08-14 (84 expiries), MO 2022-07-22→ (54), HO 2022-12-19→ (49). No network fetch; `cffex_updater.py` keeps these current.

- **Spot** is `ds.C_UND_CLOSE`, the put-call-parity forward per `(date, expiry)` — the dataset stores no index level, and the ETF OHLC caches are 前复权-adjusted and therefore incomparable to raw strikes.
- **ATM** = strike minimising `|K − forward|` among strikes with both legs quoted and tradeable, ties broken on combined volume/OI.
- **Marks**: mid-life exits at `C_CLOSE`; expiry exits at the expiring contract's `C_SETTLE`, which is the cash delivery value.
- **Return basis** is the **debit** (premium paid). A long option position posts no margin, and `cffex_margin.py` carries verified `MARGIN_PARAMS` for MO only — IO/HO are an un-sourced TODO there, so a return-on-margin figure for those two would be fabricated.
- **Costs**: headline runs are zero-fee/zero-slippage per this lab's convention (`README.md`), with a dedicated sensitivity grid at 0 / 0.2 / 0.5 / 1.0 index points of adverse half-spread per leg (CFFEX tick is 0.2), charged on entry and close-based exits but not on cash delivery.
- **Discovery/holdout** at 2024-01-01 as usual, but note this script **fits nothing** — no grid search, no frozen pick — so the split is a stability read, not an out-of-sample test.

### Validation performed

- **Expiry calendar**: every `expire_date` asserted to be its month's third Friday. Exactly three exchange-shifted exceptions exist in the data (2024-02-19, 2026-02-24, 2026-06-22); a fourth raises rather than being silently backtested.
- **Settlement**: all DTE=0 `C_SETTLE` values compared against intrinsic recomputed from the DTE=0 parity forward. Median |diff| 0.000–0.005 index points, max 0.83–1.83 across the three — the expected small gap between the 交割结算价 (last two hours' index average) and a close-derived forward, not an error.
- **Two independent entry paths agree**: the `calendar` basis (Mon/Wed/Fri of the expiry week) and the `tradingday` basis (T-4/T-2/T-0 trading days before expiry) produce **0 mismatches** on all 1,416 trades where they resolve to the same date. `tradingday` additionally recovers the ~12 holiday-shortened expiry weeks `calendar` has to skip; conclusions are the same under both.
- **Manual spot-check**: three random trades (one per underlying) reconciled line by line against the raw parquet — strikes, closes, settles, P&L all tie out.

## 3. Results and the verdict on each

Calendar basis, zero slippage, mean return on debit. `ex-top3` = the same mean with the three best trades removed.

### 3.1 Every apparently-profitable cell is three months of luck — REJECTED

| underlying | entry | exit | n | total P&L | mean r | median r | **ex-top3** | disc r | hold r |
|---|---|---|---|---|---|---|---|---|---|
| MO | Wed | to_expiry | 45 | +65,478 | **+11.9%** | −21.2% | **−1.1%** | +13.9% | +10.6% |
| MO | Mon | to_expiry | 45 | +16,088 | **+4.8%** | +0.6% | **−4.0%** | +17.6% | −2.3% |
| HO | Wed | to_expiry | 40 | −1,219 | **+1.1%** | −8.8% | **−16.3%** | −3.0% | +2.8% |

Three cells out of eighteen show a positive mean. All three have a **deeply negative median** and all three **go negative once their three best trades are removed**. That is the textbook signature of a right-skewed lottery payoff, not an edge.

At the portfolio level it is starker. MO's whole +81,910 (Mon+Wed+Fri, hold to expiry) comes from three months — 2026-07 (+62,630), 2025-11 (+41,601), 2025-10 (+26,827), together **160% of the total**. Without them MO is **−49,148**.

| underlying | P&L | premium deployed | return | positive months |
|---|---|---|---|---|
| IO 沪深300 | −79,871 | 1,021,020 | −7.8% | 32% (77 mo) |
| MO 中证1000 | +81,910 | 1,122,840 | +7.3% | 49% (47 mo) |
| HO 上证50 | −34,462 | 333,600 | −10.3% | 31% (42 mo) |
| **combined** | **−32,423** | **2,477,460** | **−1.31%** | |

**Conclusion: no cell survives the concentration check. Do not re-propose this as a standalone strategy, and do not read MO's +7.3% as a small-cap gamma premium — it is 2025-10, 2025-11 and 2026-07.**

### 3.2 The bleed is monotonic in holding time — the structure is doing exactly what theta says

Mean / median return on debit by trading days held (Mon and Wed entries, all three underlyings pooled):

| days held | n | mean r | median r |
|---|---|---|---|
| 1 | 312 | −2.6% | −10.5% |
| 2 | 471 | −2.1% | −18.3% |
| 4 | 153 | −8.8% | −22.9% |

Every additional day into expiry costs money on average, and the median loses far more than the mean throughout — you pay a near-certain small toll to buy a rare large payout, and the payout does not cover the tolls. This is the whole result in one table.

### 3.3 The Friday 0DTE leg is a data artifact, not a trade — DISCARD

The Friday numbers look benign (IO +1.8% mean, 61% win rate, stable across discovery and holdout) and must not be believed. CFFEX settles index options at the arithmetic average of the underlying index over the **last two hours** of expiry day, and trading ends at 15:00 — so entering at the 15:00 close buys a payoff that is already fully determined.

Measured: `entry_date == exit_date` for all 160 of these, and the P&L is *identically* the same-day close-minus-settlement basis — mean −0.00 / +0.08 / −0.07 index points, SD 0.22–0.34, on debits of 10–23 points. Noise around zero with no market exposure in it. Part of it is pure stale quoting: the near-worthless leg's last traded tick is a median 0.20 points against a settlement value of 0.00. Accordingly it is also the cell most destroyed by slippage (IO +1.8% → −5.5% at a single tick).

**Conclusion: a genuine 0DTE Friday test needs intraday quotes** (`Data/options/minute/index_option_minute_*.csv` via `minute_updater.py`), **which only begin 2026-08-11.** Daily bars cannot answer it.

### 3.4 Costs finish off what is left — REJECTED

Mean return on debit vs. adverse half-spread per leg:

| underlying | entry | exit | 0.00 | 0.20 | 0.50 | 1.00 |
|---|---|---|---|---|---|---|
| MO | Wed | to_expiry | +11.9% | +11.4% | +10.7% | +9.5% |
| MO | Mon | to_expiry | +4.8% | +4.4% | +3.7% | +2.7% |
| HO | Wed | to_expiry | +1.1% | −0.3% | −2.3% | −5.4% |
| IO | Fri | to_expiry | +1.8% | −5.5% | −12.5% | −20.7% |
| IO | Mon | to_expiry | −10.7% | −11.3% | −12.1% | −13.4% |
| HO | Mon | to_expiry | −19.2% | −20.2% | −21.6% | −23.8% |

Only MO's two hold-to-expiry cells stay positive under cost — and §3.1 already showed both are negative ex-top3, so cost-resilience here is just the same three months surviving. Short-dated cells decay fastest because the debit is smallest relative to a fixed per-leg spread.

## 4. What is established vs. still open

**Established.** On daily EOD data, 2019-12→2026-08, expiry-week ATM long straddles on all three CFFEX index options lose money in aggregate; the loss scales with holding time; every positive cell is three trades wide; and the Friday 0DTE cell is unmeasurable from daily bars.

**Open.**
- **Intraday 0DTE.** The only genuinely untested version of the question. Needs the minute data to accumulate.
- **The mirror trade.** These are the *buyer's* results; the systematic bleed is a seller's income. That is a different backtest with genuine tail risk and margin requirements (and IO/HO margin parameters are still un-sourced in `cffex_margin.py`), so it is not implied by anything here.
- **Conditional entry.** This tested the unconditional calendar trade. Whether an IV-percentile or RV-contraction gate rescues expiry-week timing is exactly what `iv_low_long_straddle_backtest.py` / `long_straddle_rv_contraction_backtest.py` do on ETF options; extending them to IO/MO/HO would be the natural follow-on, and now has a null baseline to beat.

## 5. Recommendation

Do not trade it. Use it as the null baseline for any future long-premium work on CFFEX index options: a candidate strategy that buys expiry-week gamma has to beat −1.31% on premium (−6.6% excluding MO's three outlier months), not zero.

## Scripts

- `expiry_week_atm_straddle_backtest.py` — the backtest (REJECTED: no edge on any of IO/MO/HO; every positive cell fails the ex-top3 check).
- `expiry_week_atm_straddle_chart.py` — 3-panel interactive charts for the concluding config (Mon+Wed+Fri, hold to expiry), one per underlying.
