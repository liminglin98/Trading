# Monthly short straddle, strike anchored to quarter-start ATM — REJECTED

**Script:** `monthly_quarter_anchored_straddle_backtest.py` · **Verdict: REJECTED** (2026-09-11)

## Motivation

`quarterly_atm_short_straddle.py --roll monthly` sells a fresh ATM straddle on the front-month
contract every ~30 days, re-centering the strike on spot at every single roll (~12x/year). The
question tested here: does decoupling the roll cadence (monthly, so each leg only carries
~20-45 DTE of gamma/theta risk, unlike the quarterly baseline's ~90 DTE) from the
strike-selection cadence (only re-center once per calendar quarter, reusing that strike for the
other ~2 monthly rolls in the quarter) capture monthly's faster theta without paying for a fresh
ATM pick on every roll?

## Method

`monthly_quarter_anchored_straddle_backtest.py`, both underlyings (510050: 2015-02-09 to
2026-09-10, 510300: 2019-12-23 to 2026-09-10). Same engine conventions as
`quarterly_atm_short_straddle.py`: fills at daily close, no fees/slippage, cash-settle at
intrinsic value on the front-month's own expiry, immediately roll into the next front-month
contract. The only change: the strike is only re-picked as ATM-to-spot on the first monthly roll
of each calendar quarter (`_quarter_key`); every other roll inside that quarter reuses the
quarter's anchor strike, clamped to the nearest strike actually listed for that month's expiry
(reusing `quarterly_atm_short_straddle._atm_strike`/`_leg_row`/`_intrinsic`, unmodified).
Discovery/holdout split at 2024-01-01, matching the rest of the family.

Compared per-trade (`mean_r` = pnl_yuan / margin_yuan) against two baselines run over the same
window: `quarterly_atm_short_straddle.py --roll monthly` (plain monthly, re-centered every roll)
and the default `--roll quarterly` (the validated, live baseline).

## Result

| underlying | config | disc mean_r | hold mean_r | total_pnl |
|---|---|---|---|---|
| 510050 | quarter-anchored monthly (this) | -0.37% | +2.73% | 6,230 |
| 510050 | monthly, re-centered every roll | -0.11% | +4.10% | 12,242 |
| 510050 | quarterly baseline (validated) | +3.71% | +11.31% | 20,717 |
| 510300 | quarter-anchored monthly (this) | -1.25% | +4.64% | 14,022 |
| 510300 | monthly, re-centered every roll | +0.87% | +3.01% | 18,917 |
| 510300 | quarterly baseline (validated) | +3.47% | +4.64% | 12,902 |

Quarter-anchoring underperforms plain monthly re-centering on **total P&L on both
underlyings**, and on **discovery-window mean_r on both underlyings** (only 510300's holdout
mean_r edges out monthly re-centering, and not by enough to close the total-P&L gap). Both
monthly variants fall well short of the quarterly baseline's per-trade return.

The by-quarter-leg breakdown explains why:

| underlying | leg | n | avg_credit | avg_pnl | avg \|moneyness\| | win% |
|---|---|---|---|---|---|---|
| 510050 | 0 (fresh ATM) | 47 | 1,242 | 210 | 0.5% | 68% |
| 510050 | 1 (reused strike) | 46 | 1,606 | 125 | 3.9% | 61% |
| 510050 | 2 (reused strike) | 45 | 1,712 | -208 | 4.6% | 51% |
| 510300 | 0 (fresh ATM) | 28 | 790 | 790 | 0.5% | 82% |
| 510300 | 1 (reused strike) | 26 | 1,942 | -13 | 2.4% | 58% |
| 510300 | 2 (reused strike) | 26 | 2,690 | -304 | 5.0% | 58% |

Leg 0 (the fresh ATM pick, functionally identical to plain monthly re-centering) performs best
on both underlyings. Legs 1 and 2 collect more nominal credit (moneyness has drifted, so the
frozen strike is often no longer the cheapest ATM pick) but their win rate and average P&L
degrade monotonically as moneyness drifts further from the quarter's anchor (0.5% -> 3.9% ->
4.6% on 510050; 0.5% -> 2.4% -> 5.0% on 510300) — the position quietly turns into a directional
bet on spot reverting toward the stale anchor strike, and that bet loses more often than not.
Unlike quarterly re-striking-within-expiry (`quarterly_restrike_short_straddle.py`, also
REJECTED), this isn't even a transaction-cost story: each monthly roll already pays away the
old legs and opens new ones regardless of whether the strike changes, so freezing the strike
saves nothing — it just forgoes the re-centering that made leg 0 the best-performing leg in the
first place.

## Verdict

**REJECTED.** Anchoring the monthly short straddle's strike to quarter-start ATM does not beat
simply re-centering ATM on every monthly roll — it loses on total P&L on both underlyings and on
discovery-window per-trade return on both underlyings, driven by moneyness drift degrading the
2nd and 3rd leg of each quarter. Plain monthly re-centering (`quarterly_atm_short_straddle.py
--roll monthly`) remains the better monthly-cadence variant, and both monthly variants still
fall short of the validated quarterly baseline's per-trade return. No chart generated per the
skip-charts-for-rejected-configs convention. Not wired into `alerts_core.py`.
