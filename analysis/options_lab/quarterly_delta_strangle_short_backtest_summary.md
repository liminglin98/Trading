# Quarterly short strangle with delta-targeted legs — PRELIMINARY / MIXED

**Script:** `quarterly_delta_strangle_short_backtest.py` · **Status: preliminary** (2026-08-31)

## Motivation

`quarterly_atm_short_straddle.py` selects both legs by ATM moneyness (nearest strike to
spot, same strike). `delta_strangle_short_backtest.py` selects legs by target Delta
independently per side, but only under a VRP-gated, opportunistic entry — not the always-on
quarterly cadence. This script isolates the pure strike-selection lever: same always-on,
no-entry-filter, hold-to-expiry, quarterly-roll cadence as the ATM baseline
(`quarterly_atm_short_straddle._target_expiries`/`_intrinsic`, unmodified), with strikes
picked via `delta_strangle_short_backtest._delta_leg` instead of ATM.

## Method

Swept `target_delta` in {0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50} on both underlyings with
reliable exchange Greeks (510050, 510300), full sample plus discovery (<2024-01-01) /
holdout (>=2024-01-01) split, then a matched-capital comparison: scaling each delta
config's size so its average margin equals the ATM straddle's (valid under linear
scaling — margin and mark-to-market P&L both scale with contract count; Sharpe and
return-on-margin are scale-invariant by construction, so matching capital only rescales
total P&L and max drawdown into directly comparable terms).

## Result

Raw (unscaled) comparison, full sample:

| underlying | config | total_pnl | sharpe | max_dd | avg_margin |
|---|---|---|---|---|---|
| 510050 | ATM straddle | 20,717 | 0.47 | -7,772 | 8,623 |
| 510050 | delta 0.40 | 17,451 | 0.41 | -8,460 | 6,866 |
| 510300 | ATM straddle | 12,902 | 0.35 | -8,766 | 12,962 |
| 510300 | delta 0.40 | 16,818 | 0.46 | -8,082 | 10,315 |

Matched to the ATM straddle's average margin (delta configs levered ~1.3-2.1x to consume
the same capital):

| underlying | config (leverage) | total_pnl (scaled) | sharpe | max_dd (scaled) |
|---|---|---|---|---|
| 510050 | ATM straddle | 20,717 | 0.47 | -7,772 |
| 510050 | delta 0.40 (x1.26) | 21,916 | 0.41 | -10,625 |
| 510050 | delta 0.10 (x2.12) | 21,466 | 0.31 | -15,114 |
| 510300 | ATM straddle | 12,902 | 0.35 | -8,766 |
| 510300 | delta 0.40 (x1.26) | 21,134 | 0.46 | -10,156 |
| 510300 | delta 0.10 (x2.15) | 13,710 | 0.30 | -9,915 |

The two underlyings disagree: on **510050**, the ATM straddle wins outright once the
delta strangle is levered to match capital — better Sharpe, better (smaller) drawdown,
similar total P&L. On **510300**, the delta=0.40 strangle wins on both total P&L (+64%)
and Sharpe (0.46 vs 0.35). Since Sharpe is scale-invariant, this isn't "matching capital
created an edge" — it's that delta=0.40 already had the better Sharpe on 510300 before any
scaling; leverage just makes that pre-existing gap visible in absolute P&L/drawdown terms.
Going further OTM (delta 0.10-0.20) is worse than delta 0.40 on both underlyings at matched
capital: total P&L is flat-to-similar vs. ATM but drawdown roughly doubles.

## Verdict

**Preliminary — not validated, not rejected.** No consistent cross-underlying edge from
delta-targeting the legs under this always-on quarterly cadence: 510050 favors staying at
ATM, 510300 favors a delta~0.40 strangle, and each underlying only offers ~26-46
independent quarterly draws (thin sample — annualized Sharpe estimates at this cadence are
noisy). Not live in `alerts_core.py`. Before either adopting a delta~0.40 strangle or
shelving the idea, this would need either a longer/wider sample (e.g. a third underlying
with reliable Greeks, once available) or a specific hypothesis for *why* 510300 and 510050
would differ here, rather than treating the disagreement as noise to average over.
