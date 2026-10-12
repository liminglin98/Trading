# Spreading the 25-lot quarterly straddle entry from T90 to T75 vs 5 lots x first 5 days of T90 — NO IMPROVEMENT

Script: `quarterly_entry_ladder_window_backtest.py`. Outputs: `out/quarterly_entry_ladder_window_<u>{,_bydte}.csv`.
Run 2026-10-12. 510050 2015-02+ (43 full cycles, 47 incl. partial), 510300 2019-12+ (25 full cycles, 27 incl. partial).

## Method
One entry-day panel: for every quarterly expiry and every eligible trading day, a 1-lot ATM short straddle (re-struck daily at
nearest raw spot, as the live book does) is carried to the same expiry with daily marks, intrinsic settlement, and broker
combo margin per day. Each arm is a lot-weight vector over those tranches, 25 lots per cycle, held to the same expiry:
L90 / L75 lump, B_settle (5/day x first 5 days after settlement, the existing ladder), B_T90 (5/day x 5 days from the first
DTE <= 90), A (25 lots over every day with 75 <= DTE <= 90, integer, evenly spread), A_frac. Measured half-spread tiers by
entry DTE, no exit fill. Reproduction: single-lot L90 / L75 totals match the entry-DTE sweep exactly on both ETFs
(510050 21,373 / 30,899; 510300 12,099 / 18,347; also at 0.005 slippage).

## Pre-registered reading (docstring, before results): A vs B_T90 at med and p90, on both ETFs
P&L edge: mean paired per-cycle P&L > 0, bootstrap CI low > 0, discovery and holdout both > 0. Capital edge: return per
margin-day higher, Sharpe not lower, higher in both discovery and holdout. Either with max DD not worse.

## Result: neither edge on either ETF (tier med; p90 within ~1%)
| | 510050 B_T90 | 510050 A | 510300 B_T90 | 510300 A |
|---|---|---|---|---|
| total P&L, 25 lots (yuan) | 586,495 | 607,886 | 382,970 | 358,495 |
| avg entry DTE | 86.0 | 82.0 | 86.1 | 82.1 |
| Sharpe (daily) | 0.560 | 0.618 | 0.458 | 0.464 |
| return per margin-day (ann) | 0.377 | 0.417 | 0.283 | 0.281 |
| ... discovery / holdout | 0.256 / 0.824 | 0.310 / 0.803 | 0.179 / 0.445 | 0.175 / 0.448 |
| max DD (yuan) | -160,900 | -156,555 | -177,000 | -177,000 |
| ann return at 15% DD cap | 4.7% | 5.0% | 4.8% | 4.5% |

Paired per-cycle A - B_T90: 510050 +497 yuan (t 0.36, CI low -2,332, better in 22/43, disc +1,275, holdout -2,069);
510300 -979 (t -0.44, CI low -4,875, better in 11/25, disc -739, holdout -1,339). Versus B_settle: +393 (t 0.27) / -2,529 (t -1.04).

* P&L edge: none (|t| < 0.5, CI includes zero, discovery/holdout signs differ on 510050 and are both negative on 510300).
* Capital edge: on 510050 A is ahead overall (return per margin-day +0.04, Sharpe +0.06, DD 3% shallower) because its average
  entry is 4 days later so margin ramps later, but it loses the holdout leg (0.803 vs 0.824), so the rule fails; on 510300 there
  is no difference at all (0.281 vs 0.283).
* The 510300 max DD is identical in every arm because all arms sold the same dividend-adjusted off-grid strike (4.923) in the
  2022 Q1 cycle that produces it; the 510050 DDs differ by arm, so DD is not a bug.
* Context: lump T75 has the highest 510050 total (648,550) but the weakest holdout (207,475 vs 252,100 for L90) and is below
  L90 on 510300 (294,950 vs 321,125), so T75's edge is not stable across the two ETFs. Both 5-day ladders beat their own lump
  entry on total P&L on both ETFs (B_T90 vs L90: +84k / +62k), consistent with the earlier ladder finding (not re-tested here).
* Entry-day diagnostic (mean single-lot P&L by entry-DTE bucket): noisy and non-monotone (510050: 584 at DTE 70-74, 496 at
  75-79, 460 at 80-84, 670 at 85-89, 334 at 90-94; 510300: 321, 315, 322, 656, 242), no smooth decay with DTE. The entry day
  inside 75-90 is mostly noise.

## Verdict
No improvement. Spreading 25 lots from T90 to T75 is a different entry schedule with later average entry (~82 DTE), not a
better one: it does not beat 5 lots x 5 days at T90 on P&L, and its capital-efficiency gain is one ETF, one sub-period.
Choose between schedules on operational grounds (fewer, larger daily orders vs. a longer build); there is no backtest
evidence for the continuous ladder. Not live; no chart.
Caveats: the first partial cycle (entry DTE < 88) is excluded from arm comparisons; chain loader is the sweep's (no
(date, expiry, strike) dedupe) for exact reproduction; 6 arms tried, one pre-registered comparison.
