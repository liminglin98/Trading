# Monthly 3% short-strangle sleeve rolled at 7 DTE, alone and added to the quarterly straddle book — NOT VALIDATED

Script: `strangle_sleeve_backtest.py` (engine reused from `rolling_itm_call_short_strangle_backtest.py`, which gained
backward-compatible `roll_dte` / `with_long` parameters). Outputs: `out/strangle_sleeve_<u>_<start>{,_combined,_corr}.csv`.
Run 2026-10-12. Windows: 510050 2019-12-23+ (primary common window), 510050 2015-02-09+, 510300 2019-12-23+.

## Structure
Short call and put at the listed strike nearest spot x (1 +/- 3%), strictly OTM, nearest expiry DTE >= 25 (~monthly),
closed at DTE <= 7 and re-entered the same day. No long leg, no signal. Measured half-spread tiers (med/p90), broker combo
margin, max DD sized to 15% of 1.0m. Robustness grid (not in the verdict): d 3/5/7% x {hold, roll 3, roll 7, roll 10}.

## Pre-registered rule (docstring, before results), primary config at BOTH med and p90
P1 stand-alone: disc > 0, holdout > 0, positive after dropping 3 best months, bootstrap CI low of monthly mean > 0.
P2 adds to book: for r in {0.5, 1.0} sleeve lots per straddle lot, re-sized to 15% DD, combined ann return AND ret_per_dep
both > straddle alone, with disc > 0 and holdout > 0.
P3 monthly P&L correlation with the straddle < 0.7. VALIDATED only if all pass on BOTH 510050 and 510300.

## Result (tier med; p90 within 0.01)
| window | P1 stand-alone | P2 adds to book | P3 corr (monthly) | verdict |
|---|---|---|---|---|
| 510050 2019+ | PASS (rpd 0.447 vs straddle 0.479; disc +0.17, hold +0.13, drop3 +0.19) | FAIL (ann 0.075/0.071 vs 0.079; rpd 0.477/0.471 vs 0.479) | PASS 0.59 | fail |
| 510050 2015+ | PASS (rpd 0.483 vs 0.311; disc +0.40, hold +0.12) | PASS (ann 0.039/0.041 vs 0.037; rpd 0.359/0.385 vs 0.311) | PASS 0.62 | pass |
| 510300 2019+ | FAIL (rpd 0.316 vs 0.208, disc/hold/drop3 all > 0, but bootstrap CI includes 0) | PASS (ann 0.038/0.038 vs 0.033; rpd 0.238/0.254 vs 0.208) | PASS 0.67 | fail |

Overall: NOT VALIDATED. The common-window 510050 and 510300 fail on different tests, and the only clean pass is the longer
510050 2015+ sample. Robustness: 5/12, 4/12 and 2/12 grid configs pass P1 (510050 2019+, 2015+, 510300); roll-3 variants are
poor everywhere; 510300 hold-to-expiry arms are strong (d5% 0.348 rpd) but 510050 hold arms are weak in 2015+, so no
roll rule is consistently best.

## What the sleeve is and is not
* Return per margin is comparable to the straddle (and above it on 510050 2015+ and 510300), net of the measured-spread costs charged here.
* It is not a diversifier: monthly P&L correlation 0.59-0.67 (daily 0.63-0.75); it lost money in 7-8 of the straddle's 10
  worst months (mean -1.3k to -1.9k per lot), and its own worst month is -6.0k to -9.4k per lot.
* Adding it improves ann return at 15% DD by only +0.2 to +0.5 pt where P2 passes, and the combined worst-20-day is
  worse than the straddle alone on two of three windows (510300 -0.135/-0.150 vs -0.118; 510050 2019+ -0.138 vs -0.132).

## Caveats
12 configs x 3 windows; the verdict uses only the pre-registered primary. Spreads: ATM table for all short legs (510300
OTM unmeasured, 510050 off-ATM was tighter). Margin: naked combo, re-marked daily. The straddle book is the backtest of
the quarterly ATM rule, not the user's live Dec ladder.

## Verdict
NOT VALIDATED (not a clean failure): the strangle is a real earner but behaves like a levered version of the straddle's
own risk. Use as a sizing alternative to the straddle if wanted, not as a diversifying sleeve. No chart.
