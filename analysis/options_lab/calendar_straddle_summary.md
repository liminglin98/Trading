# ATM straddle calendar (short near, long far, same strike) — FAILED

Script: `calendar_straddle_backtest.py`. Outputs: `out/calendar_straddle_<u>_<start>{,_bench}.csv`. Run 2026-10-12.
Windows: 510050 2019-12-23+ and 2015-02-09+, 510300 2019-12-23+.

## Structure and conventions
Short near ATM straddle (first expiry DTE >= 25) + long far ATM straddle at the same strike (first expiry with
DTE >= near DTE + GAP), re-struck every cycle, no entry signal. 4 configs per ETF: GAP {50, 80} x exit {hold to near expiry,
close at near DTE <= 7}. Measured half-spread tiers (med/p90), broker combo margin on the near straddle plus the far legs'
mark (conservative; no calendar offset assumed), sized to max DD 15% of 1.0m. Benchmarks: near-only straddle at the same
entries, live quarterly straddle.

## Pre-registered rule (docstring, before results)
Pass = at BOTH med and p90: (a) calendar ret_per_dep > near-only AND > quarterly straddle, (b) discovery and holdout > 0,
(c) positive after dropping the 3 best months, (d) far-leg monthly mean > 0 with bootstrap CI low > 0; validated only if the
same config passes on both ETFs.

## Result: NONE pass, any window
ret_per_dep (ann return / avg deposit), tier med, range over the 4 configs:

| window | quarterly straddle | calendar (conservative deposit) | calendar, net-debit deposit (upper bound) | far leg alone |
|---|---|---|---|---|
| 510050 2019+ | 0.479 | -0.03 .. +0.04 | -0.21 .. +0.19 | -0.18 .. -0.13 |
| 510050 2015+ | 0.311 | -0.01 .. +0.03 | -0.06 .. +0.17 | -0.16 .. -0.08 |
| 510300 2019+ | 0.208 | +0.01 .. +0.03 | +0.09 .. +0.17 | -0.13 .. -0.04 |

* The tail claim is true: worst month per lot is -0.8k to -2.7k for the calendar vs -2.9k to -10k for the near-only
  straddle; worst-20-day is lower in all 12 cells (510050 2015+ gap50 7dte only marginally: -0.096 vs -0.098).
* The price is the whole return. The far straddle loses money in every config on every window (it bleeds more theta and
  spread than the near leg's gains it cushions); far-leg total P&L is negative in every cell, so condition (d) fails
  everywhere. The calendar then needs 30+ lots to reach 15% DD and earns ~0.
* Even under the optimistic net-debit deposit (a broker calendar offset that was NOT verified) the best cell (0.19) is below
  the straddle in all three windows.
* Spread tier changes the numbers by < 0.02; not a cost artifact.

## Incidental (not part of the rule, not chased)
The near-only monthly ATM straddle at these entries returned 0.26-0.30 per deposit on 510050 2019+, ~0.2 on 510300 and only
0.10-0.11 (hold) / 0.22-0.29 (7dte) on 510050 2015+: unstable across windows and exits, and below the quarterly straddle on
510050. It does not overturn the earlier finding that a monthly slot is not better than T75 quarterly.

## Verdict
FAILED: long far vega protects the tail but costs more than it earns; calendar < quarterly straddle per margin. No chart.
