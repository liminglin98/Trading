# Rolling deep-ITM call + short OTM call + short OTM put (no entry signal) — FAILED

Script: `rolling_itm_call_short_strangle_backtest.py`. Outputs: `out/rolling_itm_call_short_strangle_<u>_<start>{,_bench,_regime}.csv`.
Run date 2026-10-12. Windows: 510050 2019-12-23+ and 2015-02-09+, 510300 2019-12-23+.

## Structure and conventions
Long call (vendor delta closest to 0.90, >= 0.85, nearest expiry DTE >= 60, rolled at DTE <= 25 or 45) + short call and put
at the listed strike nearest spot x (1 +/- 3/5/7%), nearest expiry DTE >= 25, rolled at expiry ("hold") or at DTE <= 7.
12 configs per ETF. Measured half-spread tiers (med/p90; flat 0.005 stress only), broker 1.15x combo margin on the shorts
plus the long call's mark value, NO long-vs-short margin offset assumed. Sized to max DD = 15% of 1.0m.
Benchmarks: live quarterly ATM short straddle (same window/margin basis), ETF buy-and-hold per deposit (research only).

## Pre-registered rule (docstring, written before results)
Pass = at BOTH med and p90: (a) full ret_per_dep > long-only AND > quarterly straddle, (b) discovery and holdout > 0,
(c) positive after dropping the 3 best months, (d) strangle-only monthly mean > 0 with bootstrap CI low > 0. Validated only
if the same config passes on both ETFs.

## Result: NONE pass on any of the three windows
ret_per_dep (ann return / avg deposit), tier med:

| window | straddle | ETF b&h | full, best row | full, 7dte d3% L45 | long-only rolled call | strangle-only 7dte d3% |
|---|---|---|---|---|---|---|
| 510050 2019+ | 0.479 | 0.013 | 0.155 (d3% 7dte L25) | 0.147 | -0.26 | 0.447 |
| 510050 2015+ | 0.311 | 0.043 | 0.281 (d3% 7dte L45) | 0.281 | +0.02 | 0.483 |
| 510300 2019+ | 0.208 | 0.035 | 0.214 (d5% hold L45; disc -0.4%) | 0.208 (ties; strangle CI low < 0) | +0.06 | 0.316 |

* The rolled ITM call alone is a poor ETF substitute: it loses the dividend and the implicit carry (510050 2019+: -26% on
  deposit vs ETF +1%). It also costs lots: the long leg's beta is what sets the 15% DD, so the strangle runs at fewer lots.
* The strangle leg is where the money is — strangle-only per deposit matches or beats the straddle (510050 2015+ 0.48 vs 0.31,
  7dte arm) — but adding the long call lowers return per deposit and per max-DD at every config. Full structure <= straddle
  on per-deposit return in 35 of 36 (config, window) cells at med; the one exception (510300 d5% hold L45, 0.214 vs 0.208)
  has a negative discovery window.
* Regime table (mean monthly P&L per lot, 510050 2019+, 3% 7dte): drops -1.9k, rallies +1.4k (long-only +1.9k: call
  capped), range +0.3k. The structure is ETF beta with the upside clipped and the downside intact.
* Hold-to-expiry vs 7-DTE roll on the shorts differs a lot on 510050 2015+ (disc -0.04 vs +0.19 on the 3% arm). Not
  chased (forking paths); flag if the strangle alone is ever revisited.
* Spread tier barely matters (med vs p90 move ret_per_dep by < 0.01); the verdict is not a cost artifact.

## Caveats
510050 deep-ITM call half-spread is a 510300 proxy (unmeasured). The broker's diagonal margin treatment is not modelled —
if it offsets the short call against the long call, the full structure's deposit falls and ret_per_dep rises, but the long
call's P&L drag (above) is unaffected. 12 configs x 3 windows tried. Forced closes (row vanished / strike changed): 3-4 per config.

## Verdict
FAILED: ETF beta plus a short-vol overlay that does not beat the quarterly short straddle per margin and gives up the ETF's
dividend. No chart (failed verdict). Not live. The only lead is a monthly 3% strangle rolled at 7 DTE as a stand-alone
sleeve, which is untested against the existing short-vol book's correlation and tail (strangle-only worst20 ~ -14.5% of account
at 15% DD sizing).
