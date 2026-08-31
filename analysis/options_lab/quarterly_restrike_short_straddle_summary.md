# Quarterly short straddle: monthly re-strike — REJECTED

**Script:** `quarterly_restrike_short_straddle.py` · **Verdict: REJECTED** (2026-08-31)

## Motivation

`quarterly_atm_short_straddle.py` opens one ATM short straddle per quarterly-cycle
expiry (season month in {3,6,9,12}) and holds it, unadjusted, to expiry. The
question tested here: does periodically re-centering the strike (closing the
current legs and reopening new ATM legs at the *same* expiry, as spot drifts)
reduce the directional/gamma risk of holding a static straddle through a full
~3-month cycle, and does that risk reduction pay for itself?

Two variants of "adjust the strike periodically" were considered up front and
one was ruled out by market structure alone, before any backtest: rolling into
a fresh "~3-months-out" *expiry* every month doesn't work on this market — SSE
50ETF/300ETF options always list exactly two quarterly-cycle expiries at once
(current-quarter, next-quarter), and the gap to the farther one swings between
~3 and ~6 months depending on where you are inside the current quarter, so
there's no clean rolling-into-fixed-tenor structure available. The only
coherent version is re-striking *within* a fixed expiry, which is what this
script backtests.

A structural concern was raised before running anything: re-striking later in
the cycle sells a straddle with much less extrinsic value (fewer days left to
decay) while ATM gamma is *rising* as DTE falls. So each successive re-strike
plausibly has worse risk/reward than the last.

## Method

`quarterly_restrike_short_straddle.py`, underlying 510050, 2015-02-09 to
2026-08-28. Same expiry selection as the baseline (`quarterly_atm_short_straddle._target_expiries`,
untouched — expiry never changes mid-cycle). A pre-committed, non-hindsight
trigger (`_should_restrike`, a pure function of calendar date / trading-day
index / DTE-remaining — never conditioned on P&L) fires the re-strike; when it
does, the current legs are closed and new ATM (or delta-target) legs reopened
at the same expiry. Swept `RESTRIKE_SPECS` (cadence x DTE floor) x slippage
(0.00 / 0.01 index points adverse half-spread per leg, charged on all 4
leg-fills of a restrike) x strike_mode=atm, split discovery (<2024-01-01) /
holdout (>=2024-01-01).

Sanity check: `--restrike-cadence none` reproduces `quarterly_atm_short_straddle.py`'s
result exactly (n=46 cycles, total_pnl=20,717 yuan, sharpe=0.47, max_dd=-7,772) —
confirms the shared expiry/settlement/margin logic wasn't altered.

## Result

| restrike spec | slippage | avg restrikes/cycle | discovery mean_r | holdout mean_r |
|---|---|---|---|---|
| never restrike (baseline) | 0.00 | 0.0 | **+3.71%** | **+11.31%** |
| never restrike (baseline) | 0.01 | 0.0 | **+1.38%** | **+9.12%** |
| monthly, no DTE floor | 0.00 | 2.9 | +3.32% | +8.19% |
| monthly, no DTE floor | 0.01 | 2.9 | -13.96% | -8.44% |
| monthly, stop inside 15 DTE | 0.00 | 2.9 | +3.32% | +8.19% |
| monthly, stop inside 15 DTE | 0.01 | 2.9 | -13.96% | -8.44% |
| monthly, stop inside 30 DTE | 0.00 | 1.9 | +4.35% | +8.37% |
| monthly, stop inside 30 DTE | 0.01 | 1.9 | -7.57% | -3.42% |
| one restrike ~1mo in, then hold | 0.00 | 0.8 | +4.40% | +8.77% |
| one restrike ~1mo in, then hold | 0.01 | 0.8 | -2.11% | +2.24% |

Full grid: `out/quarterly_restrike_short_straddle_sweep_510050.csv`.

Every re-strike variant underperforms the static baseline in **both** windows
at **both** slippage levels — including at zero slippage, where re-striking
is still slightly worse than not touching the position at all. The per-segment
table confirms the structural mechanism suspected up front: as the cycle
progresses, DTE at each new segment's entry falls (89d -> 81d -> 53d -> 24d for
monthly re-striking) and so does the credit collected (2,224 -> 2,195 -> 1,723
-> 1,134 yuan) — each re-strike sells thinner premium into rising ATM gamma
than the one before it, and that trade-off never pays for itself even before
transaction costs are counted. Adding a realistic per-leg slippage (0.01 index
points, charged on 4 fills per restrike) turns the naive monthly variant
strongly negative in both windows. The best-behaved variant ("one early
restrike, then hold to expiry", which minimizes round-trips) still loses to
the static baseline everywhere it's compared.

## Verdict

**REJECTED.** Re-striking the quarterly short straddle within a fixed expiry
— at any cadence tested, with or without a late-cycle DTE floor — does not
improve on simply holding the original ATM strike to expiry. The theoretical
concern (later re-strikes sell less premium into higher gamma) is confirmed
empirically, and it isn't just a transaction-cost artifact: the naive
zero-slippage version already underperforms. Static hold-to-expiry
(`quarterly_atm_short_straddle.py`) remains the validated approach; no
delta-mode variant was swept given the ATM-mode result was already uniformly
negative.
