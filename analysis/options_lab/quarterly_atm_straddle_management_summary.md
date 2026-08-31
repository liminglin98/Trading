# Quarterly ATM straddle: laddered entry (promising) + profit target (not robust)

**Script:** `quarterly_atm_straddle_management_backtest.py` · **Status: preliminary** (2026-08-31)

## What was tested

Two cheap tweaks to `quarterly_atm_short_straddle.py`'s plain single-day-entry,
hold-to-expiry cadence, independently and crossed:

- **Laddered entry**: spread the quarter's entry over `ladder_days` consecutive trading
  days (re-selecting ATM strike each day) instead of one lump-sum day, rescaled to the
  same total capital as a single-day entry.
- **Profit target**: close the whole position early once unrealized profit reaches
  `profit_target_pct` of credit received, then immediately re-establish a fresh ATM
  straddle at the same expiry for the rest of the cycle — the mirror image of the
  QVIX-spike stop-loss already REJECTED in `vol_targeted_short_straddle_summary.md`.

Both underlyings (510050, 510300), discovery (<2024-01-01) / holdout (>=2024-01-01) split,
then a slippage sensitivity check (0.00/0.005/0.01/0.02 index points adverse half-spread
per leg) since the restrike and vol-targeting investigations both showed that a
zero-friction edge can evaporate once real trading costs are included.

## Result

**At zero slippage, both levers look good** — nearly every cell in the 4×4
(ladder_days × profit_target) grid beats baseline in both discovery and holdout windows,
on both underlyings (`out/quarterly_atm_straddle_mgmt_sweep_510050.csv`,
`..._510300.csv`). Matched-capital full-period comparison:

| underlying | config | total_pnl | sharpe | max_dd |
|---|---|---|---|---|
| 510050 | baseline | 20,717 | 0.47 | -7,772 |
| 510050 | laddered only (5-day) | 23,844 | 0.58 | -6,436 |
| 510050 | profit-target only (50%) | 21,738 | 0.47 | -7,772 |
| 510300 | baseline | 12,902 | 0.35 | -8,766 |
| 510300 | laddered only (5-day) | 17,680 | 0.54 | -7,080 |
| 510300 | profit-target only (50%) | 14,791 | 0.42 | -8,103 |

**Slippage sensitivity separates the two levers decisively.** Profit-target requires a
genuine extra round-trip on almost every cycle (43/46 on 510050, 23/26 on 510300) to
realize its edge, and that edge does not survive realistic costs:

| underlying | slippage | baseline | profit-target 50% | laddered 5-day |
|---|---|---|---|---|
| 510050 | 0.000 | 20,717 | 21,738 | 23,844 |
| 510050 | 0.005 | 16,117 | 8,994 | 19,244 |
| 510050 | 0.010 | 11,517 | **-953** | 14,644 |
| 510050 | 0.020 | 2,317 | **-16,475** | 5,444 |
| 510300 | 0.000 | 12,902 | 14,791 | 17,680 |
| 510300 | 0.005 | 10,302 | 8,890 | 15,080 |
| 510300 | 0.010 | 7,702 | 1,464 | 12,480 |
| 510300 | 0.020 | 2,502 | **-7,416** | 7,280 |

Profit-target degrades faster than baseline and turns net-negative well before baseline
does, on both underlyings — the same failure mode as the rejected re-strike idea: real
transaction costs on extra round-trips eat a "free-looking" backtest edge. Laddering, by
contrast, degrades at roughly the *same rate* as baseline (it trades the same total
notional, just spread across more days, not genuinely extra volume) and **beats baseline
at every slippage level tested, on both underlyings**.

## Verdict

**Laddered entry: promising, preliminary.** Consistent improvement in P&L, Sharpe, and
drawdown, on both underlyings, in both discovery and holdout windows, and the edge
survives realistic transaction costs because it isn't adding real trading volume — it's
averaging the entry price across a few days instead of committing the whole quarter's
position to one day's print. Not yet live in `alerts_core.py`; worth taking seriously as a
candidate for the quarterly straddle's actual entry mechanics.

**Profit target: not robust, do not adopt.** Looks attractive at zero slippage but that
edge is an artifact of the frictionless backtest convention — it requires real extra
round-trips that a realistic spread erases, going net-negative faster than simply holding
the baseline position. Same underlying lesson as `quarterly_restrike_short_straddle_summary.md`.
