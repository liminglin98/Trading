# Vol-targeted de-risking of the quarterly ATM short straddle — verdict

**Script:** `vol_targeted_short_straddle_backtest.py` · **Status: mostly failed** (2026-08-31)

## What was tested

Two ways of closing the gap that `quarterly_atm_short_straddle.py` has neither an entry
filter nor a stop-loss — it sells every quarter regardless of conditions and holds every
position to expiry no matter how far it moves:

- **entry_gate**: skip the quarter's roll if QVIX is already elevated at entry (crossing
  above its own trailing 250-day mean + k·std), swept k in {1.0, 1.5, 2.0}.
- **stop_loss**: close the position early if QVIX spikes while it's held, instead of
  always riding it out to expiry/intrinsic settlement.

Both underlyings with reliable Greeks (510050, 510300), each underlying using its own
QVIX series.

## Result

| underlying | variant | n_trades | total_pnl | max_drawdown | n_stop_losses |
|---|---|---|---|---|---|
| 510050 | base (no rule) | 46 | 20,717 | -4,876 | 0 |
| 510050 | entry_gate k=1.5 | 43 | 19,865 | -4,616 | 0 |
| 510050 | stop_loss k=1.0 | 259 | 12,594 | -10,233 | 213 |
| 510050 | stop_loss k=1.5 | 175 | 7,773 | -9,320 | 129 |
| 510300 | base (no rule) | 26 | 12,902 | -4,185 | 0 |
| 510300 | entry_gate k=1.5 | 25 | 14,711 | -4,185 | 0 |
| 510300 | stop_loss k=1.0 | 129 | -1,801 | -9,367 | 102 |
| 510300 | stop_loss k=1.5 | 84 | 1,808 | -8,028 | 58 |

Full grid: `out/vol_targeted_straddle_summary.csv`.

**Stop-loss: REJECTED, robustly.** At every k tested, on both underlyings, adding a
QVIX-spike stop-loss makes total P&L dramatically worse (510050: -40% to -75% of baseline;
510300: swings to outright negative at k=1.0) while roughly **doubling** max drawdown. The
stop fires on the majority of quarters (n_stop_losses regularly exceeds half of n_trades) —
it's cutting the position right as a vol spike is statistically likely to mean-revert by
expiry, converting what would have been a recovered quarter into a locked-in loss. This is
the textbook failure mode of stop-losses on short-vol carry strategies, now confirmed on
this specific book rather than assumed.

**Entry gate: weak, inconclusive.** Skips very few quarters (3-5 of 46 on 510050, 1-2 of 26
on 510300) since QVIX is rarely elevated exactly at a quarterly-roll date. Effect is within
noise: mildly negative on 510050, mildly positive on 510300. Not enough signal or sample to
call this either validated or rejected — it's closer to "doesn't matter" than "helps."

## Verdict

**Failed (stop-loss mechanism) / inconclusive (entry gate).** Don't add a QVIX-spike
stop-loss to the quarterly straddle — the evidence against it is strong and consistent.
The entry gate isn't harmful but the effect is too small and too thin-sample to adopt on
its own. Not live in `alerts_core.py`. This adds to a growing pattern across this session's
investigations (re-striking rejected, delta-targeted legs mixed/no edge, now stop-losses
rejected): every tested addition to the plain quarterly ATM short straddle either hurts or
doesn't clear the bar — none has beaten simply holding the static ATM straddle to expiry.
