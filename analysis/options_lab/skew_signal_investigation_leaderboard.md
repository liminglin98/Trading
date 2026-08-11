# Skew signal investigation: results tracker

Tracks the replicated CBOE-SKEW-style volatility index (`Data/exploring/huidian_index_replication/`,
validated against China's official QVIX-adjacent benchmark) as a trading signal.
Skew captures how richly the market prices downside puts vs. upside calls -- a
"crash fear" gauge -- and is essentially uncorrelated with every other vol
signal this repo trades on (|corr| <= 0.09 with iv_pct/rv_pct/vrp), making it
a genuinely new information axis worth testing on its own. Only 510050/510300
are supported (the only underlyings with a precomputed skew index).

**Status: no validated edge found in any of the 4 approaches tested.** Every
cell that looked promising in discovery failed to replicate in holdout, or
underperformed a simpler existing baseline, or (section 4) evaporated once
directional beta was properly hedged out. Section 4 uses a different
underlying (CSI1000/MO) and different skew-measurement method (raw
option-chain BS-IV put-minus-call skew, not the precomputed SKEW index) from
sections 1-3, and an important cautionary lesson of its own: an unhedged
backtest result that looks like a clean skew-harvest win can actually be a
directional bet in disguise -- always check what the structure's net delta
actually is and what the underlying did over the sample window before
trusting a "the edge doesn't decay" read.

## 1. Directional debit spread gated on skew_pct level — DONE, REJECTED (all 4 cells)

`skew_directional_backtest.py`. Two opposite hypotheses tested independently:
high skew_pct (crash fear elevated) -> bearish put debit spread; low skew_pct
(crash fear muted) -> bullish call debit spread. Entry gated on YESTERDAY's
skew_pct (matching this repo's "rich/extreme" trigger convention). Grid: 4
thresholds x 3 wing widths x 4 max-holds x 4 profit-targets = 192 cells per
underlying/direction, discovery < 2024-01-01 <= holdout.

**Every one of the 4 underlying x direction combinations fails the exact same
way: the discovery-best pick (by construction, the frozen choice under this
repo's discipline) doesn't replicate in holdout — usually flipping sign, in
one case catastrophically:**

| Underlying | Direction | Discovery (n, mean_r) | Holdout (n, mean_r) | Eligible cells: agree-pos / agree-neg / mixed |
|---|---|---|---|---|
| 510050 | put_spread | n=67, **+21.7%** | n=18, **−45.8%** | 2 / 143 / 47 (out of 192) |
| 510050 | call_spread | n=55, **+5.2%** | n=20, **−3.2%** | 20 / 44 / 128 (out of 192) |
| 510300 | call_spread | n=15, **+9.7%** | n=13, **−3.4%** | 10 / 12 / 134 (out of 156) |
| 510300 | put_spread | n=15, **+71.4%** | n=11, **−22.4%** | 25 / 9 / 149 (out of 183) |

510050 put_spread is the most decisive rejection of the four: 143/192 eligible
cells (75%) agree-negative across the *entire* grid, not just the frozen pick
-- this isn't an isolated overfitting spike, the high-skew_pct-bearish thesis
is simply unprofitable across nearly the whole parameter space for this
underlying. The other three cells are closer to genuine discovery/holdout
sign-flips on an otherwise more mixed grid (10-25 agree-positive cells each),
consistent with an overfitting signature rather than a real, broad effect.

**Conclusion: neither direction works, on either underlying.** Don't re-propose
a skew-level-gated directional debit spread without a fundamentally different
gating construction (e.g. requiring persistence, or combining with a second
signal that's ALREADY validated on its own -- not just co-occurrence, see the
combo result below for why that specific idea also failed).

## 2. Skew spike (event trigger): contrarian short straddle vs. momentum put spread — DONE, no reliable edge

`skew_spike_backtest.py`. Discrete level-crossing event (skew crosses above
its own trailing 250-day mean + k*std, k in {1.0, 1.5, 2.0}), tested as BOTH
a contrarian fade (sell a naked ATM straddle -- crash-fear richness itself is
a mean-reversion opportunity) and a momentum follow (buy a put debit spread
-- crash fear accelerating, bet it continues) on the same trigger dates.
Entry gated on yesterday's crossing. This script deliberately doesn't freeze
a single pick (event-triggered entries aren't a threshold to grid-search);
it reports full-sample discovery/holdout stats per k/structure/underlying
instead.

**510050** (n_total ranges 17-45 depending on k):

| k | Structure | Discovery mean_r | Holdout mean_r | Read |
|---|---|---|---|---|
| 1.0 | contrarian | −1.4% | +1.4% | sign flip |
| 1.0 | momentum | −5.1% | +8.2% | sign flip |
| 1.5 | contrarian | +0.8% | +1.5% | same sign, but tiny (only genuinely consistent cell) |
| 1.5 | momentum | **−39.1%** | +16.6% | huge swing, not a real effect |
| 2.0 | contrarian | −5.3% | +3.0% | sign flip, small sample |
| 2.0 | momentum | −16.8% | +12.4% | sign flip, small sample |

**510300** is worse-conditioned (the skew index has much less usable history
here -- discovery n as low as 3-10 events): momentum shows wild, clearly
noise-driven swings (discovery mean_r up to **+95.6%**, holdout **−5.7%** at
k=2.0, explicitly flagged by the script's own SMALL SAMPLE banner at k=2.0
for both structures).

**Conclusion: no reliable edge in either direction.** Contrarian at k=1.5 is
the only cell with same-signed discovery/holdout in the entire grid, and its
magnitude (+0.8%/+1.5%) is too small to call a finding rather than noise.
Momentum's huge, unstable swings are a textbook small-event-count overfitting
signature, not a real crash-fear-continuation effect. Don't re-propose this
trigger construction without a fundamentally larger dataset or a coarser,
more persistent event definition.

## 3. skew_pct + vrp combo short straddle — DONE, REJECTED (matched-sample check both underlyings)

`skew_vrp_combo_short_straddle_backtest.py`. AND-gates yesterday's skew_pct
>= threshold with yesterday's vrp >= threshold for a naked short straddle,
testing whether "vol is rich (vrp) AND crash fear is elevated (skew)"
together produces cleaner entries than the already-validated vrp-only signal
alone ([[project-short-straddle-vol-signals]]). Since this combo's entries
are structurally a subset of the vrp-only baseline (an extra AND-gate can
only remove trades), the mandatory check is a side-by-side matched-sample
comparison, not just "does the combo look good in isolation."

**510050**: 24-cell sweep, only 1/10 eligible cells (disc_n>=15) agrees
positive in both periods -- `skew_pct>=0.70, vrp>=+3%, max_hold=20d`
(disc n=18 +1.2%, hold n=8 +2.97%). Matched-sample check against the vrp-only
baseline:

| Strategy | Discovery (n, mean_r) | Holdout (n, mean_r) |
|---|---|---|
| vrp-only (baseline) | n=54, **+3.06%** | n=23, **+2.52%** |
| skew+vrp (combo) | n=18, +1.20% | n=8, +2.97% |

The baseline beats the combo in discovery and is roughly tied in a much
thinner holdout sample. Adding the skew gate shrinks the tradeable sample by
2/3 (77 combo signals reachable from just 26, only 54% overlap with the
baseline's own fire dates) without improving returns.

**510300**: cleaner-looking sweep (both of the 2 eligible cells agree
positive, vs. 510050's 1/10), frozen pick `skew_pct>=0.70, vrp>=+0%,
max_hold=20d` (disc n=16 +3.3%, hold n=13 +2.76%). Same matched-sample check:

| Strategy | Discovery (n, mean_r) | Holdout (n, mean_r) |
|---|---|---|
| vrp-only (baseline) | n=39, **+3.60%** | n=25, **+4.78%** |
| skew+vrp (combo) | n=16, +3.30% | n=13, +2.76% |

Same story: the baseline wins on both periods here too, more clearly in
holdout (+4.78% vs +2.76%).

**Conclusion: rejected on both underlyings** -- this is exactly the
"smaller/cherry-picked sample looks better in isolation but loses to the
full baseline once matched" failure mode this repo has already hit twice in
the condor investigation (`condor_delta_investigation_summary.md` §3.5/3.10).
The skew gate doesn't add information on top of vrp; it just throws away
trades. Keep vrp-only as the frozen short-straddle signal
([[project-short-straddle-vol-signals]]); don't add a skew_pct AND-gate.

## Overall conclusion (sections 1-3, SKEW-index-gated, 510050/510300)

Despite being a genuinely new, ~uncorrelated signal axis (unlike most
"combo" ideas in this repo, which tend to overlap heavily with an existing
validated signal), skew_pct/skew-spike carries no demonstrated tradeable
edge in any of the 3 structures tested here: not as a standalone directional
trigger, not as an event-based trigger for either a contrarian or momentum
structure, and not as an additional filter on top of the already-validated
vrp short-straddle signal. This is a clean, complete negative result --
don't re-propose any of the above without a genuinely different construction
(a longer/higher-quality skew history, a persistence-based rather than
level-based gate, or a structure not yet tried here).

## 4. CSI1000 put-skew risk reversal — DONE, REJECTED (negative, confirmed across two hedging methodologies)

`csi1000_put_skew_check.py` (measurement) + `csi1000_skew_risk_reversal_backtest.py`
(harvesting). Different underlying (000852/MO, CFFEX) and different skew
measurement (per-day BS-IV inversion on the raw option chain, since no
precomputed SKEW index exists for CFFEX names) from sections 1-3 above.
Motivating measurement: CSI1000's 5%-OTM put IV is richer than call IV on
88.5% of days (mean 2.24 vol pts, n=982 days, 2022-07-22 to 2026-08-10) vs.
much weaker CSI300 (68.2%, 0.93 vol pts) and near-zero SSE50 (45.7%, -0.47
vol pts) controls -- a small-cap-hedging-demand signature, not generic index
skew, corroborated by OI put/call ratio (same ordering).

Structure: always-on-roll 20-delta risk reversal (short put + long call,
same expiry, DTE 15-45, stop-loss 2.5x credit, max_hold 25d, no profit
target -- the asymmetric payoff is the point) vs. a delta-neutral 20-delta
short strangle as the control (same engine, same parameters, profit target
50% of credit). CFFEX futures-style margin via new `cffex_margin.py`
(margin.py's SSE/SZSE formula is wrong for these contracts -- wrong
multiplier and wrong formula shape).

| Structure | n | Discovery (<2024, total pnl / mean_r) | Holdout (>=2024, total pnl / mean_r) |
|---|---|---|---|
| risk_reversal | 96 | -24,500 yuan / -0.17% | +95,060 yuan / +3.24% |
| strangle (control) | 77 | -33,160 yuan / -0.93% | -209,480 yuan / -3.66% |

At first read, risk_reversal beats the strangle control in both periods and
the gap widens in holdout. **That read turned out to be substantially wrong**
-- see the delta-hedge overlay below, which was the natural next check
(prompted by the user asking why discovery lost money, expecting it was
because the skew premium itself was weaker pre-2024 -- it wasn't: entry-time
skew was 2.76 vol pts in discovery vs. 3.18 in holdout, a modest gap, not
enough to explain a swing from -24,500 to +95,060).

**What actually explains the swing: CSI1000 fell -15.3% over the discovery
window (2022-07 to 2023-12) and rose +30.6% over the holdout window
(2024-01 to 2026-08, despite a -30% crash inside it in Feb 2024), and the
risk-reversal structure is NOT close to delta-neutral -- both legs
contribute positive delta (short put ~ long-delta, long call = long-delta),
measured net position delta ~+0.39 at entry, essentially unchanged between
periods. It's a ~40%-delta bullish bet on CSI1000 dressed up as a skew
harvest; unhedged pnl correlates +0.93 with each trade's own realized
underlying return.**

**First hedge attempt (`delta_hedge_overlay()`, KNOWN FLAWED, kept for
reference)**: daily-rebalanced BS-delta hedge layered on top of the SAME
fixed trade schedule the unhedged run above already decided (entries/exits
unchanged). Priced at the options' own parity-forward spot (no CFFEX
index-futures/IM price history exists in this repo, so this is a proxy, not
an independently-sourced futures price; no separate hedge transaction cost
modeled).

| Structure | Unhedged total | Overlay-hedged total | corr(pnl, underlying_return) unhedged -> overlay-hedged |
|---|---|---|---|
| risk_reversal | +70,560 | -70,846 yuan | +0.93 -> +0.54 |
| strangle | -242,640 | +3,975 yuan | +0.05 -> +0.32 |

This already flips both structures' sign, but the residual +0.54/+0.32
correlation to direction was suspicious -- a real hedge shouldn't leave that
much directional dependence. Root cause, found on inspection: the SAME
unhedged P&L that's supposedly being hedged away is also what decided which
trades got stopped out vs. held to expiry (`stop_loss_mult`/`max_hold`
checked against unhedged pnl). A hedged position was being exited on an
unhedged trigger -- selection bias, not just a rebalancing-frequency
limitation.

**Fix (`run(..., delta_hedge=True)`, the corrected version)**: exits are now
decided on the hedged P&L itself (option pnl + accumulated daily hedge pnl),
not the unhedged option pnl -- the hedge is now inside the strategy's own
risk management, not bolted on after the fact. Trade counts and holding
periods genuinely differ from the unhedged run (fewer stop-outs, longer
average holds), which is expected: a hedged position reacts to different
risk than an unhedged one.

| Structure | Corrected-hedged total | Discovery | Holdout | By-year sign | corr(pnl, underlying_return) |
|---|---|---|---|---|---|
| risk_reversal | **-137,099 yuan** (n=63) | -10,285 | -126,813 | negative 4 of 5 years, worst 2025 (11% win rate) | +0.14 |
| strangle | **+77,702 yuan** (n=55) | +34,779 | +42,924 | **positive every year 2022-2026** | (not separately computed; already near-neutral pre-hedge) |

The correlation drop (+0.93 -> +0.14 for risk_reversal, vs. only -> +0.54
with the flawed overlay) confirms the exit-timing bug was the dominant
source of the earlier residual, not daily-only rebalancing (which remains
an approximation -- this repo has no historical intraday CFFEX option data
to rebalance against more finely -- but is evidently a second-order effect
next to the exit-timing fix).

**Fixing the bug made the result MORE decisively negative for
risk_reversal, not less.** The properly hedged strangle, by contrast, is
now solidly and consistently profitable (positive every calendar year,
sharpe(daily,ann)=0.73) -- the generic variance risk premium (already
validated on other underlyings via this repo's vrp signal) is real and
harvestable on CSI1000 too. The skew-tilted risk reversal does not add to
it; once direction is controlled for, it actively underperforms the plain
symmetric strangle.

Entry-time skew-credit decomposition (actual put price vs. counterfactual
at the call leg's own IV) still shows a reliably positive edge at entry on
the risk_reversal's own trades: n=63, mean +1,358 yuan/trade, 94% positive,
corr=1.00 vs. the vega approximation. **This is the central finding**: the
skew premium is genuinely priced into CSI1000 puts at entry (this part of
the original claim, from `csi1000_put_skew_check.py`, was always solid and
remains so) -- but tilting a structure toward harvesting it specifically,
even properly delta-hedged, does not produce better realized P&L than just
selling vol symmetrically. Whatever eats the entry-time edge (small-cap gap
risk realized during the hold, most plausibly) eats it selectively on the
put-skewed structure and not on the symmetric one.

**Caveat, not a footnote**: CFFEX margin_ratio/min_guarantee are MO's 2022
launch-date parameters, held static across the whole window -- no verified
historical revision schedule was locatable (`cffex_margin.py` docstring).
This affects only return-on-margin/leverage figures, not the yuan P&L
totals above.

**Conclusion: REVISED TWICE, converging on a clear negative.** Unhedged,
this looked like a clean, holdout-confirmed skew harvest (+70,560 yuan,
"doesn't decay") -- that was CSI1000's realized direction (down in
discovery, up in holdout) interacting with the risk-reversal's ~+0.4 net
delta, not a skew-specific edge. The first hedge attempt (flawed
exit-timing) still showed a negative but smaller (-70,846) result. The
corrected hedge (exits on hedged P&L) is more negative still (-137,099),
while the same fix applied to the strangle control turns it solidly
profitable (+77,702, positive every year). **Don't treat the CSI1000 skew
risk reversal as a validated tradeable edge -- the entry-time pricing
observation is real, but this repo's attempts to harvest it specifically
(as opposed to just harvesting the generic variance risk premium CSI1000
already offers) have not worked, across two independent hedging
methodologies.** Remaining open question, not resolved here: whether a
different construction (tighter native gamma matching between legs, a
persistence-gated entry rather than always-on, or accepting the directional
exposure deliberately rather than fighting it) could still capture something
skew-specific -- but the burden of proof is now higher given two negative
results in a row.

## Scripts

- `skew_directional_backtest.py` — put/call debit spread gated on skew_pct level (all 4 combos rejected)
- `skew_spike_backtest.py` — contrarian short straddle / momentum put spread on a skew-spike event (no reliable edge, both underlyings)
- `skew_vrp_combo_short_straddle_backtest.py` — short straddle AND-gated on skew_pct + vrp (rejected both underlyings, matched-sample check)
- `csi1000_put_skew_check.py` — CSI1000 vs CSI300/SSE50 put-call IV skew measurement (motivating result for #4)
- `csi1000_skew_risk_reversal_backtest.py` — CSI1000 skew risk-reversal harvesting backtest, positive holdout result (#4)
- `cffex_margin.py` — CFFEX index-option margin formula (supports #4)
