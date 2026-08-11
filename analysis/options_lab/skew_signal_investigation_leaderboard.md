# Skew signal investigation: results tracker

Tracks the replicated CBOE-SKEW-style volatility index (`Data/exploring/huidian_index_replication/`,
validated against China's official QVIX-adjacent benchmark) as a trading signal.
Skew captures how richly the market prices downside puts vs. upside calls -- a
"crash fear" gauge -- and is essentially uncorrelated with every other vol
signal this repo trades on (|corr| <= 0.09 with iv_pct/rv_pct/vrp), making it
a genuinely new information axis worth testing on its own. Only 510050/510300
are supported (the only underlyings with a precomputed skew index).

**Status: no validated edge found across the 3 SKEW-index-gated structures
tested (sections 1-3, 510050/510300 only).** Every cell that looked
promising in discovery failed to replicate in holdout, or underperformed a
simpler existing baseline once checked properly. Section 4 is a separate,
positive result on a different underlying (CSI1000/MO) using a different
skew-measurement method (raw option-chain BS-IV put-minus-call skew, not the
precomputed SKEW index) and a different construction (an always-on
skew-harvesting *structure*, not a level/event-gated signal on top of an
existing structure) -- don't read its result as contradicting the
conclusion below, which is scoped to sections 1-3.

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

## 4. CSI1000 put-skew risk reversal — DONE, POSITIVE (holdout-confirmed, margin caveat applies)

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
| risk_reversal | 96 | -24,500 yuan / -0.17% | **+95,060 yuan / +3.24%** |
| strangle (control) | 77 | -33,160 yuan / -0.93% | **-209,480 yuan / -3.66%** |

risk_reversal beats the strangle control in both periods, and the gap widens
in holdout -- the skew-tilted structure isn't just "selling vol," it's doing
something the delta-neutral control isn't. Entry-time skew-credit
decomposition (actual put price vs. counterfactual at the call leg's own
IV, i.e. "what the put would cost with no skew") confirms the edge exists
at entry, independent of path: n=96, mean +1,408 yuan/trade, 91% of trades
positive, cross-checked against the vega approximation (corr=1.00). Realized
mean pnl/trade (~735 yuan) runs below the mean entry-time skew edge
(1,408 yuan) -- consistent with part of the edge being eaten by the same
small-cap gap risk that prices the put rich in the first place (the
Feb-2024 DMA-liquidation spike in the underlying skew measurement is the
textbook version of that risk).

**Caveat, not a footnote**: CFFEX margin_ratio/min_guarantee are MO's 2022
launch-date parameters, held static across the whole window -- no verified
historical revision schedule was locatable (`cffex_margin.py` docstring).
Every return-on-margin/leverage number above should be read as approximate
for this reason; the yuan P&L totals and the entry-time skew-edge
decomposition (both margin-independent) are the more trustworthy numbers
here.

**Conclusion: positive, holdout-confirmed result, distinct from sections
1-3's rejections.** This is a different underlying, different skew data
source, and a structural (always-on) rather than signal-gated construction
-- the CSI1000 small-cap-hedging premium looks like a real, harvestable
effect over this window, though capital-efficiency figures carry the margin
caveat above. A natural follow-up (not done here): sweep target_delta and
DTE window the way `delta_strangle_short_backtest.py`'s sweeps do for the
SSE strangle, to check whether 0.20/[15,45] is a stable choice or a
one-cell coincidence.

## Scripts

- `skew_directional_backtest.py` — put/call debit spread gated on skew_pct level (all 4 combos rejected)
- `skew_spike_backtest.py` — contrarian short straddle / momentum put spread on a skew-spike event (no reliable edge, both underlyings)
- `skew_vrp_combo_short_straddle_backtest.py` — short straddle AND-gated on skew_pct + vrp (rejected both underlyings, matched-sample check)
- `csi1000_put_skew_check.py` — CSI1000 vs CSI300/SSE50 put-call IV skew measurement (motivating result for #4)
- `csi1000_skew_risk_reversal_backtest.py` — CSI1000 skew risk-reversal harvesting backtest, positive holdout result (#4)
- `cffex_margin.py` — CFFEX index-option margin formula (supports #4)
