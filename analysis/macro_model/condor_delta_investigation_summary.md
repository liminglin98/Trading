# Delta-Targeted Iron Condor: Investigation Summary & Verdict

**Bottom line: no robust, out-of-sample-defensible edge found.** Every management lever tested (early exit, rolling, entry duration, wing width, a mechanism-targeted trend/momentum mid-trade exit) either hurt performance outright or looked good only in-sample and failed when tested on data it hadn't seen. Two attempts to pair the condor with something else entirely — a far-OTM tail hedge (§3.8) and a separate trend-following sleeve on the SSE Composite (§3.9) — fared no better. The apparent full-history edge is concentrated almost entirely in 2024-2026, which the data suggests was a lucky stretch (no tail events), not a structural improvement. Treat this construction as **unproven, likely negative-expectancy over a representative sample**, not a strategy ready to trade.

Scripts referenced below are all in this folder; each is independently runnable and documents its own method in its docstring.

---

## 1. What this is testing

A monthly iron condor on SSE 510050 (50ETF) options: short a put and a call at a target delta (e.g. 0.20), long protective wings `p` listed strikes further out, sized at fractional Kelly. Built on `Data/options/option_daily_510050.csv`, a local daily chain (Sina-sourced, real quotes + exchange Greeks, 2015-02-09 → 2026-07-08, gitignored, machine-local) that replaced the project's earlier reliance on live SSE API calls limited to ~50 out-of-sample months. This unlocked ~11 years / ~130 monthly entries instead of ~50 — the entire reason this investigation was possible.

Key scripts, roughly in the order they were written:
- `condor_delta_backtest.py` — baseline: monthly entry, delta-targeted strikes, hold to expiry, no macro model.
- `condor_delta_exit_backtest.py` — adds a mechanical exit (DTE / profit-target / stop-loss / delta-breach), closes to cash.
- `condor_delta_roll_backtest.py` — same triggers, but rolls into a fresh position same-day instead of going to cash; modeled as one continuous chain across the full history (not independent monthly campaigns, since an aggressive trigger can span years before naturally expiring).
- `condor_delta_roll_oos_test.py` — genuine train/test time split for the roll rule.
- `condor_delta_dte_sweep.py` — same OOS discipline, applied to entry duration (target DTE).
- `condor_delta_regime_diagnostic.py` — decomposes why the 2015-2023 vs 2024-2026 windows differ (VRP, realized vol, tail-loss frequency).
- `condor_delta_wing_sweep.py` — same OOS discipline, applied to wing width (p).
- `index_trend_signal.py` — reusable trend-persistence signal on the SSE Composite (000001), used by three of the below.
- `condor_trend_exit_backtest.py` — the trend/momentum mid-trade exit filter (§3.7).
- `sse_4000_level_read.py` — standalone (non-condor) read of the SSE Composite around the 4000 level.
- `condor_tail_hedge_backtest.py` — a far-OTM options hedge bought alongside the condor (§3.8).
- `condor_trend_sleeve_backtest.py` — a separate trend-following position on the SSE Composite, blended with the condor as a second sleeve rather than a rule on the condor itself (§3.9).

---

## 2. Methodology: why every result below has a discovery/holdout split

The first full-sample parameter sweeps (exit rules, then a stop-loss "roll" rule) found what looked like a real, tunable edge — a stop-loss around 1.6x-2.1x of credit received produced positive Calmar across two different entry-delta configs. **This did not survive a real out-of-sample test.**

**Split, fixed in advance (not tuned to make results look good):**
- Discovery: 2015-02-09 → 2023-12-31 (spans pre- and post-COVID regimes)
- Holdout: 2024-01-01 → 2026-07-08, touched once, at the end, to score whatever discovery froze

*(An earlier attempt split at 2021-01-01, putting a mostly-pre-COVID window into discovery — rejected on the reasonable objection that market structure changed materially around COVID, so that split was testing "survive a regime break," a harder and less relevant question than "generalize forward within a regime you've partly seen." All results below use the 2024-01-01 split.)*

With that discipline in place, **every positive-looking full-sample or discovery-only result died on holdout.** This methodology is the single most important output of this investigation — more valuable than any individual parameter finding.

---

## 3. What was tested, and the verdict on each

### 3.1 Exit rules (close to cash on trigger) — hurt outright, no OOS test even needed
DTE exit, profit-target, stop-loss, delta-breach — all showed **negative mean expectancy in-sample**, full stop. Mechanism: this is a high win-rate strategy (~70%+); most mid-life drawdowns recover by expiry. Any rule that reacts to the "scary middle" disproportionately cuts winners short. Confirmed at two entry configs (0.20-delta/p=3 and 0.25-delta/p=3).

### 3.2 Rolling (same triggers, but redeploy into a fresh position instead of cash)
Looked much better than closing to cash — a stop-loss-only roll (~1.6x-2.1x of credit) showed positive Calmar in-sample, replicated at a second entry config. **Failed the OOS test twice, at two different split dates:**
- 2021 split: frozen 1.8x → holdout CAGR +0.00% (Kelly f=0), worse than holdout's own no-roll baseline.
- 2024 split: frozen 1.8x → holdout CAGR +0.00% again; *every* stop-loss multiple tested was flat on holdout, while holdout's plain baseline scored the best number in the whole investigation (CAGR +10.14%, Calmar 0.70).

**Why, mechanistically (verified with real trades, not just aggregate stats):** stop-loss rolling correctly slashed every one of 5 known disaster trades (e.g. -98.6% → -40.3% on the Dec 2022 reopening-rally trade) — but it also stopped out other trades in the same windows that would have finished flat or slightly positive at expiry, converting them into real losses. Traced dollar-for-dollar over the Nov 2022-Jun 2023 window: hold-to-expiry ended net +0.0278; the same window under stop-loss=2.0x ended net **-0.0058**, despite correctly capping the real disaster. The insurance only pays off if the window you're testing happens to contain a disaster — and disasters are rare (~1 every 1.7-2 years) and clustered, so whether a backtest "shows" the rule working is largely a function of sample composition, not a stable edge.

### 3.3 Entry duration (target DTE: 14/21/30/45/60/90 days)
- Structural finding: SSE 50ETF only lists ~4 expiries at a time (near month, next month, two quarterlies), so targets below ~30 days mostly collapse to the *same actual contract* — you can't get much shorter than ~25 days without weekly options, which don't exist for this instrument.
- **The bigger finding wasn't about DTE at all:** discovery-period (2015-2023) mean expectancy was **negative for every single DTE choice** (-0.6% to -5.1% per trade), despite 61-73% win rates. This is what first revealed the strategy's fundamental expectancy problem, independent of any exit/roll mechanism.

### 3.4 Regime diagnostic: why 2015-2023 and 2024-2026 differ
| metric | discovery (2015-2023) | holdout (2024-2026) |
|---|---|---|
| mean entry ATM IV | 19.5% | 16.0% |
| mean realized vol | 18.2% | 14.4% |
| **VRP (IV − realized vol)** | **+1.3%** | **+1.6%** |
| frac. trades losing >50% of risk | 9.1% | 3.3% |
| frac. trades losing >90% of risk | 2.6% | 0.0% |

**The volatility risk premium — the actual edge this strategy tries to harvest — is essentially unchanged between periods.** What differs is tail-event frequency: discovery ate 5 large-loss trades tied to identifiable market shocks (Aug 2015 crash, Jan 2018 correction, Feb 2019 rally, Dec 2020 rally, Dec 2022 reopening rally); holdout ate none. **Four of the five worst discovery trades were violent upside rallies, not crashes** — the call side got run over, not the put side, cutting against typical crash-risk intuition. One (Jan 2018) blew through the wing with realized vol *below* what was priced in — a smooth, persistent trend, not a vol spike — meaning a realized-vol screen alone wouldn't have flagged it coming.

### 3.5 Wing width (p: 2/3/4/6/8/10 listed strikes past the short)
- p=4 mechanically softens every one of the 5 known disasters (e.g. -99% → -56% on the worst one) — real, verified trade-by-trade, no data gaps.
- **But it doesn't fix the aggregate expectancy:** p=4 discovery mean_r is -1.45%, actually *worse* than p=3's -0.70%, still Kelly-rejected (f=0). Paying for wider protection every month costs credit-efficiency on the ~59 months that were never disasters.
- p=6/p=8 looked much better in-sample (Calmar 0.25 at p=6) — **but this is a data artifact, not edge.** None of the 5 disaster dates could even be *constructed* at p=6 (insufficient strikes listed that day), so that sample got lucky by skipping the disasters via a structural/liquidity constraint, not by surviving them. Checked and ruled out a "recent-years-only" explanation — p=6 trades are spread evenly across 2015-2023.

### 3.6 Compounding
- Bankroll compounding is implemented correctly throughout (`eq = cumprod(1 + f_used*r)` — each trade sized as a fraction of the *already-compounded* bankroll, not a fixed amount).
- Caveat: `f_used` (the Kelly fraction) is fit once, in hindsight, over the entire window being scored — not walk-forward re-estimated as a live trader would have to do. This makes every absolute CAGR/Calmar number in this investigation somewhat optimistic, though it doesn't change any of the relative conclusions above (it affects every tested variant equally).

---

### 3.7 Trend/momentum filter (SSE Composite) — the one lever aimed at the actual mechanism, still failed OOS

§3.4 found the discovery period's 5 disasters were mostly persistent directional grinds, not vol spikes — Jan 2018 in particular had realized vol *below* what was priced, so a vol screen alone couldn't have flagged it. This tests the filter idea flagged as "still open" in the first version of this summary.

**Step 1 — is it predictable at entry?** Before building anything, checked whether a trend-persistence signal *as of each entry date* (Kaufman efficiency ratio — net move / sum of |daily moves|, over trailing 10-60 days on the SSE Composite (000001), deliberately the broad index rather than the 50ETF itself) distinguishes the 6 worst discovery trades from the rest. It doesn't: only 2 of 6 show elevated trailing ER at entry (Feb 2019, May 2017); Jan 2018's is *below* the sample median. **The damage happens after entry, not before** — so this was built as a reactive mid-trade monitor instead of a predictive entry filter: hold the same blind condor, watch the index's trailing ER every day of the trade's life, exit the first day it crosses a threshold (index is grinding harder than usual, checked causally — no lookahead).

**Step 2 — aggregate OOS test** (continuous-chain methodology, same universe §3.4 used to name its disasters; grid: ER window ∈ {10,20,40,60} × threshold ∈ {0.40,0.50,0.60,0.70}, frozen by best discovery mean_r since Kelly rejects most of this grid and collapses CAGR/Calmar to 0/nan — see script docstring):
- Discovery: only 1 of 16 cells beats the (already slightly negative, f=0) baseline at all, and barely (mean_r +0.17% vs baseline −0.70%), at the cost of exiting 69 of 118 trades (58%) early.
- That frozen pick (window=10, threshold=0.70) **fails OOS outright**: holdout mean_r flips from the baseline's genuinely strong +3.39% to **−3.49%** (CAGR +10.14% → 0.00%, Kelly-rejected), because it also exits 32 of 46 holdout trades (70%) early. Confirmed at a second entry config (0.25-delta/p=3): baseline +7.42% mean_r vs frozen rule −3.75%.

**Step 3 — mechanistic check, not just aggregate stats:** does the frozen rule actually engage the 5 disasters it was aimed at? Matched by exact entry date against the continuous chain — yes, decisively, on 4 of 5:

| disaster | baseline r | frozen-rule r |
|---|---|---|
| 2018-01-02 → 01-24 | −95.9% | **−6.7%** |
| 2019-02-12 → 03-27 | −71.8% | **−17.1%** |
| 2015-08-11 → 09-23 | −68.8% | **−43.1%** |
| 2020-12-23 → 2021-01-27 | −64.6% | **−27.5%** |
| 2022-12-28 → 2023-01-30 | −98.6% | (chain re-sequenced around an earlier trigger; no exact-entry analog to compare) |

So the mechanism is real — a persistent grind on the broad index does show up in the trailing ER and does trigger before the worst of the damage. **But it fires on far more than just these 4 trades** — the same low window/threshold that's sensitive enough to catch a smooth grind early is also crossed by ordinary chop constantly, which is exactly why it exits 58-70% of *all* trades, not just the disasters. This is the identical trade-off §3.2's stop-loss roll hit: the insurance only pays off in a window that happens to contain a disaster, and whether a given backtest window does is a matter of sample composition, not a stable edge — here, discovery (5 disasters in 9 years) barely broke even on the trade-off, and holdout (0 disasters in 2.5 years) paid the cost of the false triggers with nothing to show for it.

**Verdict: does not survive OOS, joining every other management lever tested.** The mechanism-targeted design doesn't change the outcome — it's a more sophisticated trigger than a PnL stop-loss, but it's still a *reactive mid-trade* rule on a strategy where most drawdowns recover by expiry (§3.1), so cutting into that recovery process to catch rare disasters costs more than it saves, on average.

*(Separately, `sse_4000_level_read.py` produced a standalone, non-condor read of the SSE Composite around the round-number 4000 level: only 9 genuine crossings — persisting ≥5 trading days, not whipsaws — in the full 2010-2026 history, because the index spent 2015-07-27 to 2025-11-14 (~10.2 years) continuously below it. It's only been back above 4000 since early 2026, and the last ~18 months show 6 genuine crossings back and forth — i.e. real, active chop right at the level, not a decisive break either way as of 2026-07-08 (last close 3970.9, −0.73%). Descriptive only, not a trading signal.)*

### 3.8 Tail hedge (far-OTM put + call bought alongside the condor) — mechanically different, still doesn't help

Every lever above reacts to or modifies the condor's *own* position. A hedge is different in kind: a separate small position bought at entry, sized to pay off exactly when the condor's wings get blown through, with no mid-trade management at all. Tests buying one far-OTM put + one far-OTM call every month (independent monthly universe, same as `condor_delta_backtest.py`; target |delta| grid 0.03/0.05/0.08/0.10/0.15), priced off the same real quoted closes.

**It is not cheap.** Even the smallest, most far-OTM hedge tested (0.03 delta) costs **27% of the condor's own credit, every month** — rising to 105%+ (more than the entire credit collected) at 0.15 delta. SSE 50ETF options carry a real volatility skew: far-OTM strikes are priced richer than flat-vol intuition suggests, precisely because everyone wants that same protection.

- Discovery: every hedge size makes mean_r *worse* than the already-negative unhedged baseline (−0.47% → −4.44% at the cheapest hedge, down to −9.52% at the most expensive).
- The frozen (cheapest, least-drag) pick fails OOS: holdout CAGR +13.58% → +0.00% (Kelly-rejected). Confirmed at a second entry config (+14.72% → 0%).
- **Mechanistic check, worst 5 monthly-universe trades:** the tiny 0.03-delta hedge barely moves anything (too far out to be reached). A bigger, pricier 0.10-delta hedge (72-82% of credit) does meaningfully cut the single worst trade in the dataset (−98.6% → −67.4%) and one other large loss — but on 2 of the same 5 trades it just adds cost with zero payout, because that particular move never got far enough to reach even a 0.10-delta strike.

**Verdict: same trade-off as the stop-loss roll and the trend-exit, just paid for in premium instead of a mid-trade rule** — selective insurance that only pays off in the specific window that contains a real blow-through, and costs money every other month. Does not survive OOS.

### 3.9 Trend-following sleeve (separate position on the SSE Composite, not a rule on the condor) — the crash-alpha thesis, tested and it didn't show up here

A genuinely different idea from everything above: instead of managing or hedging the condor, hold a **separate, independent position** on the SSE Composite that's expected to profit directly in the persistent-grind regimes that hurt the condor (4 of 5 disasters were violent rallies — a sleeve already long into that rally profits from it, rather than merely avoiding/cutting the condor's loss).

Design (time-series momentum, monthly rebalance on the condor's own entry dates, unlevered 1x notional over the condor's own entry→exit window, blended into one portfolio return `w·condor_r + (1-w)·trend_r` before Kelly-sizing):
- **Swept**: momentum lookback {1m, 3m, 6m, 12m} × condor/sleeve weight {100/0 baseline, 90/10, 80/20, 70/30, 60/40, 50/50}.
- **Fixed by convention** (not tuned, to limit an already-bigger 2-parameter search): 0% deadband (plain sign of momentum), no vol-targeting, 0.10% round-trip cost.

- Discovery: every cell lands within a hair of baseline; the "best" (1-month lookback, 50/50) only reaches −0.25% vs −0.47% baseline, and sits at the edge of the tested grid. Notably fragile: more trend allocation helps slightly at 1m/3m/12m lookbacks but *hurts* at 6m — no economic reason for that reversal, a classic overfitting tell.
- Holdout: adding *any* trend allocation, at *every* lookback, made things monotonically worse — the frozen pick drops holdout CAGR from +13.58% to +7.15%. Confirmed at a second entry config (+14.72% → +11.75%).
- **Mechanistic check, worst 5 monthly-universe trades:** the sleeve genuinely profited on only **1 of 5**. On the other 4 — including the single worst trade in the dataset (Dec 2022) — the sleeve *also lost money*, because a trailing-momentum signal was still reading the pre-rally trend right as a sharp, surprising reversal began: the textbook trend-following failure mode (whipsaw at turning points). Most of the apparent "improvement" on those trades is just the 50/50 blend halving the condor's own exposure, not the sleeve actually profiting opposite the loss.

**Verdict: the crash-alpha thesis is directionally sound in principle but didn't materialize with this simple a design, on this instrument/period** — the disasters in this dataset are disproportionately sharp reversals, which is precisely where naive sign-of-momentum trend-following struggles most. Does not survive OOS.

## 4. What's actually established vs. still open

**Established (high confidence, OOS-tested or mechanistically verified):**
- Hold-to-expiry beats every exit/roll modification tested, including the mechanism-targeted trend/momentum mid-trade exit (§3.7).
- Pairing the condor with something structurally different — a far-OTM tail hedge (§3.8) or an independent trend-following sleeve (§3.9) — was tested properly and also failed OOS, so "combine with something else" is not automatically the fix either.
- Every one of these hits variants of the same trade-off: a mechanism that would genuinely have helped the known disasters (stop-loss roll, trend-exit, tail hedge, trend sleeve) is either indiscriminate enough to cost more on ordinary months than it saves on rare ones, or — for the trend sleeve specifically — actively wrong-footed by the fact these disasters are mostly sharp reversals, not clean extended trends.
- VRP (the core edge) is stable across periods; what varies is tail-event incidence, which is low-frequency and clustered, not smoothly distributed.
- Wing width trades disaster severity against per-trade credit efficiency, and p=4's net effect on 2015-2023 aggregate expectancy is flat-to-negative.
- **The construction has negative mean expectancy over the 9-year 2015-2023 window, under every exit/roll/DTE/wing-width/trend-filter/hedge/trend-sleeve variant tested.**

**Still open / not tested:**
- Different entry delta targets (0.10-0.30 were swept pre-OOS-discipline, never re-validated with a proper split).
- Multi-underlying diversification (510300/510500 chains exist locally, unused) — different from the trend sleeve above (that was a directional bet; this would be spreading the *same* condor structure across more, imperfectly-correlated underlyings, reducing idiosyncratic tail risk without introducing a new return stream).
- A less naive trend signal (e.g. one that distinguishes a persistent grind from a sharp reversal, rather than plain sign-of-momentum) might behave differently on the specific reversal-type disasters here — untested, and would need its own honest OOS discipline given how easily this family of ideas overfits a 5-event sample.
- Whether 2024-2026 is representative going forward is fundamentally unknowable from backtesting alone — it's one realization of a low-frequency tail process, not evidence the process changed.

## 5. Recommendation

Don't trade this construction on the strength of the full-sample or 2024-2026 numbers alone — they're propped up by the absence of a tail event that has occurred roughly once every two years historically. Every lever tested — mechanical exit/roll rules, a mechanism-targeted trend exit, a far-OTM tail hedge, and an independent trend-following sleeve — either hurt outright or failed genuine OOS testing. At this point the construction itself — not its management, and not what it's paired with — is the thing that's shown negative baseline expectancy on 9 years of data; further work should treat that as the open question (a different underlying, a different structure, or genuine multi-underlying diversification), rather than searching for another overlay on this one.
