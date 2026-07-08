# Delta-Targeted Iron Condor: Investigation Summary & Verdict

**Bottom line: no robust, out-of-sample-defensible edge found.** Every management lever tested (early exit, rolling, entry duration, wing width, a mechanism-targeted trend/momentum mid-trade exit) either hurt performance outright or looked good only in-sample and failed when tested on data it hadn't seen. Two attempts to pair the condor with something else entirely — a far-OTM tail hedge (§3.8) and a separate trend-following sleeve on the SSE Composite (§3.9) — fared no better. A fourth idea, skewing the condor's own strikes by trailing momentum direction (§3.10), initially looked like the first genuine win but turned out to be a sample-composition artifact identical in kind to §3.5's wing-width finding. A fifth check — running the identical construction on a different underlying, CSI 300 ETF (510300) options, instead of another overlay on 510050 (§3.11) — also failed to find a robust edge, and its default config didn't even benefit from the 2024-2026 tailwind that flattered 510050. The apparent full-history edge is concentrated almost entirely in 2024-2026, which the data suggests was a lucky stretch (no tail events), not a structural improvement. Treat this construction as **unproven, likely negative-expectancy over a representative sample**, not a strategy ready to trade.

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
- `condor_asymmetric_skew_backtest.py` — skews the condor's own put/call strike selection by trailing momentum direction (§3.10).
- `condor_300etf_backtest.py` — identical construction run on CSI 300 ETF (510300) options instead of the 50ETF (§3.11).

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

### 3.10 Asymmetric / skewed condor (skew strikes by momentum direction) — looked like the first real win, was a sampling artifact

A different idea from §3.7-3.9: instead of a symmetric condor (same |delta| both sides) or a reactive rule/overlay, skew the strike selection itself — sell the side *with* the trailing trend closer to the money (collect more premium there) and push the side *against* it further out (more protection where the risk actually is). `condor_asymmetric_skew_backtest.py` generalizes `condor_delta_backtest.build_condor_by_delta` to independent put/call target deltas.

- **Swept**: skew magnitude {0.00 (=baseline sanity check), 0.03, 0.05, 0.08, 0.10}.
- **Fixed by convention**: direction from 3-month momentum sign only (deadband=0), `MIN_DELTA=0.02` floor so the widened side can't invert past the money.
- Sanity check: skew=0.00 exactly reproduces `condor_delta_backtest.py` (CAGR +0.49% vs +0.49%, n=77 vs 77).

**Naive result — looked like a genuine win:** frozen skew=0.03 beat the holdout baseline (CAGR +16.84% vs +13.58%), and appeared to replicate at a second entry config (+19.74% vs +14.72%) — the first and only variant in this entire investigation to clear that bar.

**It didn't survive scrutiny.** Sample size shrank monotonically with skew magnitude — discovery n: 52→49→43→39→33 as skew went 0.00→0.10; holdout n: 24→21→17→12→11 — because pushing a strike further out by `skew` sometimes runs it off that day's listed strike grid (`build_skewed_condor` can't construct the trade at all, e.g. `ip_l = ip_s - p_steps` going negative), silently dropping that entry date rather than executing at a worse price. This is the exact same failure mode as §3.5's p=6/p=8 wing-width result: a naive comparison implicitly compares different samples, and if the dropped trades are disproportionately bad ones, survivors look better for free. Checked directly: the 3 discovery trades dropped at skew=0.03 had unskewed r of −50.9%, +12.1%, +18.8% (mean −6.7%), far worse than the discovery-wide mean of −0.5% — dropping them mechanically inflates the survivors' average.

**Matched-sample comparison (same entry dates scored under both skew=0 and skew=X — the correct comparison) reverses the naive finding completely:**

| skew | discovery Δmean_r | holdout Δmean_r |
|---|---|---|
| 0.03 | +0.28% (noise) | −0.34% |
| 0.05 | −0.01% | −3.61% |
| 0.08 | −0.04% | −5.90% |
| 0.10 | −0.19% | −6.69% |

Flat-to-negative in discovery, and consistently worse — increasingly so with skew magnitude — in holdout. No genuine edge; the naive result was pure sample-composition artifact.

**Mechanistic check, worst-5 monthly-universe trades:** of the 5, one dropped off the grid entirely at skew=0.03 (can't compare); of the remaining 4, **3 showed literally identical r** — the skewed side was never reached, so the extra protection never mattered. The 2022-12-30 case is instructive: 3-month momentum read bearish (matching the pre-rally gloom), so skew protected the *put* side — but the actual disaster was a surprise reversal rally that ran over the unprotected *call* side. Same lagging-signal-whipsaw-at-reversals failure already identified for the trend sleeve in §3.9. Only 1 of 4 (2023-11-30) showed genuine improvement (−46.9% → −6.5%).

**Verdict: does not survive OOS either, once corrected for sample composition — joins every other lever tested.** Methodologically the most important part of this result is procedural, not substantive: any grid-edge-adjacent variant in this dataset (wider wings, skewed strikes, anything that can silently drop a trade at construction time rather than execute it) must be checked with a matched-sample comparison before trusting an apparent improvement, because the naive comparison is systematically biased toward making survivors look good.

### 3.11 300ETF (510300/CSI 300) standalone condor — a different underlying, not just a different overlay, and it doesn't fare better

Every lever above modifies or pairs with the 510050 condor. This instead asks whether the *instrument itself* is the problem: `condor_300etf_backtest.py` runs the identical construction (monthly entry, target-delta shorts, protective wings, hold-to-expiry-always, real quoted closes, fractional Kelly) on `Data/options/option_daily_510300.csv` — CSI 300 ETF options, a broader-based, less bank/financial-concentrated index than the SSE 50. Reuses `condor_delta_backtest.py`'s pure functions directly (they take a chain DataFrame as an argument and never reference the ticker), so this is a data swap, not a reimplementation.

**Caveat up front:** 510300 options only started listing 2019-12-23, so discovery here is ~4 years / ~48 months (2019-12-23 → 2023-12-31) vs. 510050's 9 years / ~107 months — meaningfully less data to learn from or generalize about. Same `SPLIT_DATE=2024-01-01` used throughout this investigation, so the holdout window (2024-01-01 → 2026-07-08) lines up across underlyings.

**At the established default config (delta=0.20, p=3), 300ETF is flat-to-negative in *both* windows** — it does not get the same holdout lift 510050 did:

| config | discovery mean_r | holdout mean_r |
|---|---|---|
| 510050 delta=0.20/p=3 (§3.7 baseline) | −0.70% | **+3.39%** (CAGR +10.14%) |
| 510300 delta=0.20/p=3 | **−3.03%** (f=0, Kelly-rejected) | **−0.56%** (f=0, Kelly-rejected) |

**The frozen discovery pick by mean_r (delta=0.10, p=3) does look consistent across windows** — discovery +1.17% (n=19), holdout +2.43% (n=17), same sign both times, the only case in this whole investigation where a frozen pick didn't reverse sign OOS. **It does not survive scrutiny either, for the same reason flagged in §3.9 and §3.10:** it's the *sole* Kelly-accepted cell in an 18-cell mostly-negative discovery grid, sits at the extreme edge of the swept delta range (0.10 is the grid boundary), and a one-step neighbor at the identical delta (p=2, n=35 — a larger sample) is strongly *negative* (−1.72%) in the same discovery window — a single-parameter perturbation flipping sign is the exact overfitting tell this investigation has flagged repeatedly. Separately, 14 of the 19 holdout cells across the *whole* grid are positive (not just this one) — consistent with 2024-2026 being a broadly benign period market-wide (as already established for 510050 in §3.4), not with delta=0.10/p=3 specifically having found real edge.

**Descriptive disaster-date check** (300ETF's own worst 5 discovery trades at the default config: 2020-06-30 −100%, 2022-02-28 −56.8%, 2022-12-30 −51.5%, 2020-05-29 −47.1%, 2022-03-31 −27.6%): one of these — 2022-12-30 — lines up almost exactly with 510050's own 2022-12-28 disaster, both the same Dec-2022/Jan-2023 reopening-rally event. The other four don't correspond to any of 510050's 5 known disasters. This means a genuine two-way diversified sleeve (half-size condor on each underlying, not yet built or backtested) would **not** have escaped that one shared event, but could plausibly have softened the others — an open question, not yet tested as an actual blended backtest.

**Verdict: 300ETF is not a way to route around 510050's negative-expectancy finding — if anything it's a weaker candidate on its own** (its default config doesn't even get lifted by the 2024-2026 tailwind that flattered 510050), and the one config that looked OOS-consistent fails the same robustness check that has caught every other false lead in this investigation. Genuine multi-underlying diversification (as opposed to instrument substitution) remains untested and is the one part of this that the disaster-date overlap check leaves genuinely open.

## 4. What's actually established vs. still open

**Established (high confidence, OOS-tested or mechanistically verified):**
- Hold-to-expiry beats every exit/roll modification tested, including the mechanism-targeted trend/momentum mid-trade exit (§3.7).
- Pairing the condor with something structurally different — a far-OTM tail hedge (§3.8), an independent trend-following sleeve (§3.9), or skewing the condor's own strikes by trend direction (§3.10) — was tested properly and also failed OOS, so "combine with something else" is not automatically the fix either.
- Swapping the underlying entirely — CSI 300 ETF (510300) options run through the identical construction (§3.11) — also failed to find a robust edge, and didn't even benefit from the 2024-2026 tailwind that flattered 510050's headline numbers.
- Every one of these hits variants of the same trade-off: a mechanism that would genuinely have helped the known disasters (stop-loss roll, trend-exit, tail hedge, trend sleeve, skewed strikes) is either indiscriminate enough to cost more on ordinary months than it saves on rare ones, or — for the trend sleeve and skewed condor specifically — actively wrong-footed by the fact these disasters are mostly sharp reversals, not clean extended trends.
- VRP (the core edge) is stable across periods; what varies is tail-event incidence, which is low-frequency and clustered, not smoothly distributed.
- Wing width trades disaster severity against per-trade credit efficiency, and p=4's net effect on 2015-2023 aggregate expectancy is flat-to-negative.
- **Any variant that can silently drop a trade at construction time (wide wings, skewed strikes) must be checked with a matched-sample comparison before trusting an apparent improvement** — two separate instances (§3.5, §3.10) of the same naive comparison being reversed by this check. §3.11 found a related tell: a frozen pick sitting at a swept parameter's grid boundary, with a one-step neighbor (larger sample) showing the opposite sign, should not be trusted even when it happens to hold its sign OOS.
- **The construction has negative mean expectancy over the 9-year 2015-2023 window, under every exit/roll/DTE/wing-width/trend-filter/hedge/trend-sleeve/skew variant tested — and the same default config on a different underlying (510300) stays negative-to-flat in both its shorter discovery and holdout windows too.**

**Still open / not tested:**
- Different entry delta targets (0.10-0.30 were swept pre-OOS-discipline, never re-validated with a proper split, on 510050).
- **Genuine two-way diversification (half-size condor on both 510050 and 510300 simultaneously)** — §3.11 ran 510300 as a standalone substitute, not this. Its descriptive disaster-date check found the two chains' worst discovery trades mostly fall on different calendar dates, *except* the Dec-2022/Jan-2023 reopening rally, which hit both — so a blended sleeve would not have escaped that one shared event but might plausibly soften the others. Not yet built or backtested as an actual blended-return series with its own OOS discipline.
- A less naive trend signal (e.g. one that distinguishes a persistent grind from a sharp reversal, rather than plain sign-of-momentum) might behave differently on the specific reversal-type disasters here — untested, and would need its own honest OOS discipline given how easily this family of ideas overfits a 5-event sample.
- Whether 2024-2026 is representative going forward is fundamentally unknowable from backtesting alone — it's one realization of a low-frequency tail process, not evidence the process changed.

## 5. Recommendation

Don't trade this construction on the strength of the full-sample or 2024-2026 numbers alone — they're propped up by the absence of a tail event that has occurred roughly once every two years historically. Every lever tested — mechanical exit/roll rules, a mechanism-targeted trend exit, a far-OTM tail hedge, an independent trend-following sleeve, a skewed/asymmetric condor, and the identical construction on a different underlying (510300) — either hurt outright or failed genuine OOS testing (the skewed condor in particular only *looked* like a win until a matched-sample check reversed it; 510300's one OOS-consistent cell fails the same kind of robustness check). At this point the construction itself — not its management, not what it's paired with, and not which single underlying it trades — is the thing that's shown negative baseline expectancy; further work should treat that as the open question. The one genuinely untested idea left in this family is a real blended two-underlying sleeve (510050 + 510300 simultaneously) rather than substituting one for the other, though the disaster-date overlap already found suggests it would only partially help.
