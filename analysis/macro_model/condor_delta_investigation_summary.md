# Delta-Targeted Iron Condor: Investigation Summary & Verdict

**Bottom line: no robust, out-of-sample-defensible edge found.** Every management lever tested (early exit, rolling, entry duration, wing width) either hurt performance outright or looked good only in-sample and failed when tested on data it hadn't seen. The apparent full-history edge is concentrated almost entirely in 2024-2026, which the data suggests was a lucky stretch (no tail events), not a structural improvement. Treat this construction as **unproven, likely negative-expectancy over a representative sample**, not a strategy ready to trade.

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

## 4. What's actually established vs. still open

**Established (high confidence, OOS-tested or mechanistically verified):**
- Hold-to-expiry beats every exit/roll modification tested.
- The stop-loss roll's apparent edge was in-sample overfitting to 5 specific historical tail events, not a stable rule.
- VRP (the core edge) is stable across periods; what varies is tail-event incidence, which is low-frequency and clustered, not smoothly distributed.
- Wing width trades disaster severity against per-trade credit efficiency, and p=4's net effect on 2015-2023 aggregate expectancy is flat-to-negative.
- **The construction has negative mean expectancy over the 9-year 2015-2023 window, under every exit/roll/DTE/wing-width variant tested.**

**Still open / not tested:**
- Different entry delta targets (0.10-0.30 were swept pre-OOS-discipline, never re-validated with a proper split).
- Multi-underlying diversification (510300/510500 chains exist locally, unused).
- A trend/momentum filter that specifically targets the failure mode found in 3.4 (persistent directional grind, not just elevated realized vol) — the one lever aimed at the actual mechanism rather than a generic parameter tweak, and the most promising untested idea from this investigation.
- Whether 2024-2026 is representative going forward is fundamentally unknowable from backtesting alone — it's one realization of a low-frequency tail process, not evidence the process changed.

## 5. Recommendation

Don't trade this construction on the strength of the full-sample or 2024-2026 numbers alone — they're propped up by the absence of a tail event that has occurred roughly once every two years historically. If pursuing this further, the trend/momentum entry filter (targeting the actual failure mode: persistent directional grinds that don't show elevated realized vol) is the one untested idea that attacks the mechanism rather than another parameter on top of a construction that's already shown negative baseline expectancy.
