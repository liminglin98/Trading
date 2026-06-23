# Lead-Lag Test (HFI → QVIX): Data Diagnosis & Design Memo

**Decision gate:** does the HFI shock carry information that moves implied volatility, before we spend effort on a butterfly backtest?

This memo covers what's actually in the folder, why the regression can't yet be run as specified, the two econometric design choices you flagged (announcement-only vs zero-fill), the confound to watch, and how a significant β maps to the butterfly trigger. The runnable pipeline is `lead_lag_hfi_qvix.py` in the same folder.

---

## 1. Two blockers up front

**Compute environment.** The sandboxed Linux VM that would run `statsmodels`/`matplotlib` failed to start this session, so I could not execute and hand you numbers. The script is written to run in your local Python 3.13 env exactly as-is (`python lead_lag_hfi_qvix.py`), writing tables and PNGs to `analysis/out/`.

**The HFI shock in the folder is monthly, not a daily intraday surprise.** This is the substantive issue, and it matches your hunch that "the HFI shocks probably won't work" yet.

What's present:

| File | What it actually is | Usable as the HFI shock? |
|---|---|---|
| `Data/derived/hfi_core_data.csv` → `shock_policy`, `shock_policy_change` | **Monthly** policy-rule residual (confirmed by `fr007_policy_rule_coefficients.csv`: a monthly regression on `FR007_l1`, `cpi_gap`, `fx_gap`…). One value per month, dated to month-start. | No — wrong frequency |
| `Data/derived/hfi_policy_macro_overlap.csv` | 14 actual announcement dates (2007–2023), **no magnitudes**; a list of days where a policy event overlaps a macro release. | No — it's a confound flag, not a shock |
| `Data/raw/cache/fr007_daily.csv` | Daily FR007 **levels** (FR001/007/014, FDR…). | Raw input only — needs an expectations model to become a surprise |
| `qmt_export/qvix_*.csv` | Daily QVIX OHLC, 2015-02-09 → 2026-06-22 (~2,751 rows). | Yes — clean dependent variable |

**Why monthly breaks the spec.** Your regression is a daily event study: ΔIV over k = 1, 2, 3, 5, 10 *trading days* after a shock at ~9:20 AM. A monthly residual has no trading-day timestamp and no announcement window. Forcing it onto a daily grid (which the proxy mode does, mapping each month's shock to the first trading day on/after month-start) is a **plumbing smoke test**, not a result — the "shock" is spread over a whole month and the timing is arbitrary. Read proxy-mode output only as confirmation that the merge/regression/plot pipeline works end-to-end.

---

## 2. What the real daily HFI file should contain

When you produce it, save it as `Data/derived/hfi_daily_shock.csv` and set `HFI_MODE = "external"`. Minimum schema:

```
date,hfi_shock
2017-12-14,0.041
2018-04-17,-0.022
...
```

- **`date`** = the PBoC announcement trading day (the script uses that day's QVIX close as the t anchor).
- **`hfi_shock`** = signed FR007 surprise. Document the units in your thesis (bp vs. standardized); the β interpretation depends on it. Standardizing to unit variance makes β directly comparable across horizons and is what the strategy-mapping section assumes.
- One row per announcement day. Days with no announcement simply don't appear (the script handles zero-fill internally for the robustness arm).

Everything downstream is identical between proxy and external mode, so the moment the file lands you get the real analysis with no other changes.

---

## 3. Sample choice: announcement-only vs. zero-fill (the CEF framing)

You asked for the tradeoff in conditional-expectation / OVB / measurement-error terms rather than measure-theoretic ones.

**Announcement-only.** Restrict the sample to the ~N announcement days and regress ΔIV on the shock there. You're estimating `E[ΔIV | shock, announcement day]`. Clean interpretation, no contamination from days where nothing happened, but small N → wide Newey-West bands and weak power. This is the right *primary* spec — it's the CEF you actually care about (IV response *conditional on a policy event*).

**Zero-fill.** Keep every trading day, set shock = 0 off announcement days. Now you're estimating `E[ΔIV | shock]` over the whole calendar. Two consequences worth being explicit about:

- The intercept and any constant now absorb the unconditional drift of IV on *all* days, so β is identified off the contrast between announcement and non-announcement days — a difference-in-conditional-means, which is fine but a different estimand.
- The zeros are **not** measurement error in the classical sense (they're structurally true: no announcement, no surprise), so they don't attenuate β toward zero the way noisy mismeasurement of a real regressor would. What they *do* is inflate the effective sample with uninformative rows and can shrink standard errors in a way that flatters significance. Don't read a tighter zero-fill band as "more evidence."

**Recommendation:** report announcement-only as the headline, zero-fill as a robustness cross-check. If β flips sign or significance between them, that's a red flag the result is an artifact of sample construction, not a stable IV response. The script runs both and prints them side by side.

---

## 4. The confound to watch (omitted-variable, concretely)

`hfi_policy_macro_overlap.csv` exists precisely because several PBoC announcement days **coincide with major macro releases** (industrial value-added, retail sales, GDP, FAI). On those days, ΔIV reflects both the policy surprise *and* the data surprise. If the data surprise is correlated with the policy surprise (it usually is — PBoC reacts to the same data), then β picks up an omitted variable and over- or under-states the pure policy channel.

Two defenses, in order of preference:

1. **Control for the macro surprise** on overlap days (add a column; the script's `regressors()` is built to accept extra controls). Requires a macro-surprise series you don't yet have.
2. **Drop or flag the 14 overlap days** as a robustness subsample and check β is stable without them. Cheap; do this first.

This is the same logic as your thesis's narrative-component caveat: the lead-lag test is allowed to use the contaminated shock because it's only validating information content, but you should still know how much of β is policy vs. coincident data.

---

## 5. Econometric choices baked into the script

- **HAC lags.** Overlapping k-day forward returns are mechanically MA(k−1): adjacent observations share days, so residuals are serially correlated *by construction*. The script sets Newey-West `maxlags = max(rule-of-thumb, k−1)`, where the rule-of-thumb is the standard automatic `floor(4·(n/100)^(2/9))`. This is the single most important detail — naive OLS SEs here would be badly understated.
- **Controls.** Lagged 1-day ΔIV (AR term, for IV momentum), lagged log QVIX level (mean reversion — IV is bounded and mean-reverting, so the level predicts the change), and an optional global-risk log level if you drop a VIX file in. All toggleable in CONFIG.
- **Asymmetry.** Splits the shock into positive/negative parts (tightening vs. easing surprises) with separate β's — vol often reacts more to hawkish surprises.
- **Structural break.** Pre/post 2022-01-01 subsamples, matching your thesis's break. If β is only significant in one regime, the trigger must be regime-aware or you're trading a relationship that no longer holds.

---

## 6. If β is significant: mapping to the butterfly trigger

A long butterfly profits when realized vol stays inside the wings (IV stable/falling); a short butterfly profits on a breakout (IV rising). So the *sign and size of the conditional IV drift after a shock* tells you which way to lean.

Concretely, the script's `strategy_mapping()` does this:

1. Pick the most significant horizon k\*.
2. Compute the expected ΔIV for a 1-standard-deviation shock: `β · σ_shock`, converted to a QVIX % move.
3. **Set the regime trigger on the fitted ΔIV exceeding the regression's residual σ — not on the raw shock.** This puts the threshold in the IV units the strategy actually trades, and automatically scales the trigger to how noisy the relationship is. A shock that predicts an IV move smaller than the residual noise isn't a tradeable signal even if β is "significant."
4. **Gate on subsample stability.** Only wire the trigger live if β survives pre/post-2022. The whole point of the gate is to avoid building a switch on a dead regime.

Magnitude sanity check you'll want in the thesis: if a 1-sd policy surprise moves QVIX by, say, 1–2%, that's small relative to typical butterfly P&L drivers — it would argue for the shock as a *regime tilt / position-sizing input*, not a standalone entry signal. If it's 5%+, it can anchor the entry. The script prints the number so you can make that call against your contract economics.

---

## 7. Reminder on look-ahead

This shock has a narrative component and is **not** real-time constructible — it stays in signal-validation only. The backtest must use the real-time version. Nothing here should leak into a tradeable rule; the deliverable of this gate is a yes/no on information content plus the magnitude above.

---

## Next steps

1. Run `python lead_lag_hfi_qvix.py` locally in proxy mode to confirm the pipeline executes and the plots look sane (output in `analysis/out/`).
2. Build the daily FR007-surprise file (`Data/derived/hfi_daily_shock.csv`) — this is the real unblock.
3. Flip `HFI_MODE = "external"`, rerun, and read β(k), the robustness panel, and the strategy mapping as the actual decision gate.
4. Optionally add a macro-surprise control to clean the overlap-day confound.
