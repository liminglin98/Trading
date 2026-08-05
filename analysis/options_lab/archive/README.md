# Archive

Backtest/investigation scripts moved out of `analysis/options_lab/` on
2026-08-05 because they're not validated (or their strategy was explicitly
rejected/closed) and nothing in the live app depends on them. Each still
runs standalone — a `sys.path.insert(0, ...)` shim at the top of every file
here puts `analysis/options_lab/` back on the path so imports of kept
modules (`data_store`, `quarterly_atm_short_straddle`, etc.) resolve, same
pattern already used by `Data/买断日更/materialize_szse_full_csv.py` and
`Data/exploring/huidian_index_replication/build_index.py`.

| script | verdict | citation |
|---|---|---|
| `calendar_spread_entry_timing_backtest.py` | REJECTED/CLOSED | `project_calendar_spread_investigation_closed.md`: no robust edge anywhere, closed as tested (500ETF result was a trend artifact) |
| `vol_expansion_backspread_backtest.py` | REJECTED/CLOSED | `project_backspread_investigation_closed.md`: no robust edge, closed as tested |
| `quarterly_iron_butterfly_monthly_wings.py` | REJECTED | `project_iron_butterfly_monthly_wings_rejected.md`: dominated by both the naked short straddle and the plain quarterly iron butterfly |
| `quarterly_iron_butterfly.py` | PRELIMINARY, not live | working baseline only, referenced for comparison by the rejected monthly-wings script above; never independently validated |
| `term_structure_signal_test.py` | REJECTED/no-edge | `project_long_straddle_vol_signals.md`: near-month IV level alone predicts as well as the near-far spread; don't build a signal on this |
| `signal_persistence_analysis.py` | REJECTED/no-edge (as a filter) | `project_short_straddle_vol_signals.md`: signal persistence duration is not a useful filter or predictor |
| `td9_backtest.py` | UNCLEAR, never resolved | no memory write-up reaches a conclusion; own docstring calls it "a first gut-check" |
| `iv_spike_directional_backtest.py` | UNCLEAR, never resolved | no memory write-up references this script |
| `vol_rich_short_straddle_stacked_backtest.py` | UNCLEAR, never resolved | read-only research script testing a concurrent-position variant; no recorded conclusion, no `out/` artifacts before this move |
| `iv_percentile_filter.py` | UNCLEAR, never resolved | poses an open question in its own docstring, never answered in memory |
| `vrp_analysis.py` | Foundational diagnostic, not a strategy verdict | own docstring: "treat the sign and rough magnitude as a first-order gut-check, not a tradeable P&L estimate" |

**Not archived despite being PRELIMINARY** (findings are already live in
`app.py`'s Alerts tab): `delta_strangle_short_backtest.py` (hard-imported
by `alerts_core.py` — moving it breaks the app on startup),
`iv_pct_vrp_combo_short_straddle_backtest.py`, and
`iv_spike_vrp_combo_short_straddle_backtest.py` (both reimplemented inline
in `alerts_core.py`, not directly imported, but kept alongside for
discoverability).

A daily scheduled agent re-applies this same triage to any new backtest
script added to `analysis/options_lab/` going forward — see
`project_options_lab_architecture.md`.
