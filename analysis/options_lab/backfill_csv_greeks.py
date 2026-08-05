"""
Fill missing Greeks (Delta, Theta, Gamma, Vega, Rho) in
Data/options/option_daily_{code}.csv via Black-Scholes, using each row's own
IV (real Sina-quoted where present, BS-backfilled by backfill_csv_iv.py
otherwise).

Cross-validated first (2026-07-28) against each underlying's own real
quoted Greeks before trusting this: sampled ~300 liquid, non-stale rows per
underlying (real IV in a sane 3-60% band, DTE>5, volume>20), computed all
five Greeks from the row's own (S, K, T, iv) via bs_iv.py, and compared to
the real columns.

Result: 510050/510300 (mature, long-running feed) match excellently on
BOTH call and put sides (corr >0.99 for all five Greeks, tiny errors) --
plain Black-Scholes, no rescaling, reproduces Sina's real quotes almost
exactly. 510500/588000/588080 (extended to the live fetcher 2026-07-24) are
different: PUT-side Greeks validate just as well (corr 0.98-0.998), but
CALL-side Gamma/Theta show a real, systematic ~25% understatement relative
to what their own reported IV implies (correlation still fairly high for
588000/588080, ~0.98, just a consistent scale bias; markedly worse for
510500, corr ~0.68) -- pointing at an inconsistency in Sina's own reported
call-side IV/Greeks for these three specifically (puts on the same
contracts/dates/spot validate fine, so it isn't a formula problem). Per
2026-07-28 decision: only fill PUT-side Greeks for those three; leave
call-side Greeks as a known, documented gap rather than fill with a method
that doesn't validate cleanly. 510050/510300 get both sides filled.
"""
import pandas as pd

import bs_iv
import data_store as ds

# Underlyings where BOTH call and put Greeks validated well enough to fill.
FULL_TRUST = {"510050", "510300"}
# The rest (510500, 588000, 588080): puts only -- see module docstring.
GREEK_COLS = [ds.C_DELTA, ds.C_GAMMA, ds.C_THETA, ds.C_VEGA, ds.C_RHO]


def _row_greeks(cp: str, s: float, k: float, t: float, iv: float) -> dict:
    return {
        ds.C_DELTA: bs_iv.bs_delta(cp, s, k, t, iv),
        ds.C_GAMMA: bs_iv.bs_gamma(s, k, t, iv),
        ds.C_THETA: bs_iv.bs_theta(cp, s, k, t, iv),
        ds.C_VEGA: bs_iv.bs_vega(s, k, t, iv),
        ds.C_RHO: bs_iv.bs_rho(cp, s, k, t, iv),
    }


def fill_missing_greeks(df: pd.DataFrame, code: str) -> tuple[pd.DataFrame, dict]:
    trusted_cp = {"C", "P"} if code in FULL_TRUST else {"P"}
    filled = {c: 0 for c in GREEK_COLS}

    any_missing = df[GREEK_COLS].isna().any(axis=1)
    candidates = df[any_missing & df[ds.C_CP].isin(trusted_cp)]

    for idx, row in candidates.iterrows():
        s, k, dte, iv = row[ds.C_UND_CLOSE], row[ds.C_STRIKE], row[ds.C_DTE], row[ds.C_IV]
        if pd.isna(s) or pd.isna(k) or pd.isna(dte) or pd.isna(iv) or dte <= 0 or iv <= 0:
            continue
        t = dte / 365.0
        computed = _row_greeks(row[ds.C_CP], s, k, t, iv)
        for col, val in computed.items():
            if pd.isna(row[col]) and val == val:  # missing and computable (not NaN)
                df.at[idx, col] = val
                filled[col] += 1

    return df, filled


def main():
    targets = list(ds.UNDERLYINGS) + list(ds.EXT_LIVE_UNDERLYINGS)
    for code in targets:
        path = ds.chain_path(code)
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype={ds.C_CODE: str})
        raw[ds.C_DATE] = pd.to_datetime(raw[ds.C_DATE])
        for col in (ds.C_STRIKE, ds.C_DTE, ds.C_UND_CLOSE, ds.C_IV, *GREEK_COLS):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        before = {c: raw[c].isna().sum() for c in GREEK_COLS}
        raw, filled = fill_missing_greeks(raw, code)
        after = {c: raw[c].isna().sum() for c in GREEK_COLS}

        scope = "both C/P" if code in FULL_TRUST else "P only, C left as known gap"
        print(f"{code} ({scope}):")
        for c in GREEK_COLS:
            print(f"  {c}: {before[c]:,} missing -> {after[c]:,} missing ({filled[c]:,} filled)")

        raw[ds.C_DATE] = raw[ds.C_DATE].dt.strftime("%Y-%m-%d")
        raw.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
