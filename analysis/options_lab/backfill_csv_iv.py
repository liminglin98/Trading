"""
Fill missing 隐含波动率 (C_IV) in Data/options/option_daily_{code}.csv.

Missing IV comes from two places: pre-existing gaps going back to 2015
(Sina doesn't publish IV for some illiquid/near-expiry rows) and the
2026-07-28 backfill from Data/买断日更/consolidated/ (vendor B has no IV
column at all). The Technical tab's vol-signal alerts for 510050/510300
(iv_low_long_straddle_backtest -> backtest.py's _atm_iv_series) read C_IV
directly from these CSVs, so gaps here are gaps in the live alert signal's
history, not just cosmetic.

Method: same Black-Scholes inversion + call/put averaging already validated
for the 买断日更 consolidated dataset (consolidated_store._cp_avg_iv, see
Data/买断日更/README.md and the reference_data_sources memory -- call-only
inversion has a real, small bias (+0.3-1.1 vol points); averaging both legs
at the same (date, expiry, strike) cancels most of it, cutting median error
to ~0.17 vol points near the money). Difference from that dataset: prefer
each leg's own *real* exchange-quoted IV when it's present (these CSVs have
real IV on most days) and only fall back to BS inversion for whichever leg
actually lacks it -- more accurate than inverting both legs from scratch
the way consolidated_store does (it has no real IV column to prefer at
all).

Rows with no DTE (expire_date never resolved -- true for a fraction of the
2026-07-28 backfilled rows whose matching source didn't carry expiry) can't
be inverted and are left blank, not guessed.
"""
import pandas as pd

import bs_iv
import data_store as ds


def _leg_iv(row) -> float:
    """Real IV if present, else BS-inverted from price. NaN if neither
    possible (missing DTE/price/spot)."""
    if pd.notna(row[ds.C_IV]) and row[ds.C_IV] > 0:
        return float(row[ds.C_IV])
    dte, close, spot = row[ds.C_DTE], row[ds.C_CLOSE], row[ds.C_UND_CLOSE]
    if pd.isna(dte) or pd.isna(close) or pd.isna(spot) or close <= 0 or dte <= 0:
        return float("nan")
    return bs_iv.implied_vol(row[ds.C_CP], float(spot), float(row[ds.C_STRIKE]),
                             float(dte) / 365.0, float(close))


def fill_missing_iv(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    missing_mask = df[ds.C_IV].isna()
    if not missing_mask.any():
        return df, 0

    by_key = df.groupby([ds.C_DATE, ds.C_EXPIRY, ds.C_STRIKE])
    filled = 0
    new_iv = df[ds.C_IV].copy()

    for (date, expiry, strike), grp in by_key:
        if not missing_mask.loc[grp.index].any():
            continue
        ivs = []
        for _, row in grp.iterrows():
            iv = _leg_iv(row)
            if iv == iv:  # not NaN
                ivs.append(iv)
        if not ivs:
            continue
        avg_iv = sum(ivs) / len(ivs)
        for idx in grp.index:
            if missing_mask.loc[idx]:
                new_iv.loc[idx] = avg_iv
                filled += 1

    df[ds.C_IV] = new_iv
    return df, filled


def main():
    targets = list(ds.UNDERLYINGS) + list(ds.EXT_LIVE_UNDERLYINGS)
    for code in targets:
        path = ds.chain_path(code)
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype={ds.C_CODE: str})
        raw[ds.C_DATE] = pd.to_datetime(raw[ds.C_DATE])
        for col in (ds.C_STRIKE, ds.C_CLOSE, ds.C_DTE, ds.C_UND_CLOSE, ds.C_IV):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        before = raw[ds.C_IV].isna().sum()
        raw, filled = fill_missing_iv(raw)
        after = raw[ds.C_IV].isna().sum()
        print(f"{code}: {before:,} missing -> {after:,} missing "
              f"({filled:,} filled, {after:,} left blank -- no usable DTE/price/spot)")

        raw[ds.C_DATE] = raw[ds.C_DATE].dt.strftime("%Y-%m-%d")
        raw.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
