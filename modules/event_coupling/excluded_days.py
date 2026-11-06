# excluded_days.py
# Bad recording days (black cells in the overview workbook, Yixiao's review).
# Applied in build_tables.build_wave, so every notebook uses the same list.
# Dates DD/MM/YYYY, ranges include both ends. Restart the kernel after editing.

import pandas as pd

# rat -> list of single dates and/or (first, last) inclusive ranges, DD/MM/YYYY
EXCLUDED = {
    # Yixiao, 21 August 2026
    1: [
        ("15/11/2022", "18/11/2022"),
        ("21/11/2022", "25/11/2022"),
        "28/11/2022",
        "29/11/2022",
        "01/12/2022",
        ("05/12/2022", "09/12/2022"),
        "13/12/2022",
    ],
    3: ["13/12/2022"],
    # rat 7: the same days as rat 1, plus an earlier week
    7: [
        ("07/11/2022", "11/11/2022"),
        ("15/11/2022", "18/11/2022"),
        ("21/11/2022", "25/11/2022"),
        "28/11/2022",
        "29/11/2022",
        "01/12/2022",
        ("05/12/2022", "09/12/2022"),
        "13/12/2022",
    ],
}


def _expand(spec):
    """One rat's spec -> a set of Timestamps. Ranges are inclusive."""
    out = set()
    for item in spec:
        if isinstance(item, tuple):
            lo, hi = (pd.to_datetime(x, format="%d/%m/%Y") for x in item)
            out.update(pd.date_range(lo, hi, freq="D"))
        else:
            out.add(pd.to_datetime(item, format="%d/%m/%Y"))
    return out


def excluded_table():
    """Table of every excluded rat-day."""
    rows = [(rat, d) for rat, spec in EXCLUDED.items() for d in sorted(_expand(spec))]
    return pd.DataFrame(rows, columns=["rat", "date"])


def drop_excluded(df, verbose=True):
    """Remove excluded rat-days from an event table.

    Matches on rat + calendar date (string or datetime).
    """
    if df.empty or "rat" not in df.columns or "date" not in df.columns:
        return df

    dates = pd.to_datetime(df["date"]).dt.normalize()
    rats = df["rat"].astype("int64")

    mask = pd.Series(False, index=df.index)
    for rat, spec in EXCLUDED.items():
        mask |= (rats == rat) & dates.isin(_expand(spec))

    if verbose and mask.any():
        n_days = df.loc[mask, ["rat", "date"]].drop_duplicates().shape[0]
        print(f"  excluded {int(mask.sum()):,} rows from {n_days} bad recording days")

    return df[~mask]
