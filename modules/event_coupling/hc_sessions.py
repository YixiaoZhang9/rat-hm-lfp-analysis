# hc_sessions.py
# Reads the HC / training day grid and adds goal locations to the home-cage days.
#
# Session gaps (deck, slide 4):
#     Rats 1-4:  S1 . 7d  . S2 . S3 . 48h . S4 . S5
#     Rats 5-8:  S1 . 48h . S2 . S3 . 7d  . S4 . S5
#     Rats 9-16: S1 . 48h/7d . S2 . S3
#     S1 = new goal location, S2/S4 = probe, S3/S5 = normal
#
# For every HC day: goal location, previous and next session, and the
# retention delay when the next session is a probe.
#
# Source file: grid with one column per date and one row per condition, the
# cell says which rat group did it:
#
#                 22.9.2022  23.9.2022  27.9.2022  29.9.2022
#     HC          R1-8       R1-8       R5-8       R5-8
#     Maze hab                          R1-4
#     S1                                           R1-4
#
# Two blocks, one per cohort, each with its own date row.
#
# Usage:
#     from hc_sessions import build_day_table, attach_hc_context
#     days = build_day_table("Rats_1-16__HC_vs_Training_days__.xlsx")
#     hc   = attach_hc_context(days)          # HC rows with GL context
#     events = events.merge(hc, on=["rat", "date"], how="left")

import re

import numpy as np
import pandas as pd

# Where each cohort's block sits in the sheet (0-indexed rows).
BLOCKS = {
    "rats1-8": {"conditions": range(3, 11), "dates": 13},
    "rats9-16": {"conditions": range(17, 25), "dates": 27},
}

# Rat-group labels -> the individual rats they cover.
GROUPS = {
    "R1-8": range(1, 9),
    "R1-4": range(1, 5),
    "R5-8": range(5, 9),
    "R9-16": range(9, 17),
    "R9-12": range(9, 13),
    "R13-16": range(13, 17),
}

# days with no data on the server (Yixiao, 5 Aug 2026), ignored
EXCLUDE = [
    (range(1, 9), pd.Timestamp("2022-09-22")),
    (range(9, 17), pd.Timestamp("2023-07-07")),
]

# session types (deck, slide 4); the update cohort stops after S3
SESSION_TYPE = {1: "NGL", 2: "probe", 3: "normal", 4: "probe", 5: "normal"}

# Condition label in the sheet -> the label used everywhere else.
CONDITION_MAP = {
    "hc": "HC",
    "maze hab": "Maze_Hab",
    "novelty": "Novelty",
    "s1": "S1",
    "s2": "S2",
    "s3": "S3",
    "s4": "S4",
    "s5": "S5",
}


def _parse_date(v):
    """Sheet dates are datetimes in one block and D.M.YYYY strings in the other."""
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, "year"):
        return pd.Timestamp(v).normalize()
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return pd.NaT
    for fmt in ("%d.%m.%Y", "%d.%m.%y"):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt)).normalize()
        except (ValueError, TypeError):
            pass
    try:
        return pd.Timestamp(pd.to_datetime(s, dayfirst=True)).normalize()
    except (ValueError, TypeError):
        return pd.NaT


def _expand(label):
    """'R1-4' -> [1, 2, 3, 4]. Unknown labels return an empty list."""
    key = str(label).strip().upper().replace(" ", "")
    if key in GROUPS:
        return list(GROUPS[key])
    m = re.fullmatch(r"R(\d+)-(\d+)", key)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    m = re.fullmatch(r"R(\d+)", key)
    if m:
        return [int(m.group(1))]
    return []


def build_day_table(xlsx_path, sheet=0, verbose=True):
    """Grid -> long table: one row per rat per recording day.

    Columns: rat, date, condition, cohort.
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)

    rows = []
    unknown = set()
    for cohort, spec in BLOCKS.items():
        date_row = spec["dates"]
        for col in range(1, raw.shape[1]):
            date = _parse_date(raw.iloc[date_row, col])
            if pd.isna(date):
                continue
            for r in spec["conditions"]:
                cell = raw.iloc[r, col]
                if pd.isna(cell):
                    continue
                label = str(raw.iloc[r, 0]).strip().lower()
                cond = CONDITION_MAP.get(label)
                if cond is None:
                    unknown.add(label)
                    continue
                for rat in _expand(cell):
                    rows.append((rat, date, cond, cohort))

    days = pd.DataFrame(rows, columns=["rat", "date", "condition", "cohort"])

    if unknown and verbose:
        print(f"  note: unrecognised condition labels ignored: {sorted(unknown)}")

    # drop the days Yixiao flagged as having no data on the server
    before = len(days)
    for rats, date in EXCLUDE:
        days = days[~(days["rat"].isin(rats) & (days["date"] == date))]
    if verbose and before != len(days):
        print(f"  excluded {before - len(days)} rows for days with no server data")

    days = days.drop_duplicates().sort_values(["rat", "date"]).reset_index(drop=True)
    if verbose:
        print(f"day table: {len(days)} rat-days, {days['rat'].nunique()} rats")
    return days


def add_session_type(days):
    """Label each training day NGL / probe / normal, from the deck's design.

    S1 = new goal location, S2 and S4 = probe trial, S3 and S5 = normal.
    Non-training rows get NaN.
    """
    d = days.copy()
    n = d["condition"].str.extract(r"^S(\d)$")[0]
    d["session_num"] = pd.to_numeric(n, errors="coerce")
    d["session_type"] = d["session_num"].map(SESSION_TYPE)
    return d


def attach_hc_context(days, gl_table=None, verbose=True):
    """For every HC day, work out where it sits relative to training.

    Adds, per HC day:
      hc_phase          'pre_training' before any session, else 'within_block'
      follows_gl        goal location of the most recent training session
      follows_session   session number of that session (1..5)
      follows_type      NGL / probe / normal, type of that session
      next_session      session number of the next training session
      next_type         NGL / probe / normal, type of the session after this sleep
      days_since        days since the previous session
      days_until        days until the next session
      gap_days          length of the gap this HC day sits in
      gap_type          'short' (<= 3 d), 'long' (>= 5 d), or 'other'
      retention_delay   '48h' or '7d' when next_type == 'probe', else NaN
      is_consolidation  True when the next session is a probe
      hc_index_in_gap   1st, 2nd ... HC day inside that same gap

    gl_table : optional output of gl_sessions.build_gl_sessions (otherwise the
        goal location is guessed from where the session counter restarts).
    """
    d = days.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()

    train = d[d["condition"].str.fullmatch(r"S\d")].copy()
    train["session_num"] = train["condition"].str[1].astype(int)

    if gl_table is not None and len(gl_table):
        g = gl_table[["rat", "date", "gl_num", "session_num"]].copy()
        g["date"] = pd.to_datetime(g["date"]).dt.normalize()
        train = train.drop(columns=["session_num"]).merge(
            g, on=["rat", "date"], how="left"
        )
    else:
        # fall back: a new goal location starts where the counter stops rising
        train = train.sort_values(["rat", "date"])
        train["gl_num"] = train.groupby("rat")["session_num"].transform(
            lambda s: (s.diff() <= 0).cumsum() + 1
        )

    train = train.sort_values(["rat", "date"])
    hc = d[d["condition"] == "HC"].sort_values(["rat", "date"]).copy()

    out = []
    for rat, h in hc.groupby("rat"):
        t = train[train["rat"] == rat]
        h = h.copy()

        if t.empty:
            h["hc_phase"] = "pre_training"
            for c in [
                "follows_gl",
                "follows_session",
                "next_session",
                "days_since",
                "days_until",
                "gap_days",
            ]:
                h[c] = np.nan
            out.append(h)
            continue

        tdates = t["date"].to_numpy()
        tsess = t["session_num"].to_numpy()
        tgl = t["gl_num"].to_numpy()

        prev = np.searchsorted(tdates, h["date"].to_numpy(), side="right") - 1
        ok = prev >= 0
        pi = np.clip(prev, 0, None)

        h["hc_phase"] = np.where(ok, "within_block", "pre_training")
        h["follows_gl"] = np.where(ok, tgl[pi], np.nan)
        h["follows_session"] = np.where(ok, tsess[pi], np.nan)
        h["days_since"] = np.where(
            ok, (h["date"].to_numpy() - tdates[pi]) / np.timedelta64(1, "D"), np.nan
        )

        nxt = prev + 1
        has_next = nxt < len(tdates)
        ni = np.clip(nxt, None, len(tdates) - 1)

        h["next_session"] = np.where(has_next, tsess[ni], np.nan)
        h["days_until"] = np.where(
            has_next,
            (tdates[ni] - h["date"].to_numpy()) / np.timedelta64(1, "D"),
            np.nan,
        )

        h["gap_days"] = np.where(
            ok & has_next, h["days_since"] + h["days_until"], np.nan
        )
        out.append(h)

    hc = pd.concat(out, ignore_index=True)

    hc["follows_type"] = hc["follows_session"].map(SESSION_TYPE)
    hc["next_type"] = hc["next_session"].map(SESSION_TYPE)

    hc["gap_type"] = np.select(
        [hc["gap_days"] <= 3, hc["gap_days"] >= 5], ["short", "long"], default="other"
    )
    hc.loc[hc["gap_days"].isna(), "gap_type"] = np.nan

    # label with the planned delay, not the actual gap (a few days are 4-5 d)
    hc["is_consolidation"] = hc["next_type"] == "probe"
    hc["retention_delay"] = pd.Series(pd.NA, index=hc.index, dtype="object")
    hc.loc[hc["is_consolidation"] & (hc["gap_days"] <= 3), "retention_delay"] = "48h"
    hc.loc[hc["is_consolidation"] & (hc["gap_days"] >= 5), "retention_delay"] = "7d"

    # several HC days can be in the same gap, number them
    hc = hc.sort_values(["rat", "date"])
    hc["hc_index_in_gap"] = (
        hc.groupby(["rat", "follows_gl", "follows_session"], dropna=False).cumcount()
        + 1
    )

    if verbose:
        n_pre = (hc["hc_phase"] == "pre_training").sum()
        print(
            f"HC days: {len(hc)} total | {n_pre} pre-training | "
            f"{len(hc) - n_pre} within a block"
        )
        print(
            f"  consolidation windows (next session is a probe): "
            f"{int(hc['is_consolidation'].sum())}"
        )
        print(hc["retention_delay"].value_counts(dropna=False).to_string())

    return hc.reset_index(drop=True)
