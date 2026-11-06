# gl_sessions.py
# Goal-location (GL) session labels from the lab's Overview_StudyDay_Condition
# workbooks.
#
# rat_sessions_HC_S1-5.csv merges GL1_S1 / GL2_S1 / ... into "S1", so the
# training order is read from the four workbooks (one tab per rat, SD = date,
# Condition):
#     <sub-group folder>\Rat_HM_Ephys_TD_<TAG>_Overview_StudyDay_Condition.xlsx
# Only GL training days are kept (HC goes through add_condition()). Cached.
#
# Usage in the notebook:
#     from gl_sessions import build_gl_sessions, add_gl_sessions
#     gl = build_gl_sessions(ROOT)                 # cached label table
#     ripples  = add_gl_sessions(ripples,  gl)     # adds columns, keeps originals
#     spindles = add_gl_sessions(spindles, gl)
#     deltas   = add_gl_sessions(deltas,   gl)
import re
from pathlib import Path

import pandas as pd

# which workbook is in which analysis sub-group folder
# (same as ANALYSIS_GROUPS in build_tables.py so the paths match)
CONDITION_FILES = {
    (
        "Rat_HM_Ephys_TD_Analysis_R1_8",
        "R1-4",
    ): "Rat_HM_Ephys_TD_R1_4_Overview_StudyDay_Condition.xlsx",
    (
        "Rat_HM_Ephys_TD_Analysis_R1_8",
        "R5-8",
    ): "Rat_HM_Ephys_TD_R5_8_Overview_StudyDay_Condition.xlsx",
    (
        "Rat_HM_Ephys_TD_Analysis_R9_16",
        "R9-12",
    ): "Rat_HM_Ephys_TD_R9_12_Overview_StudyDay_Condition.xlsx",
    (
        "Rat_HM_Ephys_TD_Analysis_R9_16",
        "R13-16",
    ): "Rat_HM_Ephys_TD_R13_16_Overview_StudyDay_Condition.xlsx",
}

# Matches labels like "GL1_S1", "GL10_S3": capture goal-location and session numbers.
_GL_RE = re.compile(r"^GL(\d+)_S(\d+)$")


def _parse_one_workbook(path):
    """Read every rat tab in one workbook -> long rows (rat, date, gl_session...).

    Only GL training days. Empty frame if the file is missing.
    """
    path = Path(path)
    if not path.exists():
        print(f"  (skip, not found) {path}")
        return pd.DataFrame()

    frames = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        m = re.fullmatch(r"rat(\d+)", sheet.strip(), flags=re.IGNORECASE)
        if not m:
            print(f"    (skip tab, unexpected name) {path.name} :: {sheet}")
            continue
        rat = int(m.group(1))

        df = xl.parse(sheet)
        if "SD" not in df.columns or "Condition" not in df.columns:
            print(f"    (skip tab, missing SD/Condition) {path.name} :: {sheet}")
            continue

        d = df[["SD", "Condition"]].copy()
        d["Condition"] = d["Condition"].astype(str).str.strip()

        # keep only GL training rows
        gl = d[d["Condition"].str.match(_GL_RE)].copy()
        if gl.empty:
            continue

        gl["rat"] = rat
        gl["date"] = pd.to_datetime(gl["SD"].astype(int).astype(str), format="%Y%m%d")
        gl["gl_session"] = gl["Condition"]
        nums = gl["gl_session"].str.extract(_GL_RE).astype(int)
        gl["gl_num"] = nums[0]  # goal-location number
        gl["session_num"] = nums[1]  # session within that goal location
        frames.append(gl[["rat", "date", "gl_session", "gl_num", "session_num"]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_gl_sessions(root, cache_dir="wave_cache", rebuild=False):
    """Build (or load) the GL-session label table for all rats.

    Cached to <cache_dir>/gl_sessions.pkl; rebuild=True reads the drive again.

    Columns: rat, date, gl_session, gl_num, session_num, cohort, gl_order
    (gl_order = training day number per rat, by date).
    """
    root = Path(root)
    cache = Path(cache_dir) / "gl_sessions.pkl"

    if cache.exists() and not rebuild:
        return pd.read_pickle(cache)

    parts = []
    for (analysis, sub), fname in CONDITION_FILES.items():
        path = root / analysis / sub / fname
        part = _parse_one_workbook(path)
        if not part.empty:
            parts.append(part)

    if not parts:
        print("No GL session data found. Check ROOT and that the drive is mounted.")
        return pd.DataFrame()

    gl = pd.concat(parts, ignore_index=True)

    # cohort label, matching build_tables.py convention
    gl["cohort"] = gl["rat"].apply(lambda r: "rats1-8" if r <= 8 else "rats9-16")

    # de-duplicate: one label per rat+date (a date should be one condition)
    dup = gl.duplicated(subset=["rat", "date"], keep=False)
    if dup.any():
        print(f"  note: {dup.sum()} rat/date rows had >1 GL label; keeping first each")
        gl = gl.drop_duplicates(subset=["rat", "date"], keep="first")

    # chronological order index per rat (by date)
    gl = gl.sort_values(["rat", "date"]).reset_index(drop=True)
    gl["gl_order"] = gl.groupby("rat").cumcount() + 1

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    gl.to_pickle(cache)
    print(
        f"GL sessions: {len(gl)} training days across {gl['rat'].nunique()} rats "
        f"-> cached at {cache}"
    )
    return gl


def add_gl_sessions(event_df, gl_table):
    """Add the GL-session columns to an event table (existing columns untouched).

    Adds gl_session, gl_num, session_num, gl_order (NaN for non-training days).
    """
    if event_df.empty or gl_table.empty:
        return event_df
    cols = ["rat", "date", "gl_session", "gl_num", "session_num", "gl_order"]
    return event_df.merge(gl_table[cols], on=["rat", "date"], how="left")
