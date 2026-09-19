"""
compute_spindle_ar_r_values.py

Goal
----
We already have spindle *timestamps* from the wavelet detector (the
Spindle_detection_results/.../chanX[_trial]_spindles_wavelet.csv files).
For each already-detected spindle we:

    1. finds the matching raw .mat file via `TaskLoader` / tasks_manifest.csv
    2. bandpass-filters + downsamples that raw channel once
    3. slices out exactly the [start_time, end_time] window for that spindle
       (with optional padding)
    4. fits a single Burg AR model to that window and pulls out the r-value /
       peak frequency in the spindle band (reusing `_fit_window` from
       modules.find_spindles_lfp_o_quality).

Output: one row per spindle event (all your original wavelet columns +
ar_r_value / ar_peak_freq_hz), plus a correlation/summary report.

READ THE CONFIG BLOCK BELOW BEFORE RUNNING. Channel matching between manifest
tasks and spindle CSVs is derived from `data_path`'s filename (chanN.mat /
chanN_trial.mat, per build_manifest.py) -- this field is guaranteed present
in every task dict, so no guessing needed there anymore.

(Spindle start/end column names and their units are now confirmed:
spindle_start_index / spindle_end_index are sample indices at FS=1000,
converted to spindle_start_time_s / spindle_end_time_s at load time.)
"""

import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from modules.ephys_preprocessing import bandpass_filter, downsampling
from modules.find_spindles_lfp_o_quality import _fit_window
from task_loader import TaskLoader

# --------------------------------------------------------------------------- #
# CONFIG -- check / edit these before running
# --------------------------------------------------------------------------- #
ANALYSIS_ROOTS = [
    Path("/mnt/genzel/Rat/HM/Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_R1_8"),
    Path("/mnt/genzel/Rat/HM/Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_R9_16"),
]
SUFFIX = Path("postsleep/wavelet_amp_1_ampcore_3")
MANIFEST_PATH = "tasks_manifest.csv"

FS = 1000          # raw sampling rate
TARGET_FS = 128     # AR fit sampling rate (must match calibration pipeline)
AR_ORDER = 8
SPINDLE_BAND = (9, 20)
WINDOW_PAD_SEC = 1.0   # extend each spindle window by this much on each side before fitting

# Confirmed exact column names in the wavelet spindle CSVs (spindle_start_index /
# spindle_end_index are sample indices at FS=1000 -- converted to seconds at load time).
START_COL = "spindle_start_time_s"
END_COL = "spindle_end_time_s"

# If your manifest/task dict has an explicit channel field, list its key(s) here.
# (Confirmed from build_manifest.py: the real manifest has NO channel column --
# channel lives only in the data filename, e.g. "chan51.mat" / "chan51_7.mat" --
# so this is just a fallback in case that ever changes.)
TASK_CHANNEL_KEYS = ["channel", "chan", "channel_num"]


def _norm_num(x) -> str:
    """Normalize a channel/trial identifier for matching: '' for missing,
    digit strings compared without leading zeros (e.g. '07' == '7'), same
    convention build_manifest.py uses when matching data/scoring trial suffixes."""
    if x is None:
        return ""
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return ""
    return str(int(x)) if x.isdigit() else x


def _get_task_trial(task: dict) -> str:
    # build_manifest.py writes trial="" when the data file has no trial suffix
    # (the common case), and the digit string otherwise.
    return _norm_num(task.get("trial"))

OUTPUT_DIR = Path("results_ar_per_spindle")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("spindle_ar_r")


# --------------------------------------------------------------------------- #
# Step 1: collect every detected spindle event (not aggregated -- one row each)
# --------------------------------------------------------------------------- #
def load_all_spindle_events() -> pd.DataFrame:
    rows = []
    for analysis_root in tqdm(ANALYSIS_ROOTS, desc="Analysis folders"):
        rat_groups = [d for d in analysis_root.iterdir() if d.is_dir()]
        for rat_group in tqdm(rat_groups, desc=analysis_root.name, leave=False):
            spindle_root = rat_group / "Spindle_detection_results"
            if not spindle_root.exists():
                continue
            region_dirs = [d for d in spindle_root.iterdir() if d.is_dir()]
            for region_dir in region_dirs:
                animal_dirs = [d for d in region_dir.iterdir() if d.is_dir()]
                for animal_dir in animal_dirs:
                    date_dirs = [d for d in animal_dir.iterdir() if d.is_dir()]
                    for date_dir in date_dirs:
                        csv_folder = date_dir / SUFFIX
                        if not csv_folder.exists():
                            continue
                        for csv_file in csv_folder.glob("*.csv"):
                            match = re.match(r"chan(\d+)(?:_(\d+))?_spindles_wavelet\.csv$", csv_file.name)
                            df = pd.read_csv(csv_file)
                            if df.empty:
                                continue
                            df = df.copy()
                            df["analysis"] = analysis_root.name
                            df["rat_group"] = rat_group.name
                            df["region"] = region_dir.name
                            df["rat_number"] = animal_dir.name
                            df["date"] = date_dir.name
                            df["file"] = csv_file.name
                            df["channel"] = match.group(1) if match else None
                            df["trial"] = str(match.group(2)) if match and match.group(2) else None

                            # Real wavelet CSVs store spindle timing as *sample indices*
                            # into the raw signal at FS (1000 Hz), not seconds -- convert
                            # once here so everything downstream just deals in seconds.
                            if "spindle_start_index" in df.columns and "spindle_end_index" in df.columns:
                                df["spindle_start_time_s"] = df["spindle_start_index"] / FS
                                df["spindle_end_time_s"] = df["spindle_end_index"] / FS
                            if "spindle_peak_index" in df.columns:
                                df["spindle_peak_time_s"] = df["spindle_peak_index"] / FS

                            rows.append(df)
    if not rows:
        logger.error("No spindle events found under the given ANALYSIS_ROOTS.")
        return pd.DataFrame()
    events = pd.concat(rows, ignore_index=True)
    logger.info(f"Loaded {len(events)} spindle events across {events['file'].nunique()} files.")
    return events


def resolve_time_columns(df: pd.DataFrame) -> tuple[str, str]:
    if START_COL not in df.columns or END_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{START_COL}' / '{END_COL}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )
    return START_COL, END_COL


# --------------------------------------------------------------------------- #
# Step 2: build a lookup from (rat, region, date, channel) -> manifest task
# --------------------------------------------------------------------------- #
def _get_task_channel(task: dict) -> str | None:
    for key in TASK_CHANNEL_KEYS:
        if key in task and task[key] not in (None, ""):
            return str(task[key])
    # data_path is guaranteed present (used directly by the calibration script),
    # and build_manifest.py confirms its filename format is chanN.mat / chanN_trial.mat --
    # so parse it from there rather than relying on however TaskLoader derives file_name.
    m = re.search(r"chan[_ ]?(\d+)", Path(str(task.get("data_path", ""))).name, re.IGNORECASE)
    return m.group(1) if m else None


def build_task_lookup() -> dict:
    loader = TaskLoader(MANIFEST_PATH)
    tasks = loader.to_tasks()
    lookup = {}
    for t in tasks:
        channel = _norm_num(_get_task_channel(t))
        trial = _get_task_trial(t)
        key = (str(t.get("rat")).strip(), str(t.get("region")).strip(), str(t.get("date")).strip(), trial, channel)
        lookup[key] = t
    logger.info(f"Built manifest lookup with {len(lookup)} entries.")
    return lookup


def find_task_for_group(lookup: dict, rat_number, region, date, trial, channel):
    key = (str(rat_number).strip(), str(region).strip(), str(date).strip(), _norm_num(trial), _norm_num(channel))
    if key in lookup:
        return lookup[key]
    # fall back: ignore channel if there's exactly one task for (rat, region, date, trial)
    candidates = [
        t for (r, reg, d, tr, ch), t in lookup.items()
        if r == str(rat_number).strip()
        and reg == str(region).strip()
        and d == str(date).strip()
        and tr == _norm_num(trial)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


# --------------------------------------------------------------------------- #
# Step 3: per-file signal prep + per-event AR fit
# --------------------------------------------------------------------------- #
def prepare_signal(data_path: str) -> np.ndarray:
    raw_signal = loadmat(data_path)["data"].squeeze()
    filtered = bandpass_filter(raw_signal, lowcut=0.1, highcut=100, fs=FS)
    return downsampling(filtered, FS, TARGET_FS)


def ar_fit_for_event(signal_128: np.ndarray, start_s: float, end_s: float):
    s_idx = max(0, int((start_s - WINDOW_PAD_SEC) * TARGET_FS))
    e_idx = min(len(signal_128), int((end_s + WINDOW_PAD_SEC) * TARGET_FS))
    if e_idx <= s_idx:
        return np.nan, np.nan
    window = signal_128[s_idx:e_idx]
    r_val, f_val = _fit_window(window, AR_ORDER, TARGET_FS, SPINDLE_BAND)
    return r_val, f_val


def compute_ar_r_values(events: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    start_col, end_col = resolve_time_columns(events)

    events = events.copy()
    events["ar_r_value"] = np.nan
    events["ar_peak_freq_hz"] = np.nan

    # "No trial" is the COMMON case (per build_manifest.py's own docstring), not
    # missing data -- fill with "" so pandas groupby doesn't silently drop these
    # rows (it drops NaN/None group keys by default).
    events["trial"] = events["trial"].fillna("").astype(str)

    unmatched_groups = []
    group_cols = ["rat_number", "region", "date", "trial", "channel"]

    for (rat_number, region, date, trial, channel), group in tqdm(
        events.groupby(group_cols), desc="Processing raw files"
    ):
        task = find_task_for_group(lookup, rat_number, region, date, trial, channel)
        if task is None:
            unmatched_groups.append(
                {"rat_number": rat_number, "region": region, "date": date, "trial": trial, "channel": channel}
            )
            continue

        try:
            signal_128 = prepare_signal(task["data_path"])
        except Exception as e:
            logger.warning(f"Failed to load/prepare {task.get('data_path')}: {e}")
            continue

        for idx, row in group.iterrows():
            r_val, f_val = ar_fit_for_event(signal_128, row[start_col], row[end_col])
            events.at[idx, "ar_r_value"] = r_val
            events.at[idx, "ar_peak_freq_hz"] = f_val

    if unmatched_groups:
        unmatched_df = pd.DataFrame(unmatched_groups).drop_duplicates()
        unmatched_path = OUTPUT_DIR / "unmatched_groups.csv"
        unmatched_df.to_csv(unmatched_path, index=False)
        logger.warning(
            f"{len(unmatched_df)} (rat, region, date, channel) group(s) had no manifest match. "
            f"See {unmatched_path}"
        )

    return events


# --------------------------------------------------------------------------- #
# Step 4: statistics -- correlations between AR r-value and wavelet metrics
# --------------------------------------------------------------------------- #
WAVELET_METRIC_COLS = [
    "spindle_amplitude",
    "spindle_duration_s",
    "spindle_mean_frequency_hz",
    "spindle_peak_frequency_hz",
]


def analyze_correlations(df: pd.DataFrame):
    df = df.dropna(subset=["ar_r_value"])
    if df.empty:
        logger.error("No rows with a valid ar_r_value -- nothing to correlate.")
        return

    # Overall correlations
    overall_rows = []
    for col in WAVELET_METRIC_COLS:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col, "ar_r_value"])
        if len(sub) < 3:
            continue
        r, p = pearsonr(sub["ar_r_value"], sub[col])
        overall_rows.append({"metric": col, "n": len(sub), "pearson_r": round(r, 4), "p_value": p})
    overall_df = pd.DataFrame(overall_rows)
    overall_path = OUTPUT_DIR / "correlations_overall.csv"
    overall_df.to_csv(overall_path, index=False)
    logger.info(f"Saved overall correlations to {overall_path}")
    print("\n--- Overall correlations: ar_r_value vs wavelet metrics ---")
    print(overall_df.to_string(index=False))

    # Per-region breakdown
    per_region_rows = []
    for region, region_df in df.groupby("region"):
        for col in WAVELET_METRIC_COLS:
            if col not in region_df.columns:
                continue
            sub = region_df.dropna(subset=[col, "ar_r_value"])
            if len(sub) < 3:
                continue
            r, p = pearsonr(sub["ar_r_value"], sub[col])
            per_region_rows.append(
                {"region": region, "metric": col, "n": len(sub), "pearson_r": round(r, 4), "p_value": p}
            )
    per_region_df = pd.DataFrame(per_region_rows)
    per_region_path = OUTPUT_DIR / "correlations_by_region.csv"
    per_region_df.to_csv(per_region_path, index=False)
    logger.info(f"Saved per-region correlations to {per_region_path}")

    # Summary stats of ar_r_value itself by rat/region
    summary = df.groupby(["rat_number", "region"])["ar_r_value"].agg(
        mean="mean", median="median", std="std", n="count"
    ).reset_index()
    summary_path = OUTPUT_DIR / "ar_r_value_summary_by_rat_region.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved ar_r_value summary by rat/region to {summary_path}")
    print("\n--- ar_r_value summary by rat/region ---")
    print(summary.to_string(index=False))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    events = load_all_spindle_events()
    if events.empty:
        return

    lookup = build_task_lookup()
    events_with_r = compute_ar_r_values(events, lookup)

    out_path = OUTPUT_DIR / "spindle_events_with_ar_r.csv"
    events_with_r.to_csv(out_path, index=False)
    logger.info(f"Saved per-spindle AR results to {out_path}")

    analyze_correlations(events_with_r)


if __name__ == "__main__":
    main()
