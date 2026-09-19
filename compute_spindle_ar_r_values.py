"""
compute_spindle_ar_r_values.py

For each spindle already detected by the wavelet method, find its raw LFP
file, chop the spindle's time span into overlapping 1-second windows, fit a
Burg AR model to each (same method the detection/calibration pipeline uses),
and keep the peak r-value / frequency. Output: one row per spindle with an
added ar_r_value + ar_peak_freq_hz, plus correlation/summary reports.
"""

import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
# CONFIG
# --------------------------------------------------------------------------- #
ANALYSIS_ROOTS = [
    Path("/mnt/genzel/Rat/HM/Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_R1_8"),
    Path("/mnt/genzel/Rat/HM/Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_R9_16"),
]
SUFFIX = Path("postsleep/wavelet_amp_1_ampcore_3")
MANIFEST_PATH = "tasks_manifest.csv"

FS = 1000            # raw signal sampling rate
TARGET_FS = 128       # AR fit sampling rate
AR_ORDER = 8
SPINDLE_BAND = (9, 20)
WINDOW_PAD_SEC = 0.0       # extend each spindle span by this much on each side
AR_WINDOW_SEC = 1.0        # AR fit window length (matches the detection pipeline)
AR_WINDOW_STEP_SEC = 0.5   # step between successive AR windows within a spindle's span

START_COL = "spindle_start_time_s"  # converted from spindle_start_index (samples @ FS)
END_COL = "spindle_end_time_s"

OUTPUT_DIR = Path("results_ar_per_spindle")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = max(1, os.cpu_count() - 1)

WAVELET_METRIC_COLS = [
    "spindle_amplitude",
    "spindle_duration_s",
    "spindle_mean_frequency_hz",
    "spindle_peak_frequency_hz",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("spindle_ar_r")


# --------------------------------------------------------------------------- #
# 1. Load every detected spindle event (not aggregated -- one row each)
# --------------------------------------------------------------------------- #
def load_all_spindle_events() -> pd.DataFrame:
    file_pattern = re.compile(r"chan(\d+)(?:_(\d+))?_spindles_wavelet\.csv$")
    rows = []

    for analysis_root in tqdm(ANALYSIS_ROOTS, desc="Analysis folders"):
        for rat_group in tqdm([d for d in analysis_root.iterdir() if d.is_dir()],
                               desc=analysis_root.name, leave=False):
            spindle_root = rat_group / "Spindle_detection_results"
            if not spindle_root.exists():
                continue
            for region_dir in [d for d in spindle_root.iterdir() if d.is_dir()]:
                for animal_dir in [d for d in region_dir.iterdir() if d.is_dir()]:
                    for date_dir in [d for d in animal_dir.iterdir() if d.is_dir()]:
                        csv_folder = date_dir / SUFFIX
                        if not csv_folder.exists():
                            continue
                        for csv_file in csv_folder.glob("*.csv"):
                            df = pd.read_csv(csv_file)
                            if df.empty:
                                continue
                            match = file_pattern.match(csv_file.name)
                            df["region"] = region_dir.name
                            df["rat_number"] = animal_dir.name
                            df["date"] = date_dir.name
                            df["file"] = csv_file.name
                            df["channel"] = match.group(1) if match else None
                            df["trial"] = match.group(2) if match and match.group(2) else None

                            # spindle_start_index/end_index are sample indices at FS -- convert to seconds
                            df[START_COL] = df["spindle_start_index"] / FS
                            df[END_COL] = df["spindle_end_index"] / FS

                            rows.append(df)

    if not rows:
        logger.error("No spindle events found under ANALYSIS_ROOTS.")
        return pd.DataFrame()

    events = pd.concat(rows, ignore_index=True)
    events["trial"] = events["trial"].fillna("")  # "no trial" is the common case, not missing data
    logger.info(f"Loaded {len(events)} spindle events across {events['file'].nunique()} files.")
    return events


# --------------------------------------------------------------------------- #
# 2. Manifest lookup: (rat, region, date, trial, channel) -> task dict
# --------------------------------------------------------------------------- #
def normalize_id(x) -> str:
    """'' for missing; numeric values compared without leading zeros or a
    trailing '.0' (pandas upcasts a trial/channel column to float64 once any
    row is missing, so '10' commonly arrives here as 10.0)."""
    if x is None:
        return ""
    x = str(x).strip()
    if x in ("", "nan"):
        return ""
    try:
        return str(int(float(x)))
    except ValueError:
        return x


def parse_channel_and_trial(data_path) -> tuple[str | None, str]:
    """Channel + trial live in the data filename (chanN.mat / chanN_trial.mat),
    per build_manifest.py. Parsed from data_path directly rather than trusting
    a task['trial']/['channel'] key, since TaskLoader may not populate either."""
    match = re.match(r"chan(\d+)(?:_(\d+))?\.mat$", Path(str(data_path)).name, re.IGNORECASE)
    if not match:
        return None, ""
    return match.group(1), (match.group(2) or "")


def build_task_lookup() -> dict:
    tasks = TaskLoader(MANIFEST_PATH).to_tasks()
    lookup = {}
    for task in tasks:
        channel, trial = parse_channel_and_trial(task.get("data_path", ""))
        key = (
            str(task.get("rat")).strip(),
            str(task.get("region")).strip(),
            str(task.get("date")).strip(),
            normalize_id(trial),
            normalize_id(channel),
        )
        lookup[key] = task
    logger.info(f"Built manifest lookup with {len(lookup)} entries.")
    return lookup


def find_task(lookup: dict, rat_number, region, date, trial, channel):
    key = (str(rat_number).strip(), str(region).strip(), str(date).strip(),
           normalize_id(trial), normalize_id(channel))
    return lookup.get(key)


# --------------------------------------------------------------------------- #
# 3. Per-group AR fitting (runs in worker processes)
# --------------------------------------------------------------------------- #
def prepare_signal(data_path: str) -> np.ndarray:
    raw = loadmat(data_path)["data"].squeeze()
    filtered = bandpass_filter(raw, lowcut=0.1, highcut=100, fs=FS)
    return downsampling(filtered, FS, TARGET_FS)


def ar_fit_for_event(signal_128: np.ndarray, start_s: float, end_s: float) -> tuple[float, float]:
    """Chop [start_s, end_s] (padded) into overlapping 1s AR windows and
    return the peak r-value/frequency -- mirroring how the detection pipeline
    picks the peak r within an event, rather than fitting one AR model
    across the whole variable-length span."""
    window_samples = int(AR_WINDOW_SEC * TARGET_FS)
    step_samples = max(1, int(AR_WINDOW_STEP_SEC * TARGET_FS))

    s_idx = max(0, int((start_s - WINDOW_PAD_SEC) * TARGET_FS))
    e_idx = min(len(signal_128), int((end_s + WINDOW_PAD_SEC) * TARGET_FS))
    e_idx = max(e_idx, min(len(signal_128), s_idx + window_samples))  # ensure room for one window
    if e_idx - s_idx < window_samples:
        return np.nan, np.nan  # not enough signal even after extending (recording boundary)

    best_r, best_f = 0.0, np.nan
    for w_start in range(s_idx, e_idx - window_samples + 1, step_samples):
        r_val, f_val = _fit_window(signal_128[w_start:w_start + window_samples],
                                    AR_ORDER, TARGET_FS, SPINDLE_BAND)
        if r_val > best_r:
            best_r, best_f = r_val, f_val
    return best_r, best_f


def process_group(data_path: str, event_rows: list[tuple]) -> list[tuple]:
    """Runs in a worker process: load+prep one raw file once, then AR-fit
    every spindle event that belongs to it. event_rows: [(index, start_s, end_s), ...]."""
    signal_128 = prepare_signal(data_path)
    return [(idx, *ar_fit_for_event(signal_128, start_s, end_s)) for idx, start_s, end_s in event_rows]


# --------------------------------------------------------------------------- #
# 4. Orchestration: match groups to files, fan out to workers, collect results
# --------------------------------------------------------------------------- #
def compute_ar_r_values(events: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    events = events.copy()
    events["ar_r_value"] = np.nan
    events["ar_peak_freq_hz"] = np.nan

    jobs = {}          # data_path -> list of (index, start_s, end_s)
    unmatched = []

    group_cols = ["rat_number", "region", "date", "trial", "channel"]
    for (rat_number, region, date, trial, channel), group in events.groupby(group_cols):
        task = find_task(lookup, rat_number, region, date, trial, channel)
        if task is None:
            unmatched.append({"rat_number": rat_number, "region": region, "date": date,
                               "trial": trial, "channel": channel})
            continue
        event_rows = list(zip(group.index, group[START_COL], group[END_COL]))
        jobs[task["data_path"]] = event_rows

    if unmatched:
        unmatched_path = OUTPUT_DIR / "unmatched_groups.csv"
        pd.DataFrame(unmatched).drop_duplicates().to_csv(unmatched_path, index=False)
        logger.warning(f"{len(unmatched)} group(s) had no manifest match. See {unmatched_path}")

    logger.info(f"Fitting AR models across {len(jobs)} raw files using {MAX_WORKERS} workers.")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_group, path, rows): path for path, rows in jobs.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing raw files"):
            data_path = futures[future]
            try:
                for idx, r_val, f_val in future.result():
                    events.at[idx, "ar_r_value"] = r_val
                    events.at[idx, "ar_peak_freq_hz"] = f_val
            except Exception as e:
                logger.warning(f"Failed on {data_path}: {e}")

    return events


# --------------------------------------------------------------------------- #
# 5. Statistics
# --------------------------------------------------------------------------- #
def correlate(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    rows = []
    groups = df.groupby(group_col) if group_col else [(None, df)]
    for group_val, sub_df in groups:
        for col in WAVELET_METRIC_COLS:
            sub = sub_df.dropna(subset=[col, "ar_r_value"])
            if len(sub) < 3:
                continue
            r, p = pearsonr(sub["ar_r_value"], sub[col])
            row = {"metric": col, "n": len(sub), "pearson_r": round(r, 4), "p_value": p}
            if group_col:
                row = {group_col: group_val, **row}
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_correlations(df: pd.DataFrame):
    df = df.dropna(subset=["ar_r_value"])
    if df.empty:
        logger.error("No rows with a valid ar_r_value -- nothing to correlate.")
        return

    overall = correlate(df)
    overall.to_csv(OUTPUT_DIR / "correlations_overall.csv", index=False)
    print("\n--- Overall correlations: ar_r_value vs wavelet metrics ---")
    print(overall.to_string(index=False))

    correlate(df, "region").to_csv(OUTPUT_DIR / "correlations_by_region.csv", index=False)

    summary = df.groupby(["rat_number", "region"])["ar_r_value"] \
        .agg(mean="mean", median="median", std="std", n="count").reset_index()
    summary.to_csv(OUTPUT_DIR / "ar_r_value_summary_by_rat_region.csv", index=False)
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
