"""
compute_spindle_ar_r_values.py

Calibrate AR pole magnitude (R) hysteresis thresholds using wavelet-detected
spindles as ground truth.

For each spindle event:
  1. Slides a 1.0s window across the event at high temporal resolution.
  2. Extracts R_max (peak), R_min (within-event nadir), R_start (entry),
     and R_end (exit).
  3. Interpolates a 11-point normalized temporal profile (0% to 100% duration)
     to map R evolution.
  4. Generates region-level distributions, per-rat averages, and explicit
     hysteresis recommendations (upper/lower threshold pairs).
"""

import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.io import loadmat
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

FS = 1000               # Raw LFP rate
TARGET_FS = 128         # Downsampled rate for AR fitting
AR_ORDER = 8
SPINDLE_BAND = (9, 20)
AR_WINDOW_SEC = 1.0

# Window stride in samples at TARGET_FS (2 samples = ~15.6 ms resolution)
STRIDE_SAMPLES = 2

START_COL = "spindle_start_time_s"
END_COL = "spindle_end_time_s"

OUTPUT_DIR = Path("results_ar_calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = max(1, os.cpu_count() - 1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ar_calibration")


# --------------------------------------------------------------------------- #
# 1. Load Ground-Truth Wavelet Spindles
# --------------------------------------------------------------------------- #
def load_all_spindle_events() -> pd.DataFrame:
    file_pattern = re.compile(r"chan(\d+)(?:_(\d+))?_spindles_wavelet\.csv$")
    rows = []

    for analysis_root in tqdm(ANALYSIS_ROOTS, desc="Analysis folders"):
        if not analysis_root.exists():
            continue
        for rat_group in [d for d in analysis_root.iterdir() if d.is_dir()]:
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
                            try:
                                df = pd.read_csv(csv_file)
                            except Exception:
                                continue
                            if df.empty:
                                continue

                            match = file_pattern.match(csv_file.name)
                            df["region"] = region_dir.name
                            df["rat_number"] = animal_dir.name
                            df["date"] = date_dir.name
                            df["file"] = csv_file.name
                            df["channel"] = match.group(1) if match else None
                            df["trial"] = match.group(2) if match and match.group(2) else ""

                            df[START_COL] = df["spindle_start_index"] / FS
                            df[END_COL] = df["spindle_end_index"] / FS
                            rows.append(df)

    if not rows:
        logger.error("No spindle events found under ANALYSIS_ROOTS.")
        return pd.DataFrame()

    events = pd.concat(rows, ignore_index=True)
    events["trial"] = events["trial"].fillna("")
    logger.info(f"Loaded {len(events)} spindle events across {events['file'].nunique()} files.")
    return events


# --------------------------------------------------------------------------- #
# 2. Manifest Lookup
# --------------------------------------------------------------------------- #
def normalize_id(x) -> str:
    if x is None or str(x).strip() in ("", "nan"):
        return ""
    try:
        return str(int(float(x)))
    except ValueError:
        return str(x).strip()


def parse_channel_and_trial(data_path: str) -> Tuple[str | None, str]:
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
# 3. High-Resolution Event AR Profile Fitting
# --------------------------------------------------------------------------- #
def prepare_signal(data_path: str) -> np.ndarray:
    raw = loadmat(data_path)["data"].squeeze()
    filtered = bandpass_filter(raw, lowcut=0.1, highcut=100, fs=FS)
    return downsampling(filtered, FS, TARGET_FS)


def analyze_spindle_r_dynamics(
    signal_128: np.ndarray,
    start_s: float,
    end_s: float,
    window_sec: float = AR_WINDOW_SEC,
    stride_samples: int = STRIDE_SAMPLES,
    target_fs: int = TARGET_FS,
    spindle_band: Tuple[float, float] = SPINDLE_BAND,
    ar_order: int = AR_ORDER,
) -> Dict:
    """
    Evaluates sliding 1s AR windows with centers spanning [start_s, end_s].
    Returns peak, minimum, boundary values, and normalized temporal evolution.
    """
    win_samples = int(window_sec * target_fs)
    half_win = win_samples // 2

    c_start = int(round(start_s * target_fs))
    c_end = int(round(end_s * target_fs))

    # Evaluate window centers covering onset to offset
    centers = np.arange(c_start, max(c_start + 1, c_end + 1), stride_samples)
    r_series = []
    f_series = []

    for c in centers:
        w_start = c - half_win
        w_end = w_start + win_samples

        if w_start < 0 or w_end > len(signal_128):
            continue

        r_val, f_val = _fit_window(signal_128[w_start:w_end], ar_order, target_fs, spindle_band)
        r_series.append(r_val)
        f_series.append(f_val)

    if not r_series:
        return {
            "r_max": np.nan, "r_min": np.nan, "r_mean": np.nan,
            "r_start": np.nan, "r_end": np.nan, "r_peak_freq": np.nan,
            "r_profile": [np.nan] * 11, "in_band_ratio": 0.0,
        }

    r_arr = np.array(r_series)
    f_arr = np.array(f_series)

    # Fractions where an actual pole in SPINDLE_BAND was resolved (r > 0)
    valid_mask = r_arr > 0
    in_band_ratio = float(np.mean(valid_mask))
    valid_r = r_arr[valid_mask] if np.any(valid_mask) else r_arr

    peak_idx = int(np.argmax(r_arr))
    r_max = float(r_arr[peak_idx])
    r_min = float(np.min(valid_r))
    r_mean = float(np.mean(valid_r))
    r_start = float(r_arr[0])
    r_end = float(r_arr[-1])
    r_peak_freq = float(f_arr[peak_idx])

    # Interpolate trajectory to 11 normalized timepoints (0%, 10%, ..., 100% of event)
    if len(r_arr) >= 2:
        x_norm = np.linspace(0, 1, len(r_arr))
        interpolator = interp1d(x_norm, r_arr, kind="linear", bounds_error=False, fill_value="extrapolate")
        profile = interpolator(np.linspace(0, 1, 11)).tolist()
    else:
        profile = [r_max] * 11

    return {
        "r_max": r_max,
        "r_min": r_min,
        "r_mean": r_mean,
        "r_start": r_start,
        "r_end": r_end,
        "r_peak_freq": r_peak_freq,
        "r_profile": profile,
        "in_band_ratio": in_band_ratio,
    }


def process_group(data_path: str, event_rows: List[Tuple]) -> List[Tuple]:
    """Processes all spindles belonging to one physical recording file."""
    signal_128 = prepare_signal(data_path)
    results = []
    for idx, start_s, end_s in event_rows:
        metrics = analyze_spindle_r_dynamics(signal_128, start_s, end_s)
        results.append((idx, metrics))
    return results


# --------------------------------------------------------------------------- #
# 4. Orchestration & Multiprocessing
# --------------------------------------------------------------------------- #
def compute_all_dynamics(events: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    events = events.copy()
    for col in ["r_max", "r_min", "r_mean", "r_start", "r_end", "r_peak_freq", "in_band_ratio"]:
        events[col] = np.nan

    profile_cols = [f"r_profile_{p}%" for p in range(0, 101, 10)]
    for col in profile_cols:
        events[col] = np.nan

    jobs = {}
    unmatched = []
    group_cols = ["rat_number", "region", "date", "trial", "channel"]

    for (rat_number, region, date, trial, channel), group in events.groupby(group_cols):
        task = find_task(lookup, rat_number, region, date, trial, channel)
        if task is None:
            unmatched.append({"rat": rat_number, "region": region, "date": date, "trial": trial, "chan": channel})
            continue
        jobs[task["data_path"]] = list(zip(group.index, group[START_COL], group[END_COL]))

    if unmatched:
        pd.DataFrame(unmatched).drop_duplicates().to_csv(OUTPUT_DIR / "unmatched_manifest.csv", index=False)
        logger.warning(f"{len(unmatched)} group(s) could not be matched to raw data.")

    logger.info(f"Analyzing {len(events)} events across {len(jobs)} files using {MAX_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_group, path, rows): path for path, rows in jobs.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fitting AR Trajectories"):
            try:
                for idx, metrics in future.result():
                    events.at[idx, "r_max"] = metrics["r_max"]
                    events.at[idx, "r_min"] = metrics["r_min"]
                    events.at[idx, "r_mean"] = metrics["r_mean"]
                    events.at[idx, "r_start"] = metrics["r_start"]
                    events.at[idx, "r_end"] = metrics["r_end"]
                    events.at[idx, "r_peak_freq"] = metrics["r_peak_freq"]
                    events.at[idx, "in_band_ratio"] = metrics["in_band_ratio"]
                    for col_name, val in zip(profile_cols, metrics["r_profile"]):
                        events.at[idx, col_name] = val
            except Exception as e:
                logger.error(f"Worker failed: {e}")

    return events


# --------------------------------------------------------------------------- #
# 5. Statistical Aggregations & Hysteresis Guidance
# --------------------------------------------------------------------------- #
def generate_summary_tables(df: pd.DataFrame):
    valid = df.dropna(subset=["r_max", "r_min"]).copy()
    if valid.empty:
        logger.error("No valid AR events processed.")
        return

    # A. Pooled Regional Summary across all rats
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    records = []

    for region, reg_df in valid.groupby("region"):
        row = {"Region": region, "N_Spindles": len(reg_df), "N_Rats": reg_df["rat_number"].nunique()}
        for metric in ["r_max", "r_min", "r_start", "r_end"]:
            vals = reg_df[metric].values
            row[f"{metric}_mean"] = np.mean(vals)
            row[f"{metric}_std"] = np.std(vals)
            for q in quantiles:
                row[f"{metric}_p{int(q*100)}"] = np.quantile(vals, q)
        records.append(row)

    pooled_summary = pd.DataFrame(records)
    pooled_summary.to_csv(OUTPUT_DIR / "regional_ar_pooled_summary.csv", index=False)

    # B. Rat-Averaged Summary (Prevents rats with high spindle counts from dominating)
    rat_means = valid.groupby(["region", "rat_number"])[["r_max", "r_min", "r_start", "r_end"]].mean().reset_index()
    rat_agg = rat_means.groupby("region")[["r_max", "r_min", "r_start", "r_end"]].agg(["mean", "std"]).reset_index()
    rat_agg.columns = ["_".join(filter(None, c)) for c in rat_agg.columns]
    rat_agg.to_csv(OUTPUT_DIR / "regional_ar_rat_averaged_summary.csv", index=False)

    # C. Hysteresis Threshold Calibrator
    # Upper threshold: 25th percentile of r_max (captures 75% of ground truth spindles)
    # Lower threshold: 10th percentile of r_min OR 25th percentile of r_end
    calib = []
    for region, reg_df in valid.groupby("region"):
        p25_max = np.quantile(reg_df["r_max"], 0.25)
        p50_max = np.quantile(reg_df["r_max"], 0.50)
        p10_min = np.quantile(reg_df["r_min"], 0.10)
        p25_min = np.quantile(reg_df["r_min"], 0.25)
        p25_end = np.quantile(reg_df["r_end"], 0.25)

        calib.append({
            "Region": region,
            "Target_Upper_Threshold (Catch 75%)": round(p25_max, 3),
            "Target_Upper_Threshold (Median)": round(p50_max, 3),
            "Recommended_Lower_Threshold": round(min(p10_min, p25_end), 3),
            "Empirical_Safety_Margin": round(p25_max - min(p10_min, p25_end), 3),
        })

    calib_df = pd.DataFrame(calib)
    calib_df.to_csv(OUTPUT_DIR / "recommended_hysteresis_thresholds.csv", index=False)

    # D. Mean Temporal Evolution Trajectory per Region
    profile_cols = [f"r_profile_{p}%" for p in range(0, 101, 10)]
    evolution = valid.groupby("region")[profile_cols].mean().reset_index()
    evolution.to_csv(OUTPUT_DIR / "regional_r_evolution_profile.csv", index=False)

    # Console display
    print("\n" + "=" * 80)
    print("RECOMMENDED HYSTERESIS THRESHOLDS PER REGION")
    print("=" * 80)
    print(calib_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("MEAN R EVOLUTION ACROSS SPINDLE DURATION (0% to 100%)")
    print("=" * 80)
    print(evolution.to_string(index=False))
    print("\n")


# --------------------------------------------------------------------------- #
# Main Entry Point
# --------------------------------------------------------------------------- #
def main():
    events = load_all_spindle_events()
    if events.empty:
        return

    lookup = build_task_lookup()
    analyzed_events = compute_all_dynamics(events, lookup)

    raw_out = OUTPUT_DIR / "wavelet_spindles_with_ar_dynamics.csv"
    analyzed_events.to_csv(raw_out, index=False)
    logger.info(f"Saved full event-level dynamics to {raw_out}")

    generate_summary_tables(analyzed_events)


if __name__ == "__main__":
    main()
