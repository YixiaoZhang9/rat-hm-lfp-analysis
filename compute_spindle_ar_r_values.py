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

Data-quality note (fixed):
  Events where NO sliding window resolved an in-band AR pole (in_band_ratio == 0)
  are now marked NaN for r_max/r_min/r_mean instead of silently defaulting to 0.0.
  A 0.0 R-value is indistinguishable from "no pole found" vs. "a real, very weak
  pole" unless in_band_ratio is checked, and folding failed fits in as if they
  were valid low-R spindles collapses percentile-based thresholds toward zero.
  Summary tables now (a) drop events below a minimum in_band_ratio, and
  (b) report per-region in_band_ratio / fit-failure diagnostics so failures are
  visible rather than silently baked into the calibration.
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
SPINDLE_BAND = (10, 15)
AR_WINDOW_SEC = 1.0

# Window stride in samples at TARGET_FS (2 samples = ~15.6 ms resolution)
STRIDE_SAMPLES = 4

# Minimum fraction of sliding windows within an event that must resolve an
# in-band AR pole (r > 0) for that event's r_max/r_min/etc. to be trusted.
# Events below this are excluded from calibration statistics as fit failures,
# not treated as "genuinely low R" spindles.
MIN_IN_BAND_RATIO = 0.5

START_COL = "spindle_start_time_s"
END_COL = "spindle_end_time_s"

OUTPUT_DIR = Path("results_ar_calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

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

    IMPORTANT: r_max/r_min/r_mean/r_start/r_end are only meaningful when at
    least some windows resolved an in-band pole (r > 0). If in_band_ratio is 0,
    every returned r_* value is NaN rather than 0.0, so downstream code can
    distinguish "fit failed everywhere" from "a genuinely weak/absent pole".
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
            "n_windows": 0,
        }

    r_arr = np.array(r_series)
    f_arr = np.array(f_series)
    n_windows = len(r_arr)

    # Fraction of windows where an actual pole in SPINDLE_BAND was resolved (r > 0)
    valid_mask = r_arr > 0
    in_band_ratio = float(np.mean(valid_mask))

    if not np.any(valid_mask):
        # No window in this event ever resolved an in-band pole. This is a
        # fit failure for the event, not evidence of a "zero R" spindle -
        # return NaN so it gets excluded from calibration stats via dropna,
        # instead of silently contributing a 0.0 that drags percentiles down.
        return {
            "r_max": np.nan, "r_min": np.nan, "r_mean": np.nan,
            "r_start": np.nan, "r_end": np.nan, "r_peak_freq": np.nan,
            "r_profile": [np.nan] * 11, "in_band_ratio": in_band_ratio,
            "n_windows": n_windows,
        }

    valid_r = r_arr[valid_mask]

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
        "n_windows": n_windows,
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
    for col in ["r_max", "r_min", "r_mean", "r_start", "r_end", "r_peak_freq", "in_band_ratio", "n_windows"]:
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
            unmatched.append({"rat": rat_number, "region": region, "date": date, "trial": trial, "chan": channel,
                               "n_events": len(group)})
            continue
        jobs[task["data_path"]] = list(zip(group.index, group[START_COL], group[END_COL]))

    if unmatched:
        unmatched_df = pd.DataFrame(unmatched).drop_duplicates(subset=["rat", "region", "date", "trial", "chan"])
        unmatched_df.to_csv(OUTPUT_DIR / "unmatched_manifest.csv", index=False)
        total_unmatched_events = unmatched_df["n_events"].sum()
        logger.warning(
            f"{len(unmatched_df)} group(s) ({total_unmatched_events} events) could not be matched to raw data."
        )
        # Flag if unmatched groups are concentrated in one region - this can
        # masquerade as a "low R" region if it's actually a lookup/key mismatch.
        by_region = unmatched_df.groupby("region")["n_events"].sum().sort_values(ascending=False)
        logger.warning(f"Unmatched events by region:\n{by_region.to_string()}")

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
                    events.at[idx, "n_windows"] = metrics["n_windows"]
                    for col_name, val in zip(profile_cols, metrics["r_profile"]):
                        events.at[idx, col_name] = val
            except Exception as e:
                logger.error(f"Worker failed: {e}")

    return events


# --------------------------------------------------------------------------- #
# 5. Statistical Aggregations & Hysteresis Guidance
# --------------------------------------------------------------------------- #
def generate_fit_quality_report(df: pd.DataFrame):
    """
    Reports, per region, how often the AR fit failed to resolve an in-band pole
    at all (in_band_ratio == 0) and how many events are excluded from
    calibration by the MIN_IN_BAND_RATIO gate. This is the diagnostic that
    would have surfaced the PL region issue instead of it showing up silently
    as a 0.000 threshold.
    """
    has_metrics = df.dropna(subset=["in_band_ratio"]).copy()
    if has_metrics.empty:
        logger.error("No events had AR metrics computed at all (fit stage produced nothing).")
        return

    records = []
    for region, reg_df in has_metrics.groupby("region"):
        n = len(reg_df)
        n_total_fail = int((reg_df["in_band_ratio"] == 0).sum())
        n_below_gate = int((reg_df["in_band_ratio"] < MIN_IN_BAND_RATIO).sum())
        records.append({
            "Region": region,
            "N_Events": n,
            "Mean_In_Band_Ratio": round(float(reg_df["in_band_ratio"].mean()), 3),
            "Pct_Complete_Fit_Failure (ratio==0)": round(100 * n_total_fail / n, 2),
            f"Pct_Below_Gate (<{MIN_IN_BAND_RATIO})": round(100 * n_below_gate / n, 2),
        })

    report = pd.DataFrame(records)
    report.to_csv(OUTPUT_DIR / "fit_quality_report.csv", index=False)

    print("\n" + "=" * 80)
    print("AR FIT QUALITY BY REGION (diagnose before trusting thresholds below)")
    print("=" * 80)
    print(report.to_string(index=False))
    print(
        "\nNote: events with in_band_ratio == 0 never resolved an in-band AR pole "
        "in ANY sliding window and are excluded (NaN) from all r_* statistics. "
        f"Events with in_band_ratio below {MIN_IN_BAND_RATIO} are additionally "
        "excluded from the calibration tables below as low-confidence fits. "
        "A region with a high failure/exclusion rate needs its raw data / manifest "
        "matching / band settings checked before its threshold numbers are used.\n"
    )


def generate_summary_tables(df: pd.DataFrame):
    """
    Produce compact calibration tables.

    The wavelet events are the reference spindle population. For every
    independent AR window inside those events, _fit_window returns the
    strongest 10-15 Hz pole (R and its frequency).

    We report empirical distributions rather than silently declaring one
    percentile to be "the" correct threshold.

    Upper-threshold candidates are based on event-level R_max:
        e.g. p05, p10, p15, p20, p25, p50.

    Lower-threshold candidates are based on:
        - event-level R_min
        - event-level R_end

    This lets the actual distributions be inspected before the final
    hysteresis pair is chosen.
    """
    valid = df.dropna(
        subset=["r_max", "r_min"]
    ).copy()

    valid = valid[
        valid["in_band_ratio"] >= MIN_IN_BAND_RATIO
    ].copy()

    if valid.empty:
        logger.error(
            "No valid AR events processed after quality filtering."
        )
        return

    n_fitted = int(
        df["r_max"].notna().sum()
    )

    logger.info(
        "Calibration will use %d events "
        "(in_band_ratio >= %.2f); "
        "%d fitted events excluded as low-confidence.",
        len(valid),
        MIN_IN_BAND_RATIO,
        n_fitted - len(valid),
    )

    # ------------------------------------------------------------------ #
    # A. Pooled regional summary
    # ------------------------------------------------------------------ #
    quantiles = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
    ]

    records = []

    for region, reg_df in valid.groupby(
        "region"
    ):
        row = {
            "Region": region,
            "N_Spindles": len(reg_df),
            "N_Rats": reg_df[
                "rat_number"
            ].nunique(),
        }

        for metric in [
            "r_max",
            "r_min",
            "r_mean",
            "r_start",
            "r_end",
        ]:
            vals = reg_df[
                metric
            ].values

            row[
                f"{metric}_mean"
            ] = float(
                np.mean(vals)
            )

            row[
                f"{metric}_std"
            ] = float(
                np.std(vals)
            )

            for q in quantiles:
                row[
                    f"{metric}_p{int(q * 100)}"
                ] = float(
                    np.quantile(
                        vals,
                        q,
                    )
                )

        records.append(row)

    pooled_summary = pd.DataFrame(
        records
    )

    pooled_summary.to_csv(
        OUTPUT_DIR
        / "regional_ar_pooled_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------ #
    # B. Rat-averaged summary
    # ------------------------------------------------------------------ #
    rat_means = (
        valid
        .groupby(
            [
                "region",
                "rat_number",
            ]
        )[
            [
                "r_max",
                "r_min",
                "r_start",
                "r_end",
            ]
        ]
        .mean()
        .reset_index()
    )

    rat_agg = (
        rat_means
        .groupby("region")[
            [
                "r_max",
                "r_min",
                "r_start",
                "r_end",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    rat_agg.columns = [
        "_".join(
            filter(None, c)
        )
        for c in rat_agg.columns
    ]

    rat_agg.to_csv(
        OUTPUT_DIR
        / "regional_ar_rat_averaged_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------ #
    # C. Threshold candidate table
    # ------------------------------------------------------------------ #
    #
    # Do NOT call these "recommended" thresholds yet.
    #
    # Upper candidates:
    #   quantiles of event R_max.
    #
    # Lower candidates:
    #   quantiles of event R_min and R_end.
    #
    # The final upper/lower pair can then be selected after inspecting
    # the empirical distributions and, if desired, a background/non-spindle
    # calibration.
    #
    calib = []

    for region, reg_df in valid.groupby(
        "region"
    ):
        row = {
            "Region": region,
            "N_Spindles_Used": len(
                reg_df
            ),
        }

        for q in [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.50,
        ]:
            row[
                f"Upper_Rmax_p{int(q * 100)}"
            ] = round(
                float(
                    np.quantile(
                        reg_df["r_max"],
                        q,
                    )
                ),
                4,
            )

        for metric in [
            "r_min",
            "r_end",
            "r_start",
        ]:
            for q in [
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.50,
                0.75,
            ]:
                row[
                    f"{metric}_p{int(q * 100)}"
                ] = round(
                    float(
                        np.quantile(
                            reg_df[metric],
                            q,
                        )
                    ),
                    4,
                )

        calib.append(row)

    calib_df = pd.DataFrame(
        calib
    )

    # Keep the old filename so downstream code does not break, but the
    # contents are now explicitly empirical candidates rather than a
    # hard-coded recommendation.
    calib_df.to_csv(
        OUTPUT_DIR
        / "recommended_hysteresis_thresholds.csv",
        index=False,
    )

    # ------------------------------------------------------------------ #
    # D. Mean temporal R evolution
    # ------------------------------------------------------------------ #
    profile_cols = [
        f"r_profile_{p}%"
        for p in range(0, 101, 10)
    ]

    evolution = (
        valid
        .groupby("region")[
            profile_cols
        ]
        .mean()
        .reset_index()
    )

    evolution.to_csv(
        OUTPUT_DIR
        / "regional_r_evolution_profile.csv",
        index=False,
    )

    # ------------------------------------------------------------------ #
    # E. Compact dynamics summary
    # ------------------------------------------------------------------ #
    dynamics_summary = (
        valid
        .groupby("region")[
            [
                "r_mean",
                "r_start",
                "r_end",
                "r_peak_freq",
                "in_band_ratio",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    dynamics_summary.columns = [
        "_".join(
            filter(None, c)
        )
        for c in dynamics_summary.columns
    ]

    dynamics_summary.to_csv(
        OUTPUT_DIR
        / "regional_ar_dynamics_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------ #
    # Console
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print(
        "EMPIRICAL HYSTERESIS THRESHOLD CANDIDATES"
    )
    print("=" * 80)
    print(
        calib_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print(
        "MEAN R EVOLUTION ACROSS SPINDLE DURATION"
    )
    print("=" * 80)
    print(
        evolution.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print(
        "DYNAMICS SUMMARY"
    )
    print("=" * 80)
    print(
        dynamics_summary.to_string(
            index=False
        )
    )
    print()


# --------------------------------------------------------------------------- #
# Main Entry Point
# --------------------------------------------------------------------------- #
def main():
    events = load_all_spindle_events()
    if events.empty:
        return

    lookup = build_task_lookup()
    analyzed_events = compute_all_dynamics(events, lookup)

    # Compact event-level calibration table. No raw R(t) arrays are saved,
    # so this file stays small enough to inspect/upload.
    event_cols = [
        c for c in [
            "region",
            "rat_number",
            "date",
            "file",
            "channel",
            "trial",
            START_COL,
            END_COL,
            "r_max",
            "r_min",
            "r_mean",
            "r_start",
            "r_end",
            "r_peak_freq",
            "in_band_ratio",
            "n_windows",
        ]
        if c in analyzed_events.columns
    ]

    raw_out = OUTPUT_DIR / "wavelet_spindles_ar_calibration.csv"
    analyzed_events[event_cols].to_csv(
        raw_out,
        index=False,
    )
    logger.info(
        "Saved compact event-level calibration to %s",
        raw_out,
    )

    generate_fit_quality_report(analyzed_events)
    generate_summary_tables(analyzed_events)


if __name__ == "__main__":
    main()
