"""
compute_spindle_ar_r_values.py

Use WAVELET-DETECTED spindles as the event definition and measure AR pole
dynamics inside those events.

Important scientific distinction:
    Wavelet detection -> defines spindle start/end
    AR analysis      -> measures R(t) and frequency; it does NOT define
                        whether the spindle is present.

For each wavelet-defined spindle:
  1. Slides a 1.0 s AR window across the event.
  2. Fits AR(8) and obtains all positive-frequency poles.
  3. Keeps poles in the 10-15 Hz spindle band.
  4. Tracks one spindle-frequency pole through time by frequency continuity.
  5. Extracts R_max, R_min, R_mean, R_start, R_end, peak frequency,
     peak time, and an 11-point normalized R(t) profile.
  6. Reports fit quality using the fraction of windows containing a
     valid spindle-band pole.

No R threshold is applied to the wavelet events.

The original output files are retained where possible so downstream analysis
does not need to change.
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
from scipy.io import loadmat
from scipy.signal import find_peaks
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.ephys_preprocessing import bandpass_filter, downsampling
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

FS = 1000
TARGET_FS = 128
AR_ORDER = 8
SPINDLE_BAND = (10.0, 15.0)
AR_WINDOW_SEC = 1.0

# 4 samples at 128 Hz = 31.25 ms temporal spacing.
STRIDE_SAMPLES = 4

# An event must have an in-band pole in at least this fraction of its
# analysis windows to enter calibration summaries.
MIN_IN_BAND_RATIO = 0.5

START_COL = "spindle_start_time_s"
END_COL = "spindle_end_time_s"

OUTPUT_DIR = Path("results_ar_calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Maximum allowed frequency jump when following a pole trajectory.
# This is a tracking rule, not a spindle detection threshold.
MAX_FREQUENCY_JUMP_HZ = 1.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ar_calibration")


# --------------------------------------------------------------------------- #
# 1. Load Ground-Truth Wavelet Spindles
# --------------------------------------------------------------------------- #
def load_all_spindle_events() -> pd.DataFrame:
    file_pattern = re.compile(r"chan(\d+)(?:_(\d+))?_spindles_wavelet\.csv$")
    rows = []

    for analysis_root in tqdm(ANALYSIS_ROOTS, desc="Analysis folders"):
        if not analysis_root.exists():
            logger.warning("Analysis root does not exist: %s", analysis_root)
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
                            except Exception as exc:
                                logger.warning(
                                    "Could not read %s: %s",
                                    csv_file,
                                    exc,
                                )
                                continue

                            if df.empty:
                                continue

                            match = file_pattern.match(csv_file.name)

                            df["region"] = region_dir.name
                            df["rat_number"] = animal_dir.name
                            df["date"] = date_dir.name
                            df["file"] = csv_file.name
                            df["channel"] = match.group(1) if match else None
                            df["trial"] = (
                                match.group(2)
                                if match and match.group(2)
                                else ""
                            )

                            df[START_COL] = (
                                pd.to_numeric(
                                    df["spindle_start_index"],
                                    errors="coerce",
                                )
                                / FS
                            )
                            df[END_COL] = (
                                pd.to_numeric(
                                    df["spindle_end_index"],
                                    errors="coerce",
                                )
                                / FS
                            )

                            df = df.dropna(subset=[START_COL, END_COL])
                            rows.append(df)

    if not rows:
        logger.error("No spindle events found under ANALYSIS_ROOTS.")
        return pd.DataFrame()

    events = pd.concat(rows, ignore_index=True)
    events["trial"] = events["trial"].fillna("")

    logger.info(
        "Loaded %d spindle events across %d files.",
        len(events),
        events["file"].nunique(),
    )
    return events


# --------------------------------------------------------------------------- #
# 2. Manifest Lookup
# --------------------------------------------------------------------------- #
def normalize_id(x) -> str:
    if x is None or str(x).strip() in ("", "nan"):
        return ""

    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return str(x).strip()


def parse_channel_and_trial(data_path: str) -> Tuple[str | None, str]:
    match = re.match(
        r"chan(\d+)(?:_(\d+))?\.mat$",
        Path(str(data_path)).name,
        re.IGNORECASE,
    )

    if not match:
        return None, ""

    return match.group(1), (match.group(2) or "")


def build_task_lookup() -> dict:
    tasks = TaskLoader(MANIFEST_PATH).to_tasks()

    lookup = {}

    for task in tasks:
        channel, trial = parse_channel_and_trial(
            task.get("data_path", "")
        )

        key = (
            str(task.get("rat")).strip(),
            str(task.get("region")).strip(),
            str(task.get("date")).strip(),
            normalize_id(trial),
            normalize_id(channel),
        )

        lookup[key] = task

    logger.info("Built manifest lookup with %d entries.", len(lookup))
    return lookup


def find_task(
    lookup: dict,
    rat_number,
    region,
    date,
    trial,
    channel,
):
    key = (
        str(rat_number).strip(),
        str(region).strip(),
        str(date).strip(),
        normalize_id(trial),
        normalize_id(channel),
    )

    return lookup.get(key)


# --------------------------------------------------------------------------- #
# 3. AR fitting
# --------------------------------------------------------------------------- #
def fit_window_all_poles(
    window: np.ndarray,
    ar_order: int,
    target_fs: int,
    spindle_band: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """
    Fit AR(order) using Burg and return all positive-frequency poles
    in the requested frequency band.

    Returns:
        list of (radius, frequency_hz)

    No R threshold is applied here.
    """
    try:
        import statsmodels.api as sm

        window = np.asarray(window, dtype=float).squeeze()

        if len(window) <= ar_order + 1:
            return []

        if not np.all(np.isfinite(window)):
            return []

        # Preserve the convention used by the existing detector.
        result = sm.regression.linear_model.burg(
            window,
            ar_order,
            demean=False,
        )

        # statsmodels versions can expose the AR coefficients slightly
        # differently, so handle the common forms.
        if hasattr(result, "params"):
            a = np.asarray(result.params, dtype=float).squeeze()
        elif isinstance(result, (tuple, list)) and len(result) > 0:
            a = np.asarray(result[0], dtype=float).squeeze()
        else:
            a = np.asarray(result, dtype=float).squeeze()

        # Some implementations may include an intercept; the AR model used
        # here should contain exactly ar_order coefficients.
        if len(a) != ar_order:
            if len(a) > ar_order:
                a = a[-ar_order:]
            else:
                return []

        # AR polynomial:
        # z^p - a1*z^(p-1) - ... - ap = 0
        roots = np.roots(np.r_[1.0, -a])

        low_f, high_f = spindle_band
        dt = 1.0 / float(target_fs)

        poles = []

        for root in roots:
            # Keep one member of each complex-conjugate pair.
            if np.imag(root) <= 0:
                continue

            radius = float(np.abs(root))
            phase = float(np.angle(root))

            frequency = phase / (2.0 * np.pi * dt)

            if not np.isfinite(radius) or not np.isfinite(frequency):
                continue

            if low_f <= frequency <= high_f:
                poles.append((radius, frequency))

        return poles

    except Exception:
        return []


def select_tracked_pole(
    poles: List[Tuple[float, float]],
    previous_frequency: float | None,
    spindle_band: Tuple[float, float],
    max_frequency_jump_hz: float,
):
    """
    Select the pole representing the same spindle-frequency trajectory.

    First valid window:
        choose the pole with the largest R.

    Subsequent windows:
        choose the pole closest in frequency to the previous frequency.

    If no pole is sufficiently close, return NaN for this window rather than
    silently switching to a different oscillator.

    This is a measurement/tracking rule, not an event-detection threshold.
    """
    if not poles:
        return np.nan, np.nan

    low_f, high_f = spindle_band

    candidates = [
        (float(r), float(f))
        for r, f in poles
        if np.isfinite(r)
        and np.isfinite(f)
        and low_f <= f <= high_f
    ]

    if not candidates:
        return np.nan, np.nan

    if previous_frequency is None or not np.isfinite(previous_frequency):
        return max(candidates, key=lambda x: x[0])

    candidates.sort(key=lambda x: abs(x[1] - previous_frequency))

    r, f = candidates[0]

    if abs(f - previous_frequency) > max_frequency_jump_hz:
        return np.nan, np.nan

    return r, f


# --------------------------------------------------------------------------- #
# 4. Signal preparation
# --------------------------------------------------------------------------- #
def prepare_signal(data_path: str) -> np.ndarray:
    raw = loadmat(data_path)["data"].squeeze()

    raw = np.asarray(raw, dtype=float)

    filtered = bandpass_filter(
        raw,
        lowcut=0.1,
        highcut=100,
        fs=FS,
    )

    return downsampling(
        filtered,
        FS,
        TARGET_FS,
    )


# --------------------------------------------------------------------------- #
# 5. High-resolution Event AR Profile
# --------------------------------------------------------------------------- #
def empty_dynamics_result() -> Dict:
    return {
        "r_max": np.nan,
        "r_min": np.nan,
        "r_mean": np.nan,
        "r_median": np.nan,
        "r_start": np.nan,
        "r_end": np.nan,
        "r_peak_freq": np.nan,
        "r_peak_time": np.nan,
        "r_area": np.nan,
        "r_profile": [np.nan] * 11,
        "in_band_ratio": 0.0,
        "n_windows": 0,
        "r_times": [],
        "r_values": [],
        "frequency_values": [],
    }


def make_normalized_profile(
    times: np.ndarray,
    values: np.ndarray,
    n_points: int = 11,
) -> List[float]:
    """
    Interpolate only across valid AR measurements.

    This avoids the old problem where missing poles were represented by 0
    and then interpolated into apparently real low-R values.
    """
    if n_points <= 0:
        return []

    valid = (
        np.isfinite(times)
        & np.isfinite(values)
    )

    if not np.any(valid):
        return [np.nan] * n_points

    t = times[valid]
    r = values[valid]

    if len(r) == 1:
        return [float(r[0])] * n_points

    t0 = float(t[0])
    t1 = float(t[-1])

    if t1 <= t0:
        return [float(r[0])] * n_points

    x = (t - t0) / (t1 - t0)
    x_profile = np.linspace(0.0, 1.0, n_points)

    return np.interp(
        x_profile,
        x,
        r,
    ).astype(float).tolist()


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
    Measure AR pole dynamics for a WAVELET-DEFINED spindle.

    The event is NOT accepted/rejected based on R.

    AR windows are centered from onset to offset, as in the original script.
    Therefore the 1 s windows can extend approximately 0.5 s outside the
    wavelet-defined event. This is intentional for the first analysis pass
    because it lets us inspect onset/offset R dynamics.

    Missing in-band poles are NaN.

    Returns raw trajectories in addition to summary metrics.
    """
    try:
        start_s = float(start_s)
        end_s = float(end_s)
    except (TypeError, ValueError):
        return empty_dynamics_result()

    if (
        not np.isfinite(start_s)
        or not np.isfinite(end_s)
        or end_s <= start_s
    ):
        return empty_dynamics_result()

    win_samples = int(round(window_sec * target_fs))
    half_win = win_samples // 2

    if win_samples <= 0:
        return empty_dynamics_result()

    c_start = int(round(start_s * target_fs))
    c_end = int(round(end_s * target_fs))

    centers = np.arange(
        c_start,
        max(c_start + 1, c_end + 1),
        stride_samples,
    )

    r_series = []
    f_series = []
    time_series = []

    previous_frequency = None

    for c in centers:
        w_start = c - half_win
        w_end = w_start + win_samples

        if w_start < 0 or w_end > len(signal_128):
            continue

        window = signal_128[w_start:w_end]

        poles = fit_window_all_poles(
            window,
            ar_order,
            target_fs,
            spindle_band,
        )

        r_val, f_val = select_tracked_pole(
            poles,
            previous_frequency,
            spindle_band,
            MAX_FREQUENCY_JUMP_HZ,
        )

        center_time = c / float(target_fs)

        time_series.append(center_time)
        r_series.append(r_val)
        f_series.append(f_val)

        if np.isfinite(f_val):
            previous_frequency = f_val

    if not r_series:
        return empty_dynamics_result()

    t_arr = np.asarray(time_series, dtype=float)
    r_arr = np.asarray(r_series, dtype=float)
    f_arr = np.asarray(f_series, dtype=float)

    n_windows = len(r_arr)

    valid_mask = (
        np.isfinite(r_arr)
        & np.isfinite(f_arr)
    )

    in_band_ratio = float(np.mean(valid_mask))

    if not np.any(valid_mask):
        result = empty_dynamics_result()
        result["in_band_ratio"] = in_band_ratio
        result["n_windows"] = n_windows
        result["r_times"] = t_arr.tolist()
        result["r_values"] = r_arr.tolist()
        result["frequency_values"] = f_arr.tolist()
        return result

    valid_indices = np.flatnonzero(valid_mask)
    valid_r = r_arr[valid_mask]
    valid_f = f_arr[valid_mask]
    valid_t = t_arr[valid_mask]

    peak_local_idx = int(np.argmax(valid_r))
    peak_global_idx = int(valid_indices[peak_local_idx])

    r_max = float(valid_r[peak_local_idx])
    r_min = float(np.min(valid_r))
    r_mean = float(np.mean(valid_r))
    r_median = float(np.median(valid_r))

    r_start = float(valid_r[0])
    r_end = float(valid_r[-1])

    r_peak_freq = float(valid_f[peak_local_idx])
    r_peak_time = float(valid_t[peak_local_idx])

    if len(valid_t) >= 2:
        try:
            r_area = float(np.trapezoid(valid_r, valid_t))
        except AttributeError:
            r_area = float(np.trapz(valid_r, valid_t))
    else:
        r_area = np.nan

    profile = make_normalized_profile(
        t_arr,
        r_arr,
        n_points=11,
    )

    return {
        "r_max": r_max,
        "r_min": r_min,
        "r_mean": r_mean,
        "r_median": r_median,
        "r_start": r_start,
        "r_end": r_end,
        "r_peak_freq": r_peak_freq,
        "r_peak_time": r_peak_time,
        "r_area": r_area,
        "r_profile": profile,
        "in_band_ratio": in_band_ratio,
        "n_windows": n_windows,
        "r_times": t_arr.tolist(),
        "r_values": r_arr.tolist(),
        "frequency_values": f_arr.tolist(),
    }


def process_group(
    data_path: str,
    event_rows: List[Tuple],
) -> List[Tuple]:
    """Process all wavelet spindles belonging to one physical recording."""
    signal_128 = prepare_signal(data_path)

    results = []

    for idx, start_s, end_s in event_rows:
        metrics = analyze_spindle_r_dynamics(
            signal_128,
            start_s,
            end_s,
        )
        results.append((idx, metrics))

    return results


# --------------------------------------------------------------------------- #
# 6. Orchestration & Multiprocessing
# --------------------------------------------------------------------------- #
def compute_all_dynamics(
    events: pd.DataFrame,
    lookup: dict,
) -> pd.DataFrame:
    events = events.copy()

    scalar_cols = [
        "r_max",
        "r_min",
        "r_mean",
        "r_median",
        "r_start",
        "r_end",
        "r_peak_freq",
        "r_peak_time",
        "r_area",
        "in_band_ratio",
        "n_windows",
    ]

    for col in scalar_cols:
        events[col] = np.nan

    profile_cols = [
        f"r_profile_{p}%"
        for p in range(0, 101, 10)
    ]

    for col in profile_cols:
        events[col] = np.nan

    # Raw trajectories are saved as JSON-like strings in the CSV so that
    # the event-level file retains the actual R(t), rather than only the
    # normalized 11-point profile.
    events["r_times"] = ""
    events["r_values"] = ""
    events["frequency_values"] = ""

    jobs = {}
    unmatched = []

    group_cols = [
        "rat_number",
        "region",
        "date",
        "trial",
        "channel",
    ]

    for group_key, group in events.groupby(group_cols):
        rat_number, region, date, trial, channel = group_key

        task = find_task(
            lookup,
            rat_number,
            region,
            date,
            trial,
            channel,
        )

        if task is None:
            unmatched.append(
                {
                    "rat": rat_number,
                    "region": region,
                    "date": date,
                    "trial": trial,
                    "chan": channel,
                    "n_events": len(group),
                }
            )
            continue

        data_path = task["data_path"]

        # Normally one physical recording corresponds to one group.
        # If the manifest contains repeated entries, extend rather than
        # overwrite the existing event list.
        jobs.setdefault(data_path, [])
        jobs[data_path].extend(
            list(
                zip(
                    group.index,
                    group[START_COL],
                    group[END_COL],
                )
            )
        )

    if unmatched:
        unmatched_df = pd.DataFrame(unmatched).drop_duplicates(
            subset=["rat", "region", "date", "trial", "chan"]
        )

        unmatched_df.to_csv(
            OUTPUT_DIR / "unmatched_manifest.csv",
            index=False,
        )

        total_unmatched_events = unmatched_df["n_events"].sum()

        logger.warning(
            "%d group(s) (%d events) could not be matched to raw data.",
            len(unmatched_df),
            total_unmatched_events,
        )

        by_region = (
            unmatched_df
            .groupby("region")["n_events"]
            .sum()
            .sort_values(ascending=False)
        )

        logger.warning(
            "Unmatched events by region:\n%s",
            by_region.to_string(),
        )

    logger.info(
        "Analyzing %d events across %d files using %d workers...",
        len(events),
        len(jobs),
        MAX_WORKERS,
    )

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                process_group,
                path,
                rows,
            ): path
            for path, rows in jobs.items()
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fitting AR trajectories",
        ):
            path = futures[future]

            try:
                for idx, metrics in future.result():

                    for col in scalar_cols:
                        events.at[idx, col] = metrics[col]

                    for col_name, val in zip(
                        profile_cols,
                        metrics["r_profile"],
                    ):
                        events.at[idx, col_name] = val

                    # CSV-safe representation of raw trajectories.
                    events.at[idx, "r_times"] = repr(
                        metrics["r_times"]
                    )
                    events.at[idx, "r_values"] = repr(
                        metrics["r_values"]
                    )
                    events.at[idx, "frequency_values"] = repr(
                        metrics["frequency_values"]
                    )

            except Exception as exc:
                logger.error(
                    "Worker failed for %s: %s",
                    path,
                    exc,
                )

    return events


# --------------------------------------------------------------------------- #
# 7. Fit Quality
# --------------------------------------------------------------------------- #
def generate_fit_quality_report(
    df: pd.DataFrame,
):
    """
    Diagnose AR fitting before threshold/calibration statistics are used.
    """
    has_metrics = df.dropna(
        subset=["in_band_ratio"]
    ).copy()

    if has_metrics.empty:
        logger.error(
            "No events had AR metrics computed."
        )
        return

    records = []

    for region, reg_df in has_metrics.groupby("region"):
        n = len(reg_df)

        n_total_fail = int(
            (reg_df["in_band_ratio"] == 0).sum()
        )

        n_below_gate = int(
            (reg_df["in_band_ratio"] < MIN_IN_BAND_RATIO).sum()
        )

        records.append(
            {
                "Region": region,
                "N_Events": n,
                "Mean_In_Band_Ratio": round(
                    float(
                        reg_df["in_band_ratio"].mean()
                    ),
                    3,
                ),
                "Pct_Complete_Fit_Failure (ratio==0)": round(
                    100 * n_total_fail / n,
                    2,
                ),
                f"Pct_Below_Gate (<{MIN_IN_BAND_RATIO})": round(
                    100 * n_below_gate / n,
                    2,
                ),
            }
        )

    report = pd.DataFrame(records)

    report.to_csv(
        OUTPUT_DIR / "fit_quality_report.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print(
        "AR FIT QUALITY BY REGION"
    )
    print("=" * 80)
    print(report.to_string(index=False))

    print(
        "\nEvents with no valid spindle-band pole are treated as fit failures "
        "rather than R=0. Events below the minimum in-band-ratio gate are "
        "excluded from calibration summaries.\n"
    )


# --------------------------------------------------------------------------- #
# 8. Summary Tables
# --------------------------------------------------------------------------- #
def generate_summary_tables(
    df: pd.DataFrame,
):
    valid = df.dropna(
        subset=["r_max", "r_min"]
    ).copy()

    valid = valid[
        valid["in_band_ratio"] >= MIN_IN_BAND_RATIO
    ].copy()

    if valid.empty:
        logger.error(
            "No valid AR events remained after quality filtering."
        )
        return

    n_before = int(
        df["r_max"].notna().sum()
    )

    logger.info(
        "Calibration uses %d events with in_band_ratio >= %.2f; "
        "%d fitted events excluded as low-confidence.",
        len(valid),
        MIN_IN_BAND_RATIO,
        n_before - len(valid),
    )

    quantiles = [
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
    ]

    # ------------------------------------------------------------------ #
    # A. Pooled regional summary
    # ------------------------------------------------------------------ #
    records = []

    for region, reg_df in valid.groupby("region"):
        row = {
            "Region": region,
            "N_Spindles": len(reg_df),
            "N_Rats": reg_df["rat_number"].nunique(),
        }

        for metric in [
            "r_max",
            "r_min",
            "r_start",
            "r_end",
        ]:
            vals = reg_df[metric].values

            row[f"{metric}_mean"] = np.mean(vals)
            row[f"{metric}_std"] = np.std(vals)

            for q in quantiles:
                row[
                    f"{metric}_p{int(q * 100)}"
                ] = np.quantile(vals, q)

        records.append(row)

    pooled_summary = pd.DataFrame(records)

    pooled_summary.to_csv(
        OUTPUT_DIR / "regional_ar_pooled_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------ #
    # B. Rat-averaged summary
    # ------------------------------------------------------------------ #
    rat_means = (
        valid
        .groupby(
            ["region", "rat_number"]
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
        .agg(["mean", "std"])
        .reset_index()
    )

    rat_agg.columns = [
        "_".join(
            filter(None, c)
        )
        for c in rat_agg.columns
    ]

    rat_agg.to_csv(
        OUTPUT_DIR / "regional_ar_rat_averaged_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------ #
    # C. Empirical threshold reference
    # ------------------------------------------------------------------ #
    #
    # These are deliberately presented as empirical reference values,
    # not as "the correct threshold". The wavelet events are the reference
    # events, and the R distributions describe what AR R looks like for them.
    #
    calib = []

    for region, reg_df in valid.groupby("region"):
        p25_max = np.quantile(
            reg_df["r_max"],
            0.25,
        )

        p50_max = np.quantile(
            reg_df["r_max"],
            0.50,
        )

        p10_min = np.quantile(
            reg_df["r_min"],
            0.10,
        )

        p25_end = np.quantile(
            reg_df["r_end"],
            0.25,
        )

        lower_reference = min(
            p10_min,
            p25_end,
        )

        calib.append(
            {
                "Region": region,
                "N_Spindles_Used": len(reg_df),
                "Rmax_P25": round(
                    p25_max,
                    3,
                ),
                "Rmax_Median": round(
                    p50_max,
                    3,
                ),
                "Rmin_P10": round(
                    p10_min,
                    3,
                ),
                "Rend_P25": round(
                    p25_end,
                    3,
                ),
                "Lower_Reference": round(
                    lower_reference,
                    3,
                ),
                "Upper_Lower_Separation": round(
                    p25_max - lower_reference,
                    3,
                ),
            }
        )

    calib_df = pd.DataFrame(calib)

    calib_df.to_csv(
        OUTPUT_DIR / "recommended_hysteresis_thresholds.csv",
        index=False,
    )

    # ------------------------------------------------------------------ #
    # D. Mean temporal evolution
    # ------------------------------------------------------------------ #
    profile_cols = [
        f"r_profile_{p}%"
        for p in range(0, 101, 10)
    ]

    evolution = (
        valid
        .groupby("region")[profile_cols]
        .mean()
        .reset_index()
    )

    evolution.to_csv(
        OUTPUT_DIR / "regional_r_evolution_profile.csv",
        index=False,
    )

    # ------------------------------------------------------------------ #
    # E. Additional descriptive statistics
    # ------------------------------------------------------------------ #
    dynamics_summary = (
        valid
        .groupby("region")[
            [
                "r_mean",
                "r_median",
                "r_area",
                "r_peak_freq",
                "r_peak_time",
                "in_band_ratio",
                "n_windows",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    dynamics_summary.columns = [
        "_".join(
            filter(None, c)
        )
        for c in dynamics_summary.columns
    ]

    dynamics_summary.to_csv(
        OUTPUT_DIR / "regional_ar_dynamics_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------ #
    # Console
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print(
        "EMPIRICAL R DISTRIBUTION / HYSTERESIS REFERENCE"
    )
    print("=" * 80)
    print(calib_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(
        "MEAN R EVOLUTION ACROSS WAVELET SPINDLE DURATION"
    )
    print("=" * 80)
    print(evolution.to_string(index=False))

    print("\n" + "=" * 80)
    print(
        "AR DYNAMICS SUMMARY"
    )
    print("=" * 80)
    print(dynamics_summary.to_string(index=False))
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    events = load_all_spindle_events()

    if events.empty:
        return

    lookup = build_task_lookup()

    analyzed_events = compute_all_dynamics(
        events,
        lookup,
    )

    raw_out = (
        OUTPUT_DIR
        / "wavelet_spindles_with_ar_dynamics.csv"
    )

    analyzed_events.to_csv(
        raw_out,
        index=False,
    )

    logger.info(
        "Saved full event-level dynamics to %s",
        raw_out,
    )

    generate_fit_quality_report(
        analyzed_events
    )

    generate_summary_tables(
        analyzed_events
    )


if __name__ == "__main__":
    main()
