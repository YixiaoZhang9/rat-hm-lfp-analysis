#!/usr/bin/env python3
"""
Calibrate AR spindle R thresholds from WAVELET-DETECTED spindle events.

Purpose
-------
Wavelet detection defines the reference spindle events.

For every wavelet-defined event:
    1. Load the corresponding LFP.
    2. Apply the same preprocessing used by the AR detector.
    3. Fit AR(8) in independent 1-s windows at 128 Hz.
    4. In each window, keep the strongest 10-15 Hz pole.
    5. Record that window's R and its frequency.
    6. Summarize the R distribution across wavelet-defined spindles.

No frequency-continuity tracking is used.

The output is deliberately small:
    calibration_event_summary.csv
    calibration_threshold_candidates.csv
    calibration_window_distribution.csv
    calibration_region_summary.csv

The script does NOT decide the final threshold automatically. It reports
the empirical distributions and candidate percentile-based reference values
so that the upper/lower hysteresis thresholds can be selected explicitly.
"""

import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from modules.ephys_preprocessing import bandpass_filter, downsampling
from task_loader import TaskLoader

# ============================================================================
# CONFIGURATION
# ============================================================================

ANALYSIS_ROOTS = [
    Path(
        "/mnt/genzel/Rat/HM/"
        "Rat_HM_Ephys_TD/"
        "Rat_HM_Ephys_TD_Analysis_R1_8"
    ),
    Path(
        "/mnt/genzel/Rat/HM/"
        "Rat_HM_Ephys_TD/"
        "Rat_HM_Ephys_TD_Analysis_R9_16"
    ),
]

SUFFIX = Path(
    "postsleep/wavelet_amp_1_ampcore_3"
)

MANIFEST_PATH = "tasks_manifest.csv"

FS = 1000
TARGET_FS = 128
AR_ORDER = 8
SPINDLE_BAND = (10.0, 15.0)
AR_WINDOW_SEC = 1.0

# Paper-faithful sliding window spacing.
# One sample at 128 Hz = 7.8125 ms.
STRIDE_SAMPLES = 8

START_COL = "spindle_start_time_s"
END_COL = "spindle_end_time_s"

OUTPUT_DIR = Path("results_ar_calibration")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAX_WORKERS = max(
    1,
    (os.cpu_count() or 2) - 1,
)

# Optional quality gate.
# This is NOT an R threshold.
# It only prevents events with very few usable AR windows from dominating
# the calibration statistics.
MIN_VALID_WINDOW_RATIO = 0.50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("wavelet_ar_calibration")


# ============================================================================
# WAVELET EVENTS
# ============================================================================

def load_all_spindle_events() -> pd.DataFrame:
    file_pattern = re.compile(
        r"chan(\d+)(?:_(\d+))?_spindles_wavelet\.csv$"
    )

    rows = []

    for analysis_root in tqdm(
        ANALYSIS_ROOTS,
        desc="Analysis folders",
    ):
        if not analysis_root.exists():
            logger.warning(
                "Analysis root does not exist: %s",
                analysis_root,
            )
            continue

        for rat_group in [
            d for d in analysis_root.iterdir()
            if d.is_dir()
        ]:
            spindle_root = (
                rat_group
                / "Spindle_detection_results"
            )

            if not spindle_root.exists():
                continue

            for region_dir in [
                d for d in spindle_root.iterdir()
                if d.is_dir()
            ]:
                for animal_dir in [
                    d for d in region_dir.iterdir()
                    if d.is_dir()
                ]:
                    for date_dir in [
                        d for d in animal_dir.iterdir()
                        if d.is_dir()
                    ]:
                        csv_folder = (
                            date_dir / SUFFIX
                        )

                        if not csv_folder.exists():
                            continue

                        for csv_file in csv_folder.glob(
                            "*.csv"
                        ):
                            try:
                                df = pd.read_csv(
                                    csv_file
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Could not read %s: %s",
                                    csv_file,
                                    exc,
                                )
                                continue

                            if df.empty:
                                continue

                            match = file_pattern.match(
                                csv_file.name
                            )

                            df["region"] = (
                                region_dir.name
                            )
                            df["rat_number"] = (
                                animal_dir.name
                            )
                            df["date"] = (
                                date_dir.name
                            )
                            df["file"] = (
                                csv_file.name
                            )
                            df["channel"] = (
                                match.group(1)
                                if match
                                else None
                            )
                            df["trial"] = (
                                match.group(2)
                                if match
                                and match.group(2)
                                else ""
                            )

                            df[START_COL] = (
                                pd.to_numeric(
                                    df[
                                        "spindle_start_index"
                                    ],
                                    errors="coerce",
                                )
                                / FS
                            )

                            df[END_COL] = (
                                pd.to_numeric(
                                    df[
                                        "spindle_end_index"
                                    ],
                                    errors="coerce",
                                )
                                / FS
                            )

                            df = df.dropna(
                                subset=[
                                    START_COL,
                                    END_COL,
                                ]
                            )

                            rows.append(df)

    if not rows:
        logger.error(
            "No spindle events found."
        )
        return pd.DataFrame()

    events = pd.concat(
        rows,
        ignore_index=True,
    )

    events["trial"] = (
        events["trial"].fillna("")
    )

    events["event_id"] = np.arange(
        len(events)
    )

    logger.info(
        "Loaded %d wavelet spindle events "
        "across %d files.",
        len(events),
        events["file"].nunique(),
    )

    return events


# ============================================================================
# MANIFEST
# ============================================================================

def normalize_id(x) -> str:
    if x is None:
        return ""

    if str(x).strip() in ("", "nan"):
        return ""

    try:
        return str(int(float(x)))
    except (
        ValueError,
        TypeError,
    ):
        return str(x).strip()


def parse_channel_and_trial(
    data_path: str,
) -> Tuple[str | None, str]:

    match = re.match(
        r"chan(\d+)(?:_(\d+))?\.mat$",
        Path(str(data_path)).name,
        re.IGNORECASE,
    )

    if not match:
        return None, ""

    return (
        match.group(1),
        match.group(2) or "",
    )


def build_task_lookup() -> dict:
    tasks = TaskLoader(
        MANIFEST_PATH
    ).to_tasks()

    lookup = {}

    for task in tasks:

        channel, trial = (
            parse_channel_and_trial(
                task.get(
                    "data_path",
                    "",
                )
            )
        )

        key = (
            str(
                task.get("rat")
            ).strip(),
            str(
                task.get("region")
            ).strip(),
            str(
                task.get("date")
            ).strip(),
            normalize_id(trial),
            normalize_id(channel),
        )

        lookup[key] = task

    logger.info(
        "Built manifest lookup with %d entries.",
        len(lookup),
    )

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


# ============================================================================
# AR FITTING
# ============================================================================

def fit_window_strongest_spindle_pole(
    window: np.ndarray,
) -> Tuple[float, float]:
    """
    Fit AR(8) and return the strongest positive-frequency 10-15 Hz pole.

    Returns
    -------
    r, f
        R and frequency of the strongest spindle-band pole.

    If no valid spindle-band pole exists:
        (np.nan, np.nan)

    There is NO tracking between windows.
    """

    try:
        import statsmodels.api as sm

        window = np.asarray(
            window,
            dtype=float,
        ).squeeze()

        if len(window) <= AR_ORDER + 1:
            return np.nan, np.nan

        if not np.all(
            np.isfinite(window)
        ):
            return np.nan, np.nan

        a, _ = (
            sm.regression
            .linear_model
            .burg(
                window,
                order=AR_ORDER,
                demean=False,
            )
        )

        a = np.asarray(
            a,
            dtype=float,
        ).squeeze()

        if len(a) != AR_ORDER:
            if len(a) > AR_ORDER:
                a = a[-AR_ORDER:]
            else:
                return np.nan, np.nan

        roots = np.roots(
            np.r_[1.0, -a]
        )

        low_f, high_f = SPINDLE_BAND

        frequencies = []
        radii = []

        for root in roots:

            # One member of each conjugate pair.
            if np.imag(root) <= 0:
                continue

            r = float(
                np.abs(root)
            )

            f = float(
                np.angle(root)
                * TARGET_FS
                / (2.0 * np.pi)
            )

            if not np.isfinite(r):
                continue

            if not np.isfinite(f):
                continue

            if (
                low_f <= f <= high_f
            ):
                radii.append(r)
                frequencies.append(f)

        if not radii:
            return np.nan, np.nan

        best = int(
            np.argmax(radii)
        )

        return (
            float(radii[best]),
            float(frequencies[best]),
        )

    except Exception:
        return np.nan, np.nan


# ============================================================================
# SIGNAL PREPARATION
# ============================================================================

def prepare_signal(
    data_path: str,
) -> np.ndarray:

    raw = loadmat(
        data_path
    )["data"].squeeze()

    raw = np.asarray(
        raw,
        dtype=float,
    )

    filtered = bandpass_filter(
        raw,
        lowcut=0.1,
        highcut=100.0,
        fs=FS,
    )

    return downsampling(
        filtered,
        FS,
        TARGET_FS,
    )


# ============================================================================
# EVENT-LEVEL CALIBRATION
# ============================================================================

def analyze_event(
    signal_128: np.ndarray,
    start_s: float,
    end_s: float,
):
    """
    Analyze one wavelet-defined spindle.

    Windows are placed every STRIDE_SAMPLES samples.

    The window is centered on the event timeline, as in the previous
    calibration implementation: each AR estimate is associated with the
    center of its 1-s window.

    Returns only compact scalar summaries, not the raw trajectory.
    """

    if (
        not np.isfinite(start_s)
        or not np.isfinite(end_s)
        or end_s <= start_s
    ):
        return None

    window_samples = int(
        round(
            AR_WINDOW_SEC
            * TARGET_FS
        )
    )

    half_win = (
        window_samples // 2
    )

    c_start = int(
        round(
            start_s * TARGET_FS
        )
    )

    c_end = int(
        round(
            end_s * TARGET_FS
        )
    )

    centers = np.arange(
        c_start,
        max(
            c_start + 1,
            c_end + 1,
        ),
        STRIDE_SAMPLES,
    )

    r_values = []
    f_values = []
    times = []

    for center in centers:

        w_start = (
            center - half_win
        )

        w_end = (
            w_start
            + window_samples
        )

        if (
            w_start < 0
            or w_end > len(signal_128)
        ):
            continue

        r, f = (
            fit_window_strongest_spindle_pole(
                signal_128[
                    w_start:w_end
                ]
            )
        )

        r_values.append(r)
        f_values.append(f)

        times.append(
            center / float(
                TARGET_FS
            )
        )

    if not r_values:
        return None

    r = np.asarray(
        r_values,
        dtype=float,
    )

    f = np.asarray(
        f_values,
        dtype=float,
    )

    t = np.asarray(
        times,
        dtype=float,
    )

    valid = (
        np.isfinite(r)
        & np.isfinite(f)
    )

    n_windows = len(r)
    n_valid = int(
        valid.sum()
    )

    valid_ratio = (
        n_valid / n_windows
        if n_windows
        else 0.0
    )

    if n_valid == 0:
        return {
            "r_max": np.nan,
            "r_min": np.nan,
            "r_mean": np.nan,
            "r_median": np.nan,
            "r_p05": np.nan,
            "r_p10": np.nan,
            "r_p25": np.nan,
            "r_p50": np.nan,
            "r_p75": np.nan,
            "r_p90": np.nan,
            "r_p95": np.nan,
            "r_start": np.nan,
            "r_end": np.nan,
            "peak_frequency_hz": np.nan,
            "peak_time_s": np.nan,
            "n_windows": n_windows,
            "n_valid_windows": n_valid,
            "valid_window_ratio": valid_ratio,
        }

    rv = r[valid]
    fv = f[valid]
    tv = t[valid]

    peak = int(
        np.argmax(rv)
    )

    return {
        "r_max": float(
            np.max(rv)
        ),
        "r_min": float(
            np.min(rv)
        ),
        "r_mean": float(
            np.mean(rv)
        ),
        "r_median": float(
            np.median(rv)
        ),
        "r_p05": float(
            np.quantile(
                rv, 0.05
            )
        ),
        "r_p10": float(
            np.quantile(
                rv, 0.10
            )
        ),
        "r_p25": float(
            np.quantile(
                rv, 0.25
            )
        ),
        "r_p50": float(
            np.quantile(
                rv, 0.50
            )
        ),
        "r_p75": float(
            np.quantile(
                rv, 0.75
            )
        ),
        "r_p90": float(
            np.quantile(
                rv, 0.90
            )
        ),
        "r_p95": float(
            np.quantile(
                rv, 0.95
            )
        ),
        "r_start": float(
            rv[0]
        ),
        "r_end": float(
            rv[-1]
        ),
        "peak_frequency_hz": float(
            fv[peak]
        ),
        "peak_time_s": float(
            tv[peak]
        ),
        "n_windows": n_windows,
        "n_valid_windows": n_valid,
        "valid_window_ratio": valid_ratio,
    }


def process_recording(
    data_path: str,
    event_rows: List[Tuple],
):
    signal_128 = prepare_signal(
        data_path
    )

    results = []

    for (
        event_id,
        start_s,
        end_s,
    ) in event_rows:

        metrics = analyze_event(
            signal_128,
            start_s,
            end_s,
        )

        results.append(
            (
                event_id,
                metrics,
            )
        )

    return results


# ============================================================================
# THRESHOLD / SUMMARY OUTPUT
# ============================================================================

METRIC_COLUMNS = [
    "r_max",
    "r_min",
    "r_mean",
    "r_median",
    "r_p05",
    "r_p10",
    "r_p25",
    "r_p50",
    "r_p75",
    "r_p90",
    "r_p95",
    "r_start",
    "r_end",
    "peak_frequency_hz",
    "peak_time_s",
    "n_windows",
    "n_valid_windows",
    "valid_window_ratio",
]


def build_event_results(
    events: pd.DataFrame,
    lookup: dict,
) -> pd.DataFrame:

    events = events.copy()

    for col in METRIC_COLUMNS:
        events[col] = np.nan

    jobs = {}
    unmatched = []

    for group_key, group in events.groupby(
        [
            "rat_number",
            "region",
            "date",
            "trial",
            "channel",
        ]
    ):

        (
            rat_number,
            region,
            date,
            trial,
            channel,
        ) = group_key

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
                    "channel": channel,
                    "n_events": len(group),
                }
            )
            continue

        data_path = task[
            "data_path"
        ]

        jobs.setdefault(
            data_path,
            [],
        )

        jobs[data_path].extend(
            list(
                zip(
                    group["event_id"],
                    group[START_COL],
                    group[END_COL],
                )
            )
        )

    if unmatched:
        pd.DataFrame(
            unmatched
        ).to_csv(
            OUTPUT_DIR
            / "calibration_unmatched_manifest.csv",
            index=False,
        )

        logger.warning(
            "%d recording groups "
            "could not be matched.",
            len(unmatched),
        )

    logger.info(
        "Processing %d recordings.",
        len(jobs),
    )

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                process_recording,
                path,
                rows,
            ): path
            for path, rows in jobs.items()
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Wavelet -> AR calibration",
        ):

            path = futures[
                future
            ]

            try:

                for (
                    event_id,
                    metrics,
                ) in future.result():

                    if metrics is None:
                        continue

                    mask = (
                        events["event_id"]
                        == event_id
                    )

                    for col in METRIC_COLUMNS:
                        events.loc[
                            mask,
                            col,
                        ] = metrics[col]

            except Exception as exc:

                logger.error(
                    "Worker failed for %s: %s",
                    path,
                    exc,
                )

    return events


def make_threshold_candidate_table(
    events: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for region, group in events.groupby(
        "region"
    ):

        # Quality filtering is based only on availability of a valid
        # AR measurement, not on R itself.
        qualified = group[
            group[
                "valid_window_ratio"
            ]
            >= MIN_VALID_WINDOW_RATIO
        ].copy()

        if qualified.empty:
            continue

        # Candidate upper thresholds:
        # quantiles of event-level Rmax.
        #
        # These are descriptive calibration candidates, not an automatic
        # claim of an optimal detector threshold.
        upper_percentiles = [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.50,
        ]

        # Candidate lower thresholds:
        # quantiles of within-event R values summarized by event median,
        # minimum, and end-of-event R.
        #
        # Keeping several quantities visible avoids hiding the distinction
        # between "how strong is the spindle at its peak?" and "how low does
        # R tend to fall while the wavelet event is still present?"
        row = {
            "Region": region,
            "N_wavelet_events": len(group),
            "N_qualified_events": len(
                qualified
            ),
            "Qualified_fraction": (
                len(qualified)
                / len(group)
            ),
        }

        for p in upper_percentiles:
            row[
                f"Rmax_p{int(p * 100):02d}"
            ] = float(
                np.quantile(
                    qualified["r_max"],
                    p,
                )
            )

        for source in [
            "r_min",
            "r_median",
            "r_end",
        ]:
            for p in [
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]:
                row[
                    f"{source}_p{int(p * 100):02d}"
                ] = float(
                    np.quantile(
                        qualified[source],
                        p,
                    )
                )

        rows.append(row)

    return pd.DataFrame(rows)


def make_region_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for region, group in events.groupby(
        "region"
    ):

        qualified = group[
            group[
                "valid_window_ratio"
            ]
            >= MIN_VALID_WINDOW_RATIO
        ]

        row = {
            "Region": region,
            "N_wavelet_events": len(group),
            "N_qualified_events": len(
                qualified
            ),
        }

        for col in [
            "r_max",
            "r_min",
            "r_mean",
            "r_median",
            "r_start",
            "r_end",
            "peak_frequency_hz",
            "valid_window_ratio",
        ]:

            if qualified.empty:
                row[
                    f"{col}_mean"
                ] = np.nan
                row[
                    f"{col}_median"
                ] = np.nan
                row[
                    f"{col}_std"
                ] = np.nan
            else:
                row[
                    f"{col}_mean"
                ] = float(
                    qualified[col].mean()
                )
                row[
                    f"{col}_median"
                ] = float(
                    qualified[col].median()
                )
                row[
                    f"{col}_std"
                ] = float(
                    qualified[col].std()
                )

        rows.append(row)

    return pd.DataFrame(rows)


def make_window_distribution(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact distribution of R across all valid AR windows inside
    wavelet-defined events.

    This requires re-reading raw trajectories, which we intentionally
    do NOT store. Therefore this table is generated from event-level
    summaries only and is not an all-window distribution.

    The event-level R quantiles are the primary calibration output.
    """

    rows = []

    for region, group in events.groupby(
        "region"
    ):

        qualified = group[
            group[
                "valid_window_ratio"
            ]
            >= MIN_VALID_WINDOW_RATIO
        ]

        if qualified.empty:
            continue

        for metric in [
            "r_min",
            "r_median",
            "r_end",
        ]:

            values = qualified[
                metric
            ].dropna()

            if values.empty:
                continue

            rows.append(
                {
                    "Region": region,
                    "Metric": metric,
                    "N_events": len(values),
                    "p05": values.quantile(
                        0.05
                    ),
                    "p10": values.quantile(
                        0.10
                    ),
                    "p25": values.quantile(
                        0.25
                    ),
                    "p50": values.quantile(
                        0.50
                    ),
                    "p75": values.quantile(
                        0.75
                    ),
                    "p90": values.quantile(
                        0.90
                    ),
                    "p95": values.quantile(
                        0.95
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():

    events = load_all_spindle_events()

    if events.empty:
        return

    lookup = build_task_lookup()

    results = build_event_results(
        events,
        lookup,
    )

    # Small event-level table: one row per wavelet spindle.
    event_columns = [
        "event_id",
        "region",
        "rat_number",
        "date",
        "file",
        "channel",
        "trial",
        START_COL,
        END_COL,
    ] + METRIC_COLUMNS

    event_columns = [
        c for c in event_columns
        if c in results.columns
    ]

    event_out = results[
        event_columns
    ].copy()

    event_out.to_csv(
        OUTPUT_DIR
        / "calibration_event_summary.csv",
        index=False,
    )

    threshold_candidates = (
        make_threshold_candidate_table(
            results
        )
    )

    threshold_candidates.to_csv(
        OUTPUT_DIR
        / "calibration_threshold_candidates.csv",
        index=False,
    )

    region_summary = (
        make_region_summary(
            results
        )
    )

    region_summary.to_csv(
        OUTPUT_DIR
        / "calibration_region_summary.csv",
        index=False,
    )

    window_distribution = (
        make_window_distribution(
            results
        )
    )

    window_distribution.to_csv(
        OUTPUT_DIR
        / "calibration_window_distribution.csv",
        index=False,
    )

    # Console output: this is the only part we really need to inspect
    # before deciding thresholds.
    print()
    print("=" * 80)
    print("WAVELET -> AR CALIBRATION")
    print("=" * 80)
    print(
        f"Wavelet events: {len(results):,}"
    )
    print()

    print(
        "Region summary:"
    )
    print(
        region_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Threshold candidates:"
    )
    print(
        threshold_candidates.to_string(
            index=False
        )
    )

    print()
    print(
        "Output files:"
    )

    for path in [
        OUTPUT_DIR
        / "calibration_event_summary.csv",
        OUTPUT_DIR
        / "calibration_threshold_candidates.csv",
        OUTPUT_DIR
        / "calibration_region_summary.csv",
        OUTPUT_DIR
        / "calibration_window_distribution.csv",
    ]:
        print(
            f"  {path}"
        )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "These percentile values are empirical reference values "
        "from wavelet-defined events."
    )
    print(
        "They are not automatically declared to be the final "
        "upper/lower detector thresholds."
    )
    print(
        "The final hysteresis pair should be selected after inspecting "
        "these distributions and, ideally, the corresponding "
        "non-wavelet/background R distribution."
    )


if __name__ == "__main__":
    main()
