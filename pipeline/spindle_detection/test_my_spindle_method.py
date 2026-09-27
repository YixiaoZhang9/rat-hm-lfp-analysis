import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from detector import convert_events_to_array, detect_events
from task_loader import TaskLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FS = 1000
TARGET_FS = 128

RA = 0.70
RB = 0.80

# Threshold pairs used by run_threshold_sweep().  The detector requires
# r_a < r_b; an event starts above r_b and ends below r_a.
THRESHOLD_PAIRS = [
    (0.90, 0.95),
    (0.80, 0.85),
    (0.75, 0.80),
]

# Used for the final sweep column.  Keep this explicit because
# "extremely long" is otherwise ambiguous.
EXTREMELY_LONG_DURATION_S = 4.0

SPINDLE_LOW = 9.0
SPINDLE_HIGH = 20.0


OUTPUT_DIR = Path("results")

SPINDLES_OUT_CSV = OUTPUT_DIR / "all_detected_spindles_per_region.csv"

FAILED_OUT_CSV = OUTPUT_DIR / "extraction_failed_files.csv"

THRESHOLD_SWEEP_OUT_CSV = OUTPUT_DIR / "spindle_threshold_sweep_summary.csv"


MAX_WORKERS = max(1, (os.cpu_count() or 1) - 2)

CHECKPOINT_EVERY = 25


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_path = LOG_DIR / f"extract_spindles_{run_stamp}.log"


logger = logging.getLogger("extract_spindles")

logger.setLevel(logging.DEBUG)

logger.handlers.clear()


console_handler = logging.StreamHandler(sys.stdout)

console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_path, mode="w")

file_handler.setLevel(logging.DEBUG)


logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_signal(signal):
    """
    Only resampling.

    1000 Hz -> 128 Hz
    """

    signal = np.asarray(signal, dtype=float)

    return resample_poly(signal, TARGET_FS, FS)


def select_spindles(events):
    """
    Keep only spindle-frequency events.

    detect_events() remains
    frequency independent.
    """

    selected = []

    for e in events:

        if SPINDLE_LOW <= e["frequency"] <= SPINDLE_HIGH:
            selected.append(e)

    return selected


# ---------------------------------------------------------------------------
# NREM intervals
# ---------------------------------------------------------------------------


def get_nrem_intervals(scoring_path):

    states = loadmat(scoring_path)["states"].squeeze()

    nrem_mask = (states == 3).astype(int)

    diff = np.diff(np.concatenate(([0], nrem_mask, [0])))

    starts = np.where(diff == 1)[0]

    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return np.empty((0, 2))

    return np.column_stack((starts, ends))


# ---------------------------------------------------------------------------
# Failure class
# ---------------------------------------------------------------------------


class TaskFailure(Exception):

    def __init__(self, reason: str, detail: str = ""):

        self.reason = reason
        self.detail = detail

        super().__init__(f"{reason}: {detail}" if detail else reason)


# ---------------------------------------------------------------------------
# Extract spindles from one file
# ---------------------------------------------------------------------------


def extract_spindles_for_file(
    task: Dict,
    r_a: float = RA,
    r_b: float = RB,
) -> pd.DataFrame:

    data_path = task["data_path"]
    scoring_path = task["scoring_path"]

    if not os.path.exists(data_path):
        raise TaskFailure("data file not found", data_path)

    if not os.path.exists(scoring_path):
        raise TaskFailure("scoring file not found", scoring_path)

    # ------------------------------------------------------------
    # Load LFP
    # ------------------------------------------------------------

    try:

        raw_signal = loadmat(data_path)["data"].squeeze()

    except Exception as e:

        raise TaskFailure("data read error", str(e))

    # ------------------------------------------------------------
    # Load NREM
    # ------------------------------------------------------------

    try:

        nrem_intervals = get_nrem_intervals(scoring_path)

    except Exception as e:

        raise TaskFailure("scoring read error", str(e))

    if len(nrem_intervals) == 0:

        raise TaskFailure("no NREM epochs")

    all_spindles = []

    # ------------------------------------------------------------
    # Process every NREM block
    # ------------------------------------------------------------

    for start, end in nrem_intervals:

        # NO BUFFER
        start_sample = int(start * FS)

        end_sample = int(end * FS)

        segment = raw_signal[start_sample:end_sample]

        if len(segment) < FS:

            continue

        # --------------------------------------------------------
        # 1000 Hz -> 128 Hz
        # --------------------------------------------------------

        segment = preprocess_signal(segment)

        # --------------------------------------------------------
        # Detect ALL AR events
        # --------------------------------------------------------

        events = detect_events(
            signal=segment,
            fs=TARGET_FS,
            r_a=r_a,
            r_b=r_b,
        )

        # --------------------------------------------------------
        # Keep spindle frequency only
        # --------------------------------------------------------

        events = select_spindles(events)

        spindles = convert_events_to_array(events)

        if len(spindles) == 0:

            continue

        # --------------------------------------------------------
        # Convert local NREM time -> global time
        # --------------------------------------------------------

        spindles[:, 0:3] += start_sample / FS

        # --------------------------------------------------------
        # Keep strictly inside NREM
        # --------------------------------------------------------

        keep = (spindles[:, 0] >= start) & (spindles[:, 2] <= end)

        valid = spindles[keep]

        if len(valid) > 0:

            all_spindles.append(valid)

    # ------------------------------------------------------------
    # No events
    # ------------------------------------------------------------

    if not all_spindles:

        return pd.DataFrame()

    final_spindles = np.vstack(all_spindles)

    df = pd.DataFrame(
        final_spindles,
        columns=[
            "Start_s",
            "Peak_s",
            "End_s",
            "Duration_s",
            "Max_R",
            "Peak_Freq_Hz",
        ],
    )

    # Metadata

    df.insert(0, "File", task["file_name"])

    df.insert(0, "Date", task["date"])

    df.insert(0, "Region", task["region"])

    df.insert(0, "Rat", task["rat"])

    df.insert(0, "Cohort", task["cohort"])

    df["Threshold_RB"] = r_b
    df["Threshold_RA"] = r_a

    df["Spindle_Band_Low"] = SPINDLE_LOW
    df["Spindle_Band_High"] = SPINDLE_HIGH

    return df


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def worker_process(
    task: Dict,
    r_a: float = RA,
    r_b: float = RB,
) -> Dict:

    t0 = time.time()

    try:

        df = extract_spindles_for_file(
            task,
            r_a=r_a,
            r_b=r_b,
        )

        return {
            "status": "OK",
            "task": task,
            "df": df,
            "elapsed": time.time() - t0,
        }

    except TaskFailure as e:

        return {
            "status": "SKIP",
            "task": task,
            "reason": e.reason,
            "detail": e.detail,
            "elapsed": time.time() - t0,
        }

    except Exception as e:

        return {
            "status": "ERROR",
            "task": task,
            "reason": "unexpected error",
            "detail": str(e),
            "elapsed": time.time() - t0,
        }


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def run_extraction(
    tasks: List[Dict],
    r_a: float = RA,
    r_b: float = RB,
    spindles_out_csv: Path = SPINDLES_OUT_CSV,
    failed_out_csv: Path = FAILED_OUT_CSV,
):

    if not tasks:

        logger.error("No tasks found")

        return pd.DataFrame()

    logger.info(f"Running {len(tasks)} files at r_a={r_a:.2f}, r_b={r_b:.2f}")

    all_dfs = []

    failures = []

    pbar = tqdm(
        total=len(tasks), desc="Extracting spindles", unit="file", dynamic_ncols=True
    )

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_to_task = {
            executor.submit(worker_process, task, r_a, r_b): task for task in tasks
        }

        for i, future in enumerate(as_completed(future_to_task), start=1):

            result = future.result()

            task = result["task"]

            pbar.set_postfix(
                {
                    "Rat": task["rat"],
                    "Region": task["region"],
                    "Status": result["status"],
                }
            )

            pbar.update(1)

            if result["status"] == "OK":

                df = result["df"]

                if not df.empty:

                    all_dfs.append(df)

                logger.info(f"{task['file_name']} -> " f"{len(df)} spindles")

            else:

                failures.append(
                    {
                        **task,
                        "reason": result.get("reason", ""),
                        "detail": result.get("detail", ""),
                    }
                )

            # checkpoint

            if i % CHECKPOINT_EVERY == 0 or i == len(tasks):

                if all_dfs:

                    pd.concat(all_dfs, ignore_index=True).to_csv(
                        spindles_out_csv, index=False
                    )

                if failures:

                    pd.DataFrame(failures).to_csv(failed_out_csv, index=False)

    pbar.close()

    # ------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------

    if not all_dfs:

        logger.warning("No spindles detected")

        return pd.DataFrame()

    master_df = pd.concat(all_dfs, ignore_index=True)

    master_df.to_csv(spindles_out_csv, index=False)

    logger.info(f"Saved {len(master_df)} spindles")

    return master_df


def total_nrem_minutes(tasks: List[Dict]) -> float:
    """Return the total scored NREM duration across all manifest tasks."""
    total_seconds = 0.0
    for task in tasks:
        try:
            states = np.asarray(loadmat(task["scoring_path"])["states"]).squeeze()
            total_seconds += float(np.sum(states == 3))
        except Exception as exc:
            logger.warning("Could not read scoring for %s: %s", task["file_name"], exc)
    return total_seconds / 60.0


def run_threshold_sweep(tasks: List[Dict]) -> pd.DataFrame:
    """Run each threshold pair and save one aggregate comparison row per pair."""
    nrem_minutes = total_nrem_minutes(tasks)
    rows = []

    for r_a, r_b in THRESHOLD_PAIRS:
        label = f"RA_{r_a:.2f}_RB_{r_b:.2f}"
        detections_path = OUTPUT_DIR / f"all_detected_spindles_{label}.csv"
        failures_path = OUTPUT_DIR / f"extraction_failed_files_{label}.csv"
        df = run_extraction(
            tasks,
            r_a=r_a,
            r_b=r_b,
            spindles_out_csv=detections_path,
            failed_out_csv=failures_path,
        )

        durations = (
            np.asarray(pd.to_numeric(df["Duration_s"], errors="coerce"), dtype=float)
            if not df.empty
            else np.array([], dtype=float)
        )
        durations = durations[np.isfinite(durations)]
        count = int(len(durations))
        rows.append(
            {
                "Threshold_RA": r_a,
                "Threshold_RB": r_b,
                "n_spindles": count,
                "spindle_rate_per_min_nrem": (
                    count / nrem_minutes if nrem_minutes else np.nan
                ),
                "median_duration_s": np.median(durations) if count else np.nan,
                "duration_95th_percentile_s": (
                    np.percentile(durations, 95) if count else np.nan
                ),
                "n_duration_gt_5_s": int((durations > 5).sum()),
                "n_duration_gt_10_s": int((durations > 10).sum()),
                f"n_extremely_long_gt_{EXTREMELY_LONG_DURATION_S:g}_s": int(
                    (durations > EXTREMELY_LONG_DURATION_S).sum()
                ),
                "total_nrem_minutes": nrem_minutes,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(THRESHOLD_SWEEP_OUT_CSV, index=False)
    logger.info("Saved threshold comparison to %s", THRESHOLD_SWEEP_OUT_CSV)
    return summary


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    loader = TaskLoader(
        "/home/mdadmin/Desktop/amirali/rat-hm-lfp-analysis/tasks_manifest.csv"
    )

    tasks = loader.to_tasks()

    run_threshold_sweep(tasks)
