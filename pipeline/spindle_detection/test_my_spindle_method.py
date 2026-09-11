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
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from modules.find_spindles_lfp_o_quality import find_spindles_lfp
from task_loader import TaskLoader

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
FS = 1000
BUFFER_SEC = 2.0
REGION_THRESHOLDS = {
    "HPC": 0.91,
    "PL": 0.89,
    "RSC": 0.92,
}

# "two_pass" does a coarse scan then refines only flagged regions -- this is the
# practical choice for a full-dataset run. "full" fits AR at every sample shift
# across the whole segment (what calibrate_threshold.py used on 5-min segments) --
# only use "full" if you have the compute budget and need maximal fidelity.
DETECTION_METHOD = "full"

# Each call to find_spindles_lfp does its own internal joblib parallelism across
# windows. We're already parallelizing across FILES via ProcessPoolExecutor below,
# so the inner call must be n_jobs=1 -- otherwise every outer worker also tries to
# spawn its own full set of subprocesses (oversubscription / thrashing).
INNER_N_JOBS = 1

MAX_WORKERS = max(1, os.cpu_count() - 2)
CHECKPOINT_EVERY = 25

OUTPUT_DIR = Path("results")
SPINDLES_OUT_CSV = OUTPUT_DIR / "all_detected_spindles_per_region.csv"
FAILED_OUT_CSV = OUTPUT_DIR / "extraction_failed_files.csv"

# --------------------------------------------------------------------------- #
# Logging Setup
# --------------------------------------------------------------------------- #
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / "extract_spindles_{}.log".format(run_stamp)

logger = logging.getLogger("extract_spindles")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))

file_handler = logging.FileHandler(log_path, mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"Full debug log for this run: {log_path.resolve()}")
if SPINDLES_OUT_CSV.exists():
    logger.warning(f"Output file already exists and will be OVERWRITTEN at the end of this run: {SPINDLES_OUT_CSV}")


class TaskFailure(Exception):
    """Raised with a short, categorical reason so failures can be tallied."""
    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


# --------------------------------------------------------------------------- #
# Core Detection Logic
# --------------------------------------------------------------------------- #
def get_nrem_intervals(scoring_path):
    states = loadmat(scoring_path)["states"].squeeze()
    nrem_mask = (states == 3).astype(int)
    diff = np.diff(np.concatenate(([0], nrem_mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return np.empty((0, 2))
    return np.column_stack((starts, ends))


def extract_spindles_for_file(task: Dict, threshold: float) -> pd.DataFrame:
    data_path = task["data_path"]
    scoring_path = task["scoring_path"]

    if not os.path.exists(data_path):
        raise TaskFailure("data file not found", data_path)
    if not os.path.exists(scoring_path):
        raise TaskFailure("scoring file not found", scoring_path)

    try:
        raw_signal = loadmat(data_path)["data"].squeeze()
    except Exception as e:
        raise TaskFailure("data .mat read error", str(e))

    try:
        nrem_intervals = get_nrem_intervals(scoring_path)
    except Exception as e:
        raise TaskFailure("scoring .mat read error", str(e))

    if nrem_intervals.shape[0] == 0:
        raise TaskFailure("no NREM epochs in scoring file")

    all_spindles = []
    n_blocks_too_short = 0

    for start, end in nrem_intervals:
        buf_start = max(0, int((start - BUFFER_SEC) * FS))
        buf_end = min(len(raw_signal), int((end + BUFFER_SEC) * FS))

        segment = raw_signal[buf_start:buf_end]

        if len(segment) < 2 * BUFFER_SEC * FS:
            n_blocks_too_short += 1
            continue

        spindles = find_spindles_lfp(
            segment,
            fs=FS,
            upper_threshold=threshold,
            method=DETECTION_METHOD,
            n_jobs=INNER_N_JOBS,
        )

        if len(spindles) > 0:
            spindles[:, 0:3] += buf_start / FS  # local segment time -> global recording time

            # Keep only spindles strictly within the original NREM block (ignoring buffer)
            keep_mask = (spindles[:, 0] >= start) & (spindles[:, 2] <= end)
            valid_spindles = spindles[keep_mask]

            if len(valid_spindles) > 0:
                all_spindles.append(valid_spindles)

    if n_blocks_too_short > 0:
        logger.debug(
            f"{task['file_name']}: skipped {n_blocks_too_short}/{len(nrem_intervals)} "
            f"NREM block(s) shorter than 2x buffer ({2 * BUFFER_SEC}s)"
        )

    if not all_spindles:
        return pd.DataFrame()

    final_spindles = np.vstack(all_spindles)
    df = pd.DataFrame(
        final_spindles,
        columns=["Start_s", "Peak_s", "End_s", "Duration_s", "Max_R", "Peak_Freq_Hz"]
    )
    df.insert(0, "File", task["file_name"])
    df.insert(0, "Date", task["date"])
    df.insert(0, "Region", task["region"])
    df.insert(0, "Rat", task["rat"])
    df.insert(0, "Cohort", task["cohort"])
    df["Threshold_Used"] = threshold

    return df


# --------------------------------------------------------------------------- #
# Worker Wrapper
# --------------------------------------------------------------------------- #
def worker_process(task: Dict, threshold: float) -> Dict:
    t0 = time.time()
    try:
        df_spindles = extract_spindles_for_file(task, threshold)
        elapsed = time.time() - t0
        return {
            "status": "OK",
            "task": task,
            "df": df_spindles,
            "elapsed": elapsed,
        }
    except TaskFailure as e:
        elapsed = time.time() - t0
        return {
            "status": "SKIP",
            "task": task,
            "reason": e.reason,
            "detail": e.detail,
            "elapsed": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "status": "ERROR",
            "task": task,
            "reason": "unexpected error",
            "detail": str(e),
            "elapsed": elapsed,
        }


# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #
def run_extraction(tasks: List[Dict]):
    if not tasks:
        logger.error("No valid data files provided. Nothing to do.")
        return

    logger.info(
        f"Loaded {len(tasks)} file(s) for extraction. Utilizing {MAX_WORKERS} concurrent "
        f"workers, method='{DETECTION_METHOD}', thresholds={REGION_THRESHOLDS}."
    )

    all_dfs = []
    failures = []
    failure_reason_counts = {}

    pbar = tqdm(total=len(tasks), desc="Extracting Spindles", unit="file", dynamic_ncols=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(worker_process, task, REGION_THRESHOLDS[task["Region"]]): task for task in tasks
        }

        for i, future in enumerate(as_completed(future_to_task), start=1):
            res = future.result()
            task = res["task"]
            threshold = REGION_THRESHOLDS[task["Region"]]
            status = res["status"]
            elapsed = res["elapsed"]

            pbar.set_postfix({"Rat": task["rat"], "Region": task["region"], "Status": status})
            pbar.update(1)

            if status == "OK":
                df = res["df"]
                if not df.empty:
                    all_dfs.append(df)
                logger.debug(f"[OK {i}/{len(tasks)}] {task['file_name']} with threshold: {threshold} -> {len(df)} spindles ({elapsed:.1f}s)")
            elif status == "SKIP":
                failure_reason_counts[res["reason"]] = failure_reason_counts.get(res["reason"], 0) + 1
                failures.append({**task, "reason": res["reason"], "detail": res["detail"]})
                logger.warning(f"[SKIP {i}/{len(tasks)}] {task['file_name']} -> {res['reason']} ({elapsed:.1f}s)")
            else:
                failure_reason_counts["unexpected error"] = failure_reason_counts.get("unexpected error", 0) + 1
                failures.append({**task, "reason": res["reason"], "detail": res["detail"]})
                logger.error(f"[ERROR {i}/{len(tasks)}] {task['file_name']} -> {res['detail']} ({elapsed:.1f}s)")

            # Checkpoint
            if i % CHECKPOINT_EVERY == 0 or i == len(tasks):
                if all_dfs:
                    pd.concat(all_dfs, ignore_index=True).to_csv(SPINDLES_OUT_CSV, index=False)
                    logger.debug(f"Checkpoint: saved spindles from {len(all_dfs)} file(s) so far")
                if failures:
                    pd.DataFrame(failures).to_csv(FAILED_OUT_CSV, index=False)

    pbar.close()

    # --------------------------------------------------------------------- #
    # Wrap-up
    # --------------------------------------------------------------------- #
    n_ok = len(tasks) - len(failures)
    logger.info(f"Done: {n_ok} succeeded, {len(failures)} failed/skipped out of {len(tasks)} total.")

    if failure_reason_counts:
        logger.info(
            "Failure/skip reasons: "
            + ", ".join(f"{reason}={count}" for reason, count in failure_reason_counts.items())
        )
    if failures:
        pd.DataFrame(failures).to_csv(FAILED_OUT_CSV, index=False)
        logger.info(f"Saved failure details to '{FAILED_OUT_CSV}'")

    if not all_dfs:
        logger.warning("Extraction complete, but zero spindles were found across all files.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.to_csv(SPINDLES_OUT_CSV, index=False)
    logger.info(f"Saved {len(master_df)} total spindles from {len(all_dfs)} file(s) to '{SPINDLES_OUT_CSV}'.")

    # Quick summary so you don't have to reload the CSV to sanity-check the run
    per_file_counts = master_df.groupby(["Rat", "Region", "File"]).size()
    logger.info(
        f"Spindles per file -- mean: {per_file_counts.mean():.1f}, "
        f"median: {per_file_counts.median():.1f}, "
        f"min: {per_file_counts.min()}, max: {per_file_counts.max()}"
    )
    logger.info(f"Mean spindle duration: {master_df['Duration_s'].mean():.2f}s")


if __name__ == "__main__":
    loader = TaskLoader("/home/mdadmin/Desktop/amirali/rat-hm-lfp-analysis/tasks_manifest.csv")
    tasks_to_run = loader.to_tasks()
    run_extraction(tasks_to_run)
