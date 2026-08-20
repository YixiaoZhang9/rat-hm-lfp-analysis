import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from modules.ephys_preprocessing import bandpass_filter, downsampling
from modules.find_spindles_lfp_o_quality import fit_ar_on_prepared_signal
from modules.iaaft import surrogates as iaaft_surrogates
from task_loader import TaskLoader

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
FS = 1000
TARGET_FS = 128
N_SURROGATES = 4
SEGMENT_SEC = 5 * 60
TARGET_PCT_DIFF = 92.0

MAX_WORKERS = max(1, os.cpu_count() - 2)  # Leave a couple of cores free for the OS

CHECKPOINT_EVERY = 25          # save partial results to disk every N processed files
OUTPUT_DIR = Path("results")   # where csvs + logs go

# --------------------------------------------------------------------------- #
# Logging setup: console (INFO+) and a full-detail run log file (DEBUG+)
# --------------------------------------------------------------------------- #
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"calibrate_threshold_{run_stamp}.log"

logger = logging.getLogger("calibrate_threshold")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))

file_handler = logging.FileHandler(log_path, mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"Full debug log for this run: {log_path.resolve()}")
logger.info(f"(tip: run `tail -f {log_path}` in another terminal to watch details live)")


class TaskFailure(Exception):
    """Raised with a short, categorical reason so failures can be tallied."""
    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)

# --------------------------------------------------------------------------- #
# Core logic
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


def get_random_nrem_segment(raw_signal, nrem_intervals, fs, target_sec):
    chunks = []
    for start, end in nrem_intervals:
        s_idx, e_idx = int(start * fs), int(end * fs)
        chunks.append(raw_signal[s_idx:e_idx])

    if not chunks:
        return np.array([])

    pooled_signal = np.concatenate(chunks)
    target_samples = target_sec * fs

    if len(pooled_signal) <= target_samples:
        return pooled_signal

    start_idx = np.random.randint(0, len(pooled_signal) - target_samples)
    return pooled_signal[start_idx : start_idx + target_samples]


def calculate_optimal_threshold(data_path, scoring_path):
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

    random_real_raw = get_random_nrem_segment(raw_signal, nrem_intervals, FS, SEGMENT_SEC)
    if random_real_raw.size == 0:
        raise TaskFailure("empty NREM segment after pooling")

    filtered = bandpass_filter(random_real_raw, lowcut=0.1, highcut=100, fs=FS)
    pooled_128 = downsampling(filtered, FS, TARGET_FS)

    # Note: n_jobs forced to 1 to prevent thread thrashing during parallel execution
    r_real, f_real = fit_ar_on_prepared_signal(pooled_128, TARGET_FS, n_jobs=1, verbose=False)
    valid_r_real = r_real[~np.isnan(f_real)]

    if len(valid_r_real) == 0:
        raise TaskFailure("AR fit on real data returned no valid windows")

    surrs = iaaft_surrogates(pooled_128, ns=N_SURROGATES, verbose=False)
    surrogate_valid_r_distributions = []

    for surrogate in surrs:
        r_surr, f_surr = fit_ar_on_prepared_signal(surrogate, TARGET_FS, n_jobs=1, verbose=False)
        valid = r_surr[~np.isnan(f_surr)]
        surrogate_valid_r_distributions.append(valid)

    all_surr_r = np.concatenate(surrogate_valid_r_distributions)
    if all_surr_r.size == 0:
        raise TaskFailure("AR fit on all surrogates returned no valid windows")

    thresholds = np.arange(0.5, 0.8, 0.02)
    best_threshold = None
    closest_diff = float("inf")

    for thresh in thresholds:
        count_real = np.sum(valid_r_real >= thresh)
        count_surr_avg = np.sum(all_surr_r >= thresh) / N_SURROGATES

        if count_real > 0:
            pct_diff = ((count_real - count_surr_avg) / count_real) * 100
            if abs(pct_diff - TARGET_PCT_DIFF) < closest_diff:
                closest_diff = abs(pct_diff - TARGET_PCT_DIFF)
                best_threshold = thresh

    if best_threshold is None:
        raise TaskFailure("no threshold candidate satisfied count_real > 0")

    return best_threshold


# --------------------------------------------------------------------------- #
# Worker Wrapper for Multiprocessing
# --------------------------------------------------------------------------- #
def worker_process(task: dict) -> dict:
    """Wrapper to catch exceptions and return payload back to main thread."""
    t0 = time.time()
    try:
        optimal_thresh = calculate_optimal_threshold(task["data_path"], task["scoring_path"])
        elapsed = time.time() - t0
        return {
            "status": "OK",
            "task": task,
            "threshold": float(np.round(optimal_thresh, 3)),
            "elapsed": elapsed
        }
    except TaskFailure as e:
        elapsed = time.time() - t0
        return {
            "status": "SKIP",
            "task": task,
            "reason": e.reason,
            "detail": e.detail,
            "elapsed": elapsed
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "status": "ERROR",
            "task": task,
            "reason": "unexpected error",
            "detail": str(e),
            "elapsed": elapsed
        }


# --------------------------------------------------------------------------- #
# Batch processing
# --------------------------------------------------------------------------- #
def run_batch_processing(tasks: list[dict]):
    if not tasks:
        logger.error("No valid data files provided. Nothing to do.")
        return

    logger.info(f"Loaded {len(tasks)} file(s) to process. Utilizing {MAX_WORKERS} concurrent workers.")

    results = []
    failures = []
    failure_reason_counts = Counter()

    pbar = tqdm(total=len(tasks), desc="Processing Files", unit="file", dynamic_ncols=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the process pool
        future_to_task = {executor.submit(worker_process, task): task for task in tasks}

        for i, future in enumerate(as_completed(future_to_task), start=1):
            res = future.result()
            task = res["task"]
            status = res["status"]
            elapsed = res["elapsed"]

            pbar.set_postfix({
                "Rat": task["rat"],
                "Region": task["region"],
                "Date": task["date"],
            })
            pbar.update(1)

            # Centralized logging handling
            if status == "OK":
                optimal_thresh = res["threshold"]
                results.append({
                    "Cohort": task["cohort"],
                    "Rat": task["rat"],
                    "Region": task["region"],
                    "Date": task["date"],
                    "File": task["file_name"],
                    "Threshold": optimal_thresh,
                })
                logger.debug(
                    f"[OK {i}/{len(tasks)}] {task['rat']}/{task['region']}/{task['date']}/"
                    f"{task['file_name']} -> threshold={optimal_thresh:.3f} ({elapsed:.1f}s)"
                )
            elif status == "SKIP":
                failure_reason_counts[res["reason"]] += 1
                failures.append({**task, "reason": res["reason"], "detail": res["detail"]})
                detail_str = f" ({res['detail']})" if res["detail"] else ""
                logger.warning(
                    f"[SKIP {i}/{len(tasks)}] {task['rat']}/{task['region']}/{task['date']}/"
                    f"{task['file_name']} -> {res['reason']}{detail_str} ({elapsed:.1f}s)"
                )
            elif status == "ERROR":
                failure_reason_counts["unexpected error"] += 1
                failures.append({**task, "reason": res["reason"], "detail": res["detail"]})
                logger.error(
                    f"[ERROR {i}/{len(tasks)}] {task['rat']}/{task['region']}/{task['date']}/"
                    f"{task['file_name']} -> unexpected error: {res['detail']} ({elapsed:.1f}s)"
                )

            # Checkpoint writing (grouped by Rat and Region)
            if i % CHECKPOINT_EVERY == 0 or i == len(tasks):
                if results:
                    df_tmp = pd.DataFrame(results)
                    for (rat, region), group in df_tmp.groupby(["Rat", "Region"]):
                        out_path = OUTPUT_DIR / f"thresholds_raw_Rat{rat}_{region}.csv"
                        group.to_csv(out_path, index=False)
                    logger.debug(f"Checkpoint: saved {len(results)} result(s) across rat/region files")

    pbar.close()

    # ----------------------------------------------------------------- #
    # Wrap-up
    # ----------------------------------------------------------------- #
    logger.info(
        f"Done: {len(results)} succeeded, {len(failures)} failed/skipped out of {len(tasks)} total."
    )

    if failure_reason_counts:
        logger.info("Failure/skip reasons: " + ", ".join(
            f"{reason}={count}" for reason, count in failure_reason_counts.most_common()
        ))

    if failures:
        failures_csv = OUTPUT_DIR / "failed_files.csv"
        pd.DataFrame(failures).to_csv(failures_csv, index=False)
        logger.info(f"Saved failure details to '{failures_csv}'")

    if not results:
        logger.error("No thresholds calculated successfully. Check 'failed_files.csv' and the debug log.")
        return

    # Final Raw Save
    df = pd.DataFrame(results)
    for (rat, region), group in df.groupby(["Rat", "Region"]):
        out_path = OUTPUT_DIR / f"thresholds_raw_Rat{rat}_{region}.csv"
        group.to_csv(out_path, index=False)
    logger.info(f"Saved raw thresholds to individual rat/region CSVs in '{OUTPUT_DIR}'")

    # Final Summary Save
    summary = df.groupby(["Rat", "Region"])["Threshold"].agg(
        Average="mean",
        Min="min",
        Max="max",
        N="count",
        All_Values=lambda x: list(x),
    ).reset_index()

    summary["Average"] = summary["Average"].round(3)

    for (rat, region), group in summary.groupby(["Rat", "Region"]):
        sum_path = OUTPUT_DIR / f"summary_thresholds_Rat{rat}_{region}.csv"
        group.to_csv(sum_path, index=False)
    logger.info(f"Saved summary thresholds to individual rat/region CSVs in '{OUTPUT_DIR}'")

    print("\n--- Final Threshold Summary ---")
    print(summary.to_string())


if __name__ == "__main__":
    loader = TaskLoader("tasks_manifest.csv")

    # filtered_loader = loader.filter(
    #     rat=[1, 2],
    #     region="HPC",
    #     cohort="R1-4"
    # )

    tasks_to_run = loader.to_tasks()
    run_batch_processing(tasks_to_run)
