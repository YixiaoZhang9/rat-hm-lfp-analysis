import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from modules.ephys_preprocessing import bandpass_filter, downsampling
from modules.find_spindles_lfp_o_quality import fit_ar_on_prepared_signal
from modules.iaaft import surrogates as iaaft_surrogates

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
FS = 1000
TARGET_FS = 128
N_SURROGATES = 4
SEGMENT_SEC = 10 * 60
TARGET_PCT_DIFF = 92.0

CHECKPOINT_EVERY = 25          # save partial results to disk every N processed files
OUTPUT_DIR = Path("results")   # where csvs + logs go
RAW_CSV = OUTPUT_DIR / "all_thresholds_raw.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary_thresholds_per_rat_region.csv"

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
# Task Loader Abstraction
# --------------------------------------------------------------------------- #
class TaskLoader:
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        try:
            self._df = pd.read_csv(manifest_path)
            # Standardize columns to string to avoid int/str mismatching
            for col in ["cohort", "rat", "region", "date"]:
                if col in self._df.columns:
                    self._df[col] = self._df[col].astype(str)
        except FileNotFoundError:
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    def filter(
        self,
        rat: Optional[Union[str, int, List[Union[str, int]]]] = None,
        region: Optional[Union[str, List[str]]] = None,
        cohort: Optional[Union[str, List[str]]] = None,
        date: Optional[Union[str, int, List[Union[str, int]]]] = None
    ) -> "TaskLoader":
        """Returns a new TaskLoader instance with the filtered subset of data."""
        new_loader = TaskLoader.__new__(TaskLoader)
        new_loader.manifest_path = self.manifest_path
        df = self._df.copy()

        if rat is not None:
            rats = [str(rat)] if isinstance(rat, (str, int)) else [str(r) for r in rat]
            df = df[df["rat"].isin(rats)]

        if region is not None:
            regions = [region] if isinstance(region, str) else region
            df = df[df["region"].isin(regions)]

        if cohort is not None:
            cohorts = [cohort] if isinstance(cohort, str) else cohort
            df = df[df["cohort"].isin(cohorts)]

        if date is not None:
            dates = [str(date)] if isinstance(date, (str, int)) else [str(d) for d in date]
            df = df[df["date"].isin(dates)]

        new_loader._df = df
        return new_loader

    @property
    def available_rats(self) -> List[str]:
        return sorted(self._df["rat"].unique().tolist())

    @property
    def available_regions(self) -> List[str]:
        return sorted(self._df["region"].unique().tolist())

    def __len__(self) -> int:
        return len(self._df)

    def to_tasks(self) -> List[Dict]:
        """Converts the current internal dataframe into the task dictionary format."""
        tasks = []
        for _, row in self._df.iterrows():
            data_path = Path(row["data_path"])
            tasks.append({
                "cohort": row["cohort"],
                "rat": row["rat"],
                "region": row["region"],
                "date": row["date"],
                "data_path": str(data_path),
                "file_name": data_path.name,
                "scoring_path": str(row["scoring_path"]),
            })
        return tasks


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

    logger.debug(f"NREM segment length: {random_real_raw.size / FS:.1f} s")

    filtered = bandpass_filter(random_real_raw, lowcut=0.1, highcut=100, fs=FS)
    pooled_128 = downsampling(filtered, FS, TARGET_FS)

    r_real, f_real = fit_ar_on_prepared_signal(pooled_128, TARGET_FS, n_jobs=-1, verbose=False)
    valid_r_real = r_real[~np.isnan(f_real)]

    if len(valid_r_real) == 0:
        raise TaskFailure("AR fit on real data returned no valid windows")

    logger.debug(f"Real signal: {len(valid_r_real)} valid AR windows")

    surrs = iaaft_surrogates(pooled_128, ns=N_SURROGATES, verbose=False)
    surrogate_valid_r_distributions = []

    for i, surrogate in enumerate(surrs):
        r_surr, f_surr = fit_ar_on_prepared_signal(surrogate, TARGET_FS, n_jobs=-1, verbose=False)
        valid = r_surr[~np.isnan(f_surr)]
        logger.debug(f"Surrogate {i + 1}/{N_SURROGATES}: {len(valid)} valid AR windows")
        surrogate_valid_r_distributions.append(valid)

    all_surr_r = np.concatenate(surrogate_valid_r_distributions)
    if all_surr_r.size == 0:
        raise TaskFailure("AR fit on all surrogates returned no valid windows")

    thresholds = np.arange(0.70, 0.90, 0.01)
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
# Batch processing
# --------------------------------------------------------------------------- #
def run_batch_processing(tasks: List[Dict]):
    if not tasks:
        logger.error("No valid data files provided. Nothing to do.")
        return

    logger.info(f"Loaded {len(tasks)} file(s) to process.")

    results = []
    failures = []
    failure_reason_counts = Counter()

    pbar = tqdm(tasks, desc="Processing Files", unit="file", dynamic_ncols=True)

    for i, task in enumerate(pbar, start=1):
        pbar.set_postfix({
            "Rat": task["rat"],
            "Region": task["region"],
            "Date": task["date"],
        })

        t0 = time.time()
        try:
            optimal_thresh = calculate_optimal_threshold(task["data_path"], task["scoring_path"])
            elapsed = time.time() - t0
            results.append({
                "Cohort": task["cohort"],
                "Rat": task["rat"],
                "Region": task["region"],
                "Date": task["date"],
                "File": task["file_name"],
                "Threshold": float(np.round(optimal_thresh, 3)),
            })
            logger.debug(
                f"[OK {i}/{len(tasks)}] {task['rat']}/{task['region']}/{task['date']}/"
                f"{task['file_name']} -> threshold={optimal_thresh:.3f} ({elapsed:.1f}s)"
            )

        except TaskFailure as e:
            elapsed = time.time() - t0
            failure_reason_counts[e.reason] += 1
            failures.append({**task, "reason": e.reason, "detail": e.detail})
            logger.warning(
                f"[SKIP {i}/{len(tasks)}] {task['rat']}/{task['region']}/{task['date']}/"
                f"{task['file_name']} -> {e.reason}"
                + (f" ({e.detail})" if e.detail else "")
                + f" ({elapsed:.1f}s)"
            )

        except Exception as e:
            elapsed = time.time() - t0
            failure_reason_counts["unexpected error"] += 1
            failures.append({**task, "reason": "unexpected error", "detail": str(e)})
            logger.exception(
                f"[ERROR {i}/{len(tasks)}] {task['rat']}/{task['region']}/{task['date']}/"
                f"{task['file_name']} -> unexpected error ({elapsed:.1f}s)"
            )

        if i % CHECKPOINT_EVERY == 0 or i == len(tasks):
            if results:
                pd.DataFrame(results).to_csv(RAW_CSV, index=False)
                logger.debug(f"Checkpoint: saved {len(results)} result(s) to {RAW_CSV}")

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

    df = pd.DataFrame(results)
    df.to_csv(RAW_CSV, index=False)
    logger.info(f"Saved raw thresholds to '{RAW_CSV}'")

    summary = df.groupby(["Rat", "Region"])["Threshold"].agg(
        Average="mean",
        Min="min",
        Max="max",
        N="count",
        All_Values=lambda x: list(x),
    ).reset_index()

    summary["Average"] = summary["Average"].round(3)
    summary.to_csv(SUMMARY_CSV, index=False)
    logger.info(f"Saved summary to '{SUMMARY_CSV}'")

    print("\n--- Final Threshold Summary ---")
    print(summary.to_string())


if __name__ == "__main__":
    # 1. Initialize the loader with your generated manifest
    loader = TaskLoader("tasks_manifest.csv")

    # 2. Extract the specific subset you want to process.
    # To run everything, simply use: loader.to_tasks()
    filtered_loader = loader.filter(
        rat=[1, 2],
        region="HPC",
        cohort="R1-4"
    )

    tasks_to_run = filtered_loader.to_tasks()

    # 3. Pass the resulting task list directly into the processing pipeline
    run_batch_processing(tasks_to_run)
