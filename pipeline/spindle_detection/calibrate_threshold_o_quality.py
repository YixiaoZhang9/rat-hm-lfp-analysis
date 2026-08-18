import logging
import os
import sys
import time
from collections import Counter
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
from modules.project_config import get_path

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
FS = 1000
TARGET_FS = 128
N_SURROGATES = 4
SEGMENT_SEC = 10 * 60
TARGET_PCT_DIFF = 92.0

CHECKPOINT_EVERY = 25          # save partial results to disk every N processed files
OUTPUT_DIR = Path("outputs")   # where csvs + logs go
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
    """
    Raises TaskFailure with a short reason string on any expected failure
    mode, so the caller can tally *why* things failed instead of just
    counting how many did.
    """
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
# Task discovery
# --------------------------------------------------------------------------- #
def discover_tasks(root_dirs):
    tasks = []
    skip_counts = Counter()

    for root in root_dirs:
        if not root.exists():
            logger.warning(f"Root path not found, skipping: {root}")
            continue

        cohort_dirs = [d for d in root.iterdir() if d.is_dir() and (d / "PreprocessedData").exists()]
        logger.info(f"{root}: found {len(cohort_dirs)} cohort dir(s)")

        for cohort_dir in cohort_dirs:
            preprocessed_dir = cohort_dir / "PreprocessedData"
            scoring_dir = cohort_dir / "Scoring"

            for region_path in preprocessed_dir.iterdir():
                if not region_path.is_dir():
                    continue
                region = region_path.name

                for rat_path in region_path.iterdir():
                    if not rat_path.is_dir():
                        continue
                    rat = rat_path.name

                    for date_path in rat_path.iterdir():
                        if not date_path.is_dir():
                            continue
                        date = date_path.name

                        data_dir = date_path / "postsleep"
                        scoring_date_dir = scoring_dir / rat / date / "postsleep"

                        if not data_dir.exists():
                            skip_counts["no postsleep data dir"] += 1
                            continue
                        if not scoring_date_dir.exists():
                            skip_counts["no matching scoring dir"] += 1
                            continue

                        data_files = list(data_dir.glob("*.mat"))
                        scoring_files = list(scoring_date_dir.glob("*SW-eegstates.mat"))

                        if not data_files:
                            skip_counts["no .mat data files"] += 1
                            continue
                        if not scoring_files:
                            skip_counts["no SW-eegstates scoring file"] += 1
                            continue

                        scoring_path = scoring_files[0]
                        if len(scoring_files) > 1:
                            logger.debug(
                                f"{rat}/{date}: {len(scoring_files)} scoring files found, "
                                f"using {scoring_path.name}"
                            )

                        for data_path in data_files:
                            tasks.append({
                                "cohort": cohort_dir.name,
                                "rat": rat,
                                "region": region,
                                "date": date,
                                "data_path": str(data_path),
                                "file_name": data_path.name,
                                "scoring_path": str(scoring_path),
                            })

    if skip_counts:
        logger.info("Discovery skip summary: " + ", ".join(f"{k}={v}" for k, v in skip_counts.items()))

    return tasks


# --------------------------------------------------------------------------- #
# Batch processing
# --------------------------------------------------------------------------- #
def run_batch_processing():
    r1_8_path = Path(get_path("R1_8_root"))
    r9_16_path = Path(get_path("R9_16_root"))
    root_dirs = [r1_8_path, r9_16_path]

    logger.info("Discovering tasks...")
    tasks = discover_tasks(root_dirs)

    if not tasks:
        logger.error("No valid data files discovered across roots. Nothing to do.")
        return

    logger.info(f"Discovered {len(tasks)} file(s) to process.")

    results = []
    failures = []  # list of dicts: task metadata + reason
    failure_reason_counts = Counter()

    pbar = tqdm(tasks, desc="Processing Files", unit="file", dynamic_ncols=True)

    for i, task in enumerate(pbar, start=1):
        pbar.set_postfix({
            "Cohort": task["cohort"],
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
            # Anything unexpected: log full traceback to the debug file, keep going.
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
    run_batch_processing()
