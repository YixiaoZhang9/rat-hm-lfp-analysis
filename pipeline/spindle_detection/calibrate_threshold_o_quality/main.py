import logging
import sys
import time
from collections import Counter
from datetime import datetime

import config
import pandas as pd
from core import TaskFailure, calculate_optimal_threshold
from scipy.io import loadmat
from tqdm import tqdm

from modules.project_config import get_path


def setup_logger():
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = config.LOG_DIR / f"calibrate_threshold_{run_stamp}.log"

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

    return logger, log_path


def discover_tasks(root_dirs, logger):
    tasks = []
    skip_counts = Counter()

    for root in root_dirs:
        if not root.exists():
            logger.warning(f"Root path not found, skipping: {root}")
            continue

        # Flattened directory traversal using pathlib glob
        # Matches: cohort_dir / PreprocessedData / region / rat / date / postsleep
        for data_dir in root.glob("*/PreprocessedData/*/*/*/postsleep"):
            cohort_dir = data_dir.parents[4]
            region = data_dir.parents[3].name
            rat = data_dir.parents[2].name
            date = data_dir.parents[1].name

            scoring_date_dir = cohort_dir / "Scoring" / rat / date / "postsleep"
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
                logger.debug(f"{rat}/{date}: {len(scoring_files)} scoring files found, using {scoring_path.name}")

            for data_path in data_files:
                tasks.append({
                    "cohort": cohort_dir.name,
                    "rat": rat,
                    "region": region,
                    "date": date,
                    "data_path": data_path,
                    "file_name": data_path.name,
                    "scoring_path": scoring_path,
                })

    if skip_counts:
        logger.info("Discovery skip summary: " + ", ".join(f"{k}={v}" for k, v in skip_counts.items()))

    return tasks


def process_task(task):
    """Handles file I/O safely and delegates to core logic."""
    try:
        raw_signal = loadmat(str(task["data_path"]))["data"].squeeze()
    except Exception as e:
        raise TaskFailure("data .mat read error", str(e))

    try:
        states = loadmat(str(task["scoring_path"]))["states"].squeeze()
    except Exception as e:
        raise TaskFailure("scoring .mat read error", str(e))

    return calculate_optimal_threshold(raw_signal, states)


def run_batch_processing():
    logger, log_path = setup_logger()
    logger.info(f"Full debug log for this run: {log_path.resolve()}")

    root_dirs = [Path(get_path("R1_8_root")), Path(get_path("R9_16_root"))]

    logger.info("Discovering tasks...")
    tasks = discover_tasks(root_dirs, logger)

    if not tasks:
        logger.error("No valid data files discovered across roots. Nothing to do.")
        return

    logger.info(f"Discovered {len(tasks)} file(s) to process.")

    results, failures = [], []
    failure_counts = Counter()

    pbar = tqdm(tasks, desc="Processing Files", unit="file", dynamic_ncols=True)

    for i, task in enumerate(pbar, start=1):
        pbar.set_postfix({"Rat": task["rat"], "Region": task["region"], "Date": task["date"]})
        t0 = time.time()

        try:
            optimal_thresh = process_task(task)
            elapsed = time.time() - t0

            results.append({
                "Cohort": task["cohort"],
                "Rat": task["rat"],
                "Region": task["region"],
                "Date": task["date"],
                "File": task["file_name"],
                "Threshold": float(np.round(optimal_thresh, 3)),
            })
            logger.debug(f"[OK {i}/{len(tasks)}] {task['file_name']} -> threshold={optimal_thresh:.3f} ({elapsed:.1f}s)")

        except TaskFailure as e:
            failure_counts[e.reason] += 1
            failures.append({**task, "reason": e.reason, "detail": e.detail})
            logger.warning(f"[SKIP {i}/{len(tasks)}] {task['file_name']} -> {e.reason} ({time.time() - t0:.1f}s)")

        except Exception as e:
            failure_counts["unexpected error"] += 1
            failures.append({**task, "reason": "unexpected error", "detail": str(e)})
            logger.exception(f"[ERROR {i}/{len(tasks)}] {task['file_name']} -> unexpected error")

        if i % config.CHECKPOINT_EVERY == 0 or i == len(tasks):
            if results:
                pd.DataFrame(results).to_csv(config.RAW_CSV, index=False)

    finalize_run(results, failures, failure_counts, len(tasks), logger)


def finalize_run(results, failures, failure_counts, total_tasks, logger):
    logger.info(f"Done: {len(results)} succeeded, {len(failures)} failed/skipped out of {total_tasks} total.")

    if failure_counts:
        logger.info("Failure reasons: " + ", ".join(f"{k}={v}" for k, v in failure_counts.most_common()))

    if failures:
        failures_csv = config.OUTPUT_DIR / "failed_files.csv"
        pd.DataFrame(failures).to_csv(failures_csv, index=False)

    if not results:
        logger.error("No thresholds calculated successfully.")
        return

    df = pd.DataFrame(results)
    df.to_csv(config.RAW_CSV, index=False)

    summary = df.groupby(["Rat", "Region"])["Threshold"].agg(
        Average="mean", Min="min", Max="max", N="count", All_Values=list
    ).reset_index()

    summary["Average"] = summary["Average"].round(3)
    summary.to_csv(config.SUMMARY_CSV, index=False)
    logger.info(f"Saved summary to '{config.SUMMARY_CSV}'")


if __name__ == "__main__":
    run_batch_processing()
