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
MAX_WORKERS = max(1, os.cpu_count() - 2)

OUTPUT_DIR = Path("results")
RAW_THRESHOLDS_CSV = OUTPUT_DIR / "all_thresholds_raw.csv"
SPINDLES_OUT_CSV = OUTPUT_DIR / "summary_thresholds_per_rat_region.csv"

# --------------------------------------------------------------------------- #
# Logging Setup
# --------------------------------------------------------------------------- #
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
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))

file_handler = logging.FileHandler(log_path, mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(scoring_path):
        raise FileNotFoundError(f"Scoring file not found: {scoring_path}")

    raw_signal = loadmat(data_path)["data"].squeeze()
    nrem_intervals = get_nrem_intervals(scoring_path)

    if len(nrem_intervals) == 0:
        return pd.DataFrame()

    all_spindles = []

    # Process each NREM block with buffer
    for start, end in nrem_intervals:
        buf_start = max(0, int((start - BUFFER_SEC) * FS))
        buf_end = min(len(raw_signal), int((end + BUFFER_SEC) * FS))

        segment = raw_signal[buf_start:buf_end]

        if len(segment) < 2 * BUFFER_SEC * FS:
            continue

        # Extract using the dynamically loaded threshold
        spindles = find_spindles_lfp(segment, fs=FS, upper_threshold=threshold)

        if len(spindles) > 0:
            spindles[:, 0:3] += buf_start / FS  # Adjust time to global recording time

            # Keep only spindles strictly within the original NREM block (ignoring buffer)
            keep_mask = (spindles[:, 0] >= start) & (spindles[:, 2] <= end)
            valid_spindles = spindles[keep_mask]

            if len(valid_spindles) > 0:
                all_spindles.append(valid_spindles)

    if all_spindles:
        final_spindles = np.vstack(all_spindles)

        # Structure the results
        df = pd.DataFrame(
            final_spindles,
            columns=["Start_s", "Peak_s", "End_s", "Duration_s", "Max_R", "Peak_Freq_Hz"]
        )
        # Append metadata
        df.insert(0, "File", task["file_name"])
        df.insert(0, "Date", task["date"])
        df.insert(0, "Region", task["region"])
        df.insert(0, "Rat", task["rat"])
        df.insert(0, "Cohort", task["cohort"])
        df["Threshold_Used"] = threshold

        return df
    else:
        return pd.DataFrame()


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
            "elapsed": elapsed
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "status": "ERROR",
            "task": task,
            "reason": str(e),
            "elapsed": elapsed
        }


# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #
def run_extraction(tasks: List[Dict], threshold_map: Dict[str, float]):
    if not tasks:
        logger.error("No valid data files provided. Nothing to do.")
        return

    logger.info(f"Loaded {len(tasks)} file(s) for extraction. Utilizing {MAX_WORKERS} concurrent workers.")

    all_dfs = []
    failed_files = []

    pbar = tqdm(total=len(tasks), desc="Extracting Spindles", unit="file", dynamic_ncols=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {}
        for task in tasks:
            file_name = task["file_name"]

            # Lookup calculated threshold, fallback to 0.75 if something is missing
            thresh = threshold_map.get(file_name, 0.75)

            future = executor.submit(worker_process, task, thresh)
            future_to_task[future] = task

        for future in as_completed(future_to_task):
            res = future.result()
            task = res["task"]
            status = res["status"]

            pbar.set_postfix({"File": task["file_name"], "Status": status})
            pbar.update(1)

            if status == "OK":
                df = res["df"]
                if not df.empty:
                    all_dfs.append(df)
                logger.debug(f"[OK] {task['file_name']} -> found {len(df)} spindles ({res['elapsed']:.1f}s)")
            else:
                logger.error(f"[ERROR] {task['file_name']} -> {res['reason']} ({res['elapsed']:.1f}s)")
                failed_files.append(task["file_name"])

    pbar.close()

    # Wrap-up and compile CSV
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df.to_csv(SPINDLES_OUT_CSV, index=False)
        logger.info(f"Extraction complete! Saved {len(master_df)} total spindles to '{SPINDLES_OUT_CSV}'.")
    else:
        logger.warning("Extraction complete, but zero spindles were found across all files.")

    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files. Check log for details.")


if __name__ == "__main__":
    # 1. Load the pre-calculated thresholds into a lookup dictionary
    if not RAW_THRESHOLDS_CSV.exists():
        logger.error(f"Threshold CSV not found at {RAW_THRESHOLDS_CSV}. Run calibration first.")
        sys.exit(1)

    threshold_df = pd.read_csv(RAW_THRESHOLDS_CSV)
    threshold_map = dict(zip(threshold_df["File"], threshold_df["Threshold"]))

    # 2. Use TaskLoader to filter out the target dataset
    loader = TaskLoader("/home/mdadmin/Desktop/amirali/rat-hm-lfp-analysis/tasks_manifest.csv")
    filtered_loader = loader.filter(rat=1, region="HPC")
    tasks_to_run = filtered_loader.to_tasks()

    # 3. Execute detection
    run_extraction(tasks_to_run, threshold_map)
