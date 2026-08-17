import logging
import os
import sys
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FS = 1000
TARGET_FS = 128
N_SURROGATES = 4
SEGMENT_SEC = 10 * 60
TARGET_PCT_DIFF = 92.0

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
    raw_signal = loadmat(data_path)["data"].squeeze()
    nrem_intervals = get_nrem_intervals(scoring_path)

    random_real_raw = get_random_nrem_segment(raw_signal, nrem_intervals, FS, SEGMENT_SEC)
    if random_real_raw.size == 0:
        return None

    filtered = bandpass_filter(random_real_raw, lowcut=0.1, highcut=100, fs=FS)
    pooled_128 = downsampling(filtered, FS, TARGET_FS)

    r_real, f_real = fit_ar_on_prepared_signal(pooled_128, TARGET_FS, n_jobs=-1, verbose=False)
    valid_r_real = r_real[~np.isnan(f_real)]

    if len(valid_r_real) == 0:
        return None

    surrs = iaaft_surrogates(pooled_128, ns=N_SURROGATES, verbose=False)
    surrogate_valid_r_distributions = []

    for surrogate in surrs:
        r_surr, f_surr = fit_ar_on_prepared_signal(surrogate, TARGET_FS, n_jobs=-1, verbose=False)
        surrogate_valid_r_distributions.append(r_surr[~np.isnan(f_surr)])

    all_surr_r = np.concatenate(surrogate_valid_r_distributions)

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

    return best_threshold

def run_batch_processing():
    r1_8_path = Path(get_path("R1_8_root"))
    r9_16_path = Path(get_path("R9_16_root"))

    root_dirs = [r1_8_path, r9_16_path]
    tasks = []

    # Discover and index all valid file pairs across directories
    for root in root_dirs:
        if not root.exists():
            logging.warning(f"Root path not found: {root}")
            continue

        cohort_dirs = [d for d in root.iterdir() if d.is_dir() and (d / "PreprocessedData").exists()]

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

                        if not data_dir.exists() or not scoring_date_dir.exists():
                            continue

                        data_files = list(data_dir.glob("*.mat"))
                        scoring_files = list(scoring_date_dir.glob("*SW-eegstates.mat"))

                        if not data_files or not scoring_files:
                            continue

                        scoring_path = scoring_files[0]

                        for data_path in data_files:
                            tasks.append({
                                "cohort": cohort_dir.name,
                                "rat": rat,
                                "region": region,
                                "date": date,
                                "data_path": str(data_path),
                                "file_name": data_path.name,
                                "scoring_path": str(scoring_path)
                            })

    if not tasks:
        logging.error("No valid data files discovered across roots.")
        return

    results = []
    pbar = tqdm(tasks, desc="Processing Files", unit="file", dynamic_ncols=True)

    for task in pbar:
        pbar.set_postfix({
            "Cohort": task["cohort"],
            "Rat": task["rat"],
            "Region": task["region"],
            "Date": task["date"]
        })

        optimal_thresh = calculate_optimal_threshold(task["data_path"], task["scoring_path"])

        if optimal_thresh is not None:
            results.append({
                "Cohort": task["cohort"],
                "Rat": task["rat"],
                "Region": task["region"],
                "Date": task["date"],
                "File": task["file_name"],
                "Threshold": float(np.round(optimal_thresh, 3))
            })

    if not results:
        logging.error("No thresholds calculated successfully.")
        return

    df = pd.DataFrame(results)
    df.to_csv("all_thresholds_raw.csv", index=False)
    logging.info("Saved raw thresholds to 'all_thresholds_raw.csv'")

    summary = df.groupby(["Rat", "Region"])["Threshold"].agg(
        Average="mean",
        Min="min",
        Max="max",
        All_Values=lambda x: list(x)
    ).reset_index()

    summary["Average"] = summary["Average"].round(3)
    summary.to_csv("summary_thresholds_per_rat_region.csv", index=False)
    logging.info("Saved summary to 'summary_thresholds_per_rat_region.csv'")

    print("\n--- Final Threshold Summary ---")
    print(summary.to_string())

if __name__ == "__main__":
    run_batch_processing()
