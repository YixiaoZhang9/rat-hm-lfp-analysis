import logging
import os
import sys
import time

import numpy as np
from scipy.io import loadmat

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from modules.ephys_preprocessing import bandpass_filter, downsampling
from modules.find_spindles_lfp_o_quality import fit_ar_on_prepared_signal
from modules.iaaft import surrogates as iaaft_surrogates
from modules.project_config import get_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FS = 1000
TARGET_FS = 128
N_SURROGATES = 9
SEGMENT_SEC = 10 * 60

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
        logging.warning("Total NREM duration is less than the target segment length.")
        return pooled_signal

    start_idx = np.random.randint(0, len(pooled_signal) - target_samples)
    return pooled_signal[start_idx : start_idx + target_samples]

def run_test():
    dir_base = get_path("R1_8_root")
    data_path = os.path.join(
        dir_base, "R1-4/PreprocessedData/HPC/1/20221006/postsleep/chan102_9.mat"
    )
    scoring_path = os.path.join(
        dir_base,
        "R1-4/Scoring/1/20221006/postsleep/"
        "Rat_HM_Ephys_TD_Rat1_20221006_postsleep_09_SW-eegstates.mat",
    )

    raw_signal = loadmat(data_path)["data"].squeeze()
    nrem_intervals = get_nrem_intervals(scoring_path)

    random_real_raw = get_random_nrem_segment(raw_signal, nrem_intervals, FS, SEGMENT_SEC)
    filtered = bandpass_filter(random_real_raw, lowcut=0.1, highcut=100, fs=FS)
    pooled_128 = downsampling(filtered, FS, TARGET_FS)

    # Process Real Signal
    r_real, f_real = fit_ar_on_prepared_signal(pooled_128, TARGET_FS, n_jobs=-1, verbose=False)

    # Filter out windows where no 10-15Hz pole was found
    valid_r_real = r_real[~np.isnan(f_real)]

    # Generate Surrogates
    surrs = iaaft_surrogates(pooled_128, ns=N_SURROGATES, verbose=True)
    surrogate_valid_r_distributions = []

    for i, surrogate in enumerate(surrs):
        r_surr, f_surr = fit_ar_on_prepared_signal(surrogate, TARGET_FS, n_jobs=-1, verbose=False)
        valid_r_surr = r_surr[~np.isnan(f_surr)]
        surrogate_valid_r_distributions.append(valid_r_surr)

    # Statistical Threshold Validation
    if len(valid_r_real) > 0:
        all_surr_r = np.concatenate(surrogate_valid_r_distributions)

        logging.info("Sweeping thresholds to find where Real vs Surrogate difference is ~92%...")
        logging.info("Threshold | Real Count | Surr (Avg) | % Difference")
        logging.info("-" * 55)

        # Test thresholds from 0.80 to 0.98
        thresholds = np.arange(0.70, 0.90, 0.01)

        best_threshold = None
        closest_diff = float('inf')

        for thresh in thresholds:
            # How many windows exceed this threshold?
            count_real = np.sum(valid_r_real >= thresh)

            # Average number of surrogate windows exceeding the threshold
            count_surr_avg = np.sum(all_surr_r >= thresh) / N_SURROGATES

            if count_real > 0:
                # Percentage difference (how many more real events than random noise events)
                pct_diff = ((count_real - count_surr_avg) / count_real) * 100
            else:
                pct_diff = 0.0

            logging.info(f"{thresh:.2f}      | {count_real:<10} | {count_surr_avg:<10.1f} | {pct_diff:.2f}%")

            # Find the threshold that gets closest to a 92% difference
            if count_real > 0 and abs(pct_diff - 92.0) < closest_diff:
                closest_diff = abs(pct_diff - 92.0)
                best_threshold = thresh

        logging.info("-" * 55)
        logging.info(f"Based on a target difference of 92%, your recommended threshold (rb) is: {best_threshold:.2f}")

if __name__ == "__main__":
    run_test()
