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
N_SURROGATES = 19
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


def pool_nrem_raw(raw_signal, nrem_intervals, fs, max_sec):
    chunks = []
    total_sec = 0.0
    for start, end in nrem_intervals:
        if total_sec >= max_sec:
            break
        s_idx, e_idx = int(start * fs), int(end * fs)
        chunks.append(raw_signal[s_idx:e_idx])
        total_sec += (end - start)
    return np.concatenate(chunks) if chunks else np.array([])


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
    nrem_intervals = get_nrem_intervals(scoring_path, FS)
    logging.info(f"Found {len(nrem_intervals)} NREM segments.")

    pooled_real_raw = pool_nrem_raw(raw_signal, nrem_intervals, FS, SEGMENT_SEC)
    logging.info(f"Pooled NREM (raw, {FS}Hz): {len(pooled_real_raw)/FS:.1f}s")

    filtered = bandpass_filter(pooled_real_raw, lowcut=0.1, highcut=100, fs=FS)
    pooled_128 = downsampling(filtered, FS, TARGET_FS)
    logging.info(f"Downsampled to {TARGET_FS}Hz: {len(pooled_128)} samples "
                 f"({len(pooled_128)/TARGET_FS:.1f}s)")

    t0 = time.time()
    r_real, f_real = fit_ar_on_prepared_signal(pooled_128, TARGET_FS, n_jobs=-1, verbose=False)
    t_real = time.time() - t0
    real_max_r = np.nanmax(r_real) if len(r_real) else np.nan
    logging.info(f"Real signal: max_r={real_max_r:.3f} ({t_real:.1f}s)")

    t0 = time.time()
    surrs = iaaft_surrogates(pooled_128, ns=19, verbose=True)
    t_iaaft = time.time() - t0
    logging.info(f"IAAFT surrogate generation: {t_iaaft:.2f}s")
    for i, surrogate in enumerate(surrs):
        t0 = time.time()
        r_surr, f_surr = fit_ar_on_prepared_signal(surrogate, TARGET_FS, n_jobs=-1, verbose=False)
        t_ar = time.time() - t0
        surr_max_r = np.nanmax(r_surr) if len(r_surr) else np.nan
        logging.info(f"Surrogate {i+1}: max_r={surr_max_r:.3f} (IAAFT {t_iaaft:.2f}s + AR fit {t_ar:.2f}s)")

        est_total = (t_iaaft + t_ar) * (i + 1)
        logging.info(f"Estimated total time for {i + 1} surrogates: {est_total/60:.2f} min")


if __name__ == "__main__":
    run_test()
