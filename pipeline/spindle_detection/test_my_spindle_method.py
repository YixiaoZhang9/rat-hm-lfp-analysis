import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.find_spindles_lfp_o_quality import find_spindles_lfp
from modules.project_config import get_path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
FS = 1000  # Original sampling frequency


def get_nrem_intervals(scoring_path):
    """Load states from scoring file and return list of [start, end] in seconds."""
    if not os.path.exists(scoring_path):
        logging.error(f"Scoring file not found: {scoring_path}")
        return np.empty((0, 2))
    states = loadmat(scoring_path)["states"].squeeze()
    nrem_mask = (states == 3).astype(int)
    diff = np.diff(np.concatenate(([0], nrem_mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return np.empty((0, 2))
    return np.column_stack((starts, ends)) # already in seconds, since states is 1 Hz


def run_test():
    # Targeted trial
    dir_base = get_path("R1_8_root")
    data_path = os.path.join(
        dir_base, "R1-4/PreprocessedData/PL/1/202209029/postsleep/chan108_7.mat"
    )
    scoring_path = os.path.join(
        dir_base,
        "R1-4/Scoring/1/202209029/postsleep/"
        "Rat_HM_Ephys_TD_Rat1_202209029_postsleep_07_JM-eegstates.mat",
    )


    if not os.path.exists(data_path):
        logging.error(f"Data file not found: {data_path}")
        return

    # Load data
    raw_signal = loadmat(data_path)["data"].squeeze()
    nrem_intervals = get_nrem_intervals(scoring_path)

    total_nrem_sec = np.sum(nrem_intervals[:, 1] - nrem_intervals[:, 0])
    logging.info(f"Total NREM duration: {total_nrem_sec/60:.1f} min out of "
                 f"{len(raw_signal)/FS/60:.1f} min total recording")

    logging.info(f"Found {len(nrem_intervals)} NREM segments.")

    all_spindles = []
    buffer_sec = 2.0

    # Process each NREM block with buffer
    for i, (start, end) in enumerate(nrem_intervals):
        # Add buffer
        buf_start = max(0, int((start - buffer_sec) * FS))
        buf_end = min(len(raw_signal), int((end + buffer_sec) * FS))

        segment = raw_signal[buf_start:buf_end]

        # Skip very short segments
        if len(segment) < 2 * buffer_sec * FS:
            continue

        logging.info(
            f"Processing segment {i+1}/{len(nrem_intervals)} ({start:.1f}s - {end:.1f}s)..."
        )

        spindles = find_spindles_lfp(segment, fs=FS, upper_threshold=0.70)

        if len(spindles) > 0:
            # Adjust time to global recording time
            spindles[:, 0:3] += buf_start / FS

            # Keep only spindles that are actually within the original NREM block
            keep_mask = (spindles[:, 0] >= start) & (spindles[:, 2] <= end)
            all_spindles.append(spindles[keep_mask])

            if all_spindles:
                    final_spindles = np.vstack(all_spindles)
                    logging.info(f"Detection complete. Total NREM spindles: {len(final_spindles)}")
                    # Create a DataFrame with column names
                    df_spindles = pd.DataFrame(
                        final_spindles,
                        columns=["Start_s", "Peak_s", "End_s", "Duration_s", "Max_R", "Peak_Freq_Hz"]
                    )

                    print("\n", df_spindles.head(), "\n")
            else:
                logging.info("No spindles detected in NREM.")


if __name__ == "__main__":
    run_test()
