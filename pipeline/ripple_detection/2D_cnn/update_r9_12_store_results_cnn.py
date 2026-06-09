import os
import re

import numpy as np
import pandas as pd
from demo_application_functions_m import *
from scipy.io import loadmat
from scipy.ndimage import label
from scipy.signal import convolve, hilbert
from scipy.signal.windows import gaussian

from modules.project_config import get_path
from modules.threshold_ripple_detection import filter_lfp, find_bouts


def smooth_signal(signal, fs, sigma):
    """Smooth the signal using a Gaussian filter."""
    # Define the standard deviation(sigma)
    smoothing_sigma = sigma * fs
    # Create a Gaussian window with a standard deviation
    window_size = smoothing_sigma * 3 * 2
    # in a Gaussian distribution, about 99.7 % of the distribution's area lies within ±3σ.
    std = (window_size - 1) / (2 * sigma)
    gauss_filter = gaussian(window_size, std)
    # Normalize the Gaussian filter
    gauss_filter = gauss_filter / sum(gauss_filter)
    # Apply the filter using convolution
    smoothed_lfp = convolve(signal, gauss_filter, "same")
    return smoothed_lfp


# %%
# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = get_path("RAT_HM_DATA4_ROOT")
dir_R9_12_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/PreprocessedData"
)
dir_R9_12_Scoring = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/Scoring"
)
dir_output = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/Ripple_detection_results"
)

rats = [9, 11, 12]
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 1000  # downsampled sample frequency


# %%
# prepare the spectral images for testing
# window_ms = 200  # time window of spectra image
window_ms = 400  # time window of spectra image
overlap = 0.5  # overlap of time window when computing spectra image
f_min = 100  # Hz, low cut-off frequency in spectra image
f_max = 250  # Hz, high cut-off frequency in spectra image

# Envelope-based refinement thresholds
high_th = 3.0
low_th = 0.0
pad = int(0.05 * fs)  # 50 ms padding

# Minimum NREM bout length (to avoid unstable estimation)
min_bout_samples = int(0.3 * fs)  # 300 ms

model_path = (
    Path.cwd().resolve().parents[1]
    / "ripple_detection"
    / "2D_cnn"
    / "retraining_saved_model.pth"
)

for rat in rats:
    rat_str = f"rat{rat}"
    print(f"\n===== Processing {rat_str} =====")

    for region in [regions[0]]:
        dir_R9_12_Data_perday = os.path.join(dir_R9_12_Data, region, str(rat))

        if not os.path.exists(dir_R9_12_Data_perday):
            continue

        folders_SD = [
            d
            for d in os.listdir(dir_R9_12_Data_perday)
            if os.path.isdir(os.path.join(dir_R9_12_Data_perday, d))
        ]

        for studyday in folders_SD:
            for sleep_period in sleep_periods:
                print(f"Processing {rat_str} | {studyday} | {sleep_period}")

                dir_trial = os.path.join(dir_R9_12_Data_perday, studyday, sleep_period)

                if not os.path.exists(dir_trial):
                    print(f"Missing: {dir_trial}")
                    continue

                path_scoring = os.path.join(
                    dir_R9_12_Scoring, str(rat), str(studyday), sleep_period
                )
                if not os.path.exists(path_scoring):
                    continue

                scoring_files = [
                    f for f in os.listdir(path_scoring) if f.endswith(".mat")
                ]

                # find all trial files
                for root, _, files in os.walk(dir_trial):
                    for name in files:
                        if not name.endswith(".mat"):
                            continue

                        trial_path = os.path.join(root, name)
                        print(f"  Trial: {name}")

                        trial_data = loadmat(trial_path)
                        lfp = trial_data["data"].squeeze()
                        time = np.arange(len(lfp)) / fs

                        # # load sleep scoring files
                        if len(scoring_files) == 1:
                            scoring_data = loadmat(
                                os.path.join(path_scoring, scoring_files[0])
                            )["states"].squeeze()
                        else:
                            match = re.search(r"_(\d+)\.mat$", name)
                            if not match:
                                continue

                            suffix = match.group(1).zfill(2)
                            pattern = re.compile(rf"_{suffix}_")

                            scoring_data = None
                            for f in scoring_files:
                                if pattern.search(f):
                                    scoring_data = loadmat(
                                        os.path.join(path_scoring, f)
                                    )["states"].squeeze()
                                    break

                            if scoring_data is None:
                                continue

                        # find NREM bouts
                        NREM_bouts = find_bouts(scoring_data, target_value=3, fs=fs)

                        # Store ripple results (each row = one ripple)
                        ripple_rows = []

                        # Loop over each NREM bout independently
                        for bout_idx, (b_start, b_end) in enumerate(NREM_bouts):
                            # Skip very short bouts (insufficient for spectral analysis)
                            if (b_end - b_start) < min_bout_samples:
                                continue

                            # Extract LFP segment corresponding to this NREM bout
                            lfp_bout = lfp[b_start:b_end]
                            time_bout = np.arange(len(lfp_bout)) / fs

                            # Create temporary directory for spectrogram images
                            # Each bout is processed independently to reduce memory usage
                            temp_dir = os.path.join(dir_output, f"temp_bout_{bout_idx}")
                            os.makedirs(temp_dir, exist_ok=True)

                            # Generate spectrogram images for CNN classification
                            start_time_dict, stop_time_dict = make_spectra_image_files(
                                lfp_bout,
                                time_bout,
                                out_dir=temp_dir,
                                fs=fs,
                                segment_ms=window_ms,
                                overlap=overlap,
                                f_min=f_min,
                                f_max=f_max,
                            )

                            # Run CNN inference on generated spectrograms
                            test_df = compute_CNN(
                                image_path=temp_dir,
                                start_time_dict=start_time_dict,
                                stop_time_dict=stop_time_dict,
                                model_path=model_path,
                            )

                            # Clean up temporary spectrogram images immediately
                            for f in os.listdir(temp_dir):
                                os.remove(os.path.join(temp_dir, f))
                            os.rmdir(temp_dir)

                            # Extract predicted ripple intervals (in seconds)
                            indices = np.where(test_df["prediction"].to_numpy() == 1)[0]

                            intervals = [
                                [
                                    float(test_df.iloc[i]["start time"]),
                                    float(test_df.iloc[i]["stop time"]),
                                ]
                                for i in indices
                            ]

                            # Sort intervals by start time
                            intervals.sort(key=lambda x: x[0])

                            # Merge overlapping CNN detections
                            merged_intervals = []
                            for interval in intervals:
                                if not merged_intervals:
                                    merged_intervals.append(interval)
                                else:
                                    prev_start, prev_stop = merged_intervals[-1]
                                    curr_start, curr_stop = interval

                                    if curr_start <= prev_stop:
                                        merged_intervals[-1][1] = max(
                                            prev_stop, curr_stop
                                        )
                                    else:
                                        merged_intervals.append(interval)

                            # Compute analytic signal envelope (Hilbert transform)
                            # Z-score is computed per NREM bout (local normalization)
                            filtered_lfp = filter_lfp(lfp, fs, [100, 250])
                            envelope_lfp = np.abs(hilbert(filtered_lfp))
                            envelope = envelope_lfp[b_start:b_end]

                            # Avoid numerical instability in flat signals
                            if np.std(envelope) < 1e-6:
                                continue

                            z_env = (envelope - np.mean(envelope)) / np.std(envelope)

                            # Refine CNN detections using envelope thresholding
                            for start_t, stop_t in merged_intervals:
                                # Convert time (s) to sample indices (within bout)
                                start_idx = max(0, int(start_t * fs) - pad)
                                stop_idx = min(len(z_env) - 1, int(stop_t * fs) + pad)

                                if stop_idx <= start_idx:
                                    continue

                                segment = z_env[start_idx:stop_idx]

                                # Detect supra-threshold segments
                                mask = segment > high_th
                                labeled, n_events = label(mask)

                                for i in range(1, n_events + 1):
                                    idxs = np.where(labeled == i)[0]
                                    if len(idxs) == 0:
                                        continue

                                    # Peak detection within segment
                                    peak_rel = idxs[np.argmax(segment[idxs])]
                                    peak_idx = start_idx + peak_rel

                                    # Expand to onset (backward)
                                    onset = peak_idx
                                    while onset > 0 and z_env[onset] > low_th:
                                        onset -= 1

                                    # Expand to offset (forward)
                                    offset = peak_idx
                                    while (
                                        offset < len(z_env) - 1
                                        and z_env[offset] > low_th
                                    ):
                                        offset += 1

                                    # Compute duration
                                    duration_sec = (offset - onset) / fs

                                    # Convert to global indices (relative to full recording)
                                    global_onset = onset + b_start
                                    global_offset = offset + b_start

                                    candidate_events = []
                                    candidate_events.append(
                                        [global_onset, global_offset]
                                    )

                                    # ---- merge close events ----
                                    min_gap = int(0.015 * fs)  # 15 ms

                                    candidate_events.sort(key=lambda x: x[0])

                                    merged_events = []

                                    for event in candidate_events:
                                        if not merged_events:
                                            merged_events.append(event)
                                        else:
                                            prev_start, prev_end = merged_events[-1]
                                            curr_start, curr_end = event

                                            # merge if overlap OR very close
                                            if curr_start <= prev_end + min_gap:
                                                merged_events[-1][1] = max(
                                                    prev_end, curr_end
                                                )
                                            else:
                                                merged_events.append(event)

                                    min_dur = 0.025  # 25 ms
                                    max_dur = 0.5  # 500 ms

                                    for onset, offset in merged_events:
                                        duration_sec = (offset - onset) / fs

                                        if (
                                            duration_sec < min_dur
                                            or duration_sec > max_dur
                                        ):
                                            continue

                                        ripple_rows.append(
                                            {
                                                "ripple_start": onset,
                                                "ripple_end": offset,
                                                "ripple_duration_sec": duration_sec,
                                                "nrem_bout_duration_sec": (
                                                    b_end - b_start
                                                )
                                                / fs,
                                            }
                                        )

                        # save CSV
                        df = pd.DataFrame(ripple_rows)

                        output_dir = os.path.join(
                            dir_output, str(rat), studyday, sleep_period
                        )
                        os.makedirs(output_dir, exist_ok=True)

                        trial_id = name.replace(".mat", "")
                        save_path = os.path.join(
                            output_dir, f"{trial_id}_hippocampal_ripples_cnn2.csv"
                        )

                        df.to_csv(save_path, index=False)


print("\n CNN ripple detection DONE")
