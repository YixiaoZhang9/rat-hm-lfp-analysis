# ---- Import ----
import os
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, detrend, sosfiltfilt

from modules.find_deltas_lfp import find_deltas_lfp
from modules.project_config import get_path


def filter_lfp(lfp, fs, band):
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, lfp)


def find_bouts(scoring_data, target_value=3, fs=1000):
    """
    Find all continuous segments (bouts) in scoring_data where the value equals target_value.

    Parameters
    ----------
    scoring_data : array-like
        1D array of sleep scoring values (e.g., 1=Wake, 3=NonREM, 5=REM, 4=intermediate)
    target_value : int or float, optional
        The state value to detect bouts for NREM sleep (default = 3)
    fs : float, optional
        Sampling frequency in Hz (default = 1000)

    Returns
    -------
    bouts : list of tuples
        List of (start_idx, end_idx, start_time, end_time), where:
            start_idx / end_idx are sample indices
            start_time / end_time are times in seconds
    """
    scoring_data = np.array(scoring_data)
    is_target = (scoring_data == target_value).astype(int)
    diff = np.diff(is_target, prepend=0, append=0)

    # 1 → bout starts, -1 → bout ends
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    bouts = [(s * fs, e * fs) for s, e in zip(starts, ends)]
    return bouts


# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = get_path("data4_root")
dir_R13_16_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R13-16/PreprocessedData"
)
dir_R13_16_Scoring = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R13-16/Scoring"
)
dir_output = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R13-16/Delta_detection_results"
)


rats = np.arange(13, 17)
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 1000  # downsampled sample frequency

f_plot = 0
Threshold_across_time = {}
NREM_duration_across_time = {}


# %%
# detect deltas
for rat in rats:
    rat_str = f"rat{rat}"

    for region in regions:
        print(f"\n===== Processing {rat_str} region :{region} =====")
        dir_R13_16_Data_perday = os.path.join(dir_R13_16_Data, region, str(rat))

        if not os.path.exists(dir_R13_16_Data_perday):
            print(f"Missing: {dir_R13_16_Data_perday}")
            continue

        folders_SD = [
            name
            for name in os.listdir(dir_R13_16_Data_perday)
            if os.path.isdir(os.path.join(dir_R13_16_Data_perday, name))
        ]

        for studyday in folders_SD:
            for sleep_period in sleep_periods:
                print(f"Processing {rat_str} | {studyday} | {sleep_period}")

                dir_trial = os.path.join(dir_R13_16_Data_perday, studyday, sleep_period)

                if not os.path.exists(dir_trial):
                    print(f"Missing: {dir_trial}")
                    continue

                # find all trial files
                matched_files = []
                for root, _, files in os.walk(dir_trial):
                    for name in files:
                        if name.endswith(".mat"):
                            matched_files.append(os.path.join(root, name))

                if len(matched_files) == 0:
                    print("No trial files found")
                    continue

                # load sleep scoring files
                path_scoring = os.path.join(
                    dir_R13_16_Scoring, str(rat), str(studyday), sleep_period
                )

                if not os.path.exists(path_scoring):
                    print(f"Missing scoring: {path_scoring}")
                    continue

                scoring_files = [
                    f for f in os.listdir(path_scoring) if f.endswith(".mat")
                ]

                # process each trial
                for trial_name in matched_files:
                    print(f"  Trial: {os.path.basename(trial_name)}")

                    trial_data = loadmat(trial_name)
                    data = trial_data["data"].squeeze()
                    # remove the DC drift
                    data = detrend(data)

                    # matched sleep scoring files
                    if len(scoring_files) == 1:
                        scoring_data = loadmat(
                            os.path.join(path_scoring, scoring_files[0])
                        )["states"].squeeze()
                    else:
                        match = re.search(r"_(\d+)\.mat$", trial_name)
                        if not match:
                            print("No matching scoring file")
                            continue

                        suffix = match.group(1).zfill(2)
                        pattern = re.compile(rf"_{suffix}_")

                        scoring_data = None
                        for f in scoring_files:
                            if pattern.search(f):
                                scoring_data = loadmat(os.path.join(path_scoring, f))[
                                    "states"
                                ].squeeze()
                                break

                        if scoring_data is None:
                            print("No scoring matched")
                            continue

                    # find NREM bouts
                    NREM_bouts = find_bouts(scoring_data, target_value=3, fs=fs)
                    # filter
                    filtered_data = filter_lfp(data, fs, [0.5, 4])

                    # storage
                    delta_start_list = []
                    delta_end_list = []
                    delta_peak_list = []
                    delta_duration_list = []
                    delta_negative_amplitude_list = []
                    delta_positive_amplitude_list = []
                    delta_peak_to_peak_amplitude_list = []
                    delta_rising_slope_list = []
                    delta_decreasing_slope_list = []
                    nrem_duration_list = []

                    # process each NREM bout
                    for start_sample, end_sample in NREM_bouts:
                        bout_duration_sec = (end_sample - start_sample) / fs

                        bout_data = filtered_data[start_sample:end_sample]

                        bout_raw_data = data[start_sample:end_sample]

                        # delta detection
                        deltas = find_deltas_lfp(
                            bout_raw_data,
                            bout_data,
                            fs=fs,
                            thresholds=(1, 2, 0, 1.5),
                            durations=(250, 1500),
                            f_plot=False,
                            plot_range=None,
                        )

                        if len(deltas) == 0:
                            continue

                        delta_list = list(
                            zip(
                                (deltas[:, 0] + start_sample).astype(int),
                                (deltas[:, 1] + start_sample).astype(int),
                                (deltas[:, 2] + start_sample).astype(int),
                            )
                        )

                        duration = deltas[:, 3]
                        neg_amplitude = deltas[:, 4]
                        pos_amplitude = deltas[:, 5]
                        peak_to_peak_amplitude = deltas[:, 6]
                        rising_slope = deltas[:, 7]
                        decreasing_slope = deltas[:, 8]

                        for i, delta in enumerate(delta_list):
                            start = delta[0]
                            peak = delta[1]
                            end = delta[2]

                            delta_start_list.append(start)
                            delta_peak_list.append(peak)
                            delta_end_list.append(end)

                            delta_duration_list.append(duration[i])
                            delta_negative_amplitude_list.append(neg_amplitude[i])
                            delta_positive_amplitude_list.append(pos_amplitude[i])
                            delta_peak_to_peak_amplitude_list.append(
                                peak_to_peak_amplitude[i]
                            )
                            delta_rising_slope_list.append(rising_slope[i])
                            delta_decreasing_slope_list.append(decreasing_slope[i])

                            nrem_duration_list.append(bout_duration_sec)

                    # SAVE CSV

                    if len(delta_start_list) > 0:
                        df = pd.DataFrame(
                            {
                                "delta_start_index": delta_start_list,
                                "delta_peak_index": delta_peak_list,
                                "delta_end_index": delta_end_list,
                                "delta_duration_s": delta_duration_list,
                                "delta_negative_amplitude": delta_negative_amplitude_list,
                                "delta_positive_amplitude": delta_positive_amplitude_list,
                                "delta_peak_to_peak_amplitude": delta_peak_to_peak_amplitude_list,
                                "delta_rising_slope": delta_rising_slope_list,
                                "delta_decreasing_slope": delta_decreasing_slope_list,
                                "nrem_bout_duration_s": nrem_duration_list,
                            }
                        )

                    else:
                        df = pd.DataFrame(
                            columns=[
                                "delta_start_index",
                                "delta_peak_index",
                                "delta_end_index",
                                "delta_duration_s",
                                "delta_negative_amplitude",
                                "delta_positive_amplitude",
                                "delta_peak_to_peak_amplitude",
                                "delta_rising_slope",
                                "delta_decreasing_slope",
                                "nrem_bout_duration_s",
                            ]
                        )

                        # save path
                    output_dir = os.path.join(
                        dir_output, region, str(rat), studyday, sleep_period
                    )

                    os.makedirs(output_dir, exist_ok=True)

                    trial_id = os.path.basename(trial_name).replace(".mat", "")

                    save_path = os.path.join(output_dir, f"{trial_id}_deltas.csv")

                    df.to_csv(save_path, index=False)

                    print(f"Saved: {len(df)} deltas -> {save_path}")

print("\nDone!")

# %%
