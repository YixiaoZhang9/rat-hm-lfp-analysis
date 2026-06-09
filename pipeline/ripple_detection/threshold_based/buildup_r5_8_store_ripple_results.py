# ---- Import ----
import os
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat

from modules.find_ripples_lfp import find_ripples_karlsson_Adaptive
from modules.project_config import get_path


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

    bouts = [(int(start * fs), int(end * fs)) for start, end in zip(starts, ends)]
    return bouts


# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = get_path("R1_8_root")
dir_R5_8_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/PreprocessedData"
)
dir_R5_8_Scoring = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/Scoring"
)
dir_output = os.path.join(
    dir_base1,
    "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/Ripple_detection_results",
)


rats = [5, 6, 7, 8]
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 1000  # downsampled sample frequency

f_plot = 0
Threshold_across_time = {}
NREM_duration_across_time = {}
Ripple_count_across_time = {}
Ripple_count_across_time_m2 = {}


# %%
# detect ripples
for rat in rats:
    rat_str = f"rat{rat}"
    print(f"\n===== Processing {rat_str} =====")

    for region in [regions[0]]:  # only 'HPC'
        dir_R5_8_Data_perday = os.path.join(dir_R5_8_Data, region, str(rat))

        if not os.path.exists(dir_R5_8_Data_perday):
            print(f"Missing: {dir_R5_8_Data_perday}")
            continue

        folders_SD = [
            name
            for name in os.listdir(dir_R5_8_Data_perday)
            if os.path.isdir(os.path.join(dir_R5_8_Data_perday, name))
        ]

        for studyday in folders_SD:
            for sleep_period in sleep_periods:
                print(f"Processing {rat_str} | {studyday} | {sleep_period}")

                dir_trial = os.path.join(dir_R5_8_Data_perday, studyday, sleep_period)

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
                    dir_R5_8_Scoring, str(rat), str(studyday), sleep_period
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

                    # prepare storage
                    ripple_start_list = []
                    ripple_peak_list = []
                    ripple_end_list = []
                    ripple_duration_list = []
                    ripple_amplitude_list = []
                    ripple_peak_frequency_list = []
                    ripple_mean_frequency_list = []
                    nrem_duration_list = []

                    # find NREM bouts
                    NREM_bouts = find_bouts(scoring_data, target_value=3, fs=fs)

                    for start_sample, end_sample in NREM_bouts:
                        bout_duration_sec = (end_sample - start_sample) / fs
                        bout_data = data[start_sample:end_sample]

                        Results_Threshold = find_ripples_karlsson_Adaptive(
                            bout_data, fs, f_plot=0
                        )

                        if len(Results_Threshold["StartIndex"]) == 0:
                            continue

                        ripple_karlsson = np.array(
                            list(
                                zip(
                                    (
                                        Results_Threshold["StartIndex"] + start_sample
                                    ).astype(int),
                                    (
                                        Results_Threshold["PeakIndex"] + start_sample
                                    ).astype(int),
                                    (
                                        Results_Threshold["EndIndex"] + start_sample
                                    ).astype(int),
                                )
                            )
                        )

                        duration = Results_Threshold["Duration"]
                        amplitude = Results_Threshold["Amplitude"]
                        peak_freq = Results_Threshold["Peak_Frequency"]
                        mean_freq = Results_Threshold["Mean_Frequency"]

                        # store each ripple
                        for i, ripple in enumerate(ripple_karlsson):
                            start = ripple[0]
                            peak = ripple[1]
                            end = ripple[2]

                            ripple_start_list.append(start)
                            ripple_peak_list.append(peak)
                            ripple_end_list.append(end)

                            ripple_duration_list.append(duration[i])
                            ripple_amplitude_list.append(amplitude[i])

                            ripple_peak_frequency_list.append(peak_freq[i])
                            ripple_mean_frequency_list.append(mean_freq[i])

                            nrem_duration_list.append(bout_duration_sec)

                    # save CSV
                    if len(ripple_start_list) > 0:
                        df = pd.DataFrame(
                            {
                                "ripple_start_index": ripple_start_list,
                                "ripple_peak_index": ripple_peak_list,
                                "ripple_end_index": ripple_end_list,
                                "ripple_duration_s": ripple_duration_list,
                                "ripple_amplitude": ripple_amplitude_list,
                                "ripple_peak_frequency_hz": ripple_peak_frequency_list,
                                "ripple_mean_frequency_hz": ripple_mean_frequency_list,
                                "nrem_bout_duration_s": nrem_duration_list,
                            }
                        )
                    else:
                        #
                        df = pd.DataFrame(
                            columns=[
                                "ripple_start_index",
                                "ripple_peak_index",
                                "ripple_end_index",
                                "ripple_duration_s",
                                "ripple_amplitude",
                                "ripple_peak_frequency_hz",
                                "ripple_mean_frequency_hz",
                                "nrem_bout_duration_s",
                            ]
                        )

                    # output path
                    output_dir = os.path.join(
                        dir_output, str(rat), studyday, sleep_period
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    trial_id = os.path.basename(trial_name).replace(".mat", "")
                    save_path = os.path.join(
                        output_dir, f"{trial_id}_hippocampal_ripples_threshold2.csv"
                    )

                    df.to_csv(save_path, index=False)

print("\n Done!")

# %%
