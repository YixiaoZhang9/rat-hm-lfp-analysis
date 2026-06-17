# ---- Import ----
import os
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat

from modules.lfp_artifact_MAD_detection import mad_artifact_detector
from modules.project_config import get_path

import matplotlib.pyplot as plt


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
dir_base1 = get_path("R9_16_root")
dir_R13_16_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16/PreprocessedData"
)
dir_R13_16_Scoring = os.path.join(dir_base1, "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16/Scoring")
dir_output = os.path.join(
    dir_base1,
    "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16/Artifact_detection_results",
)


rats = np.arange(13, 17)
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 1000  # downsampled sample frequency


# %%
for rat in rats:
    rat_str = f"rat{rat}"
    print(f"\n===== Processing {rat_str} =====")

    for region in regions:
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
                print(f"Processing {rat_str} |{region}| {studyday} | {sleep_period}")

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
                    artifact_rows = []

                    # find NREM bouts
                    NREM_bouts = find_bouts(scoring_data, target_value=3, fs=fs)

                    for bout_idx, (start_sample, end_sample) in enumerate(NREM_bouts):
                        bout_start_s = start_sample / fs
                        bout_end_s = end_sample / fs
                        bout_duration_sec = bout_end_s - bout_start_s

                        bout_data = data[start_sample:end_sample]

                        # Detect LFP artifacts using the median absolute deviation method
                        valid_times, artifact_intervals_s = mad_artifact_detector(
                            bout_data,
                            mad_thresh=6.0,
                            proportion_above_thresh=0.1,
                            removal_window_ms=100.0,
                            sampling_frequency=fs,
                        )

                        # artifact_intervals_s are relative to the current NREM bout
                        for artifact_idx, (artifact_start_rel_s, artifact_end_rel_s) in enumerate(artifact_intervals_s):

                            artifact_start_s = bout_start_s + artifact_start_rel_s
                            artifact_end_s = bout_start_s + artifact_end_rel_s

                            artifact_rows.append(
                                {
                                    "artifact_start_index": int(round(artifact_start_s * fs)),
                                    "artifact_end_index": int(round(artifact_end_s * fs)),
                                }
                            )

                        # # plot and check the detected artifact
                        # t_bout = np.arange(start_sample, end_sample) / fs
                        #
                        # plt.figure(figsize=(14, 4))
                        # plt.plot(t_bout, bout_data, linewidth=0.8)
                        #
                        # for artifact_start_rel_s, artifact_end_rel_s in artifact_intervals_s:
                        #     plt.axvspan(
                        #         bout_start_s + artifact_start_rel_s,
                        #         bout_start_s + artifact_end_rel_s,
                        #         color="red",
                        #         alpha=0.3,
                        #     )
                        #
                        # plt.xlabel("Time (s)")
                        # plt.ylabel("LFP")
                        # plt.title(
                        #     f"{rat_str} | {region} | {studyday} | {sleep_period} | "
                        #     f"bout {bout_idx}"
                        # )
                        # plt.tight_layout()
                        # plt.show()


                    # save CSV
                    df = pd.DataFrame(artifact_rows)

                    # output path
                    output_dir = os.path.join(
                        dir_output, region, str(rat), studyday, sleep_period
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    trial_id = os.path.basename(trial_name).replace(".mat", "")
                    save_path = os.path.join(
                        output_dir, f"{trial_id}_artifact_time_intervals.csv"
                    )

                    df.to_csv(save_path, index=False)

print("\n Done!")