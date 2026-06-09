# ---- Import ----
import os

import numpy as np
from scipy.io import loadmat, savemat

from modules import ephys_preprocessing as processing
from modules.project_config import get_path

"""
This script performs the following preprocessing steps on the dataset:

1) Downsamples the data from 20000Hz to 1000Hz.

2) Removes power line artifacts (50Hz).

3) Segments each trial into 30-minute intervals (make sure the signal length is the same).

4) Saves the cleaned and processed results to the designated results directory.
"""


# ---- Set base paths, date lists, and constants for data processing ----
dir_base = get_path("R9_16_root")
dir_R13_16_RawData = os.path.join(
    dir_base, "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16/RawData"
)
dir_R13_16_Scoring = os.path.join(
    dir_base, "Rat_HM_Ephys_TD_Sleepscoring_results_R13-16"
)

dir_R13_16_Preprec_withartifacts = os.path.join(
    dir_base, "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16/Preprec_withartifacts"
)
# this folder will store the preprocessed data
dir_R13_16_Scoring_withartifacts = os.path.join(
    dir_base, "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16/Scoring"
)


rats = np.arange(13, 17)
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 20000  # sampling frequency is 20000Hz
ds_fs = 1000  # target sample frequency

dates_special_handles1 = ["20230710"]
# theis date, recording 1 was seperated into two parts bacause of expirimental issues
dates_special_handles2 = ["20230711"]
# theis date, recording 8 was seperated into two parts bacause of expirimental issues


dates_special_handels3 = ["20230724"]
# on this day recording5 and recording 6 in rat 14 were discarded because of too noisy
dates_special_handels4 = ["20230801"]
# on this day recording5  in rat 14 were discarded because of too noisy
dates_special_handels5 = ["20230710", "20230720"]
# on these days recording6 in rat 15 were discarded because of too noisy
dates_special_handels6 = [
    "20230711",
    "20230716",
    "20230717",
    "20230721",
    "20230724",
    "20230801",
    "20230804",
]
# on these days all the recordings in rat 15 were discarded because of too noisy
dates_special_handels7 = ["20230829"]
# on this day recording9 in rat 15 were discarded because of too noisy


# -------------------------------Preprocesing--------------------------------------
for rat in [rats[3]]:  # for rat in [rats[0]]   rat in rats
    for region in [regions[0]]:  # for region in [regions[0]]  region in regions
        dir_R13_16_RawData_perday = os.path.join(dir_R13_16_RawData, region, str(rat))
        folders_SD = [
            name
            for name in os.listdir(dir_R13_16_RawData_perday)
            if os.path.isdir(os.path.join(dir_R13_16_RawData_perday, name))
        ]

        for (
            studyday
        ) in folders_SD:  # for studyday in [folders_SD[0]] studyday in folders_SD
            print(
                f"\nProcessing Ephys Recordings on {studyday} | Rat {rat} | Region {region}"
            )

            for sleep_period in sleep_periods:  # for sleep_period in sleep_periods
                # choose the max_length according to the data and sleep period

                max_length = 30 * 60 * ds_fs  # 30min
                max_length_scoring = 30 * 60

                # find all the trial recordings (suffix = ".mat")
                dir_R13_16_RawData_pertrial = os.path.join(
                    dir_R13_16_RawData_perday, studyday, sleep_period
                )
                matched_files = []
                for root, dirs, files in os.walk(dir_R13_16_RawData_pertrial):
                    for name in files:
                        if name.endswith(".mat"):
                            matched_files.append(os.path.join(root, name))

                # find all the scoring results (suffix = ".mat")
                dir_R13_16_Scoring_pertrial = os.path.join(
                    dir_R13_16_Scoring, str(rat), studyday, sleep_period
                )
                matched_scoringfiles = []
                for root, dirs, files in os.walk(dir_R13_16_Scoring_pertrial):
                    for name in files:
                        if name.endswith(".mat"):
                            matched_scoringfiles.append(os.path.join(root, name))

                # group the trial according to the suffix ("a","b","c" should be the same trial)
                grouped = processing.group_files(matched_files)

                for (
                    trial_name,
                    files,
                ) in grouped.items():  # for trial_name, files in [list(grouped.items())[0]]: for trial_name, files in grouped.items()
                    filtered_segments = []
                    for suffix, filepath in files:
                        trial_data = loadmat(filepath)
                        data = trial_data["data"]
                        # downsampling the data
                        data_ds = processing.downsampling(
                            data, fs, ds_fs, plot_response=False
                        )
                        # remove powerline artifacts
                        Powerline_freq = [50, 100, 150, 200, 250, 300, 350, 400, 450]

                        data_filtered = processing.powerline_filter(
                            data_ds,
                            ds_fs,
                            Powerline_freq,
                            Method="Adaptive_RLS",
                            plot_data=False,
                        )
                        filtered_segments.append(data_filtered)

                    # merge the data if one trial is spilt into several recordings bacause of experimental issure
                    if len(filtered_segments) == 1:
                        combined_data = filtered_segments[0]
                    else:
                        trial_name = trial_name + ".mat"
                        combined_data = filtered_segments[0]
                        for seg in filtered_segments[1:]:
                            combined_data = processing.smooth_transition(
                                combined_data,
                                seg,
                                smooth_points=50,
                                window_size=5,
                                plot_comparison=False,
                            )
                            # avoid abrupt changes

                    # Keep only the first 30 minutes of data
                    combined_data = combined_data.squeeze()
                    if combined_data.shape[-1] < max_length:
                        print(
                            f"The duration of trial in {rat} {studyday} {trial_name} is shorter than 30min."
                        )
                    elif combined_data.shape[-1] > max_length:
                        combined_data = combined_data[:max_length]
                    # save the downsampled and cutted data
                    save_dir = os.path.join(
                        dir_R13_16_Preprec_withartifacts,
                        region,
                        str(rat),
                        studyday,
                        sleep_period,
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    savemat(os.path.join(save_dir, trial_name), {"data": combined_data})
                    print(f"\nprocessed data  saved on {save_dir}")

                # ------------------------ also cut the sleep scoring files into 30 min---------------------------------
                for scoring_name in matched_scoringfiles:
                    trial_scoring = loadmat(scoring_name)
                    scoring = trial_scoring["states"]

                    scoring = scoring.squeeze()
                    # Keep only the first 30 minutes of scoring (scoring epoch is 1s)
                    if scoring.shape[-1] < max_length_scoring:
                        print(
                            f"The duration of scoring in {rat} {studyday} {scoring_name} is shorter than 30min."
                        )
                    elif scoring.shape[-1] > max_length_scoring:
                        scoring = scoring[:max_length_scoring]
                    # save the downsampled and cutted scoring
                    save_dir_scoring = os.path.join(
                        dir_R13_16_Scoring_withartifacts,
                        str(rat),
                        studyday,
                        sleep_period,
                    )
                    os.makedirs(save_dir_scoring, exist_ok=True)
                    savemat(
                        os.path.join(save_dir_scoring, os.path.basename(scoring_name)),
                        {"states": scoring},
                    )
                    # print(f"\nSleep scoring saved on {save_dir_scoring}")
