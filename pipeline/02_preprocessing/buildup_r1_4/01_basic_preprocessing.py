# ---- Import ----
import os

import numpy as np
from scipy.io import loadmat, savemat

from modules import ephys_preprocessing as processing

"""
This script performs the following preprocessing steps on the dataset:

1) Downsamples the data from 20000Hz to 1000Hz.

2) Removes power line artifacts (50Hz).

3) Segments each trial into 30-minute intervals (make sure the signal length is the same).

4) Saves the cleaned and processed results to the designated results directory.
"""


# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = "/media/yixiao/GL14_RAT_FA/"
dir_base2 = "/media/yixiao/Data5/"
dir_R1_4_RawData = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/RawData"
)
dir_R1_4_Scoring = os.path.join(
    dir_base2, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Sleepscoring_results_R1-4"
)

dir_R1_4_Preprec_withartifacts = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/Preprec_withartifacts"
)
# this folder will store the preprocessed data
dir_R1_4_Scoring_withartifacts = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/Scoring"
)


rats = np.arange(1, 5)
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 20000  # sampling frequency is 20000Hz
ds_fs = 1000  # target sample frequency

dates_special_handels1 = ["20220923", "20220926", "20220927"]
# On these dates, sleep session recordings are not split into standard 30-minute segments.

dates_special_handels2 = ["20220929", "20221006"]
# On these dates, a single trial may contain multiple recordings in the same directory.


# -------------------------------Preprocesing--------------------------------------
for rat in rats:  # for rat in [rats[0]]   rat in rats
    for region in regions:  # for region in [regions[0]]  region in regions
        dir_R1_4_RawData_perday = os.path.join(dir_R1_4_RawData, region, str(rat))
        folders_SD = [
            name
            for name in os.listdir(dir_R1_4_RawData_perday)
            if os.path.isdir(os.path.join(dir_R1_4_RawData_perday, name))
        ]

        for (
            studyday
        ) in folders_SD:  # for studyday in [folders_SD[0]] studyday in folders_SD
            for sleep_period in sleep_periods:  # for sleep_period in sleep_periods
                # choose the max_length according to the data and sleep period
                if studyday in dates_special_handels1:
                    if sleep_period == "presleep":
                        max_length = 2 * 30 * 60 * ds_fs  # 60min
                        max_length_scoring = 2 * 30 * 60
                    elif sleep_period == "postsleep":
                        max_length = 8 * 30 * 60 * ds_fs  # 240min
                        max_length_scoring = 8 * 30 * 60
                else:
                    max_length = 30 * 60 * ds_fs  # 30min
                    max_length_scoring = 30 * 60

                # find all the trial recordings (suffix = ".mat")
                dir_R1_4_RawData_pertrial = os.path.join(
                    dir_R1_4_RawData_perday, studyday, sleep_period
                )
                matched_files = []
                for root, dirs, files in os.walk(dir_R1_4_RawData_pertrial):
                    for name in files:
                        if name.endswith(".mat"):
                            matched_files.append(os.path.join(root, name))

                # find all the scoring results (suffix = ".mat")
                dir_R1_4_Scoring_pertrial = os.path.join(
                    dir_R1_4_Scoring, str(rat), studyday, sleep_period
                )
                matched_scoringfiles = []
                for root, dirs, files in os.walk(dir_R1_4_Scoring_pertrial):
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
                        dir_R1_4_Preprec_withartifacts,
                        region,
                        str(rat),
                        studyday,
                        sleep_period,
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    savemat(os.path.join(save_dir, trial_name), {"data": combined_data})

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
                        dir_R1_4_Scoring_withartifacts, str(rat), studyday, sleep_period
                    )
                    os.makedirs(save_dir_scoring, exist_ok=True)
                    savemat(
                        os.path.join(save_dir_scoring, os.path.basename(scoring_name)),
                        {"states": scoring},
                    )
