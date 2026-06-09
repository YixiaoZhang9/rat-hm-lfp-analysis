# ---- Import ----
import importlib
import os
import re

import numpy as np
from scipy.io import savemat

from modules import ephys_data_io as fstore
from modules.project_config import get_path

importlib.reload(fstore)

# ---- Set base paths, date lists, and constants for data processing ----
dir_base = get_path("data5_root")
dir_R1_4_Data = os.path.join(
    dir_base, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_OpenEphysRecordings_R1-4/"
)
dir_results_base = get_path("gl14_root")
dir_R1_4_results = os.path.join(
    dir_results_base, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/"
)
path_csv_overview_StudyDay = os.path.join(
    dir_R1_4_results, "Rat_HM_Ephys_TD_R1_4_Overview_StudyDay.csv"
)
path_csv_SelectedChannel = os.path.join(
    dir_R1_4_results, "Rat_HM_Ephys_TD_R1_8_SelectedChannel.csv"
)
path_csv_postfix_trial = os.path.join(
    dir_R1_4_results, "Rat_HM_Ephys_TD_R1_4_Suffix_Trial.csv"
)
date_prefix = "Rat_HM_Ephys_TD_R1-4_"
rats = np.arange(1, 5)


dates_all = [
    "20220923",
    "20220926",
    "20220927",
    "20220929",
    "20221003",
    "20221006",
    "20221010",
    "20221012",
    "20221013",
    "20221014",
    "20221018",
    "20221021",
    "20221024",
    "20221025",
    "20221026",
    "20221028",
    "20221031",
    "20221101",
    "20221103",
    "20221110",
    "20221111",
    "20221115",
    "20221116",
    "20221117",
    "20221122",
    "20221123",
    "20221124",
    "20221128",
    "20221201",
    "20221206",
    "20221207",
    "20221208",
    "20221209",
    "20221212",
]

dates_special_handels1 = ["20220923", "20220926", "20220927"]
# On these dates, sleep session recordings are not split into standard 30-minute segments.

dates_special_handels2 = ["20220929", "20221006"]
# On these dates, a single trial may contain multiple recordings in the same directory.

dates_fullname = [f"{date_prefix}{d}" for d in dates_all]
regions = ["HPC", "PL", "RSC"]
sleep_period = ["presleep", "postsleep"]

result_dirs = {
    "HPC": os.path.join(dir_R1_4_results, "RawData", "HPC"),
    "PL": os.path.join(dir_R1_4_results, "RawData", "PL"),
    "RSC": os.path.join(dir_R1_4_results, "RawData", "RSC"),
}

# ---- load and store the ephys recordings ----

for rat in rats:
    for region in regions:
        channel = fstore.get_selected_channel(path_csv_SelectedChannel, rat, region)

        for date in dates_all:
            if not fstore.check_studyDay(path_csv_overview_StudyDay, rat, date):
                continue

            pre_suffix, post_suffix = fstore.get_pre_post_suffixes(
                path_csv_postfix_trial, rat
            )

            print(f"\nLoading Ephys Recordings on {date} | Rat {rat} | Region {region}")
            region_result_dir = os.path.join(result_dirs[region], str(rat), date)
            path_recording = os.path.join(dir_R1_4_Data, f"{date_prefix}{date}")

            # Case 1: Special days with unsegmented recordings (e.g., 20220923)
            if date in dates_special_handels1:
                for period in sleep_period:
                    matched_dir = fstore.find_path_by_suffix(
                        path_recording, period, search_dir=True
                    )

                    # Edge case: 20220923 presleep contains two .continuous files
                    if date == "20220923" and period == "presleep":
                        suffixes = ["", "_2"]
                        for i, sfx in enumerate(suffixes):
                            pattern = re.compile(
                                rf"(?<![A-Z0-9])(?:CH)?{channel}{sfx}.continuous$"
                            )
                            recording = fstore.read_recording(matched_dir, pattern)
                            save_dir = os.path.join(region_result_dir, period)
                            os.makedirs(save_dir, exist_ok=True)
                            save_name = (
                                f"chan{channel}{chr(97 + i)}.mat"  # e.g., _a, _b
                            )
                            savemat(os.path.join(save_dir, save_name), recording)
                    else:
                        fstore.save_sleep_recording(
                            path_recording, period, channel, region_result_dir, period
                        )

            # Case 2: Normal segmented recordings with trial suffixes
            else:
                trial_suffixes = pre_suffix + post_suffix
                for trial in trial_suffixes:
                    period = "presleep" if trial in pre_suffix else "postsleep"

                    # Edge case: split continuous files
                    if (date == "20220929" and trial == "_6") or (
                        date == "20221006" and trial == "_4"
                    ):
                        matched_dir = fstore.find_path_by_suffix(
                            path_recording, trial, search_dir=True
                        )
                        suffixes = ["", "_2"]
                        for i, sfx in enumerate(suffixes):
                            pattern = re.compile(
                                rf"(?<![A-Z0-9])(?:CH)?{channel}{sfx}.continuous$"
                            )
                            recording = fstore.read_recording(matched_dir, pattern)
                            save_dir = os.path.join(region_result_dir, period)
                            os.makedirs(save_dir, exist_ok=True)
                            save_name = (
                                f"chan{channel}{trial}{chr(97 + i)}.mat"  # e.g., _a, _b
                            )
                            savemat(os.path.join(save_dir, save_name), recording)

                    else:
                        fstore.save_sleep_recording(
                            path_recording,
                            trial,
                            channel,
                            region_result_dir,
                            period,
                            filename_suffix=trial,
                        )
