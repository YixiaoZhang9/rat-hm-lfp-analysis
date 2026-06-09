# ---- Import ----
import importlib
import os

import numpy as np

from modules import ephys_data_io as fstore
from modules.project_config import get_path

importlib.reload(fstore)

# ---- Set base paths, date lists, and constants for data processing ----
dir_base = get_path("RAT_HM_DATA4_ROOT")
dir_R13_16_Data = os.path.join(
    dir_base, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_OpenEphysRecordings_R13-16/"
)
dir_results_base = get_path("RAT_HM_DATA4_ROOT")
dir_R13_16_results = os.path.join(
    dir_results_base, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R13-16/"
)
path_csv_SelectedChannel = os.path.join(
    dir_R13_16_results, "Rat_HM_Ephys_TD_R9_16_SelectedChannel.csv"
)
path_csv_postfix_trial = os.path.join(
    dir_R13_16_results, "Rat_HM_Ephys_TD_R13_16_Suffix_Trial.csv"
)
date_prefix = "Rat_HM_Ephys_TD_R13-16_"
rats = np.arange(13, 17)


dates_all = [
    "20230710",
    "20230711",
    "20230713",
    "20230716",
    "20230717",
    "20230718",
    "20230720",
    "20230721",
    "20230724",
    "20230725",
    "20230728",
    "20230801",
    "20230803",
    "20230804",
    "20230808",
    "20230815",
    "20230816",
    "20230817",
    "20230821",
    "20230823",
    "20230824",
    "20230825",
    "20230828",
    "20230829",
    "20230830",
    "20230901",
    "20230903",
    "20230905",
    "20230906",
]

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


dates_fullname = [f"{date_prefix}{d}" for d in dates_all]
regions = ["HPC", "PL", "RSC"]
sleep_period = ["presleep", "postsleep"]

result_dirs = {
    "HPC": os.path.join(dir_R13_16_results, "RawData", "HPC"),
    "PL": os.path.join(dir_R13_16_results, "RawData", "PL"),
    "RSC": os.path.join(dir_R13_16_results, "RawData", "RSC"),
}

# ---- load and store the ephys recordings ----

for rat in rats:
    for region in regions:
        channel = fstore.get_selected_channel(path_csv_SelectedChannel, rat, region)

        for date in dates_all:
            # on these days all the recordings in rat 15 were discarded because of too noisy
            if rat == 15 and date in [
                "20230711",
                "20230716",
                "20230717",
                "20230721",
                "20230724",
                "20230801",
                "20230804",
            ]:
                continue

            pre_suffix, post_suffix = fstore.get_pre_post_suffixes(
                path_csv_postfix_trial, rat
            )

            print(f"\nLoading Ephys Recordings on {date} | Rat {rat} | Region {region}")
            region_result_dir = os.path.join(result_dirs[region], str(rat), date)
            path_recording = os.path.join(dir_R13_16_Data, f"{date_prefix}{date}")

            if date in dates_special_handles1:
                if "_1" in pre_suffix:
                    # repalce _1 with _1a/_1b
                    index = pre_suffix.index("_1")
                    pre_suffix[index : index + 1] = ["_1a", "_1b"]
            elif date in dates_special_handles2:
                if "_8" in post_suffix:
                    # repalce _1 with _8a/_8b
                    index = post_suffix.index("_8")
                    post_suffix[index : index + 1] = ["_8a", "_8b"]

            trial_suffixes = pre_suffix + post_suffix
            for trial in trial_suffixes:
                period = "presleep" if trial in pre_suffix else "postsleep"

                # on 20230724 recording_5 and _6 in rat 14 were too noisy, skip it
                if rat == 14 and date == "20230724" and trial in ["_5", "_6"]:
                    continue
                # on 20230801 recording_5 in rat 14 is too noisy, skip it
                if rat == 14 and date == "20230801" and trial == "_5":
                    continue
                # on 20230710 and 20230720 recording_6 in rat 15 were too noisy, skip it
                if rat == 15 and date in ["20230710", "20230720"] and trial == "_6":
                    continue
                # on 20230829 recording_9 in rat 15 is too noisy, skip it
                if rat == 15 and date == "20230829" and trial == "_9":
                    continue

                fstore.save_sleep_recording(
                    path_recording,
                    trial,
                    channel,
                    region_result_dir,
                    period,
                    filename_suffix=trial,
                )
