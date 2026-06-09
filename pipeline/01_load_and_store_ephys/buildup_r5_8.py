# ---- Import ----
import importlib
import os

import numpy as np

from modules import ephys_data_io as fstore
from modules.project_config import get_path

importlib.reload(fstore)

# ---- Set base paths, date lists, and constants for data processing ----

dir_base = get_path("data5_root")
dir_R5_8_Data = os.path.join(
    dir_base, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_OpenEphysRecordings_R5-8/"
)
dir_results_base = get_path("gl14_root")
dir_R5_8_results = os.path.join(
    dir_results_base, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/"
)
path_csv_overview_StudyDay = os.path.join(
    dir_R5_8_results, "Rat_HM_Ephys_TD_R5_8_Overview_StudyDay.csv"
)
path_csv_SelectedChannel = os.path.join(
    dir_R5_8_results, "Rat_HM_Ephys_TD_R1_8_SelectedChannel.csv"
)
path_csv_postfix_trial = os.path.join(
    dir_R5_8_results, "Rat_HM_Ephys_TD_R5_8_Suffix_Trial.csv"
)
path_csv_R5_8_20221005_Folder_SelectedChannel = os.path.join(
    dir_R5_8_results, "Rat_HM_Ephys_TD_R5_8_20221005_Folder_SelectedChannel.csv"
)
path_csv_R5_20221007_Folder_SelectedChannel = os.path.join(
    dir_R5_8_results, "Rat_HM_Ephys_TD_R5_20221007_Folder_SelectedChannel.csv"
)
date_prefix = "Rat_HM_Ephys_TD_R5-8_"
rats = np.arange(5, 9)  # already modified, original rats = np.arange(5,9)


dates_all = [
    "20220927",
    "20220929",
    "20220930",
    "20221003",
    "20221005",
    "20221007",
    "20221012",
    "20221014",
    "20221017",
    "20221018",
    "20221019",
    "20221021",
    "20221024",
    "20221028",
    "20221031",
    "20221101",
    "20221102",
    "20221107",
    "20221108",
    "20221109",
    "20221111",
    "20221117",
    "20221118",
    "20221121",
    "20221123",
    "20221124",
    "20221125",
    "20221128",
    "20221129",
    "20221205",
    "20221206",
    "20221207",
    "20221209",
]

# not taking date '20220923' because of the number of channels in presleep period is incorrect (does not have channel 124
# which is selected in HPC region)

dates_special_handles1 = ["20220927"]
# On these dates, sleep session recordings are not split into standard 30-minute segments.

dates_special_handles2 = ["20220929", "20220930"]
# These dates all the rats has the same pre_sufix and post_suffix

dates_special_handles3 = ["20221005", "20221007"]
# these dates, the trials and chosed channels are seperately decided bacause of expirimental issues

dates_fullname = [f"{date_prefix}{d}" for d in dates_all]
regions = ["HPC", "PL", "RSC"]
sleep_period = ["presleep", "postsleep"]

result_dirs = {
    "HPC": os.path.join(dir_R5_8_results, "RawData", "HPC"),
    "PL": os.path.join(dir_R5_8_results, "RawData", "PL"),
    "RSC": os.path.join(dir_R5_8_results, "RawData", "RSC"),
}

# ---- load and store the ephys recordings ----

for rat in rats:
    for region in regions:
        default_channel = fstore.get_selected_channel(
            path_csv_SelectedChannel, rat, region
        )

        for date in dates_all:
            if not fstore.check_studyDay(path_csv_overview_StudyDay, rat, date):
                continue
            pre_suffix, post_suffix = fstore.get_pre_post_suffixes(
                path_csv_postfix_trial, rat
            )
            print(f"\nLoading Ephys Recordings on {date} | Rat {rat} | Region {region}")
            region_result_dir = os.path.join(result_dirs[region], str(rat), date)

            # Case 1: Special days with unsegmented recordings (e.g., 20220923)
            if date in dates_special_handles1:
                path_recording = os.path.join(dir_R5_8_Data, f"{date_prefix}{date}")
                for period in sleep_period:
                    fstore.save_sleep_recording(
                        path_recording,
                        period,
                        default_channel,
                        region_result_dir,
                        period,
                    )

            # Case 2: Special days with specific filepath and selected channel (e.g., 20221005,20221007)
            elif date == "20221005":
                # in 20221005,trial 2 is split into 2a,2b,2c
                if rat in [6, 7] and "_2" in pre_suffix:
                    folder, channel = fstore.get_folder_selected_channel_r5_8_20221005(
                        path_csv_R5_8_20221005_Folder_SelectedChannel, rat, region, "_2"
                    )
                    # repalce _2 with _2a/_2b/_2c
                    index = pre_suffix.index("_2")
                    pre_suffix[index : index + 1] = ["_2a", "_2b", "_2c"]
                else:
                    folder, channel = None, None

                for idx in pre_suffix + post_suffix:
                    if idx in ["_2a", "_2b", "_2c"]:
                        this_folder, this_channel = folder, channel
                    else:
                        this_folder, this_channel = (
                            fstore.get_folder_selected_channel_r5_8_20221005(
                                path_csv_R5_8_20221005_Folder_SelectedChannel,
                                rat,
                                region,
                                idx,
                            )
                        )

                    path_recording = os.path.join(dir_R5_8_Data, this_folder)
                    period = "presleep" if idx in pre_suffix else "postsleep"
                    fstore.save_sleep_recording(
                        path_recording,
                        idx,
                        this_channel,
                        region_result_dir,
                        period,
                        filename_suffix=idx,
                    )

            # Case 3: Special days with specific filepath and selected channel (20221007)
            elif rat == 5 and date == "20221007":
                for idx in pre_suffix + post_suffix:
                    folder, channel = fstore.get_folder_selected_channel_r5_20221007(
                        path_csv_R5_20221007_Folder_SelectedChannel, region, idx
                    )
                    path_recording = os.path.join(dir_R5_8_Data, folder)
                    period = "presleep" if idx in pre_suffix else "postsleep"
                    fstore.save_sleep_recording(
                        path_recording,
                        idx,
                        channel,
                        region_result_dir,
                        period,
                        filename_suffix=idx,
                    )

            # Case 4: normal days
            else:
                path_recording = os.path.join(dir_R5_8_Data, f"{date_prefix}{date}")

                if date == "20220929":
                    pre_suffix = ["_1", "_2"]
                    post_suffix = ["_4", "_5", "_6", "_7", "_8", "_9", "_10", "_11"]
                elif date == "20220930":
                    pre_suffix = ["_1", "_2"]
                    post_suffix = ["_3", "_4", "_5", "_6", "_7", "_8", "_9", "_10"]

                for idx in pre_suffix + post_suffix:
                    period = "presleep" if idx in pre_suffix else "postsleep"
                    fstore.save_sleep_recording(
                        path_recording,
                        idx,
                        default_channel,
                        region_result_dir,
                        period,
                        filename_suffix=idx,
                    )
