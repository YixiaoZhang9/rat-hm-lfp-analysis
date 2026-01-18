# ---- Import ----
import os
import numpy as np
import importlib
from modules import ephys_data_io as fstore

importlib.reload(fstore)

# ---- Set base paths, date lists, and constants for data processing ----
dir_base = '/media/yixiao/Data4/'
dir_R9_12_Data = os.path.join(dir_base,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_OpenEphysRecordings_R9-12/')
dir_results_base = '/media/yixiao/Data4/'
dir_R9_12_results = os.path.join(dir_results_base, 'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/')
path_csv_SelectedChannel = os.path.join(dir_R9_12_results, 'Rat_HM_Ephys_TD_R9_16_SelectedChannel.csv')
path_csv_postfix_trial = os.path.join(dir_R9_12_results, 'Rat_HM_Ephys_TD_R9_12_Suffix_Trial.csv')
date_prefix = 'Rat_HM_Ephys_TD_R9-12_'
rats = np.arange(9,13)


dates_all = ['20230710','20230711','20230712','20230715','20230717','20230718','20230720','20230724','20230725','20230726',
             '20230728','20230730','20230731','20230801','20230802','20230803','20230804','20230807','20230808',
             '20230809','20230815','20230816','20230819','20230822','20230823','20230824','20230826','20230829','20230830',
             '20230831','20230901']

dates_special_handels1 = ['20230710']
# On this day, recording_3 is broken (rat 10 and 11 was affected)
dates_special_handels2 = ['20230717']
# On this day, recording 5-9 was unplugged in rat 10
dates_special_handels3 = ['20230815']
# On this day, recording 2 and 6 was unplugged in rat 10
dates_special_handels4 = ['20230819']
# On this day, recording 1 and 7 was unplugged in rat 10

dates_fullname = [f"{date_prefix}{d}" for d in dates_all]
regions = ['HPC','PL','RSC']
sleep_period = ['presleep','postsleep']

result_dirs = {
    'HPC': os.path.join(dir_R9_12_results, 'RawData', 'HPC'),
    'PL': os.path.join(dir_R9_12_results, 'RawData', 'PL'),
    'RSC': os.path.join(dir_R9_12_results, 'RawData', 'RSC')
}

# ---- load and store the ephys recordings ----

for rat in rats:

    for region in regions:
        channel = fstore.get_selected_channel(path_csv_SelectedChannel, rat, region)

        for date in dates_all:

            pre_suffix, post_suffix = fstore.get_pre_post_suffixes(path_csv_postfix_trial, rat)

            print(f"\nLoading Ephys Recordings on {date} | Rat {rat} | Region {region}")
            region_result_dir = os.path.join(result_dirs[region], str(rat), date)
            path_recording = os.path.join(dir_R9_12_Data, f"{date_prefix}{date}")

            trial_suffixes = pre_suffix + post_suffix
            for trial in trial_suffixes:
                period = 'presleep' if trial in pre_suffix else 'postsleep'

                # on 20230710 recording_3 is broken, skip it
                if rat in [10, 11] and date == "20230710" and trial == "_3":
                    continue
                # on 20230717 recording_5 to _9 were unplugged, skip it
                if rat == 10 and date == "20230717" and trial in ["_5", "_6", "_7", "_8", "_9"]:
                    continue
                # on 20230815 recording_2 and _6 were unplugged, skip it
                if rat == 10 and date == "20230815" and trial in ["_2", "_6"]:
                    continue
                # on 20230819 recording_1 and _7 were unplugged, skip it
                if rat == 10 and date == "20230819" and trial in ["_1", "_7"]:
                    continue

                fstore.save_sleep_recording(path_recording, trial, channel, region_result_dir, period,
                                             filename_suffix=trial)
