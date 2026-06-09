# ---- Import ----
import os
import random
import re

import numpy as np
from scipy.io import loadmat

from modules.project_config import get_path

# ---- Set base paths ----
dir_data = {}
dir_data["rat1"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/PreprocessedData/HPC/1/",
)
dir_data["rat2"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/PreprocessedData/HPC/2/",
)
dir_data["rat3"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/PreprocessedData/HPC/3/",
)
dir_data["rat5"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/PreprocessedData/HPC/5/",
)
dir_data["rat6"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/PreprocessedData/HPC/6/",
)
dir_data["rat7"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/PreprocessedData/HPC/7/",
)
dir_data["rat8"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/PreprocessedData/HPC/8/",
)
dir_data["rat11"] = os.path.join(
    get_path("R9_16_root"),
    "Rat_HM_Ephys_TD_Analysis_R9_16/R9-12/PreprocessedData/HPC/11/",
)
dir_data["rat12"] = os.path.join(
    get_path("R9_16_root"),
    "Rat_HM_Ephys_TD_Analysis_R9_16/R9-12/PreprocessedData/HPC/12/",
)
dir_data["rat13"] = os.path.join(
    get_path("R9_16_root"),
    "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16/PreprocessedData/HPC/13/",
)
dir_data["rat14"] = os.path.join(
    get_path("R9_16_root"),
    "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16/PreprocessedData/HPC/14/",
)
dir_data["rat15"] = os.path.join(
    get_path("R9_16_root"),
    "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16/PreprocessedData/HPC/15/",
)
dir_scoring = {}
dir_scoring["rat1"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/Scoring/1/",
)
dir_scoring["rat2"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/Scoring/2/",
)
dir_scoring["rat3"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/Scoring/3/",
)
dir_scoring["rat5"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/Scoring/5/",
)
dir_scoring["rat6"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/Scoring/6/",
)
dir_scoring["rat7"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/Scoring/7/",
)
dir_scoring["rat8"] = os.path.join(
    get_path("R1_8_root"),
    "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/Scoring/7/",
)
dir_scoring["rat11"] = os.path.join(
    get_path("R9_16_root"),
    "Rat_HM_Ephys_TD_Analysis_R9_16/R9-12/Scoring/11/",
)
dir_scoring["rat12"] = os.path.join(
    get_path("R9_16_root"),
    "Rat_HM_Ephys_TD_Analysis_R9_16/R9-12/Scoring/12/",
)
dir_scoring["rat13"] = os.path.join(
    get_path("R9_16_root"),
    "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16/Scoring/13/",
)
dir_scoring["rat14"] = os.path.join(
    get_path("R9_16_root"),
    "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16/Scoring/14/",
)
dir_scoring["rat15"] = os.path.join(
    get_path("R9_16_root"),
    "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16/Scoring/15/",
)

rats = [1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 15]

sleep_periods = ["presleep", "postsleep"]

common_seeds = [1, 137, 999, 7, 271828, 2718, 1314, 1234, 3, 42, 1, 9001]

# ---- find all the available trial for each rat
all_selected_trials = []

for i, rat in enumerate(rats):
    random.seed(common_seeds[i])
    all_trials = []
    selected_trials = []
    dir_path = dir_data["rat" + str(rat)]
    dir_scoring_path = dir_scoring["rat" + str(rat)]
    # find all the available date
    folders_SD = [
        name
        for name in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, name))
    ]

    for studyday in folders_SD:
        for sleep_period in sleep_periods:
            # find all the available trial
            dir_data_pertrial = os.path.join(dir_path, studyday, sleep_period)
            for root, dirs, files in os.walk(dir_data_pertrial):
                for name in files:
                    if name.endswith(".mat"):
                        full_path = os.path.join(root, name)

                        dir_scoring_pertrial = os.path.join(
                            dir_scoring_path, studyday, sleep_period
                        )
                        scoring_files = [
                            f
                            for f in os.listdir(dir_scoring_pertrial)
                            if f.endswith(".mat")
                        ]
                        if len(scoring_files) == 1:
                            scoring_data = loadmat(
                                os.path.join(dir_scoring_pertrial, scoring_files[0])
                            )["states"].squeeze()
                        else:
                            # Find matching scoring (suffix match)
                            match = re.search(r"_(\d+)\.mat$", name)
                            if match:
                                suffix = match.group(1)
                            suffix = suffix.zfill(2)  # from '6' to '06'
                            pattern = re.compile(rf"_{suffix}_")  # '_06_'
                            for f in os.listdir(dir_scoring_pertrial):
                                if pattern.search(f):
                                    trial_scoring_data = loadmat(
                                        os.path.join(dir_scoring_pertrial, f)
                                    )
                                    scoring_data = trial_scoring_data[
                                        "states"
                                    ].squeeze()
                                    f_scoring = f
                                    break
                        # calculate the nrem duration of this trial
                        nrem_seconds = np.sum(scoring_data == 3)
                        nrem_minutes = nrem_seconds / 60.0

                        all_trials.append((full_path, nrem_minutes))

    random.shuffle(all_trials)
    selected_trials = random.sample(all_trials, k=9)

    for item in selected_trials:
        print(item)
    a = 1
    total_NREM_duration = sum(trial[1] for trial in selected_trials)
    print(f"{name}: total NREM duration = {total_NREM_duration} minutes")
    all_selected_trials.append(selected_trials)

# check the NREM duration of each trial


# version 2
## assign the lfp file for each reviewer
reviewer = ["A", "K", "L", "S"]
flat_trials = [trial for sublist in all_selected_trials for trial in sublist]

double_all_selected_trials = flat_trials

random.seed(42)
random.shuffle(double_all_selected_trials)

reviewer_dict = {r: [] for r in reviewer}

num_trials = len(double_all_selected_trials) // len(
    reviewer
)  # number of trial per reviewer
remainder = len(double_all_selected_trials) % len(reviewer)

for i, trial in enumerate(double_all_selected_trials):
    if i < num_trials * len(reviewer):
        for revi in reviewer:
            if (
                len(reviewer_dict[revi]) < num_trials
                and trial not in reviewer_dict[revi]
            ):
                reviewer_dict[revi].append(trial)
                break

    else:
        reviewer_dict["Y"].append(trial)


for item in reviewer_dict.values():
    print(item)

# calculate the total NREM duration for each reviewer
for name in reviewer_dict:
    total_NREM_duration = sum(trial[1] for trial in reviewer_dict[name])
    print(f"{name}: total NREM duration = {total_NREM_duration} minutes")

a = 1


"""
# version 1
## assign the lfp file for each reviewer
reviewer = ['A','K','L','Y','S']
flat_trials = [trial for sublist in all_selected_trials for trial in sublist]

double_all_selected_trials = flat_trials * 2

random.seed(42)
random.shuffle(double_all_selected_trials)

reviewer_dict = {r: [] for r in reviewer}

num_trials = len(double_all_selected_trials)//len(reviewer) # number of trial per reviewer
remainder = len(double_all_selected_trials)%len(reviewer)

for i,trial in enumerate(double_all_selected_trials):
    if i < num_trials * len(reviewer):
        for revi in reviewer:
            if len(reviewer_dict[revi])<num_trials and trial not in reviewer_dict[revi]:
                reviewer_dict[revi].append(trial)
                break

    else:
        reviewer_dict['Y' ].append(trial)


for item in reviewer_dict.values():
    print(item)

# calculate the total NREM duration for each reviewer
for name in reviewer_dict:
    total_NREM_duration = sum(trial[1] for trial in reviewer_dict[name])
    print(f"{name}: total NREM duration = {total_NREM_duration} minutes")

a = 1
"""
