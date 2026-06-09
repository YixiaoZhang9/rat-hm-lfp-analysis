# ---- Import ----
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog
from scipy.io import loadmat
from signal_viewer import DeltaViewer

from modules.project_config import get_path
from modules.threshold_ripple_detection import (
    filter_lfp,
)

# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = get_path("RAT_HM_DATA4_ROOT")
dir_R13_16_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R13-16/PreprocessedData"
)
dir_R13_16_Scoring = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R13-16/Scoring"
)
dir_R13_16_Delta = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R13-16/Delta_detection_results"
)

rats = np.arange(13, 17)
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 1000  # downsampled sample frequency

# -----------------------------------------------------------------------------------
# ------------------------------plot the delta rate per day per rat-----------------


# %%
# Compute delta rate (deltas / min in NREM)
def compute_delta_rate(csv_path, scoring):

    nrem_seconds = np.sum(scoring == 3)
    nrem_minutes = nrem_seconds / 60

    if nrem_minutes == 0:
        return np.nan

    if not os.path.exists(csv_path):
        return np.nan

    if os.path.getsize(csv_path) == 0:
        return 0.0

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return 0.0

    if len(df) == 0:
        return 0.0

    n_deltas = len(df)

    return n_deltas / nrem_minutes


# Collect results
results = []

for rat in rats:
    rat_str = f"rat{rat}"

    for region in ["RSC"]:
        dir_data = os.path.join(dir_R13_16_Data, region, str(rat))

        if not os.path.exists(dir_data):
            continue

        for studyday in os.listdir(dir_data):
            for sleep_period in sleep_periods:
                dir_trial = os.path.join(dir_data, studyday, sleep_period)
                dir_delta = os.path.join(
                    dir_R13_16_Delta, region, str(rat), studyday, sleep_period
                )

                # find all trial files
                matched_files = []
                for root, _, files in os.walk(dir_trial):
                    for name in files:
                        if name.endswith(".mat"):
                            matched_files.append(os.path.join(root, name))

                # load sleep scoring files
                dir_scoring = os.path.join(
                    dir_R13_16_Scoring, str(rat), str(studyday), sleep_period
                )

                if not os.path.exists(dir_scoring):
                    print(f"Missing scoring: {dir_scoring}")
                    continue

                scoring_files = [
                    f for f in os.listdir(dir_scoring) if f.endswith(".mat")
                ]

                # process each trial
                for trial_name in matched_files:
                    # matched sleep scoring files
                    if len(scoring_files) == 1:
                        scoring = loadmat(os.path.join(dir_scoring, scoring_files[0]))[
                            "states"
                        ].squeeze()
                    else:
                        match = re.search(r"_(\d+)\.mat$", trial_name)
                        if not match:
                            print("No matching scoring file")
                            continue

                        suffix = match.group(1).zfill(2)
                        pattern = re.compile(rf"_{suffix}_")

                        scoring = None
                        for f in scoring_files:
                            if pattern.search(f):
                                scoring = loadmat(os.path.join(dir_scoring, f))[
                                    "states"
                                ].squeeze()
                                break

                    trial_id = Path(trial_name).stem
                    # --- CSV paths ---
                    csv_threshold = os.path.join(dir_delta, f"{trial_id}_deltas.csv")

                    rate_th = compute_delta_rate(csv_threshold, scoring)

                    results.append(
                        {
                            "rat": rat_str,
                            "day": studyday,
                            "sleep": sleep_period,
                            "trial": trial_id,
                            "threshold": rate_th,
                        }
                    )


df = pd.DataFrame(results)


# #Plot (per day average)

# df_day = df.groupby(["rat", "day"]).mean(numeric_only=True).reset_index()
#
# rats_unique = df_day["rat"].unique()
#
# fig, axes = plt.subplots(len(rats_unique), 1, figsize=(10, 4*len(rats_unique)), sharex=True)
#
# if len(rats_unique) == 1:
#     axes = [axes]
#
# for ax, rat in zip(axes, rats_unique):
#
#     df_r = df_day[df_day["rat"] == rat]
#
#     ax.plot(df_r["day"], df_r["threshold"], marker='o', label="threshold")
#
#     ax.set_ylim(45, 80)
#     ax.set_title(f"{rat}")
#     ax.set_ylabel("deltas / min")
#
#     ax.legend()
#
# axes[-1].set_xlabel("Day")
# plt.xticks(rotation=45)
#
# plt.tight_layout()
# plt.show()

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


##----------------------------------------------------------------------
##--------------------------plot the signal-----------------------------
def load_events(csv_path):
    df = pd.read_csv(csv_path)
    return list(zip(df["delta_start"].values, df["delta_end"].values))


# ---- Open file dialog to select a trial .mat file ----
file_path, _ = QFileDialog.getOpenFileName(
    None,
    "Select trial .mat file",
    dir_R13_16_Data,  # starting directory
    "MAT files (*.mat)",
)

if not file_path:
    print("No file selected.")
    sys.exit()

trial_path = Path(file_path)
trial_id = trial_path.stem
parts = trial_path.parts

# ---- Deduce rat, studyday, sleep_period, and region from path ----
try:
    region = parts[-5]
    rat = parts[-4]
    studyday = parts[-3]
    sleep_period = parts[-2]  # if trial is in folder named presleep/postsleep
except IndexError:
    print("Path does not match expected format.")
    sys.exit()

# ---- Find scoring file ----
dir_scoring = os.path.join(dir_R13_16_Scoring, rat, studyday, sleep_period)
scoring_files = [f for f in os.listdir(dir_scoring) if f.endswith(".mat")]
# matched sleep scoring files
if len(scoring_files) == 1:
    scoring = loadmat(os.path.join(dir_scoring, scoring_files[0]))["states"].squeeze()
else:
    match = re.search(r"_(\d+)$", trial_id)
    if not match:
        print("No matching scoring file")

    suffix = match.group(1).zfill(2)
    pattern = re.compile(rf"_{suffix}_")

    scoring = None
    for f in scoring_files:
        if pattern.search(f):
            scoring = loadmat(os.path.join(dir_scoring, f))["states"].squeeze()
            break

# ---- Find ripple CSVs ----
dir_delta = os.path.join(dir_R13_16_Delta, region, rat, studyday, sleep_period)
csv_threshold = os.path.join(dir_delta, f"{trial_id}_deltas.csv")

threshold_events = load_events(csv_threshold) if os.path.exists(csv_threshold) else []

lfp = loadmat(trial_path)["data"].squeeze()
filtered_lfp = filter_lfp(lfp, fs, [0.3, 30])

# ---- Launch viewer ----

# check if a QApplication already exists
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

event_sets = {
    "Threshold": threshold_events,
}

viewer = DeltaViewer(lfp=filtered_lfp, scoring=scoring, fs=fs, event_sets=event_sets)
viewer.show()
sys.exit(app.exec_())
