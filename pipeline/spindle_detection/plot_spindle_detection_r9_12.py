# ---- Import ----
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog
from scipy.io import loadmat
from signal_viewer import SpindleViewer

from modules.project_config import get_path
from modules.threshold_ripple_detection import (
    filter_lfp,
)

# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = get_path("R9_16_root")
dir_R9_12_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/PreprocessedData"
)
dir_R9_12_Scoring = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/Scoring"
)
dir_R9_12_Spindle = os.path.join(
    dir_base1,
    "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/Spindle_detection_results",
)


rats = np.arange(9, 13)
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 1000  # downsampled sample frequency

# -----------------------------------------------------------------------------------
# ------------------------------plot the ripple rate per day per rat-----------------


# %%
# Compute ripple rate (ripples / min in NREM)
def compute_spindle_rate(csv_path, scoring):
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

    n_spindles = len(df)

    return n_spindles / nrem_minutes


# Collect results
results = []

for rat in rats:
    rat_str = f"rat{rat}"

    for region in ["HPC"]:
        dir_data = os.path.join(dir_R9_12_Data, region, str(rat))

        if not os.path.exists(dir_data):
            continue

        for studyday in os.listdir(dir_data):
            for sleep_period in sleep_periods:
                dir_trial = os.path.join(dir_data, studyday, sleep_period)
                dir_spindle = os.path.join(
                    dir_R9_12_Spindle, region, str(rat), studyday, sleep_period
                )

                # find all trial files
                matched_files = []
                for root, _, files in os.walk(dir_trial):
                    for name in files:
                        if name.endswith(".mat"):
                            matched_files.append(os.path.join(root, name))

                # load sleep scoring files
                dir_scoring = os.path.join(
                    dir_R9_12_Scoring, str(rat), str(studyday), sleep_period
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
                    csv_threshold = os.path.join(
                        dir_spindle,
                        "envelop_thr_1_peak_3.0",
                        f"{trial_id}_spindles.csv",
                    )
                    csv_wavelet = os.path.join(
                        dir_spindle,
                        "wavelet_amp_1_ampcore_3",
                        f"{trial_id}_spindles_wavelet.csv",
                    )
                    csv_wavelet_optimal = os.path.join(
                        dir_spindle,
                        "wavelet_optimal_thrL_0.8_thrH_3.0",
                        f"{trial_id}_spindles_wavelet_optimal.csv",
                    )

                    rate_th = compute_spindle_rate(csv_threshold, scoring)
                    rate_wavelet = compute_spindle_rate(csv_wavelet, scoring)
                    rate_wavelet_opt = compute_spindle_rate(
                        csv_wavelet_optimal, scoring
                    )

                    results.append(
                        {
                            "rat": rat_str,
                            "day": studyday,
                            "sleep": sleep_period,
                            "trial": trial_id,
                            "threshold": rate_th,
                            "wavelet": rate_wavelet,
                            "wavelet_optimal": rate_wavelet_opt,
                        }
                    )

df = pd.DataFrame(results)

# #Plot (per day average)
#
# df_day = df.groupby(["rat", "day"]).mean(numeric_only=True).reset_index()
#
# rats_unique = df_day["rat"].unique()
#
# fig, axes = plt.subplots(len(rats_unique), 1, figsize=(10, 4 * len(rats_unique)), sharex=True)
#
# if len(rats_unique) == 1:
#     axes = [axes]
#
# for ax, rat in zip(axes, rats_unique):
#     df_r = df_day[df_day["rat"] == rat]
#
#     ax.plot(df_r["day"], df_r["threshold"], marker='o', label="envelop_threshold")
#     ax.plot(df_r["day"], df_r["wavelet"], marker='o', label="wavelet")
#     ax.plot(df_r["day"], df_r["wavelet_optimal"], marker='o', label="wavelet_optimal")
#
#     ax.set_title(f"{rat}")
#     ax.set_ylabel("spindles / min")
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
    return list(zip(df["spindle_start"].values, df["spindle_end"].values))


# ---- Open file dialog to select a trial .mat file ----
file_path, _ = QFileDialog.getOpenFileName(
    None,
    "Select trial .mat file",
    dir_R9_12_Data,  # starting directory
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
dir_scoring = os.path.join(dir_R9_12_Scoring, rat, studyday, sleep_period)
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
dir_spindle = os.path.join(dir_R9_12_Spindle, region, rat, studyday, sleep_period)
csv_threshold = os.path.join(
    dir_spindle, "envelop_thr_1_peak_3.0", f"{trial_id}_spindles.csv"
)
csv_wavelet = os.path.join(
    dir_spindle, "wavelet_amp_1_ampcore_3", f"{trial_id}_spindles_wavelet.csv"
)
csv_wavelet_optimal = os.path.join(
    dir_spindle,
    "wavelet_optimal_thrL_0.8_thrH_3.0",
    f"{trial_id}_spindles_wavelet_optimal.csv",
)

threshold_events = load_events(csv_threshold) if os.path.exists(csv_threshold) else []
wavelet_events = load_events(csv_wavelet) if os.path.exists(csv_wavelet) else []
wavelet_optimal_events = (
    load_events(csv_wavelet_optimal) if os.path.exists(csv_wavelet_optimal) else []
)

lfp = loadmat(trial_path)["data"].squeeze()
filtered_lfp = filter_lfp(lfp, fs, [0.3, 30])

# ---- Launch viewer ----

# check if a QApplication already exists
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

event_sets = {
    "Threshold": threshold_events,
    "Threshold_wavelet": wavelet_events,
    "Threshold_wavelet_optimal": wavelet_optimal_events,
}

viewer = SpindleViewer(lfp=filtered_lfp, scoring=scoring, fs=fs, event_sets=event_sets)
viewer.show()
sys.exit(app.exec_())
