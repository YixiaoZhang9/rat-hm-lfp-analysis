from modules.project_config import get_path

"""
This script assesses the reliability of spindle detection using two validation criteria:

1) Specificity check: quantify spindle events detected during non-NREM sleep,
   under the assumption that true spindles primarily occur during NREM sleep.

2) Condition comparison: compare spindle rates across behavioral conditions,
   where higher spindle activity is expected during training compared to home sessions.
"""

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5.QtWidgets import QApplication, QFileDialog
from scipy.io import loadmat
from signal_viewer import SpindleViewer

from modules.threshold_ripple_detection import filter_lfp

# Paths
dir_base1 = get_path("R9_16_root")

dir_data_root = os.path.join(dir_base1, "Rat_HM_Ephys_TD_Analysis_R9_16/R9-12/")
dir_R9_12_Data = os.path.join(dir_data_root, "PreprocessedData")
dir_R9_12_Scoring = os.path.join(dir_data_root, "Scoring")
dir_R9_12_Spindle = os.path.join(dir_data_root, "Spindle_detection_results")
dir_R9_12_condition = os.path.join(
    dir_data_root, "Rat_HM_Ephys_TD_R9_12_Overview_StudyDay_Condition.xlsx"
)

rats = np.arange(9, 13)
sleep_periods = ["presleep", "postsleep"]
regions = ["HPC", "PL", "RSC"]
fs = 1000


# Spindle rate computatio
def compute_spindle_rate(csv_path, scoring, use_nrem=True):
    if use_nrem:
        valid_time = np.sum(scoring == 3)  # NREM
    else:
        valid_time = np.sum(scoring != 3)  # non-NREM

    valid_minutes = valid_time / 60

    if valid_minutes == 0:
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

    return n_spindles / valid_minutes


# Load condition table
xls = pd.ExcelFile(dir_R9_12_condition)
rat_conditions = {}

for sheet in xls.sheet_names:
    df_cond = pd.read_excel(dir_R9_12_condition, sheet_name=sheet)
    df_cond.columns = df_cond.columns.str.strip()

    df_sub = df_cond[["SD", "Condition"]].dropna()
    df_sub["SD"] = df_sub["SD"].astype(str)

    rat_conditions[sheet] = dict(zip(df_sub["SD"], df_sub["Condition"]))


# Main extraction loop
results = []

for rat in rats:
    rat_str = f"rat{rat}"

    for region in ["HPC"]:
        dir_rat = os.path.join(dir_R9_12_Data, region, str(rat))

        if not os.path.exists(dir_rat):
            continue

        for studyday in os.listdir(dir_rat):
            for sleep_period in sleep_periods:
                dir_trial = os.path.join(dir_rat, studyday, sleep_period)
                dir_spindle = os.path.join(
                    dir_R9_12_Spindle, region, str(rat), studyday, sleep_period
                )
                dir_scoring = os.path.join(
                    dir_R9_12_Scoring, str(rat), str(studyday), sleep_period
                )

                if not os.path.exists(dir_scoring):
                    continue

                scoring_files = [
                    f for f in os.listdir(dir_scoring) if f.endswith(".mat")
                ]

                trial_files = []
                for root, _, files in os.walk(dir_trial):
                    for f in files:
                        if f.endswith(".mat"):
                            trial_files.append(os.path.join(root, f))

                for trial_path in trial_files:
                    trial_id = Path(trial_path).stem

                    # Load sleep scoring
                    if len(scoring_files) == 1:
                        scoring = loadmat(os.path.join(dir_scoring, scoring_files[0]))[
                            "states"
                        ].squeeze()
                    else:
                        match = re.search(r"_(\d+)\.mat$", trial_path)
                        if not match:
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

                    # --- CSV paths ---
                    csv_threshold = os.path.join(
                        dir_spindle,
                        "envelop_thr_1_peak_3.0",
                        f"{trial_id}_spindles_non_NREM.csv",
                    )
                    csv_wavelet = os.path.join(
                        dir_spindle,
                        "wavelet_amp_1_ampcore_3",
                        f"{trial_id}_spindles_wavelet_non_NREM.csv",
                    )
                    csv_wavelet_optimal = os.path.join(
                        dir_spindle,
                        "wavelet_optimal_thrL_0.8_thrH_3.0",
                        f"{trial_id}_spindles_wavelet_optimal_non_NREM.csv",
                    )

                    rate_th = compute_spindle_rate(
                        csv_threshold, scoring, use_nrem=False
                    )
                    rate_wavelet = compute_spindle_rate(
                        csv_wavelet, scoring, use_nrem=False
                    )
                    rate_wavelet_opt = compute_spindle_rate(
                        csv_wavelet_optimal, scoring, use_nrem=False
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

# Load condition mapping
df["day"] = df["day"].astype(str)

df["condition_raw"] = np.nan

for i in range(len(df)):
    rat = df.loc[i, "rat"]
    day = df.loc[i, "day"]

    if rat in rat_conditions and day in rat_conditions[rat]:
        df.loc[i, "condition_raw"] = rat_conditions[rat][day]

df["condition"] = np.nan

df.loc[df["condition_raw"] == "HC", "condition"] = "HC"

df.loc[df["condition_raw"].str.match(r"GL\d+_S\d+", na=False), "condition"] = "Training"

# Long format conversion (include condition)
df_long = []

for i in range(len(df)):
    rat = df.loc[i, "rat"]
    cond = df.loc[i, "condition"]

    if pd.isna(cond):
        continue

    for method in ["threshold", "wavelet", "wavelet_optimal"]:
        df_long.append([rat, cond, method, df.loc[i, method]])


df_long = pd.DataFrame(
    df_long, columns=["rat", "condition", "method", "spindle_rate"]
).dropna()


# ------------------------------------------------------------
# -----------------------------------------------------------
# Plot: per rat

for r in df_long["rat"].unique():
    plt.figure(figsize=(8, 5))
    sub = df_long[df_long["rat"] == r]

    # Raw data
    sns.stripplot(
        data=sub,
        x="method",
        y="spindle_rate",
        hue="condition",
        dodge=True,
        alpha=0.35,
        size=5,
        jitter=0.15,
    )

    # Mean ± SEM
    sns.pointplot(
        data=sub,
        x="method",
        y="spindle_rate",
        hue="condition",
        dodge=0.4,
        errorbar="se",
        markers="D",
        linestyles="none",
        capsize=0.15,
    )

    plt.title(f"Spindle rate by method ({r})")

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# -----------------------------------------------------------
# Plot (per day average)

df_day = df.groupby(["rat", "day"]).mean(numeric_only=True).reset_index()

rats_unique = df_day["rat"].unique()

fig, axes = plt.subplots(
    len(rats_unique), 1, figsize=(10, 4 * len(rats_unique)), sharex=True
)

if len(rats_unique) == 1:
    axes = [axes]

for ax, rat in zip(axes, rats_unique):
    df_r = df_day[df_day["rat"] == rat]

    ax.plot(df_r["day"], df_r["threshold"], marker="o", label="envelop_threshold")
    ax.plot(df_r["day"], df_r["wavelet"], marker="o", label="wavelet")
    ax.plot(df_r["day"], df_r["wavelet_optimal"], marker="o", label="wavelet_optimal")

    ax.set_title(f"{rat}")
    ax.set_ylabel("spindles / min")

    ax.legend()

axes[-1].set_xlabel("Day")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


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
    dir_spindle, "envelop_thr_1_peak_3.0", f"{trial_id}_spindles_non_NREM.csv"
)
csv_wavelet = os.path.join(
    dir_spindle, "wavelet_amp_1_ampcore_3", f"{trial_id}_spindles_wavelet_non_NREM.csv"
)
csv_wavelet_optimal = os.path.join(
    dir_spindle,
    "wavelet_optimal_thrL_0.8_thrH_3.0",
    f"{trial_id}_spindles_wavelet_optimal_non_NREM.csv",
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
