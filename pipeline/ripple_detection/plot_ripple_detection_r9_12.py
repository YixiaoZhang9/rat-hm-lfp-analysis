# ---- Import ----
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
from signal_viewer import RippleViewer

from modules.project_config import get_path

# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = get_path("R9_16_root")
dir_data_root = os.path.join(dir_base1, "Rat_HM_Ephys_TD_Analysis_R9_16/R9-12")
dir_R9_12_Data = os.path.join(dir_data_root, "PreprocessedData")
dir_R9_12_Scoring = os.path.join(dir_data_root, "Scoring")
dir_R9_12_Ripple = os.path.join(dir_data_root, "Ripple_detection_results")
dir_condition = os.path.join(
    dir_data_root, "Rat_HM_Ephys_TD_R9_12_Overview_StudyDay_Condition.xlsx"
)


rats = np.arange(9, 13)
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 1000  # downsampled sample frequency

# -----------------------------------------------------------------------------------
# ------------------------------plot the ripple rate per day per rat-----------------


# %%
# #Compute ripple rate (ripples / min in NREM)
# Compute ripple rate (ripples / min in NREM)
def compute_ripple_rate(csv_path, scoring):

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

    n_ripples = len(df)

    return n_ripples / nrem_minutes


# Load condition table
xls = pd.ExcelFile(dir_condition)
rat_conditions = {}

for sheet in xls.sheet_names:
    df_cond = pd.read_excel(dir_condition, sheet_name=sheet)
    df_cond.columns = df_cond.columns.str.strip()

    df_sub = df_cond[["SD", "Condition"]].dropna()
    df_sub["SD"] = df_sub["SD"].astype(str)

    rat_conditions[sheet] = dict(zip(df_sub["SD"], df_sub["Condition"]))

# main loop
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
                dir_ripple = os.path.join(
                    dir_R9_12_Ripple, str(rat), studyday, sleep_period
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
                        dir_ripple, f"{trial_id}_hippocampal_ripples_threshold.csv"
                    )
                    csv_threshold2 = os.path.join(
                        dir_ripple, f"{trial_id}_hippocampal_ripples_threshold2.csv"
                    )
                    csv_threshold3 = os.path.join(
                        dir_ripple, f"{trial_id}_hippocampal_ripples_threshold3.csv"
                    )
                    csv_cnn = os.path.join(
                        dir_ripple, f"{trial_id}_hippocampal_ripples_cnn.csv"
                    )
                    csv_ripplenet = os.path.join(
                        dir_ripple, f"{trial_id}_hippocampal_ripples_ripplenet.csv"
                    )

                    rate_th = compute_ripple_rate(csv_threshold, scoring)
                    rate_th2 = compute_ripple_rate(csv_threshold2, scoring)
                    rate_cnn = compute_ripple_rate(csv_cnn, scoring)
                    rate_rn = compute_ripple_rate(csv_ripplenet, scoring)

                    results.append(
                        {
                            "rat": rat_str,
                            "day": studyday,
                            "sleep": sleep_period,
                            "trial": trial_id,
                            "threshold": rate_th,
                            "threshold2": rate_th2,
                            "cnn": rate_cnn,
                            "ripplenet": rate_rn,
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

    df_long.append([rat, cond, df.loc[i, "threshold2"]])


df_long = pd.DataFrame(df_long, columns=["rat", "condition", "ripple_rate"]).dropna()

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# -------------------------------plot-------------------------------------

# Plot (per day average)
df_day = df.groupby(["rat", "day"]).mean(numeric_only=True).reset_index()
df_day["rat_num"] = df_day["rat"].str.extract(r"(\d+)").astype(int)

df_day = df_day.sort_values(by=["rat_num", "day"])

rats_unique = df_day["rat"].unique()

fig, axes = plt.subplots(
    len(rats_unique), 1, figsize=(10, 4 * len(rats_unique)), sharex=True
)

if len(rats_unique) == 1:
    axes = [axes]

for ax, rat in zip(axes, rats_unique):
    df_r = df_day[df_day["rat"] == rat]

    # ax.plot(df_r["day"], df_r["threshold"], marker='o', label="Threshold")
    ax.plot(df_r["day"], df_r["threshold2"], marker="o", label="Threshold2")
    # ax.plot(df_r["day"], df_r["cnn"], marker='o', label="CNN")
    # ax.plot(df_r["day"], df_r["ripplenet"], marker='o', label="RippleNet")

    ax.set_title(f"{rat}")
    ax.set_ylabel("Ripples / min")

    ax.legend()

axes[-1].set_xlabel("Day")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


for r in df_long["rat"].unique():
    plt.figure(figsize=(8, 5))
    sub = df_long[df_long["rat"] == r]

    sns.stripplot(
        data=sub,
        x="condition",
        y="ripple_rate",
        hue="condition",
        dodge=False,
        alpha=0.35,
        size=5,
        jitter=0.15,
    )

    sns.pointplot(
        data=sub,
        x="condition",
        y="ripple_rate",
        hue="condition",
        dodge=False,
        errorbar="se",
        markers="D",
        linestyles="none",
        capsize=0.15,
    )

    plt.title(f"Ripple rate by method ({r})")

    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 0:
        plt.legend(handles[:2], labels[:2], bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.show()


##----------------------------------------------------------------------
##--------------------------plot the signal-----------------------------
def load_events(csv_path):
    df = pd.read_csv(csv_path)
    return list(zip(df["ripple_start"].values, df["ripple_end"].values))


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
dir_ripple = os.path.join(dir_R9_12_Ripple, rat, studyday, sleep_period)
csv_threshold = os.path.join(
    dir_ripple, f"{trial_id}_hippocampal_ripples_threshold.csv"
)
csv_threshold2 = os.path.join(
    dir_ripple, f"{trial_id}_hippocampal_ripples_threshold2.csv"
)
csv_threshold3 = os.path.join(
    dir_ripple, f"{trial_id}_hippocampal_ripples_threshold3.csv"
)
csv_cnn = os.path.join(dir_ripple, f"{trial_id}_hippocampal_ripples_cnn.csv")
csv_ripplenet = os.path.join(
    dir_ripple, f"{trial_id}_hippocampal_ripples_ripplenet.csv"
)

threshold_events = load_events(csv_threshold) if os.path.exists(csv_threshold) else []
threshold2_events = (
    load_events(csv_threshold2) if os.path.exists(csv_threshold2) else []
)
threshold3_events = (
    load_events(csv_threshold3) if os.path.exists(csv_threshold3) else []
)
cnn_events = load_events(csv_cnn) if os.path.exists(csv_cnn) else []
ripplenet_events = load_events(csv_ripplenet) if os.path.exists(csv_ripplenet) else []


# ---- Launch viewer ----

# check if a QApplication already exists
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

event_sets = {
    "Threshold": threshold_events,
    "Threshold2": threshold2_events,
    "Threshold3": threshold3_events,
    # "CNN": cnn_events,
    # "RippleNet": ripplenet_events,
}

viewer = RippleViewer(
    lfp=loadmat(trial_path)["data"].squeeze(),
    scoring=scoring,
    fs=fs,
    event_sets=event_sets,
)
viewer.show()
sys.exit(app.exec_())
