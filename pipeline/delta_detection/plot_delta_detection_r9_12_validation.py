from modules.project_config import get_path

"""
This script assesses the reliability of delta detection using two validation criteria:

1) Specificity check: quantify delta events detected during non-NREM sleep,
   under the assumption that true delta primarily occur during NREM sleep.

2) Condition comparison: compare delta rates across behavioral conditions,
   where higher delta activity is expected during training compared to home sessions.
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
from signal_viewer import DeltaViewer

from modules.threshold_ripple_detection import filter_lfp

# Paths
dir_base1 = get_path("R9_16_root")

dir_data_root = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12"
)
dir_R9_12_Data = os.path.join(dir_data_root, "PreprocessedData")
dir_R9_12_Scoring = os.path.join(dir_data_root, "Scoring")
dir_R9_12_Delta = os.path.join(dir_data_root, "Delta_detection_results")
dir_condition = os.path.join(
    dir_data_root, "Rat_HM_Ephys_TD_R9_12_Overview_StudyDay_Condition.xlsx"
)

rats = np.arange(9, 13)
sleep_periods = ["presleep", "postsleep"]
regions = ["HPC", "PL", "RSC"]
fs = 1000


# Delta rate computation
def compute_delta_rate(csv_path, scoring, use_nrem=True):

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

    n_deltas = len(df)

    return n_deltas / valid_minutes


# Load condition table
xls = pd.ExcelFile(dir_condition)
rat_conditions = {}

for sheet in xls.sheet_names:
    df_cond = pd.read_excel(dir_condition, sheet_name=sheet)
    df_cond.columns = df_cond.columns.str.strip()

    df_sub = df_cond[["SD", "Condition"]].dropna()
    df_sub["SD"] = df_sub["SD"].astype(str)

    rat_conditions[sheet] = dict(zip(df_sub["SD"], df_sub["Condition"]))


# Main extraction loop
results = []
duration_results = []

for rat in rats:
    rat_str = f"rat{rat}"

    for region in ["PL"]:
        dir_rat = os.path.join(dir_R9_12_Data, region, str(rat))

        if not os.path.exists(dir_rat):
            continue

        for studyday in sorted(os.listdir(dir_rat)):
            for sleep_period in sleep_periods:
                dir_trial = os.path.join(dir_rat, studyday, sleep_period)
                dir_delta = os.path.join(
                    dir_R9_12_Delta, region, str(rat), studyday, sleep_period
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

                    # -----------------------------
                    # Load scoring
                    # -----------------------------
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
                    csv_threshold = os.path.join(dir_delta, f"{trial_id}_deltas.csv")
                    rate_th = compute_delta_rate(csv_threshold, scoring)

                    # calculate the duration of deltas
                    if os.path.exists(csv_threshold):
                        try:
                            df_evt = pd.read_csv(csv_threshold)

                            if (
                                "delta_start" in df_evt.columns
                                and "delta_end" in df_evt.columns
                            ):
                                # duration in ms (because fs=1000)
                                durations = (
                                    df_evt["delta_end"] - df_evt["delta_start"]
                                ) / fs  # s

                                for d in durations:
                                    duration_results.append(
                                        {
                                            "rat": rat_str,
                                            "day": studyday,
                                            "sleep": sleep_period,
                                            "trial": trial_id,
                                            "duration_s": d,
                                        }
                                    )

                        except Exception as e:
                            print(f"Failed reading {csv_threshold}: {e}")

                    results.append(
                        {
                            "rat": rat_str,
                            "day": studyday,
                            "sleep": sleep_period,
                            "trial": trial_id,
                            "threshold": rate_th,
                        }
                    )

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
df_dur = pd.DataFrame(duration_results).dropna()


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

    df_long.append([rat, cond, df.loc[i, "threshold"]])


df_long = pd.DataFrame(df_long, columns=["rat", "condition", "delta_rate"]).dropna()


# ------------------------------------------------------------
# -----------------------------------------------------------
# Plot: condition comparison
rats = df_long["rat"].unique()

fig, axes = plt.subplots(len(rats), 1, figsize=(8, 5 * len(rats)), sharex=True)

# if only one rat
if len(rats) == 1:
    axes = [axes]

for ax, r in zip(axes, rats):
    sub = df_long[df_long["rat"] == r]

    # Raw data
    sns.stripplot(
        data=sub,
        x="condition",
        y="delta_rate",
        hue="condition",
        dodge=False,
        alpha=0.35,
        size=5,
        jitter=0.15,
        ax=ax,
    )

    # Mean ± SEM
    sns.pointplot(
        data=sub,
        x="condition",
        y="delta_rate",
        hue="condition",
        dodge=False,
        errorbar="se",
        markers="D",
        linestyles="none",
        capsize=0.15,
        ax=ax,
    )

    ax.set_title(f"Delta rate by condition ({r})")

    ax.set_ylabel("Delta rate")

    # remove duplicated legends
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles[:2], labels[:2], bbox_to_anchor=(1.02, 1), loc="upper left")

axes[-1].set_xlabel("Condition")

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

    ax.plot(df_r["day"], df_r["threshold"], marker="o", label="threshold")

    ax.set_title(f"{rat}")
    ax.set_ylabel("deltas / min")

    ax.legend()

axes[-1].set_xlabel("Day")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

##-------------------------plot the distribution of duration-----------
plt.figure(figsize=(10, 6))

sns.violinplot(data=df_dur, x="rat", y="duration_s", inner="box", cut=0)

plt.title("Delta duration distribution per rat")
plt.ylabel("Duration (s)")
plt.tight_layout()
plt.show()


df_dur["day"] = df_dur["day"].astype(str)
df["day"] = df["day"].astype(str)

df_cond = df[["rat", "day", "condition"]].dropna()

df_dur = df_dur.merge(df_cond, on=["rat", "day"], how="left")
df_dur = df_dur.dropna(subset=["condition"])

plt.figure(figsize=(12, 6))

sns.violinplot(
    data=df_dur,
    x="rat",
    y="duration_s",
    hue="condition",
    split=True,
    inner="quartile",
    cut=0,
)

plt.title("Delta duration by condition per rat")
plt.ylabel("Duration (s)")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


##----------------------------------------------------------------------
##--------------------------plot the signal-----------------------------
def load_events(csv_path):
    df = pd.read_csv(csv_path)
    return list(zip(df["delta_start"].values, df["delta_end"].values))


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
dir_delta = os.path.join(dir_R9_12_Delta, region, rat, studyday, sleep_period)
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
