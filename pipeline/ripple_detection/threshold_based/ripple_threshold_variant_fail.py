# ---- Import ----
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.stats import zscore

from modules.ephys_signal_scoring_view import SignalPlotViewer
from modules.project_config import get_path
from modules.threshold_ripple_detection import (
    estimate_noise_mirror_threshold,
    filter_lfp,
    find_bouts,
    find_ripples_karlsson,
    find_ripples_karlsson_modified,
)

# def bandpass_filter(data, lowcut=0.5, highcut=30.0, fs=1000, order=4):

#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     y = filtfilt(b, a, data)
#     return y


# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = get_path("gl14_root")
dir_R1_4_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/PreprocessedData"
)
dir_R1_4_Scoring = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/Scoring"
)


rats = np.arange(1, 5)
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 1000  # downsampled sample frequency

f_plot = 0
Threshold_across_time = {}
NREM_duration_across_time = {}
Ripple_count_across_time = {}

# %%
# find the threshold per day
for rat in rats:  # rat in [rats[3]]
    rat_str = f"rat{rat}"
    print(f"\n===== find the threshold {rat_str} =====")
    threshold_perrat = []

    for region in [regions[0]]:  # e.g. 'HPC'
        dir_R1_4_Data_perday = os.path.join(dir_R1_4_Data, region, str(rat))
        folders_SD = [
            name
            for name in os.listdir(dir_R1_4_Data_perday)
            if os.path.isdir(os.path.join(dir_R1_4_Data_perday, name))
        ]

        for studyday in folders_SD:
            envelope_per_day = []
            for sleep_period in sleep_periods:
                dir_R1_4_Data_pertrial = os.path.join(
                    dir_R1_4_Data_perday, studyday, sleep_period
                )
                if not os.path.exists(dir_R1_4_Data_pertrial):
                    print(f"path : {dir_R1_4_Data_pertrial} does not exist")

                matched_files = []
                for root, _, files in os.walk(dir_R1_4_Data_pertrial):
                    for name in files:
                        if name.endswith(".mat"):
                            matched_files.append(os.path.join(root, name))
                if len(matched_files) == 0:
                    print(f"matched_files does not exist")

                # --- Load corresponding scoring ---
                path_scoring = os.path.join(
                    dir_R1_4_Scoring, str(rat), str(studyday), sleep_period
                )
                if not os.path.exists(path_scoring):
                    print(f"scoring path : {path_scoring} does not exist")

                scoring_files = [
                    f for f in os.listdir(path_scoring) if f.endswith(".mat")
                ]
                if len(scoring_files) == 0:
                    continue

                # --- Ripple detection across all trials ---

                for trial_name in matched_files:
                    trial_data = loadmat(trial_name)
                    data = trial_data["data"].squeeze()

                    # Find matching scoring (suffix match)
                    if len(scoring_files) == 1:
                        scoring_data = loadmat(
                            os.path.join(path_scoring, scoring_files[0])
                        )["states"].squeeze()
                    else:
                        match = re.search(r"_(\d+)\.mat$", trial_name)
                        if match:
                            suffix = match.group(1)
                        suffix = suffix.zfill(2)  # from '6' to '06'
                        pattern = re.compile(rf"_{suffix}_")  # '_06_'
                        for f in os.listdir(path_scoring):
                            if pattern.search(f):
                                trial_scoring_data = loadmat(
                                    os.path.join(path_scoring, f)
                                )
                                scoring_data = trial_scoring_data["states"].squeeze()
                                f_scoring = f
                                break

                    # --- Find NREM bouts ---
                    NREM_bouts = find_bouts(scoring_data, target_value=3, fs=1000)
                    for bout_idx, (start_sample, end_sample) in enumerate(NREM_bouts):
                        bout_data = data[start_sample:end_sample]
                        envelope_bout = find_ripples_karlsson(bout_data, fs, f_plot=0)

                        envelope_per_day.extend(envelope_bout["smoothed_envelope"])
            zscored_envelope = zscore(envelope_per_day)
            threshold_perday_z = estimate_noise_mirror_threshold(
                zscored_envelope, bins=500, percentile=0.9999, plot=0
            )

            threshold_perday = threshold_perday_z * np.std(envelope_per_day) + np.mean(
                envelope_per_day
            )
            threshold_perrat.append(threshold_perday)
    Threshold_across_time[rat_str] = threshold_perrat


# %%
# detect ripples
for rat in rats:  # rat in [rats[3]]
    rat_str = f"rat{rat}"
    print(f"\n===== Processing {rat_str} =====")
    threshold_perrat = []
    NREM_duration_across_time[rat_str] = {"presleep": [], "postsleep": []}
    Ripple_count_across_time[rat_str] = {"presleep": [], "postsleep": []}

    for region in [regions[0]]:  # e.g. 'HPC'
        dir_R1_4_Data_perday = os.path.join(dir_R1_4_Data, region, str(rat))
        folders_SD = [
            name
            for name in os.listdir(dir_R1_4_Data_perday)
            if os.path.isdir(os.path.join(dir_R1_4_Data_perday, name))
        ]

        for studyday in folders_SD:
            threshold_perday = []
            for sleep_period in sleep_periods:
                dir_R1_4_Data_pertrial = os.path.join(
                    dir_R1_4_Data_perday, studyday, sleep_period
                )
                if not os.path.exists(dir_R1_4_Data_pertrial):
                    print(f"path : {dir_R1_4_Data_pertrial} does not exist")

                matched_files = []
                for root, _, files in os.walk(dir_R1_4_Data_pertrial):
                    for name in files:
                        if name.endswith(".mat"):
                            matched_files.append(os.path.join(root, name))
                if len(matched_files) == 0:
                    print(f"matched_files does not exist")

                # --- Load corresponding scoring ---
                path_scoring = os.path.join(
                    dir_R1_4_Scoring, str(rat), str(studyday), sleep_period
                )
                if not os.path.exists(path_scoring):
                    print(f"scoring path : {path_scoring} does not exist")

                scoring_files = [
                    f for f in os.listdir(path_scoring) if f.endswith(".mat")
                ]
                if len(scoring_files) == 0:
                    continue

                all_scoring = []
                for f in scoring_files:
                    scoring_data = loadmat(os.path.join(path_scoring, f))[
                        "states"
                    ].squeeze()
                    all_scoring.append(scoring_data)
                all_scoring = np.concatenate(all_scoring)

                # --- Compute NREM duration ---
                nrem_seconds = np.sum(all_scoring == 3)
                nrem_minutes = nrem_seconds / 60.0
                NREM_duration_across_time[rat_str][sleep_period].append(nrem_minutes)

                # --- Ripple detection across all trials ---
                ripple_count_per_period = 0

                for trial_name in matched_files:
                    trial_data = loadmat(trial_name)
                    data = trial_data["data"].squeeze()
                    filtered_data = filter_lfp(data, fs, [80, 250])
                    instantaneous_amplitude = np.abs(hilbert(filtered_data))
                    # -------- plot the distribution of filtered data--------
                    # plt.figure(figsize=(6, 4))
                    # sns.histplot(instantaneous_amplitude, bins=150, kde=True)
                    # plt.title("Distribution of Filtered Data (80–250 Hz)")
                    # plt.xlabel("Amplitude")
                    # plt.ylabel("Count")
                    # plt.show()
                    # --------------------

                    ripple_pertrial = []
                    thresholds_per_trial = []

                    # Find matching scoring (suffix match)
                    if len(scoring_files) == 1:
                        scoring_data = loadmat(
                            os.path.join(path_scoring, scoring_files[0])
                        )["states"].squeeze()
                    else:
                        match = re.search(r"_(\d+)\.mat$", trial_name)
                        if match:
                            suffix = match.group(1)
                        suffix = suffix.zfill(2)  # from '6' to '06'
                        pattern = re.compile(rf"_{suffix}_")  # '_06_'
                        for f in os.listdir(path_scoring):
                            if pattern.search(f):
                                trial_scoring_data = loadmat(
                                    os.path.join(path_scoring, f)
                                )
                                scoring_data = trial_scoring_data["states"].squeeze()
                                f_scoring = f
                                break

                    # --- Find NREM bouts ---
                    NREM_bouts = find_bouts(scoring_data, target_value=3, fs=1000)
                    for bout_idx, (start_sample, end_sample) in enumerate(NREM_bouts):
                        bout_data = data[start_sample:end_sample]
                        Results_Threshold = find_ripples_karlsson_modified(
                            bout_data, fs, f_plot=0
                        )

                        ripple_karlsson = np.array(
                            list(
                                zip(
                                    (
                                        Results_Threshold["StartIndex"] * fs
                                        + start_sample
                                    ).astype(int),
                                    (
                                        Results_Threshold["EndIndex"] * fs
                                        + start_sample
                                    ).astype(int),
                                )
                            )
                        )

                        ripple_count_per_period += len(ripple_karlsson)
                        ripple_pertrial.extend(ripple_karlsson)
                        thresholds_per_trial.append(
                            Results_Threshold["thresh_envelope"]
                        )
                    threshold_perday.append(thresholds_per_trial)
                    # -------plot the signal and sleep scoring results-----
                    if f_plot:
                        app_created = False
                        app = QApplication.instance()
                        if app is None:
                            app = QApplication(sys.argv)
                            app_created = True

                        upsampled_scoring_data = np.repeat(scoring_data, fs)
                        # create dic
                        data_dict = {
                            "signal": data,
                            "scoring": upsampled_scoring_data,
                            # "filtered_signal":filtered_data
                        }

                        window = SignalPlotViewer(data_dict, fs, window_sec=5)
                        window.set_event_intervals(ripple_pertrial)
                        window.show()
                        if app_created:
                            app.exec()
                        print("end")

                Ripple_count_across_time[rat_str][sleep_period].append(
                    ripple_count_per_period
                )

            threshold_perrat.append(threshold_perday)
    Threshold_across_time[rat_str] = threshold_perrat


# %%
## plot the threshold across time
for rat, rat_data in Threshold_across_time.items():
    mean_thresh_perday = rat_data

    days = range(1, len(mean_thresh_perday) + 1)
    plt.plot(days, mean_thresh_perday, marker="o", label=rat)


plt.xlabel("Study Day", fontsize=22)
plt.ylabel("Mean Ripple Threshold ± Std", fontsize=22)
plt.title("Ripple Threshold Across Days (Rat 1-4)", fontsize=26)
plt.legend(fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.show()


# %%
## --- Plot results (stacked bars per rat) ---

for rat in rats:
    rat_str = f"rat{rat}"

    presleep_vals_nrem = NREM_duration_across_time[rat_str]["presleep"]
    postsleep_vals_nrem = NREM_duration_across_time[rat_str]["postsleep"]

    presleep_vals_ripples = Ripple_count_across_time[rat_str]["presleep"]
    postsleep_vals_ripples = Ripple_count_across_time[rat_str]["postsleep"]

    num_days = max(len(presleep_vals_nrem), len(postsleep_vals_nrem))
    days = np.arange(1, num_days + 1)

    # ---------- Plot 1: NREM duration ----------
    plt.figure(figsize=(8, 6))
    plt.bar(days, presleep_vals_nrem, label="Presleep", color="skyblue")
    plt.bar(
        days,
        postsleep_vals_nrem,
        bottom=presleep_vals_nrem,
        label="Postsleep",
        color="lightcoral",
        alpha=0.8,
    )

    plt.xlabel("Study Day", fontsize=18)
    plt.ylabel("NREM Duration (min)", fontsize=18)
    plt.title(f"NREM Duration per Day - {rat_str}", fontsize=20)
    plt.legend(fontsize=14)
    plt.xticks(days, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ---------- Plot 2: Ripple count ----------
    plt.figure(figsize=(8, 6))
    plt.bar(days, presleep_vals_ripples, label="Presleep", color="skyblue")
    plt.bar(
        days,
        postsleep_vals_ripples,
        bottom=presleep_vals_ripples,
        label="Postsleep",
        color="lightcoral",
        alpha=0.8,
    )

    plt.xlabel("Study Day", fontsize=18)
    plt.ylabel("Ripple Count", fontsize=18)
    plt.title(f"Ripple Count per Day - {rat_str}", fontsize=20)
    plt.legend(fontsize=14)
    plt.xticks(days, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
