# ---- Import ----
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication
from scipy.io import loadmat
from scipy.signal import hilbert

from modules.ephys_signal_scoring_view import SignalPlotViewer
from modules.project_config import get_path
from modules.threshold_ripple_detection import (
    filter_lfp,
    find_bouts,
    find_ripples_karlsson,
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
Ripple_count_across_time_m2 = {}


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

                all_scoring = []
                for f in scoring_files:
                    scoring_data = loadmat(os.path.join(path_scoring, f))[
                        "states"
                    ].squeeze()
                    all_scoring.append(scoring_data)
                all_scoring = np.concatenate(all_scoring)

                # ------- Compute NREM duration -------
                nrem_seconds = np.sum(all_scoring == 3)
                nrem_minutes = nrem_seconds / 60.0
                NREM_duration_across_time[rat_str][sleep_period].append(nrem_minutes)

                # --- Ripple detection across all trials ---
                ripple_count_per_period = 0

                for trial_name in matched_files:
                    trial_data = loadmat(trial_name)
                    data = trial_data["data"].squeeze()
                    filtered_data = filter_lfp(data, fs, [100, 250])
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
                        Results_Threshold = find_ripples_karlsson(
                            bout_data, fs, f_plot=0
                        )

                        ripple_karlsson = np.array(
                            list(
                                zip(
                                    (
                                        Results_Threshold["StartIndex"] + start_sample
                                    ).astype(int),
                                    (
                                        Results_Threshold["EndIndex"] + start_sample
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
                            "filtered_signal": filtered_data,
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
mean_thresh_perrat = {}
mean_thresh_perday = {}
std_thresh_perday = {}
for rat, rat_data in Threshold_across_time.items():
    mean_thresh_perday[rat] = []
    std_thresh_perday[rat] = []

    for day_data in rat_data:  # day_data = [presleep_thresh, postsleep_thresh]
        all_bout_means = []

        for trial_data in day_data:  # trial_data = list of bouts
            bout_means = [np.mean(bout) for bout in trial_data]
            all_bout_means.extend(bout_means)

        mean_day = np.mean(all_bout_means)
        std_day = np.std(all_bout_means)

        mean_thresh_perday[rat].append(mean_day)
        std_thresh_perday[rat].append(std_day)

    mean_thresh_perrat[rat] = np.mean(mean_thresh_perday[rat])

    days = range(1, len(mean_thresh_perday[rat]) + 1)
    plt.plot(days, mean_thresh_perday[rat], marker="o", label=rat)
    plt.fill_between(
        days,
        np.array(mean_thresh_perday[rat]) - np.array(std_thresh_perday[rat]),
        np.array(mean_thresh_perday[rat]) + np.array(std_thresh_perday[rat]),
        alpha=0.2,
    )

plt.xlabel("Study Day", fontsize=22)
plt.ylabel("Mean Ripple Threshold ± Std", fontsize=22)
plt.title("Ripple Threshold Across Days (Rat 1-4)", fontsize=26)
plt.legend(fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.show()

# %%
## detect ripples based on the average threshold per day
for rat in rats:  # rat in [rats[3]]
    rat_str = f"rat{rat}"
    print(f"\n===== Processing {rat_str} =====")
    threshold_perrat = []

    Ripple_count_across_time_m2[rat_str] = {"presleep": [], "postsleep": []}

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

                # --- Ripple detection across all trials ---
                ripple_count_per_period_m2 = 0

                for trial_name in matched_files:
                    trial_data = loadmat(trial_name)
                    data = trial_data["data"].squeeze()

                    ripple_pertrial_m2 = []

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
                        threhold_day = mean_thresh_perrat[rat_str]
                        Results_Threshold_m2 = find_ripples_karlsson(
                            bout_data, fs, f_plot=0, threshold=threhold_day
                        )

                        ripple_karlsson = np.array(
                            list(
                                zip(
                                    (
                                        Results_Threshold_m2["StartIndex"] * fs
                                        + start_sample
                                    ).astype(int),
                                    (
                                        Results_Threshold_m2["EndIndex"] * fs
                                        + start_sample
                                    ).astype(int),
                                )
                            )
                        )

                        ripple_count_per_period_m2 += len(ripple_karlsson)
                        ripple_pertrial_m2.extend(ripple_karlsson)
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
                        window.set_event_intervals(ripple_pertrial_m2)
                        window.show()
                        if app_created:
                            app.exec()
                        print("end")

                Ripple_count_across_time_m2[rat_str][sleep_period].append(
                    ripple_count_per_period_m2
                )


# %%
## --- Plot results (stacked bars per rat) ---
for rat in rats:
    rat_str = f"rat{rat}"

    presleep_vals_nrem = NREM_duration_across_time[rat_str]["presleep"]
    postsleep_vals_nrem = NREM_duration_across_time[rat_str]["postsleep"]

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

    plt.xlabel("Study Day", fontsize=22)
    plt.ylabel("NREM Duration (min)", fontsize=22)
    plt.title(f"NREM Duration per Day - {rat_str}", fontsize=24)
    plt.legend(fontsize=20)
    plt.xticks(days, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ---------- Plot 2: RIpple count ----------
    ripple_count_pre = Ripple_count_across_time[rat_str]["presleep"]
    ripple_count_post = Ripple_count_across_time[rat_str]["postsleep"]

    plt.figure(figsize=(8, 6))
    plt.bar(days, ripple_count_pre, label="Presleep", color="skyblue")
    plt.bar(
        days,
        ripple_count_post,
        bottom=ripple_count_pre,
        label="Postsleep",
        color="lightcoral",
        alpha=0.8,
    )

    plt.xlabel("Study Day", fontsize=22)
    plt.ylabel("Ripple count", fontsize=22)
    plt.title(f"Ripple count per Day - {rat_str}", fontsize=24)
    plt.legend(fontsize=20)
    plt.xticks(days, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ---------- Plot 3: Ripple rate ----------
    Total_Ripple_count_across_time = (
        np.array(Ripple_count_across_time[rat_str]["presleep"])
    ) + np.array(Ripple_count_across_time[rat_str]["postsleep"])
    Total_nrem = np.array(presleep_vals_nrem) + np.array(postsleep_vals_nrem)

    ripple_rate = Total_Ripple_count_across_time / Total_nrem

    plt.figure(figsize=(8, 6))
    plt.plot(days, ripple_rate, marker="o", linewidth=3)

    plt.ylim(0, 40)
    plt.xlabel("Study Day", fontsize=22)
    plt.ylabel("Ripple rate (events/min)", fontsize=22)
    plt.title(f"Ripple rate per Day - {rat_str}", fontsize=24)
    plt.xticks(days, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


## --- compare the ripple count in different threshold ---
# %%
ripple_count_perday = {}
ripple_count_perday_m2 = {}

for rat in rats:
    rat_str = f"rat{rat}"

    ripple_count_perday[rat_str] = [
        a + b
        for a, b in zip(
            Ripple_count_across_time[rat_str]["presleep"],
            Ripple_count_across_time[rat_str]["postsleep"],
        )
    ]

    ripple_count_perday_m2[rat_str] = [
        a + b
        for a, b in zip(
            Ripple_count_across_time_m2[rat_str]["presleep"],
            Ripple_count_across_time_m2[rat_str]["postsleep"],
        )
    ]

    thresh_day = np.ravel(mean_thresh_perday[rat_str])
    thresh_day_m2 = np.repeat(mean_thresh_perrat[rat_str], len(thresh_day))

    ripple_day = ripple_count_perday[rat_str]
    ripple_day_m2 = ripple_count_perday_m2[rat_str]

    days = np.arange(1, len(mean_thresh_perday[rat_str]) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # -------- left：threshold --------
    ax1.plot(days, thresh_day, "o-", color="blue", label="Threshold (Method 1)")
    ax1.plot(
        days, thresh_day_m2, "s--", color="green", label="Threshold (Method 2 avg)"
    )
    ax1.set_ylabel("Threshold", fontsize=20, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue", labelsize=20)

    # -------- right：ripple count --------
    ax2 = ax1.twinx()
    width = 0.35
    ax2.bar(
        days - width / 2,
        ripple_day,
        width=width,
        color="blue",
        alpha=0.7,
        label="Ripple Count (Method 1)",
    )
    ax2.bar(
        days + width / 2,
        ripple_day_m2,
        width=width,
        color="green",
        alpha=0.7,
        label="Ripple Count (Method 2)",
    )
    ax2.set_ylabel("Ripple Count", fontsize=20, color="red")
    ax2.tick_params(axis="y", labelcolor="red", labelsize=20)

    ax1.set_xlabel("Study Day", fontsize=20)
    ax1.set_title(f"Ripple Count & Threshold per Day - {rat_str}", fontsize=24)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=20, loc="upper left")

    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
