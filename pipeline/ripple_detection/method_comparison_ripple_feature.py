import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.errors import EmptyDataError
from scipy.io import loadmat
from scipy.signal import convolve, hilbert
from scipy.signal.windows import gaussian


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def smooth_signal(signal, fs, sigma):
    """Smooth the signal using a Gaussian filter."""
    # Define the standard deviation(sigma)
    smoothing_sigma = sigma * fs
    # Create a Gaussian window with a standard deviation
    window_size = smoothing_sigma * 3 * 2
    # in a Gaussian distribution, about 99.7 % of the distribution's area lies within ±3σ.
    std = (window_size - 1) / (2 * sigma)
    gauss_filter = gaussian(window_size, std)
    # Normalize the Gaussian filter
    gauss_filter = gauss_filter / sum(gauss_filter)
    # Apply the filter using convolution
    smoothed_lfp = convolve(signal, gauss_filter, "same")
    return smoothed_lfp


def extract_features(df, fs, filtered_lfp):

    features = {}
    durations = []
    freqs = []
    # find peaks in each ripple interval
    instantaneous_amplitude = np.abs(hilbert(filtered_lfp))
    smoothed_envelope = smooth_signal(instantaneous_amplitude, fs, sigma=0.004)

    for _, row in df.iterrows():
        onset = int(row["ripple_start"])
        offset = int(row["ripple_end"])
        seg_env = smoothed_envelope[onset : offset + 1]
        peak_rel = np.argmax(seg_env)
        peak_idx = onset + peak_rel
        # duration
        durations.append(((offset - onset) / fs) * 1000)  # ms
        # wavelet frequency
        # f = compute_wavelet_frequency(filtered_lfp, onset, offset, peak_idx,fs)
        # freqs.append(f)

    features["duration"] = np.array(durations)
    # features["frequency"] = np.array(freqs)

    return features


def compute_wavelet_frequency(signal, onset, offset, peak_idx, fs):

    half_win = int(0.5 * fs)

    win_start = max(0, peak_idx - half_win)
    win_end = min(len(signal), peak_idx + half_win)

    seg = signal[win_start:win_end]

    freqs = np.linspace(100, 250, 80)

    power = mne.time_frequency.tfr_array_morlet(
        seg[np.newaxis, np.newaxis, :],
        sfreq=fs,
        freqs=freqs,
        n_cycles=7,
        output="power",
    )[0, 0]

    # normalize the power
    mean_p = np.mean(power, axis=1, keepdims=True)
    std_p = np.std(power, axis=1, keepdims=True)
    norm_power = (power - mean_p) / (std_p + 1e-10)

    rel_onset = onset - win_start
    rel_offset = offset - win_start
    ripple_spec = norm_power[:, rel_onset:rel_offset]

    if ripple_spec.size == 0:
        return np.nan

    avg_spec = np.mean(ripple_spec, axis=1)
    peak_freq = freqs[np.argmax(avg_spec)]

    return peak_freq


# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = "/media/yixiao/GL14_RAT_FA/"
dir_R1_4_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/PreprocessedData"
)
dir_R1_4_Scoring = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/Scoring"
)
dir_R1_4_Ripple = os.path.join(
    dir_base1,
    "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/Ripple_detection_results",
)
dir_R5_8_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/PreprocessedData"
)
dir_R5_8_Scoring = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/Scoring"
)
dir_R5_8_Ripple = os.path.join(
    dir_base1,
    "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/Ripple_detection_results",
)

rats = [1, 3, 7, 8]
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 1000  # downsampled sample frequency

# ---------------------------------------------------------------------------------------------------------
# ---------------------------- collect the detection results and ground truth-----------------------------
detection_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for rat in rats:
    rat_str = f"rat{rat}"

    for region in ["HPC"]:
        if rat in [1, 3]:
            dir_data = os.path.join(dir_R1_4_Data, region, str(rat))
        elif rat in [7, 8]:
            dir_data = os.path.join(dir_R5_8_Data, region, str(rat))

        if not os.path.exists(dir_data):
            continue

        for studyday in os.listdir(dir_data):
            for sleep_period in sleep_periods:
                dir_trial = os.path.join(dir_data, studyday, sleep_period)
                if rat in [1, 3]:
                    dir_ripple = os.path.join(
                        dir_R1_4_Ripple, str(rat), studyday, sleep_period
                    )
                elif rat in [7, 8]:
                    dir_ripple = os.path.join(
                        dir_R5_8_Ripple, str(rat), studyday, sleep_period
                    )

                # find all trial files
                matched_files = []
                for root, _, files in os.walk(dir_trial):
                    for name in files:
                        if name.endswith(".mat"):
                            matched_files.append(os.path.join(root, name))

                # process each trial
                for trial_name in matched_files:
                    trial_id = Path(trial_name).stem
                    trial_data = loadmat(trial_name)
                    pre_lfp = trial_data["data"].squeeze()
                    trial_dict = {
                        "methods": {},
                        "pre_signal": pre_lfp,
                    }
                    # --- CSV paths ---

                    csv_threshold = os.path.join(
                        dir_ripple, f"{trial_id}_hippocampal_ripples.csv"
                    )
                    csv_threshold2 = os.path.join(
                        dir_ripple, f"{trial_id}_hippocampal_ripples_threshold2.csv"
                    )
                    csv_cnn = os.path.join(
                        dir_ripple, f"{trial_id}_hippocampal_ripples_cnn.csv"
                    )
                    csv_ripplenet = os.path.join(
                        dir_ripple, f"{trial_id}_hippocampal_ripples_ripplenet.csv"
                    )

                    trial_dict["methods"]["threshold"] = safe_read_csv(csv_threshold)
                    trial_dict["methods"]["threshold2"] = safe_read_csv(csv_threshold2)
                    trial_dict["methods"]["cnn"] = safe_read_csv(csv_cnn)
                    trial_dict["methods"]["ripplenet"] = safe_read_csv(csv_ripplenet)

                    detection_results[rat][studyday][sleep_period][trial_id] = (
                        trial_dict
                    )


# %%
# ---------------------------------------------------------------------------------------------------------
# ------------------------feature distribution of ground truth and other detection method-------------------
feature_rows = []

for rat in rats:
    for studyday in detection_results[rat]:
        for sleep_period in detection_results[rat][studyday]:
            for trial_id in detection_results[rat][studyday][sleep_period]:
                pre_lfp = detection_results[rat][studyday][sleep_period][trial_id][
                    "pre_signal"
                ]

                # methods duration
                for method, pred_df in detection_results[rat][studyday][sleep_period][
                    trial_id
                ]["methods"].items():
                    feat = extract_features(pred_df, fs, pre_lfp)

                    for d in feat["duration"]:
                        feature_rows.append(
                            {
                                "rat": rat,
                                "studyday": studyday,
                                "sleep_period": sleep_period,
                                "method": method,
                                "feature": "duration",
                                "value": d,
                            }
                        )

                    # #methods frequency
                    # for f in feat["frequency"]:
                    #     feature_rows.append({
                    #         "rat": rat,
                    #         "studyday": studyday,
                    #         "sleep_period": sleep_period,
                    #         "method": method,
                    #         "feature": "frequency",
                    #         "value": f
                    #     })


# %%
df_feat = pd.DataFrame(feature_rows)

plt.figure(figsize=(10, 5))
sns.violinplot(
    data=df_feat[df_feat["feature"] == "duration"], x="method", y="value", cut=0
)
plt.ylabel("Duration (ms)")
plt.xlabel("")
plt.show()


plt.figure(figsize=(8, 5))
sns.ecdfplot(data=df_feat[df_feat["feature"] == "duration"], x="value", hue="method")
plt.xlabel("Duration (ms)")
plt.ylabel("Cumulative proportion")
plt.show()


# -----------------------frequency-------------------------
# plt.figure(figsize=(10, 5))
# sns.violinplot(
#     data=df_feat[df_feat["feature"] == "frequency"],
#     x="method",
#     y="value",
#     cut=0
# )
#
# plt.ylabel("Frequency (Hz)")
# plt.xlabel("")
# plt.show()
#
# plt.figure(figsize=(8, 5))
#
# sns.ecdfplot(
#     data=df_feat[df_feat["feature"] == "frequency"],
#     x="value",
#     hue="method"
# )
#
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Cumulative proportion")
# plt.show()


# ---------------------------------------------------------------------------------------------------------
# ---------------------------- evaluate the consensus of those methods and ground truth -------------------
