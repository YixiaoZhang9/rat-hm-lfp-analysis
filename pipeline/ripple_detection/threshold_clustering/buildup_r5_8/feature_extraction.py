import os
import re

import EntropyHub as EH
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_array_stockwell
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
from scipy.ndimage import label
from skimage.measure import regionprops, shannon_entropy

from modules import ripple_features
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
dir_base1 = get_path("R1_8_root")
dir_R5_8_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/PreprocessedData"
)
dir_R5_8_Scoring = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/Scoring"
)

rats = np.arange(5, 9)
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
all_ripple_features = []
for rat in rats:  # rat in [rats[3]]
    rat_str = f"rat{rat}"
    print(f"\n===== Processing {rat_str} =====")
    threshold_perrat = []
    NREM_duration_across_time[rat_str] = {"presleep": [], "postsleep": []}
    Ripple_count_across_time[rat_str] = {"presleep": [], "postsleep": []}

    for region in [regions[0]]:  # e.g. 'HPC'
        dir_R5_8_Data_perday = os.path.join(dir_R5_8_Data, region, str(rat))
        folders_SD = [
            name
            for name in os.listdir(dir_R5_8_Data_perday)
            if os.path.isdir(os.path.join(dir_R5_8_Data_perday, name))
        ]

        for studyday in folders_SD:
            print(f"\n===== Processing studyday: {studyday} =====")
            threshold_perday = []
            for sleep_period in sleep_periods:
                dir_R5_8_Data_pertrial = os.path.join(
                    dir_R5_8_Data_perday, studyday, sleep_period
                )
                if not os.path.exists(dir_R5_8_Data_pertrial):
                    print(f"path : {dir_R5_8_Data_pertrial} does not exist")

                matched_files = []
                for root, _, files in os.walk(dir_R5_8_Data_pertrial):
                    for name in files:
                        if name.endswith(".mat"):
                            matched_files.append(os.path.join(root, name))
                if len(matched_files) == 0:
                    print(f"matched_files does not exist")

                # --- Load corresponding scoring ---
                path_scoring = os.path.join(
                    dir_R5_8_Scoring, str(rat), str(studyday), sleep_period
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

                # --- Compute NREM duration ---
                nrem_seconds = np.sum(all_scoring == 3)
                nrem_minutes = nrem_seconds / 60.0
                NREM_duration_across_time[rat_str][sleep_period].append(nrem_minutes)

                # --- Ripple detection across all trials ---
                ripple_count_per_period = 0

                for trial_name in matched_files:
                    trial_data = loadmat(trial_name)
                    data = trial_data["data"].squeeze()
                    filtered_data = filter_lfp(data, fs, [100, 250])

                    ripple_pertrial = []
                    thresholds_per_trial = []
                    ripple_envelop_pertrial = []

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
                    ripple_karlsson = []
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
                                        Results_Threshold["PeakIndex"] + start_sample
                                    ).astype(int),
                                    (
                                        Results_Threshold["EndIndex"] + start_sample
                                    ).astype(int),
                                )
                            )
                        )

                        ripple_count_per_period += len(ripple_karlsson)
                        ripple_pertrial.extend(ripple_karlsson)
                        ripple_envelop_pertrial.extend(
                            Results_Threshold["ripple_envelop"]
                        )
                        thresholds_per_trial.append(
                            Results_Threshold["thresh_envelope"]
                        )
                    threshold_perday.append(thresholds_per_trial)
                    # -------plot the signal and sleep scoring results-----
                    # if f_plot:
                    #     app_created = False
                    #     app = QApplication.instance()
                    #     if app is None:
                    #         app = QApplication(sys.argv)
                    #         app_created = True
                    #
                    #     upsampled_scoring_data = np.repeat(scoring_data, fs)
                    #     # create dic
                    #     data_dict = {
                    #         "signal": data,
                    #         "scoring": upsampled_scoring_data
                    #         # "filtered_signal":filtered_data
                    #     }
                    #
                    #     window = SignalPlotViewer(data_dict, fs, window_sec=5)
                    #     window.set_event_intervals(ripple_pertrial)
                    #     window.show()
                    #     if app_created:
                    #         app.exec()
                    #     print("end")

                    window_ms = 500  # 500ms on both left and right
                    half_window_samples = int(fs * window_ms / 1000)
                    # plot the detected ripples and stockwell time frequency analysis

                    ripple_features = []

                    for i, ripple in enumerate(ripple_pertrial):
                        start_idx, peak_idx, end_idx = ripple
                        ripple_data = data[start_idx:end_idx]
                        ripple_data_filt = filtered_data[start_idx:end_idx]

                        seg_start = max(0, peak_idx - half_window_samples)
                        seg_end = min(len(data), peak_idx + half_window_samples + 1)

                        seg_start = min(start_idx, seg_start)
                        seg_end = max(end_idx, seg_end)

                        ripple_segment = data[seg_start:seg_end]
                        ripple_segment_filt = filtered_data[seg_start:seg_end]
                        ripple_envelop = ripple_envelop_pertrial[i]
                        # ====================================== stockwell transform and related features=================================

                        ripple_segment_reshaped = ripple_segment.reshape(
                            1, 1, len(ripple_segment)
                        )
                        # ===  Stockwell time frequency transform ===
                        tfr_results = tfr_array_stockwell(
                            ripple_segment_reshaped, fs, fmin=100, fmax=250, width=1.0
                        )
                        power = np.abs(tfr_results[0]).squeeze()  # (n_freqs, n_times)
                        freqs = tfr_results[2]

                        ripple_time_indices = np.arange(seg_start, seg_end)
                        ripple_mask_time = (ripple_time_indices >= start_idx) & (
                            ripple_time_indices <= end_idx
                        )

                        # === Calculated the time point corresponding to the maximum value of the envelope max_t===
                        envelope = np.sqrt(np.sum(power, axis=0))
                        envelope_masked = envelope[ripple_mask_time]
                        max_t_local_idx = np.argmax(envelope_masked)
                        max_t_idx = np.where(ripple_mask_time)[0][max_t_local_idx]

                        # === Calculated the frequency value corresponding to the maximum power above 80 Hz at max_t (max_f ) ===
                        high_freq_mask = freqs >= 100
                        max_f_idx = np.argmax(power[high_freq_mask, max_t_idx])
                        max_f_idx = np.where(high_freq_mask)[0][max_f_idx]

                        # === Binarized the time-frequency distribution using 50% of st(max_t, max_f) ===
                        threshold = 0.5 * power[max_f_idx, max_t_idx]
                        binary_mask = (power >= threshold).astype(int)

                        # === Selected the region of interest (ROI). Among all the connected components in the binary image,
                        # the ROI corresponded to the region including the (max_t, max_f ) coordinates ===
                        binary_mask[:, ~ripple_mask_time] = 0

                        # === selected ROI ===
                        labeled_mask, num = label(binary_mask)
                        roi_label = labeled_mask[max_f_idx, max_t_idx]
                        roi_mask = labeled_mask == roi_label

                        # === calculate the features of ROI ===
                        props = regionprops(roi_mask.astype(int))
                        area = props[0].area

                        # Shannon entropy
                        entropy_val = shannon_entropy(roi_mask.astype(int))

                        # time width (TW) 和 frequency width (FW)
                        freq_indices, time_indices = np.where(roi_mask)
                        TW = (time_indices.max() - time_indices.min()) / fs
                        FW = freqs[freq_indices.max()] - freqs[freq_indices.min()]

                        # === plot ===
                        if f_plot:
                            t = (np.arange(seg_start, seg_end) - peak_idx) / fs
                            start_time = (start_idx - peak_idx) / fs
                            end_time = (end_idx - peak_idx) / fs

                            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

                            # original signal
                            axs[0].plot(t, ripple_segment.squeeze(), "k")
                            axs[0].plot(t, ripple_segment_filt.squeeze(), "b")
                            axs[0].axvline(0, color="r", linestyle="--", linewidth=2)
                            axs[0].axvline(
                                start_time, color="r", linestyle="--", linewidth=2
                            )
                            axs[0].axvline(
                                end_time, color="r", linestyle="--", linewidth=2
                            )
                            axs[0].set_ylabel("Amplitude")
                            axs[0].set_title(f"Ripple {i + 1} ±500 ms around peak")

                            # Stockwell tf  + ROI
                            im = axs[1].pcolormesh(
                                t,
                                freqs,
                                10 * np.log10(power + 1e-12),
                                shading="gouraud",
                                cmap="jet",
                            )
                            axs[1].axvline(0, color="r", linestyle="--", linewidth=2)
                            axs[1].set_ylim(80, 250)
                            axs[1].set_xlabel("Time (s)")
                            axs[1].set_ylabel("Frequency (Hz)")

                            # add ROI contour
                            axs[1].contour(
                                t,
                                freqs,
                                roi_mask,
                                levels=[0.5],
                                colors="white",
                                linewidths=1.5,
                            )

                            divider = make_axes_locatable(axs[1])
                            cax = divider.append_axes("right", size="3%", pad=0.05)
                            plt.colorbar(im, cax=cax, label="Power (dB)")

                            plt.tight_layout()
                            plt.show()

                        # ============================================ time domain features ===========================================
                        rel_start_ind = start_idx - seg_start
                        rel_end_ind = end_idx - seg_start

                        envelop_features = Ripple_features.ripple_envelop_features(
                            ripple_envelop,
                            ripple_segment,
                            ripple_segment_filt,
                            fs,
                            rel_start_ind,
                            rel_end_ind,
                            plot=1,
                        )

                        time_domain_features = (
                            Ripple_features.ripple_time_domain_features(
                                ripple_segment,
                                ripple_segment_filt,
                                fs,
                                rel_start_ind,
                                rel_end_ind,
                                plot=1,
                            )
                        )

                        # ============================================ other features ===========================================
                        # fuzzy entropy
                        Fuzz, ps1, ps2 = EH.FuzzEn(
                            ripple_data, m=2, tau=1, r=(0.2, 2.0), Fx="default"
                        )
                        Fuzz_value = Fuzz[1]

                        # # short time energy
                        # # calculated in filtered data
                        #
                        # short_time_energy = Ripple_features.short_time_energy(ripple_data_filt, fs, window_ms=15, window_type='hanning')
                        # short_time_energy_feature = np.mean(short_time_energy)
                        #
                        # # spectral centroid
                        # spectro_centroid = Ripple_features.spectral_centroid_short(ripple_data_filt, fs, window_type='hamming')

                        # ================ organize all the features ================
                        ripple_features.append(
                            {
                                "rat": rat,
                                "region": region,
                                "studyday": studyday,
                                "sleep_period": sleep_period,
                                "trial": os.path.basename(trial_name),
                                "ripple_index": i + 1,
                                "ripple_start": start_idx,
                                "ripple_peak": peak_idx,
                                "ripple_end": end_idx,
                                "area": area,
                                "entropy": entropy_val,
                                "TW": TW,
                                "FW": FW,
                                "max_f": freqs[max_f_idx],
                                "duration_ms": envelop_features["duration_ms"],
                                "peak_env": envelop_features["peak_env"],
                                "rms_env": envelop_features["rms_env"],
                                "relative_rise_time": envelop_features[
                                    "relative_rise_time"
                                ],
                                "relative_fall_time": envelop_features[
                                    "relative_fall_time"
                                ],
                                "env_skewness": envelop_features["env_skewness"],
                                "peak_energy_fraction": envelop_features[
                                    "peak_energy_fraction"
                                ],
                                "peak_to_trough": time_domain_features[
                                    "peak_to_trough"
                                ],
                                "num_peaks": time_domain_features["num_peaks"],
                                "zero_crossing_rate": time_domain_features[
                                    "zero_crossing_rate"
                                ],
                                "good_cycle_ratio": time_domain_features[
                                    "good_cycle_ratio"
                                ],
                                "Fuzz_value": Fuzz_value,
                            }
                        )

                    ripple_df = pd.DataFrame(ripple_features)
                    all_ripple_features.append(ripple_df)

total_df = pd.concat(all_ripple_features, ignore_index=True)
total_df.to_csv("All_Rat5-8_Ripple_Features_17features_test_0.5SD.csv", index=False)
print(f" Ripple features saved! Total {len(total_df)} ripples.")
