import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros_like

from demo_application_functions_m import *
from pathlib import Path
from torchvision import models
from detection_evaluation import *
import math
from scipy.signal.windows import gaussian
from scipy.ndimage import label
from scipy.signal import iirnotch,butter, filtfilt, hilbert,convolve, get_window


def smooth_signal(signal, fs, sigma):
    """Smooth the signal using a Gaussian filter."""
    # Define the standard deviation(sigma)
    smoothing_sigma = sigma * fs
    # Create a Gaussian window with a standard deviation
    window_size = smoothing_sigma * 3 * 2
    # in a Gaussian distribution, about 99.7 % of the distribution's area lies within ±3σ.
    std = (window_size-1)/(2*sigma)
    gauss_filter = gaussian(window_size, std)
    # Normalize the Gaussian filter
    gauss_filter = gauss_filter / sum(gauss_filter)
    # Apply the filter using convolution
    smoothed_lfp = convolve(signal, gauss_filter, 'same')
    return smoothed_lfp

def filter_lfp(lfp, fs,freq_range):
    # Bandpass filter
    b, a = butter(4, np.array(freq_range)/(fs/2), btype='band')
    return filtfilt(b, a, lfp)

#%%
# simulated signal
project_root = Path.cwd().resolve().parents[1]
sim_root = project_root/"data"/"simulation"
out_dir_root =  sim_root/"spectral_image"/"main"/"evaluation"


snr_list = [-10, -5, 0, 5]
noise_types = {
    "baseline": {'brown': 0.2, 'pink': 0.7, 'white': 0.1, 'powerline': 0, 'emg': 0},
    "baseline_powerline": {'brown': 0.15,'pink':0.55,'white': 0.1,'powerline': 0.2,'emg': 0},
    "baseline_powerline_emg": {'brown': 0.1,'pink':0.5,'white': 0.1,'powerline': 0.2,'emg': 0.1}}

test_filr_records = []

for snr in snr_list:
    snr_dir = sim_root / f"SNR{snr}"
    if not snr_dir.is_dir():
        print(f"[skip] missing SNR folder: {snr_dir}")
        continue

    for noise_name in noise_types.keys():
        noise_dir = snr_dir / noise_name
        if not noise_dir.is_dir():
            print(f"[skip] missing noise folder: {noise_dir}")
            continue

        npz_files = sorted(noise_dir.glob("trial_*.npz"))
        if len(npz_files) == 0:
            print(f"[warn] no npz files under: {noise_dir}")
            continue

        train_count = math.floor(len(npz_files) * 0.8)
        val_count = max(1, math.floor(len(npz_files) * 0.1))
        test_count = max(1, math.floor(len(npz_files) * 0.1))

        test_files = npz_files[train_count + val_count:train_count + val_count + test_count]

        for file in test_files:
            test_filr_records.append({
                "snr": snr,
                "noise_name": noise_name,
                "file": file,
                "out_dir": out_dir_root / f"SNR{snr}" / noise_name,
            })

print(f"collect {len(test_filr_records)} npz files in test dataset.")

# %%
# prepare the spectral images for testing
# window_ms = 200  # time window of spectra image
window_ms = 400  # time window of spectra image
overlap = 0.5  # overlap of time window when computing spectra image
f_min = 100  # Hz, low cut-off frequency in spectra image
f_max = 250  # Hz, high cut-off frequency in spectra image

model_path = Path.cwd().resolve() / "retraining_saved_model.pth"

# store results for each condition
condition_stats = {}
file_level_results = []

for record in test_filr_records:
    snr = record["snr"]
    noise_name = record["noise_name"]
    file = record["file"]
    out_image_path = record["out_dir"]

    condition_key = (snr, noise_name)

    if condition_key not in condition_stats:
        condition_stats[condition_key] = {"TP": 0,"FP": 0,"FN": 0,"n_files": 0}

    print(f"\nProcessing file: {file}")
    print(f"Condition: SNR={snr}, noise={noise_name}")

    lfp, true_ripple, ripple_peaks, fs, ripple_frequency = load_npz_trial(file)
    # gt is the index of [start,end] of ripple; peaks is the index ripple peak

    # make spectra images
    time = np.arange(len(lfp)) / fs

    os.makedirs(out_image_path, exist_ok=True)

    start_time_dict, stop_time_dict = make_spectra_image_files(lfp,time,out_dir=out_image_path,fs = fs, segment_ms=window_ms,overlap=overlap,
                                                               f_min=f_min,f_max=f_max,width=1.0,decim=1,log_power=True,smooth_sigma=1)

    test_df = compute_CNN(image_path=out_image_path, start_time_dict=start_time_dict, stop_time_dict=stop_time_dict,
                          model_path=model_path)

    # %%
    # find index of the image with detected ripples
    indices = np.where(test_df["prediction"].to_numpy() == 1)[0]

    # collect [start, stop] intervals
    intervals = []
    for i in indices:
        start = float(test_df.iloc[i]["start time"])
        stop = float(test_df.iloc[i]["stop time"])
        intervals.append([start, stop])

    # sort by start time
    intervals.sort(key=lambda x: x[0])

    # merge overlapping intervals
    merged_intervals = []
    for interval in intervals:
        if not merged_intervals:
            merged_intervals.append(interval)
        else:
            last_start, last_stop = merged_intervals[-1]
            curr_start, curr_stop = interval

            if curr_start <= last_stop:
                merged_intervals[-1][1] = max(last_stop, curr_stop)
            else:
                merged_intervals.append(interval)

    # compute ripple envelope
    filtered_lfp = filter_lfp(lfp, fs, [100, 250])
    instantaneous_amplitude = np.abs(hilbert(filtered_lfp))
    smoothed_envelope = smooth_signal(instantaneous_amplitude, fs, sigma=0.004)
    zscored_envelope = zscore(smoothed_envelope)

    # refinement of the ripples
    high_th = 3.0
    low_th = 0
    pad = int(0.05 * fs)  # 50 ms padding
    min_dur = 25  # ms
    max_dur = 200  # ms
    merge_gap = int(round(0.01 * fs)) # 10ms

    detected_events = []
    Y_cont_pred_binary = np.zeros(len(lfp), dtype=int)

    for interval in merged_intervals:
        start_t, stop_t = interval

        start_idx = max(0, int(round(start_t * fs)) - pad)
        stop_idx = min(len(zscored_envelope) - 1,
                       int(round(stop_t * fs)) + pad)

        if stop_idx <= start_idx:
            continue

        segment = zscored_envelope[start_idx:stop_idx + 1]

        # find the ripple candidate
        mask = segment > high_th
        labeled, n_events = label(mask)

        if n_events == 0:
            continue

        # refine each ripple candidate
        candidates = []
        for i in range(1, n_events + 1):
            indexs = np.where(labeled == i)[0]

            if len(indexs) == 0:
                continue

            peak_rel = indexs[np.argmax(segment[indexs])]
            peak_idx = start_idx + peak_rel

            onset = peak_idx
            while onset > start_idx and zscored_envelope[onset] > low_th:
                onset -= 1

            offset = peak_idx
            while offset < stop_idx and zscored_envelope[offset] > low_th:
                offset += 1

            candidates.append({
                "onset": onset,
                "offset": offset,
                "peak": peak_idx
            })

        if len(candidates) == 0:
            continue

        # merge close events
        candidates = sorted(candidates, key=lambda x: x["onset"])
        merged = [candidates[0]]

        for curr in candidates[1:]:
            prev = merged[-1]

            gap = curr["onset"] - prev["offset"]

            if gap <= merge_gap:
                new_event = {
                    "onset": prev["onset"],
                    "offset": max(prev["offset"], curr["offset"]),
                    "peak": prev["peak"] if zscored_envelope[prev["peak"]] >= zscored_envelope[curr["peak"]] else curr[
                        "peak"]
                }
                merged[-1] = new_event
            else:
                merged.append(curr)

        # duration filtering
        for event in merged:
            onset = event["onset"]
            offset = event["offset"]
            peak_idx = event["peak"]

            duration_ms = (offset - onset) / fs * 1000.0

            if duration_ms < min_dur or duration_ms > max_dur:
                continue

            Y_cont_pred_binary[onset:offset + 1] = 1

            detected_events.append({
                "ripple_start": onset,
                "ripple_peak": peak_idx,
                "ripple_end": offset,
            })

    detected_events_df = pd.DataFrame(detected_events)


    # event-based evaluation
    # TPs, FPs, FNs, stats = get_tps_fps_fns_event_based(zscored_envelope.flatten(), true_ripple[:, 0], true_ripple[:, 1],
    #                                                    detected_events_df["ripple_peak"].to_numpy())


    # time-interval based evaluation
    TPs, FPs, FNs, stats = get_tps_fps_fns_time_interval_based(zscored_envelope.flatten(),Y_cont_pred_binary,true_ripple[:, 0],
                                                              true_ripple[:, 1],ripple_peaks,iou_threshold=0.5,peak_tol=10)


    ##---------------------------- save the detected results--------------------------------
    save_dir = file.parent
    base_name = file.stem
    save_path = save_dir / f"{base_name}_ripples_CNN.csv"
    # save to csv
    detected_events_df.to_csv(save_path, index=False)
    print(f"ripplenet saved CSV -> {save_path}")


    # Add metadata for later grouping
    stats = stats.copy()
    stats["snr"] = snr
    stats["noise_type"] = noise_name

    all_stats = []
    all_stats.append(stats)

    print(f"\n[Test] {file}")
    print(stats.to_string(index=False))

    # convert to counts if returned objects are lists
    tp_count = len(TPs) if hasattr(TPs, "__len__") and not isinstance(TPs, (int, np.integer)) else int(TPs)
    fp_count = len(FPs) if hasattr(FPs, "__len__") and not isinstance(FPs, (int, np.integer)) else int(FPs)
    fn_count = len(FNs) if hasattr(FNs, "__len__") and not isinstance(FNs, (int, np.integer)) else int(FNs)


    condition_stats[condition_key]["TP"] = tp_count
    condition_stats[condition_key]["FP"] = fp_count
    condition_stats[condition_key]["FN"] = fn_count
    condition_stats[condition_key]["n_files"] = 1

    file_level_results.append({
        "snr": snr,
        "noise_name": noise_name,
        "file": file.name,
        "TP": tp_count,
        "FP": fp_count,
        "FN": fn_count,
    })


# %%
# summarize each condition
summary_rows = []

overall_TP = 0
overall_FP = 0
overall_FN = 0

for (snr, noise_name), counts in sorted(condition_stats.items()):
    TP = counts["TP"]
    FP = counts["FP"]
    FN = counts["FN"]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    summary_rows.append({
        "snr": snr,
        "noise_name": noise_name,
        "n_files": counts["n_files"],
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    })

    overall_TP += TP
    overall_FP += FP
    overall_FN += FN

summary_df = pd.DataFrame(summary_rows)

overall_precision = overall_TP / (overall_TP + overall_FP) if (overall_TP + overall_FP) > 0 else 0.0
overall_recall = overall_TP / (overall_TP + overall_FN) if (overall_TP + overall_FN) > 0 else 0.0
overall_f1 = (
    2 * overall_precision * overall_recall / (overall_precision + overall_recall)
    if (overall_precision + overall_recall) > 0 else 0.0
)

overall_df = pd.DataFrame([{
    "snr": "ALL",
    "noise_name": "ALL",
    "n_files": sum(v["n_files"] for v in condition_stats.values()),
    "TP": overall_TP,
    "FP": overall_FP,
    "FN": overall_FN,
    "precision": overall_precision,
    "recall": overall_recall,
    "F1": overall_f1,
}])

print("\nPer-condition results:")
print(summary_df)

print("\nOverall results:")
print(overall_df)

# # optional: save results
# summary_df.to_csv(project_root / "condition_level_metrics.csv", index=False)
# overall_df.to_csv(project_root / "overall_metrics.csv", index=False)
# pd.DataFrame(file_level_results).to_csv(project_root / "file_level_metrics.csv", index=False)




#%%
#
# '''
# # Event-based evaluation
# TPs, FPs, FNs, stats = get_tps_fps_fns_event_based(zscored_envelope.flatten(), true_ripple[:,0], true_ripple[:,1], detected_events_df["peak_idx"].to_numpy())
# '''
#
# # time-interval based evaluation
# TPs, FPs, FNs, stats = get_tps_fps_fns_time_interval_based(zscored_envelope.flatten(), Y_cont_pred_binary, true_ripple[:,0], true_ripple[:,1], ripple_peaks, iou_threshold = 0.5,peak_tol=10)




