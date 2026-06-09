import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
from tensorflow import keras

from modules.project_config import get_path

# Custom modules
from modules.threshold_ripple_detection import find_bouts

# load neural networks
project_root = Path.cwd().resolve().parents[1]
ripplenet_model_dir = project_root / "ripple_detection" / "RNN_LSTM"
# model_path = RIPPLENET_MODEL_DIR /"best_model.pkl"
model_path = ripplenet_model_dir / "best_model_finetuned_sim_main3.pkl"
# load info on best model (path, threhsold settings)
with open(model_path, "rb") as f:
    best_model = pickle.load(f)
    print(best_model)
# Threshold for detecting event from prediction
threshold = best_model["threshold"]
distance = best_model["distance"]
width = best_model["width"]
# load the 'best' performing model on the validation sets
# model = keras.models.load_model(RIPPLENET_MODEL_DIR /best_model['model_file'])
model = keras.models.load_model(
    ripplenet_model_dir / "ripplenet_finetuned_sim_main3.h5", compile=False
)
model.summary()


# RippleNet inference function (core)
def ripplenet_predict(lfp_segment, fs, model):

    # Z-score normalization (recording-wise)
    lfp_z = zscore(lfp_segment)

    # Handle rare NaNs
    if np.any(np.isnan(lfp_z)):
        m = np.mean(lfp_segment)
        s = np.std(lfp_segment) + 1e-8
        lfp_z = (lfp_segment - m) / s

    # Reshape to (1, T, 1)
    X = np.expand_dims(np.expand_dims(lfp_z, 0), -1)
    T = X.shape[1]

    segment_length = int(fs * 1.0)  # 1 second window
    n_shifts = 5
    shift = int(segment_length / n_shifts)

    container = []

    # Shift ensemble inference
    for i in range(n_shifts):
        # Shift via zero padding
        X_pad = np.concatenate(
            (np.zeros((1, i * shift, 1), dtype=np.float32), X), axis=1
        )

        # Pad to multiple of segment length
        remainder = X_pad.shape[1] % segment_length
        if remainder != 0:
            pad_len = segment_length - remainder
            X_pad = np.concatenate(
                (X_pad, np.zeros((1, pad_len, 1), dtype=np.float32)), axis=1
            )

        # Segment into chunks
        X_seg = X_pad.reshape((-1, segment_length, 1))

        # Predict
        Y_seg = model.predict(X_seg, verbose=0)

        # Reconstruct continuous signal
        Y_pad = Y_seg.reshape((1, -1, 1))
        Y_pad = Y_pad[:, : X_pad.shape[1], :]

        # Remove shift padding
        start = i * shift
        end = start + T

        container.append(Y_pad[:, start:end, :])

    # Median ensemble
    Y_cont = np.median(np.stack(container, axis=0), axis=0)

    return Y_cont[0, :, 0]


#  Convert prediction resulst to ripple events


def detect_ripples_from_prediction(pred, fs, threshold, distance, width, offset=0):

    events = []

    Y_pred_binary, predicted_peaks = get_binary_predictions(
        pred.flatten(), threshold, distance, width
    )

    padded = np.concatenate([[0], Y_pred_binary, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = (
        np.where(diff == -1)[0] - 1
    )  # inclusive, because offset in the ground truth is also inclusive

    # Convert segments into events
    for start, end in zip(starts, ends):
        global_start = start + offset
        global_end = end + offset

        duration = (global_end - global_start) / fs

        events.append([global_start, global_end, duration])

    return events


def get_binary_predictions(y_prob, threshold, min_distance, min_width):
    """
    Convert a probability trace into a binary event trace with post-processing.

    Parameters
    ----------
    y_prob : ndarray, shape (T,)
        Model output probability at each time point. Values should be in [0, 1]

    threshold : float
        Probability threshold for event detection
        Time points with probability >= threshold are treated as candidate event points.

    min_distance : int
        Minimum allowed gap (in time points) between two events
        If the gap (start_next - end_prev) between two events is smaller than this value, they will be merged into one event

    min_width : int
        Minimum event duration (in time points)
    Events shorter than this value will be removed.

    Returns
    -------
    y_binary : ndarray, shape (T,)
        Binary prediction trace after post-processing (0/1).
     pred_peaks : ndarray, shape (N_events,)
        Peak indices (in original time index) for each detected event.
    """
    y_prob = np.asarray(y_prob).flatten()
    T = y_prob.shape[0]

    # thresholding
    binary = (y_prob >= threshold).astype(np.int32)

    # find contiguous segments
    # diff == 1 -> event starts, diff == -1 -> event ends
    padded = np.concatenate([[0], binary, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]  # end is exclusive

    raw_events = []
    for start, end in zip(starts, ends):
        raw_events.append([start, end])

    # merge events that are too close
    merged_events = []
    for start, end in raw_events:
        if not merged_events:
            merged_events.append([start, end])
        else:
            prev_s, prev_e = merged_events[-1]
            gap = start - prev_e
            if gap < min_distance:
                # Merge by extending the previous event's end
                merged_events[-1][1] = end
            else:
                merged_events.append([start, end])

    # filter out events that are too short
    final_events = []
    for start, end in merged_events:
        if (end - start) >= min_width:
            final_events.append([start, end])

    # build final binary trace
    y_binary = np.zeros(T, dtype=np.int32)
    detected_peaks = []
    for start, end in final_events:
        y_binary[start:end] = 1

        seg = y_prob[start:end]  # end is exclusive
        if seg.size == 0:
            continue

        max_val = float(seg.max())
        max_pos = np.where(seg == max_val)[0]  # indices within seg
        pos = int(max_pos[len(max_pos) // 2])

        detected_peaks.append(start + pos)

    return y_binary, np.asarray(detected_peaks)


dir_base1 = get_path("R9_16_root")
dir_R9_12_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/PreprocessedData"
)
dir_R9_12_Scoring = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/Scoring"
)
dir_output = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/Ripple_detection_results"
)

rats = [9, 11, 12]
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 1000  # downsampled sample frequency

f_plot = 1

for rat in rats:
    rat_str = f"rat{rat}"
    print(f"\n===== Processing {rat_str} =====")

    for region in [regions[0]]:
        dir_R9_12_Data_perday = os.path.join(dir_R9_12_Data, region, str(rat))
        if not os.path.exists(dir_R9_12_Data_perday):
            continue

        folders_SD = [
            d
            for d in os.listdir(dir_R9_12_Data_perday)
            if os.path.isdir(os.path.join(dir_R9_12_Data_perday, d))
        ]

        for studyday in folders_SD:
            for sleep_period in sleep_periods:
                print(f"Processing {rat_str} | {studyday} | {sleep_period}")

                dir_trial = os.path.join(dir_R9_12_Data_perday, studyday, sleep_period)
                path_scoring = os.path.join(
                    dir_R9_12_Scoring, str(rat), studyday, sleep_period
                )

                if not os.path.exists(dir_trial) or not os.path.exists(path_scoring):
                    continue

                scoring_files = [
                    f for f in os.listdir(path_scoring) if f.endswith(".mat")
                ]

                for root, _, files in os.walk(dir_trial):
                    for name in files:
                        if not name.endswith(".mat"):
                            continue

                        print(f"  Trial: {name}")

                        trial_path = os.path.join(root, name)
                        trial_data = loadmat(trial_path)

                        lfp = trial_data["data"].squeeze()

                        # Load scoring
                        if len(scoring_files) == 1:
                            scoring = loadmat(
                                os.path.join(path_scoring, scoring_files[0])
                            )["states"].squeeze()
                        else:
                            match = re.search(r"_(\d+)\.mat$", name)
                            if not match:
                                continue

                            suffix = match.group(1).zfill(2)
                            scoring = None

                            for f in scoring_files:
                                if f"_{suffix}_" in f:
                                    scoring = loadmat(os.path.join(path_scoring, f))[
                                        "states"
                                    ].squeeze()
                                    break

                            if scoring is None:
                                continue

                        # Find NREM bouts
                        bouts = find_bouts(scoring, target_value=3, fs=fs)

                        ripple_rows = []

                        # Process each bout
                        for b_start, b_end in bouts:
                            lfp_bout = lfp[b_start:b_end]

                            predicted_ripples = ripplenet_predict(lfp_bout, fs, model)

                            # Convert to events
                            events = detect_ripples_from_prediction(
                                predicted_ripples,
                                fs,
                                threshold,
                                distance,
                                width,
                                offset=b_start,
                            )

                            for onset, offset_, duration in events:
                                if duration < 0.025 or duration > 0.5:
                                    continue

                                ripple_rows.append(
                                    {
                                        "ripple_start": onset,
                                        "ripple_end": offset_,
                                        "ripple_duration_sec": duration,
                                        "nrem_bout_duration_sec": (b_end - b_start)
                                        / fs,
                                    }
                                )

                        # Save results
                        df = pd.DataFrame(ripple_rows)

                        output_dir = os.path.join(
                            dir_output, str(rat), studyday, sleep_period
                        )
                        os.makedirs(output_dir, exist_ok=True)

                        trial_id = name.replace(".mat", "")
                        save_path = os.path.join(
                            output_dir, f"{trial_id}_hippocampal_ripples_ripplenet.csv"
                        )

                        df.to_csv(save_path, index=False)


print("\n RippleNet detection DONE")
