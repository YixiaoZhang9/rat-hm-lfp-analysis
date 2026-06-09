import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore
from tensorflow import keras
from tqdm import tqdm

from pipeline.ripple_detection.detection_evaluation import *

# path
project_root = Path.cwd().resolve().parents[1]
data_root = project_root / "training_data"

val_list = sorted(data_root.glob("trial_005.npz"))

# grid search parameters
threshold_list = np.linspace(0.1, 0.9, 17)
distance_list = [0, 10, 20, 30, 40, 50, 60, 70, 80]
width_list = [0, 5, 10, 15, 20, 25, 30, 35]

# load the model
model = keras.models.load_model(
    project_root / "RNN_LSTM" / "model_training" / "retrained_model.h5", compile=False
)

segment_seconds = 1.0
n_shifts = 5

cache = []


for file in tqdm(val_list, desc="Precompute predictions"):
    d = np.load(file, allow_pickle=True)

    pre = d["preprocessed_data"].astype(np.float32)
    y_gt = d["consensus_trace"].astype(np.float32)
    scoring = d["scoring"].astype(np.int8)

    fs = 1000

    # NREM mask
    scoring = np.repeat(scoring, fs)
    mask = scoring == 3

    # zscore
    X = np.expand_dims(np.expand_dims(pre, 0), -1)
    X = zscore(X, axis=1)
    X = np.nan_to_num(X).astype(np.float32)

    segment_length = int(segment_seconds * fs)
    shift = segment_length // n_shifts

    container = []

    for i_shift in range(n_shifts):
        X_pad = np.concatenate(
            (np.zeros((1, i_shift * shift, 1), dtype=np.float32), X), axis=1
        )

        remainder = X_pad.shape[1] % segment_length
        if remainder != 0:
            pad_len = segment_length - remainder
            X_pad = np.concatenate(
                (X_pad, np.zeros((1, pad_len, 1), dtype=np.float32)), axis=1
            )

        X_seg = X_pad.reshape((-1, segment_length, 1))

        Y_pred = model.predict(X_seg, verbose=0)
        Y_pad = Y_pred.reshape((1, -1, 1))
        Y_pad = Y_pad[:, : X_pad.shape[1], :]

        start = i_shift * shift
        end = start + len(pre)

        container.append(Y_pad[:, start:end, :])

    Y_pred_cont = np.median(np.stack(container, axis=0), axis=0).flatten()

    # only keep NREM segments
    Y_pred_cont = Y_pred_cont[mask]
    y_gt = y_gt[mask]

    gt_binary = (y_gt >= 0.5).astype(int)

    gt_onset = np.where(np.diff(np.concatenate([[0], gt_binary])) == 1)[0]
    gt_offset = np.where(np.diff(np.concatenate([gt_binary, [0]])) == -1)[0]

    cache.append(
        {
            "Y_pred": Y_pred_cont,
            "gt_onset": gt_onset,
            "gt_offset": gt_offset,
        }
    )

print("Cached:", len(cache))


# Grid search
results = []

total = len(threshold_list) * len(distance_list) * len(width_list)
pbar = tqdm(total=total, desc="Grid search")

for th in threshold_list:
    for dist in distance_list:
        for wid in width_list:
            TP_sum = FP_sum = FN_sum = 0

            for item in cache:
                Y = item["Y_pred"]

                Y_bin, _ = get_binary_predictions(Y, th, dist, wid)

                gt_on = item["gt_onset"]
                gt_off = item["gt_offset"]

                TPs, FPs, FNs, stats = get_tps_fps_fns_time_interval_based(
                    Y, Y_bin, gt_on, gt_off, gt_on, iou_threshold=0.5, peak_tol=1000
                )

                TP_sum += int(stats["TP"])
                FP_sum += int(stats["FP"])
                FN_sum += int(stats["FN"])

            precision = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) else 0
            recall = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) else 0
            F1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0
            )

            results.append({"threshold": th, "distance": dist, "width": wid, "F1": F1})

            pbar.update(1)

pbar.close()

df = pd.DataFrame(results)
best = df.sort_values("F1", ascending=False).iloc[0]

print("\nBEST:")
print(best)


# save the results
retrained_model_pkl = (
    project_root / "RNN_LSTM" / "model_training" / "retrained_model.pkl"
)

with open(retrained_model_pkl, "rb") as f:
    retrained_model = pickle.load(f)

print(retrained_model)

retrained_model["model_fime"] = "retrained_model"
retrained_model["threshold"] = float(best["threshold"])
retrained_model["distance"] = int(best["distance"])
retrained_model["width"] = int(best["width"])

with open(retrained_model_pkl, "wb") as f:
    pickle.dump(retrained_model, f)

print("Saved updated params")
