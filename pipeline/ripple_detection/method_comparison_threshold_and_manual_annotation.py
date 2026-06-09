# ---- Import ----
import os
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat

from modules.project_config import get_path


def match_events(gt_arr, pred_arr, iou_th=0.5):

    if gt_arr is None:
        gt_arr = np.zeros((0, 2))
    if pred_arr is None:
        pred_arr = np.zeros((0, 2))

    gt_arr = np.asarray(gt_arr)
    pred_arr = np.asarray(pred_arr)

    # sort by start time
    gt_arr = gt_arr[np.argsort(gt_arr[:, 0])]
    pred_arr = pred_arr[np.argsort(pred_arr[:, 0])]

    used_pred = set()
    matched = []

    pred_ptr = 0

    for i, gt in enumerate(gt_arr):
        gt_start, gt_end = gt

        best_iou = 0
        best_j = None

        # skip impossible overlaps
        while pred_ptr < len(pred_arr) and pred_arr[pred_ptr][1] < gt_start:
            pred_ptr += 1

        j = pred_ptr

        while j < len(pred_arr):
            if j in used_pred:
                j += 1
                continue

            pred_start, pred_end = pred_arr[j]

            if pred_start > gt_end:
                break

            inter = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
            union = max(gt_end, pred_end) - min(gt_start, pred_start)

            iou = inter / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_j = j

            j += 1

        if best_iou >= iou_th:
            matched.append((i, best_j))
            used_pred.add(best_j)

    tp = len(matched)
    fp = len(pred_arr) - tp
    fn = len(gt_arr) - tp

    return tp, fp, fn


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = get_path("R1_8_root")
dir_R1_4_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/PreprocessedData"
)
dir_R1_4_Scoring = os.path.join(dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/Scoring")
dir_R1_4_Ripple = os.path.join(
    dir_base1,
    "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/Ripple_detection_results",
)
dir_R5_8_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/PreprocessedData"
)
dir_R5_8_Scoring = os.path.join(dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/Scoring")
dir_R5_8_Ripple = os.path.join(
    dir_base1,
    "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/Ripple_detection_results",
)

# the path storing the ripple marking results
root_annotation = get_path("ripple_marking_root")
annotators = "Yixiao"

rats = [3, 7]
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
trials = {
    "Rat3": ["20221010_trial13", "20221013_trial8"],
    "Rat7": ["20221014_trial11", "20221017_trial12"],
}
fs = 1000  # downsampled sample frequency


for rat in rats:
    rat_key = f"Rat{rat}"
    annotation_path = os.path.join(root_annotation, annotators, rat_key)
    if rat == 3:
        threshold_result_path = dir_R1_4_Ripple
        data_path = dir_R1_4_Data
        scoring_path = dir_R1_4_Scoring
    elif rat == 7:
        threshold_result_path = dir_R5_8_Ripple
        data_path = dir_R5_8_Data
        scoring_path = dir_R5_8_Scoring

    for trial_name in trials[rat_key]:
        annotated_ripples = []
        threshold_ripples = []
        threshold2_ripples = []
        for root, dirs, files in os.walk(annotation_path):
            if trial_name in root:
                for file in files:
                    if file.endswith("events.csv"):
                        file_path = os.path.join(root, file)
                        print(file_path)
                        # store the results
                        # read ripple start/end times from csv
                        ripple_df = pd.read_csv(file_path)
                        # first column = ripple start time, second column = ripple end time
                        annotated_ripples = np.column_stack(
                            [
                                (ripple_df["start_time"].to_numpy() * 1000).astype(int),
                                (ripple_df["end_time"].to_numpy() * 1000).astype(int),
                            ]
                        )

        # parse date and trial number from trial_name
        date_str, trial_str = trial_name.split("_")
        trial_id = trial_str.replace("trial", "")

        rat_threshold_path = os.path.join(threshold_result_path, str(rat), date_str)

        file_sleep_period = []
        for sleep_period in ["presleep", "postsleep"]:
            sleep_path = os.path.join(rat_threshold_path, sleep_period)

            if not os.path.exists(sleep_path):
                continue

            for ripple_file in os.listdir(sleep_path):
                ripple_file_path = os.path.join(sleep_path, ripple_file)

                if f"_{trial_id}_hippocampal_ripples." in ripple_file:
                    file_sleep_period = sleep_period
                    print("threshold1:", ripple_file_path)
                    threshold_df = pd.read_csv(ripple_file_path)
                    threshold1_ripples = threshold_df[
                        ["ripple_start", "ripple_end"]
                    ].to_numpy()

                if f"_{trial_id}_hippocampal_ripples_threshold2." in ripple_file:
                    print("threshold2:", ripple_file_path)
                    threshold_df2 = pd.read_csv(ripple_file_path)
                    threshold2_ripples = threshold_df2[
                        ["ripple_start", "ripple_end"]
                    ].to_numpy()

        # Find scoring file
        dir_scoring = os.path.join(scoring_path, str(rat), date_str, file_sleep_period)
        scoring_files = [f for f in os.listdir(dir_scoring) if f.endswith(".mat")]
        # matched sleep scoring files
        if len(scoring_files) == 1:
            scoring = loadmat(os.path.join(dir_scoring, scoring_files[0]))[
                "states"
            ].squeeze()
        else:
            suffix = trial_id.zfill(2)
            pattern = re.compile(rf"_{suffix}_")

            scoring = None
            for f in scoring_files:
                if pattern.search(f):
                    scoring = loadmat(os.path.join(dir_scoring, f))["states"].squeeze()
                    break

        # remove the annotations that are out of NREM stage
        nrem_label = 3
        filtered_ripples = []

        for r in annotated_ripples:
            start_idx, end_idx = r
            # convert to seconds for scoring alignment
            start_s = int(start_idx // 1000)
            end_s = int(end_idx // 1000)

            start_s = max(start_s, 0)
            end_s = min(end_s, len(scoring) - 1)

            # keep only fully NREM ripples
            if np.all(scoring[start_s : end_s + 1] == nrem_label):
                filtered_ripples.append(r)

        annotated_ripples = np.array(filtered_ripples)

        # compare the results
        gt = annotated_ripples
        pred1 = threshold1_ripples if len(threshold1_ripples) > 0 else np.zeros((0, 2))
        pred2 = threshold2_ripples if len(threshold2_ripples) > 0 else np.zeros((0, 2))

        tp1, fp1, fn1 = match_events(gt, pred1, iou_th=0.5)
        tp2, fp2, fn2 = match_events(gt, pred2, iou_th=0.5)

        p1, r1, f1_1 = compute_metrics(tp1, fp1, fn1)
        p2, r2, f1_2 = compute_metrics(tp2, fp2, fn2)

        print(f"\n{rat_key},Trial: {trial_name}")

        print("Threshold1:")
        print(f"TP={tp1}, FP={fp1}, FN={fn1}")
        print(f"P={p1:.3f}, R={r1:.3f}, F1={f1_1:.3f}")

        print("Threshold2:")
        print(f"TP={tp2}, FP={fp2}, FN={fn2}")
        print(f"P={p2:.3f}, R={r2:.3f}, F1={f1_2:.3f}")
