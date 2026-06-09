import glob
import os

import numpy as np
import pandas as pd
from IPython.display import display
from scipy.io import loadmat
from sklearn.metrics import cohen_kappa_score

from modules.project_config import get_path


def filter_nrem(starts, ends):
    keep = []

    for s, e in zip(starts, ends):
        s_sec = int(np.floor(s))
        e_sec = int(np.floor(e))

        s_sec = max(s_sec, 0)
        e_sec = min(e_sec, len(stage) - 1)

        # in NREM
        if np.all(stage[s_sec : e_sec + 1] == nrem_label):
            keep.append(True)
        else:
            keep.append(False)

    keep = np.array(keep, dtype=bool)
    return starts[keep], ends[keep]


# set the root path
root = get_path("ripple_marking_root")

annotators = ["Anumita", "Kjell", "Lisa", "Sachuriga", "Yixiao"]
main_annotator = "Yixiao"
other_annotators = [a for a in annotators if a != main_annotator]

rats = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
fs = 1000

data = {}

# collect all the paths
for rat in rats:
    rat_key = f"Rat{rat}"
    data[rat_key] = []

    main_path = os.path.join(root, main_annotator, rat_key)
    if not os.path.exists(main_path):
        continue

    for trial_folder in os.listdir(main_path):
        trial_path = os.path.join(main_path, trial_folder)
        if not os.path.isdir(trial_path):
            continue

        # main annotator csv
        main_csv_list = glob.glob(os.path.join(trial_path, "*.csv"))
        if len(main_csv_list) == 0:
            continue
        results_annotator1 = main_csv_list[0]

        # sleep scoring results
        sleep_scoring_list = glob.glob(os.path.join(trial_path, "*eegstates.mat"))
        sleep_scoring_file = (
            sleep_scoring_list[0] if len(sleep_scoring_list) > 0 else None
        )

        # other annotator
        for other in other_annotators:
            other_trial_path = os.path.join(root, other, rat_key, trial_folder)
            if not os.path.exists(other_trial_path):
                continue

            other_csv_list = glob.glob(os.path.join(other_trial_path, "*.csv"))
            if len(other_csv_list) == 0:
                continue

            results_annotator2 = other_csv_list[0]

            data[rat_key].append(
                {
                    "trial": trial_folder,
                    "annotator1": main_annotator,
                    "annotator2": other,
                    "results_annotator1": results_annotator1,
                    "results_annotator2": results_annotator2,
                    "sleep_scoring_results": sleep_scoring_file,
                }
            )

# calculate consistence of the annotation results
results = []

# %%
for rat_key, sessions in data.items():
    for session in sessions:
        # NREM sleep
        nrem_duration_min = np.nan

        if session["sleep_scoring_results"] is not None:
            scoring_mat = loadmat(session["sleep_scoring_results"])
            scoring = scoring_mat["states"].squeeze()

            nrem_seconds = np.sum(scoring == 3)
            nrem_duration_min = nrem_seconds / 60

        df_annotator1 = pd.read_csv(session["results_annotator1"], header=0)
        df_annotator2 = pd.read_csv(session["results_annotator2"], header=0)

        starts_annotator1 = pd.to_numeric(df_annotator1.iloc[:, 0])
        ends_annotator1 = pd.to_numeric(df_annotator1.iloc[:, 1])
        starts_annotator2 = pd.to_numeric(df_annotator2.iloc[:, 0])
        ends_annotator2 = pd.to_numeric(df_annotator2.iloc[:, 1])

        if session["sleep_scoring_results"] is not None:
            stage = scoring  # already loaded above
            nrem_label = 3

            starts_annotator1, ends_annotator1 = filter_nrem(
                starts_annotator1.values, ends_annotator1.values
            )

            starts_annotator2, ends_annotator2 = filter_nrem(
                starts_annotator2.values, ends_annotator2.values
            )

        else:
            starts_annotator1 = starts_annotator1.values
            ends_annotator1 = ends_annotator1.values
            starts_annotator2 = starts_annotator2.values
            ends_annotator2 = ends_annotator2.values

        n_ripples_annotator1 = len(starts_annotator1)
        n_ripples_annotator2 = len(starts_annotator2)

        # ripple rate
        rate_annotator1 = np.nan
        rate_annotator2 = np.nan

        if nrem_duration_min > 0:
            rate_annotator1 = n_ripples_annotator1 / nrem_duration_min
            rate_annotator2 = n_ripples_annotator2 / nrem_duration_min

        # ---- ONE-TO-ONE IoU matching ----
        starts1 = starts_annotator1
        ends1 = ends_annotator1
        starts2 = starts_annotator2
        ends2 = ends_annotator2
        order1 = np.argsort(starts1)
        starts1, ends1 = starts1[order1], ends1[order1]

        order1 = np.argsort(starts1)
        order2 = np.argsort(starts2)

        starts1, ends1 = starts1[order1], ends1[order1]
        starts2, ends2 = starts2[order2], ends2[order2]

        matched_1 = np.zeros(len(starts1), dtype=bool)
        matched_2 = np.zeros(len(starts2), dtype=bool)

        iou_th = 0.5

        for i in range(len(starts1)):
            best_j = -1
            best_iou = 0

            for j in range(len(starts2)):
                if matched_2[j]:
                    continue

                inter = max(0, min(ends1[i], ends2[j]) - max(starts1[i], starts2[j]))
                union = max(ends1[i], ends2[j]) - min(starts1[i], starts2[j])

                iou = inter / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_th:
                matched_1[i] = True
                matched_2[best_j] = True

        tp = np.sum(matched_1)
        fp = len(starts2) - tp
        fn = len(starts1) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # 2. TIME-LEVEL COHEN'S KAPPA
        nrem_mask_sec = stage == 3

        nrem_mask = np.repeat(nrem_mask_sec, fs)
        total_time_sec = len(stage)
        total_samples = total_time_sec * 1000

        timeline1 = np.zeros(total_samples, dtype=np.uint8)
        timeline2 = np.zeros(total_samples, dtype=np.uint8)

        for s, e in zip(starts1, ends1):
            s_idx = int(s * 1000)
            e_idx = int(e * 1000)
            timeline1[s_idx:e_idx] = 1

        for s, e in zip(starts2, ends2):
            s_idx = int(s * 1000)
            e_idx = int(e * 1000)
            timeline2[s_idx:e_idx] = 1

            timeline1_nrem = timeline1[nrem_mask]
            timeline2_nrem = timeline2[nrem_mask]

        # Cohen's kappa
        kappa = cohen_kappa_score(timeline1_nrem, timeline2_nrem)

        results.append(
            {
                "rat": rat_key,
                "trial": session["trial"],
                "annotator1": session["annotator1"],
                "annotator2": session["annotator2"],
                "n_ripples_annotator1": n_ripples_annotator1,
                "n_ripples_annotator2": n_ripples_annotator2,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "kappa": kappa,
                "nrem_duration_min": nrem_duration_min,
                "rate_annotator1_per_min": rate_annotator1,
                "rate_annotator2_per_min": rate_annotator2,
            }
        )

df_results = pd.DataFrame(results)
pd.set_option("display.max_columns", None)
display(df_results)
# df_results.to_csv("ripple_annotation_agreement.csv", index=False)

print("consistence calculation done.")

# %%
