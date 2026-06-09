import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.errors import EmptyDataError
from scipy.io import loadmat

from modules.project_config import get_path


def match_events(gt_df, pred_df, iou_th=0.5):

    required_cols = ["ripple_start", "ripple_end"]

    if gt_df is None:
        gt_df = pd.DataFrame(columns=required_cols)
    if pred_df is None:
        pred_df = pd.DataFrame(columns=required_cols)

    for col in required_cols:
        if col not in gt_df.columns:
            gt_df[col] = pd.Series(dtype=float)
        if col not in pred_df.columns:
            pred_df[col] = pd.Series(dtype=float)

    gt_df = gt_df[required_cols].copy()
    pred_df = pred_df[required_cols].copy()

    gt_df = gt_df.sort_values("ripple_start").reset_index(drop=True)
    pred_df = pred_df.sort_values("ripple_start").reset_index(drop=True)

    matched = []
    used_pred = set()

    pred_ptr = 0

    for i, gt in gt_df.iterrows():
        gt_start = gt["ripple_start"]
        gt_end = gt["ripple_end"]

        best_iou = 0
        best_j = None

        # move pointer until pred event could overlap
        while (
            pred_ptr < len(pred_df) and pred_df.loc[pred_ptr, "ripple_end"] < gt_start
        ):
            pred_ptr += 1

        j = pred_ptr

        # only examine events that could overlap
        while j < len(pred_df):
            pred = pred_df.loc[j]

            pred_start = pred["ripple_start"]
            pred_end = pred["ripple_end"]

            # no possible overlap anymore
            if pred_start > gt_end:
                break

            if j not in used_pred:
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
    fp = len(pred_df) - tp
    fn = len(gt_df) - tp

    return tp, fp, fn


def get_voted_events(
    trial_methods_dict, selected_methods, anchor_order=["ripplenet", "cnn"], min_votes=2
):
    """
    Perform event-based voting across multiple detection methods.

    Parameters:
    -----------
    trial_methods_dict : dict
        Dictionary containing DataFrames for each detection method.
    anchor_order : list
        Priority list for determining the final event boundaries (start/end).
    min_votes : int
        Minimum number of methods required to agree for a valid event.

    Returns:
    --------
    pd.DataFrame : Consolidated events with boundaries defined by the anchor method.
    """
    all_events = []
    for m in selected_methods:
        if m in trial_methods_dict:
            df = trial_methods_dict[m].copy()
            df["method_source"] = m
            all_events.append(df)

    if not all_events:
        return pd.DataFrame(columns=["ripple_start", "ripple_end"])

    # Merge all detections and sort by onset time
    combined = pd.concat(all_events).sort_values("ripple_start")

    voted_events = []
    if combined.empty:
        return pd.DataFrame(voted_events)

    # Cluster overlapping events into groups
    groups = []
    curr_group = [combined.iloc[0]]
    for i in range(1, len(combined)):
        # Calculate the latest end time in the current cluster to check for overlap
        prev_max_end = max([e["ripple_end"] for e in curr_group])
        curr_event = combined.iloc[i]

        # If the current event starts before the cluster ends, they overlap
        if curr_event["ripple_start"] < prev_max_end:
            curr_group.append(curr_event)
        else:
            groups.append(curr_group)
            curr_group = [curr_event]
    groups.append(curr_group)  # Add the final group

    # Process each cluster to apply voting logic and boundary anchoring
    for group in groups:
        methods_in_group = {e["method_source"] for e in group}

        # Check if the consensus meets the minimum vote threshold
        if len(methods_in_group) >= min_votes:
            final_start, final_end = None, None

            # Boundary Selection: Use the boundary from the highest priority anchor method
            for anchor in anchor_order:
                anchor_matches = [e for e in group if e["method_source"] == anchor]
                if anchor_matches:
                    final_start = anchor_matches[0]["ripple_start"]
                    final_end = anchor_matches[0]["ripple_end"]
                    break

            voted_events.append(
                {
                    "ripple_start": final_start,
                    "ripple_end": final_end,
                    "vote_count": len(methods_in_group),
                }
            )

    if len(voted_events) == 0:
        return pd.DataFrame(columns=["ripple_start", "ripple_end", "vote_count"])

    return pd.DataFrame(voted_events)


def safe_read_csv(path):
    required_cols = ["ripple_start", "ripple_end"]

    try:
        df = pd.read_csv(path)
        for col in required_cols:
            if col not in df.columns:
                df[col] = []

        return df[required_cols]

    except (EmptyDataError, FileNotFoundError, pd.errors.ParserError):
        return pd.DataFrame(columns=required_cols)


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

rats = [1, 3, 7, 8]
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
fs = 1000  # downsampled sample frequency

# ---------------------------------------------------------------------------------------------------------
# ---------------------------- collect the detection results and ground truth-----------------------------
detection_methods = ["threshold", "threshold_advanced", "ripplenet", "cnn"]
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
# ---------------------------- evaluate the consensus of those methods and ground truth -------------------

event_evaluation_rows = []

for rat in rats:
    for studyday in detection_results[rat]:
        for sleep_period in detection_results[rat][studyday]:
            for trial_id in detection_results[rat][studyday][sleep_period]:
                methods = detection_results[rat][studyday][sleep_period][trial_id][
                    "methods"
                ]
                # compare the consensus between different detection method
                method_names = list(methods.keys())

                for i in range(len(method_names)):
                    for j in range(i + 1, len(method_names)):
                        m1 = method_names[i]
                        m2 = method_names[j]

                        tp, fp, fn = match_events(methods[m1], methods[m2])
                        precision = tp / (tp + fp + 1e-8)
                        recall = tp / (tp + fn + 1e-8)
                        event_evaluation_rows.append(
                            {
                                "rat": rat,
                                "studyday": studyday,
                                "sleep_period": sleep_period,
                                "method": f"{m1}_vs_{m2}",
                                "tp": tp,
                                "fp": fp,
                                "fn": fn,
                                "precision": precision,
                                "recall": recall,
                                "f1": 2
                                * precision
                                * recall
                                / (precision + recall + 1e-8),
                            }
                        )

event_eval_df = pd.DataFrame(event_evaluation_rows)

# %%
for rat in sorted(event_eval_df["rat"].unique()):
    subset = event_eval_df[event_eval_df["rat"] == rat].copy()

    if subset.empty:
        continue

    # average over studyday / sleep_period / trials
    summary = (
        subset.groupby("method")[["tp", "fp", "fn", "precision", "recall", "f1"]]
        .mean()
        .reset_index()
    )

    summary[["tp", "fp", "fn"]] = summary[["tp", "fp", "fn"]].round(1)
    summary[["precision", "recall", "f1"]] = summary[
        ["precision", "recall", "f1"]
    ].round(3)

    # rename comparison names
    summary["comparison"] = summary["method"].apply(lambda x: x.replace("_vs_", " vs "))

    # preserve method ordering
    comparison_order = {
        "threshold_vs_ripplenet": 0,
        "threshold_vs_cnn": 1,
        "ripplenet_vs_cnn": 2,
    }

    summary["sort_order"] = summary["method"].apply(
        lambda x: comparison_order.get(x, 999)
    )
    summary = summary.sort_values("sort_order")
    summary = summary[["comparison", "tp", "fp", "fn", "precision", "recall", "f1"]]

    print(" " + " = " * 120)
    print(f"Rat = {rat}")
    print("=" * 120)
    print(summary.to_string(index=False))

    # ------------------------------heatmap plot------------------------------------------------------
    plot_data = []

    mean_f1_df = subset.groupby("method")["f1"].mean().reset_index()

    for _, row in mean_f1_df.iterrows():
        m = row["method"]
    f1 = row["f1"]

    if "_vs_" in m:
        m1, m2 = m.split("_vs_")

    plot_data.append({"m1": m1, "m2": m2, "f1": f1})
    plot_data.append({"m1": m2, "m2": m1, "f1": f1})

    df_plot = pd.DataFrame(plot_data)

    if df_plot.empty:
        continue

    matrix = df_plot.pivot(index="m1", columns="m2", values="f1")

    ordered_methods = ["threshold", "ripplenet", "cnn"]
    available_methods = [
        m for m in ordered_methods if m in matrix.index or m in matrix.columns
    ]

    matrix = matrix.reindex(index=available_methods, columns=available_methods)

    # diagonal = 1
    for method in available_methods:
        matrix.loc[method, method] = 1.0

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Mean F1 Score"},
    )

    plt.title(f"Pairwise F1 Agreement (Rat={rat})")
    plt.tight_layout()
    plt.show()
    plt.close("all")

# %%
