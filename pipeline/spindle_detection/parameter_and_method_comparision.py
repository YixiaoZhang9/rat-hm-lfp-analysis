import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# bases = [
#     '/media/yixiao/GL14_RAT_FA/Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/Spindle_detection_results',
#     '/media/yixiao/GL14_RAT_FA/Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/Spindle_detection_results',
#     '/media/yixiao/Data4/Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/Spindle_detection_results',
#     '/media/yixiao/Data4/Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R13-16/Spindle_detection_results'
# ]
#
# regions = ['HPC','PL','RSC']
# sleep_periods = ['presleep','postsleep']
#
# all_rows = []
#
# # =========================
# # COUNT TOTAL FILES FIRST (for tqdm)
# # =========================
#
# all_files = []
#
# for base in bases:
#     if not os.path.exists(base):
#         continue
#
#     for region in os.listdir(base):
#         region_path = os.path.join(base, region)
#         if not os.path.isdir(region_path):
#             continue
#
#         for rat in os.listdir(region_path):
#             rat_path = os.path.join(region_path, rat)
#
#             for studyday in os.listdir(rat_path):
#                 day_path = os.path.join(rat_path, studyday)
#
#                 for sleep_period in os.listdir(day_path):
#                     sp_path = os.path.join(day_path, sleep_period)
#
#                     for param_folder in os.listdir(sp_path):
#                         param_path = os.path.join(sp_path, param_folder)
#
#                         if not os.path.isdir(param_path):
#                             continue
#
#                         for file in os.listdir(param_path):
#                             if file.endswith(".csv"):
#                                 all_files.append(os.path.join(param_path, file))
#
# print(f"Total CSV files: {len(all_files)}")
#
# # =========================
# # MAIN LOOP WITH PROGRESS
# # =========================
#
# for fpath in tqdm(all_files, desc="Processing all CSV files"):
#
#     parts = fpath.split(os.sep)
#
#     # try-safe parsing
#     try:
#         region = parts[-6]
#         rat = parts[-5]
#         studyday = parts[-4]
#         sleep_period = parts[-3]
#         param_folder = parts[-2]
#     except:
#         continue
#
#     # optional: show current rat (lightweight, not tqdm spam)
#     if 'last_rat' not in locals() or rat != last_rat:
#         print(f"\n===== Processing rat: {rat} =====")
#         last_rat = rat
#
#     df = pd.read_csv(fpath)
#
#     if len(df) == 0:
#         continue
#
#     durations = df["spindle_duration_sec"].values
#
#     if "nrem_bout_duration_sec" in df.columns:
#         total_nrem_sec = np.sum(df["nrem_bout_duration_sec"].unique())
#     else:
#         total_nrem_sec = np.nan
#
#     n_spindles = len(df)
#
#     rate_per_min = n_spindles / (total_nrem_sec / 60) if total_nrem_sec > 0 else np.nan
#
#     all_rows.append({
#         "base": parts[0],
#         "rat": rat,
#         "region": region,
#         "studyday": studyday,
#         "sleep_period": sleep_period,
#         "method": param_folder,
#         "params": param_folder,
#         "n_spindles": n_spindles,
#         "nrem_sec": total_nrem_sec,
#         "rate_per_min": rate_per_min,
#         "mean_duration": np.mean(durations),
#         "median_duration": np.median(durations),
#         "p90_duration": np.percentile(durations, 90),
#         "p10_duration": np.percentile(durations, 10),
#     })
#
# summary = pd.DataFrame(all_rows)
# summary.to_csv("spindle_parameter_summary.csv", index=False)
#
# print("Saved summary table:", summary.shape)
# =========================

summary = pd.read_csv("spindle_parameter_summary.csv")


# %%
# =========================
# 1. RATE COMPARISON PLOT
summary_region = (
    summary.groupby(["rat", "region", "params"])
    .agg({"rate_per_min": "mean", "mean_duration": "mean"})
    .reset_index()
)
methods = {
    "Envelope": summary_region[summary_region["params"].str.contains("envelop")],
    "Wavelet": summary_region[summary_region["params"].str.contains("wavelet_amp")],
    "WaveletOptimal": summary_region[
        summary_region["params"].str.contains("wavelet_optimal")
    ],
}
for method_name, method_df in methods.items():
    # order = (
    #     method_df.groupby("params")["rate_per_min"]
    #     .median()
    #     .sort_values()
    #     .index
    # )
    order = sorted(
        method_df["params"].unique(),
        key=lambda x: tuple(float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", str(x))),
    )
    plt.figure(figsize=(18, 8))
    # violin
    sns.violinplot(
        data=method_df, x="params", y="rate_per_min", order=order, inner=None, cut=0
    )
    # rat points
    sns.stripplot(
        data=method_df,
        x="params",
        y="rate_per_min",
        hue="region",
        order=order,
        dodge=True,
        size=7,
        alpha=0.8,
    )
    # literature range
    plt.axhspan(2, 6, alpha=0.15, color="green")
    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("Mean spindle rate (/min NREM)", fontsize=20)
    plt.xlabel("Parameters", fontsize=20)
    plt.title(f"{method_name} Rate Stability Across Rats & Regions", fontsize=22)
    plt.legend(
        title="Region",
        fontsize=20,
        title_fontsize=20,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()
    plt.show()
# 2. DURATION DISTRIBUTION
for method_name, method_df in methods.items():
    # order = (
    #     method_df.groupby("params")["mean_duration"]
    #     .median()
    #     .sort_values()
    #     .index
    # )
    order = sorted(
        method_df["params"].unique(),
        key=lambda x: tuple(float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", str(x))),
    )
    plt.figure(figsize=(18, 8))
    sns.violinplot(
        data=method_df, x="params", y="mean_duration", order=order, inner=None, cut=0
    )
    sns.stripplot(
        data=method_df,
        x="params",
        y="mean_duration",
        hue="region",
        order=order,
        dodge=True,
        size=7,
        alpha=0.8,
    )
    plt.axhspan(0.4, 3.5, alpha=0.15, color="green")
    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("Mean spindle duration (s)", fontsize=20)
    plt.xlabel("Parameters", fontsize=20)
    plt.title(f"{method_name} Duration Stability Across Rats & Regions", fontsize=22)
    plt.legend(
        title="Region",
        fontsize=20,
        title_fontsize=20,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()
    plt.show()

# ====================================
# 3. OVERALL RATE DISTRIBUTION
# ====================================
methods = {
    "Envelope": summary[summary["params"].str.contains("envelop")],
    "Wavelet": summary[summary["params"].str.contains("wavelet_amp")],
    "WaveletOptimal": summary[summary["params"].str.contains("wavelet_optimal")],
}

for method_name, method_df in methods.items():
    order = sorted(
        method_df["params"].unique(),
        key=lambda x: tuple(float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", str(x))),
    )

    plt.figure(figsize=(18, 8))

    sns.violinplot(
        data=method_df,
        x="params",
        y="rate_per_min",
        order=order,
        inner="quartile",
        cut=0,
    )

    # sns.swarmplot(
    #     data=method_df,
    #     x="params",
    #     y="rate_per_min",
    #     order=order,
    #     size=3,
    #     color='k',
    #     alpha=0.4
    # )

    plt.axhspan(2, 6, alpha=0.15, color="green")

    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(fontsize=15)

    plt.ylabel("Spindle rate (/min NREM)", fontsize=20)
    plt.xlabel("Parameters", fontsize=20)

    plt.title(f"{method_name} Overall Rate Distribution", fontsize=22)

    plt.tight_layout()
    plt.show()


# ====================================
# 4. OVERALL DURATION DISTRIBUTION
# ====================================
methods = {
    "Envelope": summary[summary["params"].str.contains("envelop")],
    "Wavelet": summary[summary["params"].str.contains("wavelet_amp")],
    "WaveletOptimal": summary[summary["params"].str.contains("wavelet_optimal")],
}

for method_name, method_df in methods.items():
    order = sorted(
        method_df["params"].unique(),
        key=lambda x: tuple(float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", str(x))),
    )

    plt.figure(figsize=(18, 8))

    sns.violinplot(
        data=method_df,
        x="params",
        y="mean_duration",
        order=order,
        inner="quartile",
        cut=0,
    )

    # sns.swarmplot(
    #     data=method_df,
    #     x="params",
    #     y="mean_duration",
    #     order=order,
    #     color='black',
    #     size=3,
    #     alpha=0.4
    # )

    plt.axhspan(0.4, 3.5, alpha=0.15, color="green")

    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(fontsize=15)

    plt.ylabel("Mean spindle duration (s)", fontsize=20)
    plt.xlabel("Parameters", fontsize=20)

    plt.title(f"{method_name} Overall Duration Distribution", fontsize=22)

    plt.tight_layout()
    plt.show()
