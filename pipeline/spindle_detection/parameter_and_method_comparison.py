import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
