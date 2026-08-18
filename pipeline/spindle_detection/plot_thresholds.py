from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

THRESHOLD_CSV = Path("results/all_thresholds_raw.csv")
OUTPUT_DIR = Path("results/threshold_plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(THRESHOLD_CSV)

print(f"Loaded {len(df):,} threshold results")

print()
print(df.head())

print()
print("Threshold statistics:")
print(df["Threshold"].describe())


# ============================================================
# 1. HISTOGRAM OF THRESHOLDS
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["Threshold"],
    bins=30,
)

plt.xlabel("Calibrated threshold")
plt.ylabel("Number of recordings")
plt.title("Distribution of Calibrated Spindle Detection Thresholds")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "threshold_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. THRESHOLD COUNTS
# ============================================================

threshold_counts = (
    df["Threshold"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 6))

plt.bar(
    threshold_counts.index.astype(str),
    threshold_counts.values,
)

plt.xlabel("Threshold")
plt.ylabel("Number of recordings")
plt.title("Number of Recordings at Each Threshold")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "threshold_counts.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. THRESHOLD BY RAT
# ============================================================

rat_stats = (
    df.groupby("Rat")["Threshold"]
    .agg(["mean", "std", "min", "max", "count"])
)

print()
print("Threshold statistics by rat:")
print(rat_stats)


plt.figure(figsize=(10, 6))

df.boxplot(
    column="Threshold",
    by="Rat",
)

plt.xlabel("Rat")
plt.ylabel("Threshold")
plt.title("Threshold Distribution by Rat")

plt.suptitle("")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "threshold_by_rat.png",
    dpi=300
)

plt.close()


# ============================================================
# 4. THRESHOLD BY REGION
# ============================================================

region_stats = (
    df.groupby("Region")["Threshold"]
    .agg(["mean", "std", "min", "max", "count"])
)

print()
print("Threshold statistics by region:")
print(region_stats)


plt.figure(figsize=(8, 6))

df.boxplot(
    column="Threshold",
    by="Region",
)

plt.xlabel("Region")
plt.ylabel("Threshold")
plt.title("Threshold Distribution by Brain Region")

plt.suptitle("")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "threshold_by_region.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. THRESHOLD BY RAT × REGION
# ============================================================

group_stats = (
    df.groupby(
        ["Rat", "Region"]
    )["Threshold"]
    .agg(
        Mean="mean",
        SD="std",
        Min="min",
        Max="max",
        N="count",
    )
    .reset_index()
)

print()
print("Threshold statistics by Rat × Region:")
print(group_stats.to_string(index=False))


# ============================================================
# 6. THRESHOLD OVER RECORDINGS
# ============================================================

df = df.sort_values(
    ["Rat", "Region", "Date"]
)

plt.figure(figsize=(14, 6))

plt.scatter(
    range(len(df)),
    df["Threshold"],
)

plt.xlabel("Recording")
plt.ylabel("Threshold")
plt.title("Calibrated Threshold Across Recordings")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "threshold_across_recordings.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. SUMMARY TABLE
# ============================================================

overall = pd.DataFrame({
    "Statistic": [
        "N recordings",
        "Mean threshold",
        "Median threshold",
        "SD",
        "Minimum",
        "Maximum",
    ],
    "Value": [
        len(df),
        df["Threshold"].mean(),
        df["Threshold"].median(),
        df["Threshold"].std(),
        df["Threshold"].min(),
        df["Threshold"].max(),
    ],
})

print()
print("=" * 60)
print("OVERALL THRESHOLD SUMMARY")
print("=" * 60)

print(overall.to_string(index=False))

overall.to_csv(
    OUTPUT_DIR / "threshold_statistics.csv",
    index=False
)


print()
print(f"Plots saved to: {OUTPUT_DIR}")
