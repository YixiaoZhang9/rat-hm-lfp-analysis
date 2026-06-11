import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from modules.project_config import get_path
from modules.threshold_ripple_detection import filter_lfp

# ====================== Basic settings ======================
fs = 1000  # Sampling frequency (Hz)
window_ms = 500  # Time window around ripple (ms)
half_window_samples = int(fs * window_ms / 1000)
n_samples_per_cluster = 10  # Number of example ripples to plot per cluster

# Base directory for data
dir_base1 = get_path("R1_8_root")
dir_R1_4_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/PreprocessedData"
)

# ====================== Load feature data ======================
total_df = pd.read_csv(
    "../../../../archive/Ripple_feature_results/All_Rat1-4_Ripple_Features_17features_test_3.5SD_0.5.csv"
)

# List of features to use for clustering
features = [
    "area",
    "entropy",
    "TW",
    "FW",
    "max_f",
    "duration_ms",
    "peak_env",
    "rms_env",
    "relative_rise_time",
    "relative_fall_time",
    "env_skewness",
    "peak_energy_fraction",
    "peak_to_trough",
    "num_peaks",
    "zero_crossing_rate",
    "good_cycle_ratio",
]  # ,'Fuzz_value'

# Standardize features
X = total_df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====================== UMAP dimensionality reduction ======================
reducer = umap.UMAP(
    n_neighbors=50,  # Controls local vs. global structure preservation
    min_dist=0.01,  # Controls clustering tightness in low-dimensional space
    n_components=3,  # Embed into 3D for visualization
    random_state=42,
)
X_umap = reducer.fit_transform(X_scaled)

# Add UMAP results to dataframe
total_df["UMAP_1"], total_df["UMAP_2"], total_df["UMAP_3"] = X_umap.T

# ====================== Gaussian Mixture Model (GMM) clustering ======================
print("\n=== Running GMM clustering ===")
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
labels = gmm.fit_predict(X_scaled)
total_df["GMM_label"] = labels

# ====================== 3D UMAP visualization ======================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    total_df["UMAP_1"],
    total_df["UMAP_2"],
    total_df["UMAP_3"],
    c=total_df["GMM_label"],
    cmap="Set2",
    s=10,
    alpha=0.8,
)
ax.set_title("GMM Clustering on 3D UMAP Projection", fontsize=12)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_zlabel("UMAP 3")

legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
ax.add_artist(legend)

plt.tight_layout()
plt.show()

# ====================== Plot sample ripples for each GMM cluster ======================
label_col = "GMM_label"
print("\n================ Plotting GMM clusters ================")

for cluster_id in sorted(total_df[label_col].unique()):
    cluster_data = total_df[total_df[label_col] == cluster_id]
    if len(cluster_data) == 0:
        continue

    print(f"\n Plotting Cluster {cluster_id} (total {len(cluster_data)} samples)")
    samples = cluster_data.sample(
        n=min(n_samples_per_cluster, len(cluster_data)), random_state=42
    )

    for _, row in samples.iterrows():
        rat = int(row["rat"])
        region = row["region"]
        studyday = row["studyday"]
        sleep_period = row["sleep_period"]
        trial_name = row["trial"]
        ripple_index = row["ripple_index"]
        peak_idx = int(row["ripple_peak"])
        start_idx = int(row["ripple_start"])
        end_idx = int(row["ripple_end"])

        # Construct full path to .mat file
        mat_path = os.path.join(
            dir_R1_4_Data, region, str(rat), str(studyday), sleep_period, trial_name
        )

        if not os.path.exists(mat_path):
            print(f"File not found: {mat_path}")
            continue

        # Load LFP data
        mat_data = loadmat(mat_path)
        data = mat_data["data"].squeeze()

        # Extract ripple-centered segment
        seg_start = max(0, peak_idx - half_window_samples)
        seg_end = min(len(data), peak_idx + half_window_samples)
        segment = data[seg_start:seg_end]

        # Bandpass filter (100–250 Hz)
        filtered_segment = filter_lfp(segment, fs, [100, 250])

        # Time vector (centered at ripple peak)
        t = (np.arange(seg_start, seg_end) - peak_idx) / fs

        # Plot the raw and filtered LFP
        plt.figure(figsize=(6, 3))
        plt.plot(t, segment, "k", linewidth=1, label="Raw")
        plt.plot(t, filtered_segment, "b", linewidth=1, label="Filtered")
        plt.axvline(0, color="r", linestyle="--", label="Peak")

        # Highlight ripple region
        ripple_start_t = (start_idx - peak_idx) / fs
        ripple_end_t = (end_idx - peak_idx) / fs
        plt.axvspan(
            ripple_start_t, ripple_end_t, color="green", alpha=0.3, label="Ripple"
        )

        # Title and formatting
        plt.title(
            f"GMM - Cluster {cluster_id}\nRat{rat}-{region}-{studyday}-{sleep_period}-{trial_name}-{ripple_index}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (a.u.)")
        plt.xlim(-0.5, 0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
