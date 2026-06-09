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

# ======================= Load Data ==========================
total_df = pd.read_csv("../../../../results/All_Rat9-12_Ripple_Features_17features.csv")

# Select features for clustering
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

X = total_df[features].copy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================================================
# 3. UMAP Dimensionality Reduction (for clustering + visualization)
# ===============================================================
print("\n=== Performing UMAP dimensionality reduction ===")
reducer = umap.UMAP(
    n_neighbors=50,  # Controls local vs. global structure
    min_dist=0.01,  # Controls clustering tightness
    n_components=3,  # Reduce to 3D
    random_state=42,
)
X_umap = reducer.fit_transform(X_scaled)

# Add UMAP results to dataframe
total_df["UMAP_1"], total_df["UMAP_2"], total_df["UMAP_3"] = X_umap.T

# ===============================================================
# 4. GMM Clustering on UMAP-Reduced Data
# ===============================================================
print("\n=== Running Gaussian Mixture Model (GMM) on UMAP embedding ===")
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
labels = gmm.fit_predict(X_umap)
total_df["GMM_label"] = labels

# ===============================================================
# 5. 3D Visualization of UMAP + GMM Clusters
# ===============================================================
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
ax.set_title("GMM Clustering on 3D UMAP Embedding", fontsize=12)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_zlabel("UMAP 3")

legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
ax.add_artist(legend)

plt.tight_layout()
plt.show()


# %%
# -------------- plot the ripple --------------------
dir_base1 = get_path("R9_16_root")
dir_R9_12_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/PreprocessedData"
)
fs = 1000
window_ms = 500
half_window_samples = int(fs * window_ms / 1000)

n_samples_per_cluster = 10

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
            dir_R9_12_Data, region, str(rat), str(studyday), sleep_period, trial_name
        )

        if not os.path.exists(mat_path):
            print(f"⚠️ File not found: {mat_path}")
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

        # Plot raw and filtered signals
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
            f"GMM - Cluster {cluster_id}\n"
            f"Rat{rat}-{region}-{studyday}-{sleep_period}-{trial_name}-{ripple_index}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (a.u.)")
        plt.xlim(-0.5, 0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
