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
total_df = pd.read_csv(
    "../../../../results/All_Rat5-8_Ripple_Features_17features_test.csv"
)

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
    "Fuzz_value",
]
# features = ['area', 'entropy', 'TW', 'FW','Fuzz_value','short_time_energy','spectro_centroid']
X = total_df[features].copy()
valid_idx = X.index

# ======================= Standardize ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

umap_reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, n_components=3, random_state=42)

X_umap = umap_reducer.fit_transform(X_scaled)

total_df[["UMAP1", "UMAP2", "UMAP3"]] = X_umap

# for n in range(2, 7):
#     gmm_test = GaussianMixture(n_components=n, covariance_type='full')
#     labels = gmm_test.fit_predict(X_umap)
#     sil = silhouette_score(X_umap, labels)
#     print(f"{n} clusters → silhouette = {sil:.4f}")
#
# n_components = range(1, 31)
# bics = []
# aics = []
#
# for n in n_components:
#     gmm_test = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
#     gmm_test.fit(X_umap)
#     bics.append(gmm_test.bic(X_umap))
#     aics.append(gmm_test.aic(X_umap))
#
# plt.plot(n_components, bics, label='BIC')
# plt.plot(n_components, aics, label='AIC')
# plt.xlabel("n_components")
# plt.ylabel("Score")
# plt.legend()
# plt.show()

gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
gmm.fit(X_umap)

labels = gmm.predict(X_umap)
probs = gmm.predict_proba(X_umap)

total_df["cluster_label"] = labels
total_df["prob_ripple"] = probs[:, np.argmax(gmm.means_[:, 0])]

means = pd.DataFrame(gmm.means_, columns=["UMAP1", "UMAP2", "UMAP3"])
print("Cluster means:\n", means)

likely_ripple_cluster = means.mean(axis=1).idxmin()
total_df["is_ripple"] = (labels == likely_ripple_cluster).astype(int)

print(f"→ Cluster {likely_ripple_cluster} is Ripple")


# -------------- plot the ripple --------------------
dir_base1 = get_path("R1_8_root")
dir_R1_4_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/PreprocessedData"
)
fs = 1000
window_ms = 500
half_window_samples = int(fs * window_ms / 1000)

n_samples_per_cluster = 10

for cluster_id in sorted(total_df["cluster_label"].unique()):
    cluster_data = total_df[total_df["cluster_label"] == cluster_id]

    print(f"\nPlot sample of Cluster {cluster_id} (total {len(cluster_data)})")
    samples = cluster_data.sample(n=min(n_samples_per_cluster, len(cluster_data)))

    for idx, row in samples.iterrows():
        rat = int(row["rat"])
        region = row["region"]
        studyday = row["studyday"]
        sleep_period = row["sleep_period"]
        trial_name = row["trial"]
        ripple_index = row["ripple_index"]
        peak_idx = int(row["ripple_peak"])
        start_idx = int(row["ripple_start"])
        end_idx = int(row["ripple_end"])

        mat_path = os.path.join(
            dir_R1_4_Data, region, str(rat), str(studyday), sleep_period, trial_name
        )

        if not os.path.exists(mat_path):
            print(f" File not exist: {mat_path}")
            continue

        mat_data = loadmat(mat_path)
        data = mat_data["data"].squeeze()

        # --- ±500ms window ---
        seg_start = max(0, peak_idx - half_window_samples)
        seg_end = min(len(data), peak_idx + half_window_samples)
        segment = data[seg_start:seg_end]
        filtered_segment = filter_lfp(segment, fs, [100, 250])
        t = (np.arange(seg_start, seg_end) - peak_idx) / fs

        plt.figure(figsize=(6, 3))
        plt.plot(t, segment, "k", linewidth=1, label="Raw")
        plt.plot(t, filtered_segment, "b", linewidth=1, label="Filtered")
        plt.axvline(0, color="r", linestyle="--", label="Peak")

        ripple_start_t = (start_idx - peak_idx) / fs
        ripple_end_t = (end_idx - peak_idx) / fs
        plt.axvspan(
            ripple_start_t, ripple_end_t, color="green", alpha=0.3, label="Ripple"
        )

        plt.title(
            f"Rat{rat}-{region}-{studyday}-{sleep_period}-{trial_name}-{ripple_index}\nCluster {cluster_id}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (a.u.)")
        plt.xlim(-0.5, 0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ------------------- 3D plot in UMAP space ---------------------
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    total_df["UMAP1"],
    total_df["UMAP2"],
    total_df["UMAP3"],
    c=labels,
    cmap="Set2",
    alpha=0.6,
)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_zlabel("UMAP3")
ax.set_title("UMAP 3D Embedding + GMM Clustering")
plt.show()
