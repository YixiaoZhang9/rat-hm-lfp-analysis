import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

# Update path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from modules.ephys_preprocessing import bandpass_filter, downsampling
from modules.find_spindles_lfp_o_quality import fit_ar_on_prepared_signal
from modules.project_config import get_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FS = 1000
TARGET_FS = 128
INSPECTION_SEC = 45  # 45-second window is ideal for visual inspection

def get_nrem_intervals(scoring_path):
    states = loadmat(scoring_path)["states"].squeeze()
    nrem_mask = (states == 3).astype(int)
    diff = np.diff(np.concatenate(([0], nrem_mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return np.empty((0, 2))
    return np.column_stack((starts, ends))

def get_random_nrem_segment(raw_signal, nrem_intervals, fs, target_sec):
    chunks = []
    for start, end in nrem_intervals:
        s_idx, e_idx = int(start * fs), int(end * fs)
        chunks.append(raw_signal[s_idx:e_idx])

    if not chunks:
        return np.array([])

    pooled_signal = np.concatenate(chunks)
    target_samples = target_sec * fs

    if len(pooled_signal) <= target_samples:
        logging.warning("Total NREM duration is less than the target segment length.")
        return pooled_signal

    start_idx = np.random.randint(0, len(pooled_signal) - target_samples)
    return pooled_signal[start_idx : start_idx + target_samples]

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def interactive_visual_threshold(raw_chunk, fs, target_fs, r_timeseries, window_sec=1.0):
    sigma_filtered = butter_bandpass_filter(raw_chunk, 10, 15, fs)

    t_raw = np.arange(len(raw_chunk)) / fs

    window_samples = int(window_sec * target_fs)
    t_r = (np.arange(len(r_timeseries)) + (window_samples / 2)) / target_fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    plt.subplots_adjust(bottom=0.2)

    axes[0].plot(t_raw, raw_chunk, color='black', linewidth=0.8)
    axes[0].set_ylabel('Raw Signal')
    axes[0].set_title('Interactive Spindle Threshold Selection')

    axes[1].plot(t_raw, sigma_filtered, color='blue', linewidth=1)
    axes[1].set_ylabel('10-15 Hz Filtered')

    axes[2].plot(t_r, r_timeseries, color='red', linewidth=1.5)
    axes[2].set_ylabel('Damping (r-value)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylim(0.7, 1.0)

    init_threshold = 0.8
    hline = axes[2].axhline(init_threshold, color='green', linestyle='--', linewidth=2)

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    threshold_slider = Slider(
        ax=ax_slider,
        label='Threshold (rb)',
        valmin=0.70,
        valmax=0.90,
        valinit=init_threshold,
        valstep=0.005
    )

    def update(val):
        hline.set_ydata([val, val])
        fig.canvas.draw_idle()

    threshold_slider.on_changed(update)
    plt.show()

def run_visual_calibration():
    dir_base = get_path("R1_8_root")
    data_path = os.path.join(
        dir_base, "R1-4/PreprocessedData/HPC/1/20221006/postsleep/chan102_9.mat"
    )
    scoring_path = os.path.join(
        dir_base,
        "R1-4/Scoring/1/20221006/postsleep/"
        "Rat_HM_Ephys_TD_Rat1_20221006_postsleep_09_SW-eegstates.mat",
    )

    logging.info("Loading signal and scoring data...")
    raw_signal = loadmat(data_path)["data"].squeeze()
    nrem_intervals = get_nrem_intervals(scoring_path)

    logging.info(f"Extracting a {INSPECTION_SEC}-second random NREM chunk for visualization...")
    inspection_raw = get_random_nrem_segment(raw_signal, nrem_intervals, FS, INSPECTION_SEC)

    if len(inspection_raw) == 0:
        logging.error("Failed to extract NREM data.")
        return

    logging.info("Preprocessing and downsampling...")
    filtered = bandpass_filter(inspection_raw, lowcut=0.1, highcut=100, fs=FS)
    pooled_128 = downsampling(filtered, FS, TARGET_FS)

    logging.info("Fitting AR model to compute r-values...")
    r_real, f_real = fit_ar_on_prepared_signal(pooled_128, TARGET_FS, n_jobs=-1, verbose=False)

    logging.info("Launching interactive visualizer...")
    interactive_visual_threshold(inspection_raw, FS, TARGET_FS, r_real, window_sec=1.0)

if __name__ == "__main__":
    run_visual_calibration()
