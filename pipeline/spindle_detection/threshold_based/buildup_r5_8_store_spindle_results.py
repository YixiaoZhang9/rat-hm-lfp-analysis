# ---- Import ----
import re
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
import sys
from modules.ephys_signal_scoring_view import SignalPlotViewer
import pandas as pd
from scipy.signal import iirnotch,butter, filtfilt, hilbert,convolve, get_window
from modules.find_spindles_lfp_envelope import find_spindles_lfp_envelope

def filter_lfp(lfp, fs,freq_range):
    # Bandpass filter
    b, a = butter(4, np.array(freq_range)/(fs/2), btype='band')
    return filtfilt(b, a, lfp)

def find_bouts(scoring_data, target_value=3, fs=1000):
    """
    Find all continuous segments (bouts) in scoring_data where the value equals target_value.

    Parameters
    ----------
    scoring_data : array-like
        1D array of sleep scoring values (e.g., 1=Wake, 3=NonREM, 5=REM, 4=intermediate)
    target_value : int or float, optional
        The state value to detect bouts for NREM sleep (default = 3)
    fs : float, optional
        Sampling frequency in Hz (default = 1000)

    Returns
    -------
    bouts : list of tuples
        List of (start_idx, end_idx, start_time, end_time), where:
            start_idx / end_idx are sample indices
            start_time / end_time are times in seconds
    """
    scoring_data = np.array(scoring_data)
    is_target = (scoring_data == target_value).astype(int)
    diff = np.diff(is_target, prepend=0, append=0)

    # 1 → bout starts, -1 → bout ends
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    bouts = [(s*fs, e*fs) for s, e in zip(starts, ends)]
    return bouts

def find_non_nrem_bouts(scoring_data, nrem_value=3, fs=1000):

    scoring_data = np.array(scoring_data)

    is_target = (scoring_data != nrem_value).astype(int)

    diff = np.diff(is_target, prepend=0, append=0)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    bouts = [(s*fs, e*fs) for s, e in zip(starts, ends)]

    return bouts

# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = '/media/yixiao/GL14_RAT_FA/'
dir_R5_8_Data = os.path.join(dir_base1,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/PreprocessedData')
dir_R5_8_Scoring = os.path.join(dir_base1,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/Scoring')
dir_output = os.path.join(dir_base1,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/Spindle_detection_results')


rats = np.arange(5,9)
regions = ['HPC','PL','RSC']
sleep_periods = ['presleep','postsleep']
fs = 1000 # downsampled sample frequency

f_plot = 0
Threshold_across_time = {}
NREM_duration_across_time = {}


#%%

# parameter grid
threshold_grid = [1, 1.1, 1.3, 1.5]
minpeak_grid = [2, 2.5, 3]


# detect spindles
for rat in rats:
    rat_str = f'rat{rat}'

    for region in regions:
        print(f"\n===== Processing {rat_str} region :{region} =====")
        dir_R5_8_Data_perday = os.path.join(dir_R5_8_Data, region, str(rat))

        if not os.path.exists(dir_R5_8_Data_perday):
            print(f"Missing: {dir_R5_8_Data_perday}")
            continue

        folders_SD = [name for name in os.listdir(dir_R5_8_Data_perday)
                      if os.path.isdir(os.path.join(dir_R5_8_Data_perday, name))]

        for studyday in folders_SD:

            for sleep_period in sleep_periods:

                print(f"Processing {rat_str} | {studyday} | {sleep_period}")

                dir_trial = os.path.join(dir_R5_8_Data_perday, studyday, sleep_period)

                if not os.path.exists(dir_trial):
                    print(f"Missing: {dir_trial}")
                    continue

                # find all trial files
                matched_files = []
                for root, _, files in os.walk(dir_trial):
                    for name in files:
                        if name.endswith(".mat"):
                            matched_files.append(os.path.join(root, name))

                if len(matched_files) == 0:
                    print("No trial files found")
                    continue

                # load sleep scoring files
                path_scoring = os.path.join(dir_R5_8_Scoring, str(rat), str(studyday), sleep_period)

                if not os.path.exists(path_scoring):
                    print(f"Missing scoring: {path_scoring}")
                    continue

                scoring_files = [f for f in os.listdir(path_scoring) if f.endswith(".mat")]

                # process each trial
                for trial_name in matched_files:

                    print(f"  Trial: {os.path.basename(trial_name)}")

                    trial_data = loadmat(trial_name)
                    data = trial_data['data'].squeeze()

                    # matched sleep scoring files
                    if len(scoring_files) == 1:
                        scoring_data = loadmat(os.path.join(path_scoring, scoring_files[0]))['states'].squeeze()
                    else:
                        match = re.search(r'_(\d+)\.mat$', trial_name)
                        if not match:
                            print("No matching scoring file")
                            continue

                        suffix = match.group(1).zfill(2)
                        pattern = re.compile(rf'_{suffix}_')

                        scoring_data = None
                        for f in scoring_files:
                            if pattern.search(f):
                                scoring_data = loadmat(os.path.join(path_scoring, f))['states'].squeeze()
                                break

                        if scoring_data is None:
                            print("No scoring matched")
                            continue


                    # find NREM bouts
                    NREM_bouts = find_bouts(scoring_data, target_value=3, fs=fs)
                    # find non-NREM bouts
                    nonNREM_bouts = find_non_nrem_bouts(scoring_data, nrem_value=3, fs=fs)

                    # filter
                    filtered_data = filter_lfp(data, fs, [9, 20])

                    # grid search
                    for threshold in threshold_grid:

                        for minpeak in minpeak_grid:

                            print(f"threshold={threshold} | "f"minpeak={minpeak}")

                            # storage
                            spindle_start_list = []
                            spindle_peak_list = []
                            spindle_end_list = []
                            spindle_duration_list = []
                            spindle_amplitude_list = []
                            spindle_peak_frequency_list = []
                            spindle_mean_frequency_list = []
                            nrem_duration_list = []

                            # process each NREM bout
                            for (start_sample, end_sample) in NREM_bouts:

                                bout_duration_sec = (end_sample - start_sample) / fs

                                bout_data = filtered_data[start_sample:end_sample]

                                bout_raw_data = data[start_sample:end_sample]

                                # spindle detection
                                spindles = find_spindles_lfp_envelope(bout_raw_data, bout_data, fs, threshold=threshold,
                                                                      minpeak=minpeak)

                                if len(spindles) == 0:
                                    continue

                                spindle_list = list(zip(
                                    (spindles[:, 0] + start_sample).astype(int),
                                    (spindles[:, 1] + start_sample).astype(int),
                                    (spindles[:, 2] + start_sample).astype(int)
                                ))

                                duration = spindles[:, 3]
                                amplitude = spindles[:, 4]
                                peak_freq = spindles[:, 5]
                                mean_freq = spindles[:, 6]

                                for i, spindle in enumerate(spindle_list):
                                    start = spindle[0]
                                    peak = spindle[1]
                                    end = spindle[2]

                                    spindle_start_list.append(start)
                                    spindle_peak_list.append(peak)
                                    spindle_end_list.append(end)

                                    spindle_duration_list.append(duration[i])
                                    spindle_amplitude_list.append(amplitude[i])

                                    spindle_peak_frequency_list.append(peak_freq[i])
                                    spindle_mean_frequency_list.append(mean_freq[i])

                                    nrem_duration_list.append(bout_duration_sec)

                            # SAVE CSV

                            if len(spindle_start_list) > 0:

                                df = pd.DataFrame({
                                    "spindle_start_index": spindle_start_list,
                                    "spindle_peak_index": spindle_peak_list,
                                    "spindle_end_index": spindle_end_list,

                                    "spindle_duration_s": spindle_duration_list,

                                    "spindle_amplitude": spindle_amplitude_list,

                                    "spindle_peak_frequency_hz": spindle_peak_frequency_list,

                                    "spindle_mean_frequency_hz": spindle_mean_frequency_list,

                                    "nrem_bout_duration_s": nrem_duration_list
                                })

                            else:

                                df = pd.DataFrame(columns=[
                                    "spindle_start_index",
                                    "spindle_peak_index",
                                    "spindle_end_index",
                                    "spindle_duration_s",
                                    "spindle_amplitude",
                                    "spindle_peak_frequency_hz",
                                    "spindle_mean_frequency_hz",
                                    "nrem_bout_duration_s"
                                ])

                            # save path
                            output_dir = os.path.join(dir_output, region, str(rat),
                                                      studyday, sleep_period,
                                                      f"envelop_thr_{threshold}_peak_{minpeak}"
                                                      )

                            os.makedirs(output_dir, exist_ok=True)

                            trial_id = os.path.basename(trial_name).replace(".mat", "")

                            save_path = os.path.join(output_dir, f"{trial_id}_spindles.csv")

                            df.to_csv(save_path, index=False)

                            print(f"Saved: " f"{len(df)} spindles -> {save_path}")

print("\nDone!")

#%%




