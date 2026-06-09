# ---- Import ----
import os

import numpy as np
from scipy.io import loadmat

from modules import Function_reduce_powerline_harmonics as powerline
from modules import ephys_preprocessing as processing
from modules.project_config import get_path

"""
This script performs the following preprocessing steps on the dataset:

1) Downsamples the data from 20000Hz to 1000Hz.

2) Removes power line artifacts (50Hz).

3) Segments each trial into 30-minute intervals (mae sure the signal length is the same).

4) Detects and removes artifacts from the segmented data.

5) Saves the cleaned and processed results to the designated results directory.
"""


# ---- Set base paths, date lists, and constants for data processing ----
dir_base = get_path("gl14_root")
dir_R1_4_RawData = os.path.join(
    dir_base, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/RawData"
)

dir_R1_4_PreprocessedData = os.path.join(
    dir_base, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/PreprocessedData"
)
# this folder will store the preprocessed data

rats = np.arange(1, 5)
regions = ["HPC", "PL", "RSC"]
sleep_period = ["presleep", "postsleep"]
fs = 20000  # sampling frequency is 20000Hz
ds_fs = 1000  # target sample frequency

dates_special_handels1 = ["20220923", "20220926", "20220927"]
# On these dates, sleep session recordings are not split into standard 30-minute segments.

dates_special_handels2 = ["20220929", "20221006"]
# On these dates, a single trial may contain multiple recordings in the same directory.


# -------------------------------Preprocesing----------------------------------

for rat in [rats[3]]:  # for rat in rats
    for region in [regions[0]]:  # for region in regions
        dir_R1_4_RawData_perday = os.path.join(dir_R1_4_RawData, region, str(rat))
        folders_SD = [
            name
            for name in os.listdir(dir_R1_4_RawData_perday)
            if os.path.isdir(os.path.join(dir_R1_4_RawData_perday, name))
        ]

        for studyday in [folders_SD[20]]:  # for studyday in folders_SD
            for sleep_period in [sleep_period[1]]:  # for sleep_period in sleep_period
                # find all the trial recordings (suffix = ".mat")
                dir_R1_4_RawData_pertrial = os.path.join(
                    dir_R1_4_RawData_perday, studyday, sleep_period
                )
                matched_files = []
                for root, dirs, files in os.walk(dir_R1_4_RawData_pertrial):
                    for name in files:
                        if name.endswith(".mat"):
                            matched_files.append(os.path.join(root, name))

                # group the trial according to the suffix ("a","b","c" should be the same trial)
                grouped = processing.group_files(matched_files)

                all_trials_processed = {}
                # for trial_name, files in grouped.items():
                for trial_name, files in [
                    list(grouped.items())[0]
                ]:  # for the testing and debuging
                    filtered_segments = []
                    for suffix, filepath in files:
                        trial_data = loadmat(filepath)
                        data = trial_data["data"]
                        # downsampling the data
                        data_ds = processing.downsampling(data, fs, ds_fs)
                        # remove powerline artifacts
                        Powerline_freq = [50, 100, 150, 200, 250, 300, 350, 400, 450]
                        # methods = ["Notch", "DFT", "Adaptive_LMS", "Adaptive_RLS"]
                        methods = ["Adaptive_LMS"]

                        for method in methods:
                            try:
                                data_filtered = processing.powerline_filter(
                                    data_ds,
                                    ds_fs,
                                    Powerline_freq,
                                    Method=method,
                                    plot_data=True,
                                )

                                squeezed_data = data_ds.squeeze()
                                squeezed_filt_data = data_filtered.squeeze()

                                parr, freqs = powerline.calculate_parr(
                                    squeezed_data,
                                    squeezed_filt_data,
                                    ds_fs,
                                    powerline_freq=50,
                                    harmonic_n=9,
                                    bandwidth=2.0,
                                )
                                print(
                                    f"Method: {method} PARR: {parr:.2f} dB at frequencies {freqs}"
                                )

                                ppr, effective_bands = (
                                    powerline.calculate_ppr_specific_band(
                                        squeezed_data,
                                        squeezed_filt_data,
                                        ds_fs,
                                        target_band=(0, 30),
                                    )
                                )
                                print(f"target_band=(0, 30) PPR: {ppr:.2f} dB")

                                ppr, effective_bands = (
                                    powerline.calculate_ppr_specific_band(
                                        squeezed_data,
                                        squeezed_filt_data,
                                        ds_fs,
                                        target_band=(100, 300),
                                        exclude_harmonics=[100, 150, 200, 250, 300],
                                        exclude_bw=0.5,
                                    )
                                )
                                print(f"target_band=(100, 300) PPR: {ppr:.2f} dB")

                                # print(powerline.calculate_sampen_avg(squeezed_data))
                                # print(powerline.calculate_sampen_avg(squeezed_filt_data))

                            except Exception as e:
                                print(f"Method {method} failed: {e}")
                        """
                        data_filtered = processing.powerline_filter(data_ds, ds_fs, Powerline_freq,Method = "Adaptive_RLS",plot_data = True)
                        filtered_segments.append(data_filtered)
                        """

                    # merge the data if one trial is spilt into several recordings bacause of experimental issure
                    if len(filtered_segments) == 1:
                        combined_data = filtered_segments[0]
                    else:
                        combined_data = filtered_segments[0]
                        for seg in filtered_segments[1:]:
                            combined_data = processing.smooth_transition(
                                combined_data, seg, n=100
                            )
                            # avoid abrupt changes

                    # Keep only the first 30 minutes of data
                    combined_data = combined_data.squeeze()
                    max_length = int(30 * 60 * ds_fs)
                    if combined_data.shape[-1] > max_length:
                        combined_data = combined_data[:max_length]

                    all_trials_processed[trial_name] = combined_data
