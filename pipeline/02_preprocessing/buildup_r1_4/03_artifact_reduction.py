# ---- Import ----
import os
import re
import sys

# from Signal_scoring_Plot_pyqt import SignalPlotViewer
import matplotlib
import numpy as np
from PyQt5.QtWidgets import QApplication
from scipy.io import loadmat, savemat

from modules import ephys_preprocessing as processing
from modules.ephys_signal_view import SignalPlotViewer
from modules.project_config import get_path

matplotlib.use("Qt5Agg")


"""
This script performs the following preprocessing steps on the dataset:

1) Reduce the artifacts.

2) Saves the preprocessed results to the designated results directory.
"""

# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = get_path("R1_8_root")
dir_R1_4_filteredData = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/Preprec_withartifacts"
)
dir_R1_4_Scoring = os.path.join(dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/Scoring")

# this folder will store the preprocessed data and
dir_R1_4_PreprocessedData = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/PreprocessedData"
)

rats = np.arange(1, 5)
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
ds_fs = 1000  # downsampled frequency

dates_special_handels = ["20220929", "20221003", "20221006", "20221021", "20221103"]
# On these dates, sleep session recordings are shorter than 30-minute segments.

plot_data = 0  # plot the signal


# -------------------------------Preprocesing-----------------------------------------
for rat in rats:  # for rat in [rats[0]]   for rat in rats
    for region in regions:  # for region in [regions[0]] for region in regions
        dir_R1_4_filteredData_perday = os.path.join(
            dir_R1_4_filteredData, region, str(rat)
        )
        folders_SD = [
            name
            for name in os.listdir(dir_R1_4_filteredData_perday)
            if os.path.isdir(os.path.join(dir_R1_4_filteredData_perday, name))
        ]

        for (
            studyday
        ) in folders_SD:  # for studyday in [folders_SD[0]] for studyday in folders_SD
            for sleep_period in sleep_periods:  # for sleep_period in sleep_periods
                # find all the trial recordings (suffix = ".mat")
                dir_R1_4_RawData_pertrial = os.path.join(
                    dir_R1_4_filteredData_perday, studyday, sleep_period
                )
                All_files = []
                for root, dirs, files in os.walk(dir_R1_4_RawData_pertrial):
                    for name in files:
                        if name.endswith(".mat"):
                            All_files.append(os.path.join(root, name))

                # artifact removal
                for file in All_files:
                    trial_data = loadmat(file)
                    data = trial_data["data"]
                    squeezed_data = data.squeeze()

                    # matched scoring results
                    path_scoring = os.path.join(
                        dir_R1_4_Scoring, str(rat), str(studyday), sleep_period
                    )

                    match = re.search(r"(\d+)\.mat$", file)
                    if len(os.listdir(path_scoring)) == 1:
                        trial_scoring_data = loadmat(
                            os.path.join(path_scoring, os.listdir(path_scoring)[0])
                        )
                        scoring_data = trial_scoring_data["states"].squeeze()

                    else:
                        if match:
                            suffix = match.group(1)
                        suffix = suffix.zfill(2)  # from '6' to '06'
                        pattern = re.compile(rf"_{suffix}_")  # '_06_'
                        for f in os.listdir(path_scoring):
                            if pattern.search(f):
                                trial_scoring_data = loadmat(
                                    os.path.join(path_scoring, f)
                                )
                                scoring_data = trial_scoring_data["states"].squeeze()
                                f_scoring = f
                                break

                    # artifacts = processing.find_artifact_zscore(squeezed_data,threshold = 10, fs = ds_fs)
                    # clean_signal = processing.remove_artifacts_by_interpolation(squeezed_data, artifacts,fs = ds_fs)
                    # clean_signal, artifacts = processing.remove_artifacts_eemd_ica_segmented(squeezed_data, ds_fs)

                    clean_signal = processing.artifact_removal_wavelet(squeezed_data)

                    save_dir = os.path.join(
                        dir_R1_4_PreprocessedData,
                        region,
                        str(rat),
                        studyday,
                        sleep_period,
                    )
                    trial_name = os.path.basename(file)
                    os.makedirs(save_dir, exist_ok=True)
                    savemat(os.path.join(save_dir, trial_name), {"data": clean_signal})
                    print(f"save the preprocessed data to {save_dir}")

                    # Optional: plot signal of before and after artifact removal
                    if plot_data:
                        app_created = False
                        app = QApplication.instance()
                        if app is None:
                            app = QApplication(sys.argv)
                            app_created = True

                        upsampled_scoring_data = np.repeat(scoring_data, ds_fs)
                        # create dic
                        data_dict = {
                            # "Signal before artifacts removal": squeezed_data,
                            # "scoring":upsampled_scoring_data
                            "Signal before artifacts removal": squeezed_data,
                            "Signal after artifacts removal": clean_signal,
                        }

                        window = SignalPlotViewer(data_dict, ds_fs, window_sec=5)
                        # window.set_event_intervals(artifacts)
                        window.show()

                        if app_created:
                            app.exec()

                        a = 1
