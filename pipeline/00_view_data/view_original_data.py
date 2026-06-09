# ---- Import ----
import os
import re
import sys

import matplotlib
import numpy as np
from PyQt5.QtWidgets import QApplication

from modules import OpenEphys
from modules import ephys_data_io as fstore
from modules import ephys_preprocessing as processing
from modules.ephys_signal_view import SignalPlotViewer
from modules.project_config import get_path

matplotlib.use("Qt5Agg")
from scipy.signal import filtfilt, iirnotch

# ---- Set base paths, date lists, and constants for data processing ----
dir_base = get_path("RAT_HM_DATA4_ROOT")
dir_R13_16_Data = os.path.join(
    dir_base, "Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_OpenEphysRecordings_R13-16/"
)
rats = np.arange(13, 17)
date_prefix = "Rat_HM_Ephys_TD_R13-16_"
study_day = "20230905"
dates_fullname = f"{date_prefix}{study_day}"
regions = ["HPC", "PL", "RSC"]
sleep_period = ["presleep", "postsleep"]
trial_suffix = "13"
channels = [57]
plot_data = 1
fs = 20000
ds_fs = 1000
data_filtered = {}

# ---- load and plot the ephys recordings ----

for rat in [rats[2]]:
    for region in [regions[1]]:
        for channel in channels:
            print(
                f"\nLoading Ephys Recordings on {study_day} | Rat {rat} | Region {region}"
            )

            path_recording = os.path.join(dir_R13_16_Data, dates_fullname)

            matched_dir = fstore.find_path_by_suffix(
                path_recording, trial_suffix, search_dir=True
            )

            pattern = re.compile(rf"(?<![A-Z0-9])(?:CH)?{channel}\.continuous$")
            str_matched_file = fstore.find_path_by_suffix(
                matched_dir, pattern, search_dir=False, use_regex=True
            )
            # read recording
            Recording = OpenEphys.load(str_matched_file)
            signal = Recording["data"]
            data_ds = processing.downsampling(signal, fs, ds_fs, plot_response=False)
            q = 30  # Quality factor
            b_notch, a_notch = iirnotch(50, q, ds_fs)
            data_filtered[channel] = filtfilt(b_notch, a_notch, data_ds)

            # remove powerline artifacts
            # Powerline_freq = [50, 100, 150, 200, 250, 300, 350, 400, 450]
            #
            # data_filtered[channel] = processing.powerline_filter(data_ds, ds_fs, Powerline_freq,
            #                                             Method="Adaptive_RLS", plot_data=False)

        # Optional: plot signal of before and after artifact removal
        if plot_data:
            app_created = False
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
                app_created = True

            # create dic
            data_dict = {
                # "Signal before artifacts removal": squeezed_data,
                # "scoring":upsampled_scoring_data
                "Original Signal1 ": data_filtered[channels[0]],
                # "Original Signal2 ": data_filtered[channels[1]],
                # "Original Signal3 ": data_filtered[channels[2]],
                # "Original Signal4 ": data_filtered[channels[3]],
                # "Original Signal5 ": data_filtered[channels[4]],
                # "Original Signal6 ": data_filtered[channels[5]],
                # "Original Signal7 ": data_filtered[channels[6]],
                # "Original Signal8 ": data_filtered[channels[7]],
            }

            window = SignalPlotViewer(data_dict, ds_fs, window_sec=5)
            window.show()

            if app_created:
                app.exec()

            a = 1
