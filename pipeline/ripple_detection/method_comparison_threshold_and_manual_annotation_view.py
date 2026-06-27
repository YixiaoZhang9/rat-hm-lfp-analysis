# ---- Import ----
import os
import re
import sys

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.io import loadmat
from signal_viewer import RippleViewer

from modules.project_config import get_path


class TrialSelector(QWidget):
    def __init__(self, trials_data):
        super().__init__()

        self.setWindowTitle("Select Trial to View")
        self.resize(500, 600)

        self.trials_data = trials_data
        self.viewers = []

        layout = QVBoxLayout()

        # list
        self.list_widget = QListWidget()
        for t in trials_data:
            self.list_widget.addItem(t["name"])

        layout.addWidget(self.list_widget)

        # botton
        btn = QPushButton("Open Selected Trial")
        btn.clicked.connect(self.open_trial)
        layout.addWidget(btn)

        self.setLayout(layout)

        # double click to open
        self.list_widget.itemDoubleClicked.connect(self.open_trial)

    def open_trial(self):
        idx = self.list_widget.currentRow()
        if idx < 0:
            return

        data = self.trials_data[idx]

        viewer = RippleViewer(
            lfp=data["lfp"],
            scoring=data["scoring"],
            fs=data["fs"],
            event_sets=data["event_sets"],
        )

        viewer.show()
        self.viewers.append(viewer)


# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = get_path("R1_8_root")
dir_R1_4_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/PreprocessedData/HPC"
)
dir_R1_4_Scoring = os.path.join(dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/Scoring")
dir_R1_4_Ripple = os.path.join(
    dir_base1,
    "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/Ripple_detection_results",
)
dir_R5_8_Data = os.path.join(
    dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/PreprocessedData/HPC"
)
dir_R5_8_Scoring = os.path.join(dir_base1, "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/Scoring")
dir_R5_8_Ripple = os.path.join(
    dir_base1,
    "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8/Ripple_detection_results",
)

# the path storing the ripple marking results
root_annotation = get_path("ripple_marking_root")
annotator = ["Lisa", "Yixiao"]

rats = [3, 7]
regions = ["HPC", "PL", "RSC"]
sleep_periods = ["presleep", "postsleep"]
trials = {
    "Rat3": ["20221010_trial13", "20221013_trial8"],
    "Rat7": ["20221014_trial11", "20221017_trial12"],
}

fs = 1000  # downsampled sample frequency

trials_data = []
for rat in rats:
    rat_key = f"Rat{rat}"
    if rat == 3:
        threshold_result_path = dir_R1_4_Ripple
        data_path = dir_R1_4_Data
        scoring_path = dir_R1_4_Scoring
    elif rat == 7:
        threshold_result_path = dir_R5_8_Ripple
        data_path = dir_R5_8_Data
        scoring_path = dir_R5_8_Scoring

    for trial_name in trials[rat_key]:
        annotated_ripples = {}
        threshold_ripples = []
        threshold2_ripples = []
        for name in annotator:
            annotation_path = os.path.join(root_annotation, name, rat_key)
            for root, dirs, files in os.walk(annotation_path):
                if trial_name in root:
                    for file in files:
                        if file.endswith("events.csv"):
                            file_path = os.path.join(root, file)
                            print(file_path)
                            # store the results
                            # read ripple start/end times from csv
                            ripple_df = pd.read_csv(file_path)
                            # first column = ripple start time, second column = ripple end time
                            ripples = np.column_stack(
                                [
                                    (ripple_df["start_time"].to_numpy() * 1000).astype(
                                        int
                                    ),
                                    (ripple_df["end_time"].to_numpy() * 1000).astype(
                                        int
                                    ),
                                ]
                            )
                            annotated_ripples[name] = ripples

        # parse date and trial number from trial_name
        date_str, trial_str = trial_name.split("_")
        trial_id = trial_str.replace("trial", "")

        rat_threshold_path = os.path.join(threshold_result_path, str(rat), date_str)

        file_sleep_period = []
        for sleep_period in ["presleep", "postsleep"]:
            sleep_path = os.path.join(rat_threshold_path, sleep_period)

            if not os.path.exists(sleep_path):
                continue

            for ripple_file in os.listdir(sleep_path):
                ripple_file_path = os.path.join(sleep_path, ripple_file)

                if f"_{trial_id}_hippocampal_ripples_threshold." in ripple_file:
                    file_sleep_period = sleep_period
                    print("threshold1:", ripple_file_path)
                    threshold_df = pd.read_csv(ripple_file_path)
                    threshold1_ripples = threshold_df[
                        ["ripple_start", "ripple_end"]
                    ].to_numpy()

                if f"_{trial_id}_hippocampal_ripples_threshold2." in ripple_file:
                    print("threshold2:", ripple_file_path)
                    threshold_df2 = pd.read_csv(ripple_file_path)
                    threshold2_ripples = threshold_df2[
                        ["ripple_start_index", "ripple_end_index"]
                    ].to_numpy()

        # find the corresponding data and sleep scoring results
        dir_data = os.path.join(data_path, str(rat), date_str, file_sleep_period)
        trial_path = os.path.join(
            dir_data,
            next(f for f in os.listdir(dir_data) if f.endswith(f"{trial_id}.mat")),
        )

        # Find scoring file
        dir_scoring = os.path.join(scoring_path, str(rat), date_str, file_sleep_period)
        scoring_files = [f for f in os.listdir(dir_scoring) if f.endswith(".mat")]
        # matched sleep scoring files
        if len(scoring_files) == 1:
            scoring = loadmat(os.path.join(dir_scoring, scoring_files[0]))[
                "states"
            ].squeeze()
        else:
            suffix = trial_id.zfill(2)
            pattern = re.compile(rf"_{suffix}_")

            scoring = None
            for f in scoring_files:
                if pattern.search(f):
                    scoring = loadmat(os.path.join(dir_scoring, f))["states"].squeeze()
                    break

        # ---- Launch viewer ----

        # check if a QApplication already exists
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        event_sets = {
            "Lisa": annotated_ripples["Lisa"],
            "Yixiao": annotated_ripples["Yixiao"],
            "Threshold1": threshold1_ripples,
            "Threshold2": threshold2_ripples,
        }

        trials_data.append(
            {
                "name": f"{rat_key}_{trial_name}_{file_sleep_period}",
                "lfp": loadmat(trial_path)["data"].squeeze(),
                "scoring": scoring,
                "fs": fs,
                "event_sets": event_sets.copy(),
            }
        )

# ---- Launch selector GUI ----
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

selector = TrialSelector(trials_data)
selector.show()

sys.exit(app.exec_())
