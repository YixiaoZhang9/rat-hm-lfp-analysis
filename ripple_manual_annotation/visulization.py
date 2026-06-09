import glob
import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QFileDialog
from Ripple_marking_viewer import RippleViewer, load_events
from scipy.io import loadmat

# -------------------------------
# Config
# -------------------------------
root = "/mnt/genzel/Rat/HM/Rat_HM_Ephys_TD/Rat_HM_Ripple_Detection/Ripple_Marking"
annotators = ["Anumita", "Kjell", "Lisa", "Sachuriga", "Yixiao"]
main_annotator = "Yixiao"
other_annotators = [a for a in annotators if a != main_annotator]
fs = 1000  # sampling rate

# -------------------------------
# Launch QApplication
# -------------------------------
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

# -------------------------------
# Open file dialog to select a trial folder
# -------------------------------
trial_folder_path = QFileDialog.getExistingDirectory(None, "Select trial folder", root)

if not trial_folder_path:
    print("No folder selected. Exiting.")
    sys.exit()

trial_folder = Path(trial_folder_path)
rat_key = trial_folder.parts[-2]  # folder structure assumed: root/annotator/RatX/trial
trial_name = trial_folder.name


# Load LFP (from main annotator)
mat_files = glob.glob(
    os.path.join(root, main_annotator, rat_key, trial_name, "chan*.mat")
)
if len(mat_files) == 0:
    print(f"No 'chan*.mat' file found in {trial_folder}")
    sys.exit()

lfp_mat = loadmat(mat_files[0])

if "data" in lfp_mat:
    lfp = lfp_mat["data"].squeeze()
else:
    print(f"No 'data' variable in {mat_files[0]}")
    sys.exit()

# -------------------------------
# Load sleep scoring
# -------------------------------
sleep_scoring_list = glob.glob(
    os.path.join(root, main_annotator, rat_key, trial_name, "*eegstates.mat")
)
if len(sleep_scoring_list) == 0:
    print(f"No sleep scoring file found in {trial_folder}")
    sys.exit()

scoring = loadmat(sleep_scoring_list[0])["states"].squeeze()

# -------------------------------
# Load ripple events for main annotator + one other
# -------------------------------
events_dict = {}
# main annotator
csv_main = glob.glob(os.path.join(root, main_annotator, rat_key, trial_name, "*.csv"))
if csv_main:
    events_dict[main_annotator] = load_events(csv_main[0])

# pick first other annotator that exists
for other in other_annotators:
    csv_other = glob.glob(os.path.join(root, other, rat_key, trial_name, "*.csv"))
    if csv_other:
        events_dict[other] = load_events(csv_other[0])
        break

if not events_dict:
    print("No ripple CSVs found for this trial.")
    sys.exit()

# -------------------------------
# Launch RippleViewer
# -------------------------------
viewer = RippleViewer(lfp=lfp, scoring=scoring, fs=fs, events_dict=events_dict)
viewer.show()
sys.exit(app.exec_())
