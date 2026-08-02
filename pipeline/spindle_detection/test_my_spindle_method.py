import numpy as np
import os
import sys
from scipy.io import loadmat

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.find_spindles_lfp_o_quality import find_spindles_lfp
from modules.project_config import get_path

# Constants
FS = 1000  # Original sampling frequency

def find_first_trial_path():
    dir_base = get_path("R1_8_root")
    # Path: /Volumes/genzel/Rat/HM/Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_R1_8/R1-4/PreprocessedData/HPC
    # Target: rat 1, date 20221006, postsleep, chan102_9.mat
    hpc_data = os.path.join(dir_base, "R1-4/PreprocessedData/HPC/1/20221006/postsleep")
    file_name = "chan102_9.mat"
    target_path = os.path.join(hpc_data, file_name)
    
    print(f"Targeting: {target_path}")
    
    if os.path.exists(target_path):
        return target_path
    else:
        print(f"File not found at: {target_path}")
        return None

def run_test():
    file_path = find_first_trial_path()
    if not file_path:
        print("No .mat files found in the preprocessed data directory.")
        return

    # 2. Load data
    print(f"Loading {file_path}...")
    try:
        data_dict = loadmat(file_path)
        raw_signal = data_dict['data'].squeeze()
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 3. Run your new spindle detection
    print("Running spindle detection...")
    spindles = find_spindles_lfp(raw_signal, fs=FS)

    print(f"Detected {len(spindles)} spindles.")
    if len(spindles) > 0:
        print("First 5 detected spindles (start_sec, peak_sec, end_sec, duration, max_r, freq):")
        print(spindles[:5])

if __name__ == "__main__":
    run_test()
