import os
import re
import glob
import pandas as pd
import numpy as np
from scipy.io import loadmat
from IPython.display import display
from sklearn.metrics import cohen_kappa_score
from scipy.ndimage import gaussian_filter1d

# Only the trial that the consensus between two annotator is higher than 0.5 for both the event based overlap (IOU) and
# cohen's kappa coefficient
# The following are the trials that match the criteria:
# rat 3: 20221013_triL8; 20221010_trial13
# rat 7: 20221017_trial12 ; 20221014_trial11
# rat 12: 20230731_trial7


# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = '/media/yixiao/GL14_RAT_FA/'
dir_R1_4_Data = os.path.join(dir_base1,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/PreprocessedData/HPC')
dir_R1_4_Scoring = os.path.join(dir_base1,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/Scoring')
dir_R1_4_Ripple = os.path.join(dir_base1,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R1-4/Ripple_detection_results')
dir_R5_8_Data = os.path.join(dir_base1,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/PreprocessedData/HPC')
dir_R5_8_Scoring = os.path.join(dir_base1,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/Scoring')
dir_R5_8_Ripple = os.path.join(dir_base1,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/Ripple_detection_results')
dir_base2 = '/media/yixiao/Data4/'
dir_R9_12_Data = os.path.join(dir_base2,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/PreprocessedData/HPC')
dir_R9_12_Scoring = os.path.join(dir_base2, 'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/Scoring')
dir_R9_12_Ripple = os.path.join(dir_base2,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis/R9-12/Ripple_detection_results')


# the path storing the ripple marking results
root_annotation = '/mnt/genzel/Rat/HM/Rat_HM_Ephys_TD/Rat_HM_Ripple_Detection/Ripple_Marking'
annotator = ["Lisa","Yixiao"]

rats = [3,7,12]
regions = ['HPC','PL','RSC']
sleep_periods = ['presleep','postsleep']
trials = {'Rat3': ['20221010_trial13','20221013_trial8'],
          'Rat7': ['20221014_trial11','20221017_trial12'],
          'Rat12': ['20230731_trial7']}
fs = 1000 # downsampled sample frequency

trial_counter = 1
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

    elif rat == 12:
        threshold_result_path = dir_R9_12_Ripple
        data_path = dir_R9_12_Data
        scoring_path = dir_R9_12_Scoring

    for trial_name in trials[rat_key]:
        annotated_ripples = {}
        for name in annotator:
            annotation_path = os.path.join(root_annotation, name, rat_key)
            for root, dirs, files in os.walk(annotation_path):
                if trial_name in root:
                    for file in files:
                        if file.endswith('events.csv'):
                            file_path = os.path.join(root, file)
                            print(file_path)
                            # store the results
                            # read ripple start/end times from csv
                            ripple_df = pd.read_csv(file_path)
                            # first column = ripple start time, second column = ripple end time
                            ripples = np.column_stack([
                                (ripple_df["start_time"].to_numpy() * 1000).astype(int),
                                (ripple_df["end_time"].to_numpy() * 1000).astype(int)
                            ])
                            annotated_ripples[name] = ripples

        # parse date and trial number from trial_name
        date_str, trial_str = trial_name.split('_')
        trial_id = trial_str.replace('trial', '')

        rat_threshold_path = os.path.join(threshold_result_path, str(rat), date_str)

        file_sleep_period = []
        for sleep_period in ['presleep', 'postsleep']:
            sleep_path = os.path.join(rat_threshold_path, sleep_period)
            for ripple_file in os.listdir(sleep_path):
                ripple_file_path = os.path.join(sleep_path, ripple_file)
                if f'_{trial_id}_hippocampal_ripples.' in ripple_file:
                    file_sleep_period = sleep_period

        # Find scoring file
        dir_scoring = os.path.join(scoring_path, str(rat), date_str, file_sleep_period)
        scoring_files = [f for f in os.listdir(dir_scoring) if f.endswith(".mat")]
        # matched sleep scoring files
        if len(scoring_files) == 1:
            scoring = loadmat(os.path.join(dir_scoring, scoring_files[0]))['states'].squeeze()
        else:
            suffix = trial_id.zfill(2)
            pattern = re.compile(rf'_{suffix}_')

            scoring = None
            for f in scoring_files:
                if pattern.search(f):
                    scoring = loadmat(os.path.join(dir_scoring, f))['states'].squeeze()
                    break

        # find the preprocessed lfp data
        dir_data = os.path.join(data_path, str(rat), date_str, file_sleep_period)
        trial_path = os.path.join(dir_data,next(f for f in os.listdir(dir_data) if f.endswith(f"{trial_id}.mat")))
        preprocessed_data = loadmat(trial_path)["data"].squeeze()

        # remove the annotations that are out of NREM stage
        nrem_label = 3

        for name in annotator:
            filtered_ripples = []
            for r in annotated_ripples[name]:

                start_idx, end_idx = r
                # convert to seconds for scoring alignment
                start_s = int(start_idx // 1000)
                end_s = int(end_idx // 1000)

                start_s = max(start_s, 0)
                end_s = min(end_s, len(scoring) - 1)

                # keep only fully NREM ripples
                if np.all(scoring[start_s:end_s + 1] == nrem_label):
                    filtered_ripples.append(r)

            annotated_ripples[name] = np.array(filtered_ripples)

        # created the weighted consensus for the training of model
        length = len(preprocessed_data)
        trace1 = np.zeros(length, dtype=np.float32)
        trace2 = np.zeros(length, dtype=np.float32)

        for r in annotated_ripples["Lisa"]:
            start_idx, end_idx = r

            start_idx = max(0, int(start_idx))
            end_idx = min(length - 1, int(end_idx))

            trace1[start_idx:end_idx + 1] = 1.0

        for r in annotated_ripples["Yixiao"]:
            start_idx, end_idx = r

            start_idx = max(0, int(start_idx))
            end_idx = min(length - 1, int(end_idx))

            trace2[start_idx:end_idx + 1] = 1.0

        consensus = (trace1 + trace2) / 2.0
        consensus = gaussian_filter1d(consensus, sigma=5)


        # save the data
        save_root = "/home/yixiao/PycharmProjects/Rat_HM/pipeline/ripple_detection/training_data"
        os.makedirs(save_root, exist_ok=True)


        preprocessed_data = preprocessed_data.astype(np.float32)
        consensus = consensus.astype(np.float32)
        scoring = scoring.astype(np.int8)

        file_name = f"trial_{trial_counter:03d}.npz"
        save_path = os.path.join(save_root, file_name)

        np.savez_compressed(
            save_path,
            preprocessed_data=preprocessed_data,
            consensus_trace=consensus,
            scoring=scoring,
            fs=fs,
            rat=rat,
            trial_name=trial_name
        )

        print(f"[Saved] {save_path}")

        trial_counter += 1