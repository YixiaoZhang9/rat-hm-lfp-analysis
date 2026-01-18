# ---- Import ----
import re
import os
import numpy as np
from scipy.io import loadmat
import math
from scipy.io import savemat


'''
In some study days, the duration of the recordings and the 
associated scoring files are shorter than 30min and they do not match. To ensure
consistency, the script aligns the length of each sleep scoring file with its
corresponding recording before saving the results.
'''

# ---- Set base paths, date lists, and constants for data processing ----
dir_base1 = '/media/yixiao/GL14_RAT_FA/'

dir_R5_8_Data = os.path.join(dir_base1,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/Preprec_withartifacts')
dir_R5_8_Scoring = os.path.join(dir_base1,'Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_New/R5-8/Scoring')

rats = np.arange(5,9)
regions = ['HPC','PL','RSC']
sleep_periods = ['presleep','postsleep']

fs = 1000 # downsampled sample frequency

dates = ['20221005','20221007','20221102','20221108','20221123','20221207']
# On these dates, the duration of some recordings are shorter than 30 min

expected_lengths = [
    2 * 30 * 60 * fs,
    30 * 60 * fs,
    8 * 30 * 60 * fs
]

#------------------------------------------------------------------------
for rat in rats:  #for rat in [rats[0]]   rat in rats

    for region in regions: #for region in [regions[0]]  region in regions

        dir_R5_8_RawData_perday = os.path.join(dir_R5_8_Data, region, str(rat))
        folders_SD = [name for name in os.listdir(dir_R5_8_RawData_perday)
                      if os.path.isdir(os.path.join(dir_R5_8_RawData_perday, name))]

        for studyday in folders_SD: #for studyday in [folders_SD[0]] studyday in folders_SD

            if studyday in dates:

                for sleep_period in sleep_periods:  #for sleep_period in sleep_periods
                    dir_R5_8_RawData_pertrial = os.path.join(dir_R5_8_RawData_perday, studyday, sleep_period)
                    matched_files = []
                    for root, dirs, files in os.walk(dir_R5_8_RawData_pertrial):
                        for name in files:
                            if name.endswith(".mat"):
                                matched_files.append(os.path.join(root, name))

                    for file in matched_files:
                        trial_data = loadmat(file)
                        data = trial_data['data'].squeeze()
                        if len(data) not in expected_lengths:
                            match = re.search(r'(\d+)\.mat$', file)
                            if match:
                                suffix = match.group(1)  # '6'
                            print(f"Data length in {file} doesn't match any expected duration")
                            path_scoring  = os.path.join(dir_R5_8_Scoring,str(rat),str(studyday),sleep_period)
                            suffix = suffix.zfill(2) # from '6' to '06'
                            pattern = re.compile(rf'_{suffix}_')  # '_06_'
                            for f in os.listdir(path_scoring):
                                if pattern.search(f):
                                    trial_scoring_data = loadmat(os.path.join(path_scoring ,f))
                                    scoring_data = trial_scoring_data['states'].squeeze()
                                    f_scoring  = f
                                    break


                             # cut data and scoring results and save the file
                            if len(data) > len(scoring_data)*fs:
                                pad_len = math.floor(len(data) / fs) - len(scoring_data)
                                scoring_data_new = np.concatenate([scoring_data, np.full(pad_len, scoring_data[-1])])
                                data_new  = data[:len(scoring_data_new)*fs]
                                savemat(file, {'data': data_new})
                                savemat(os.path.join(path_scoring, f_scoring), {'states': scoring_data_new})
                            elif len(data) < len(scoring_data)*fs:
                                scoring_data_new = scoring_data[:math.floor(len(data)/fs)]
                                data_new = data[:len(scoring_data_new) * fs]
                                savemat(file, {'data': data_new})
                                savemat(os.path.join(path_scoring, f_scoring), {'states': scoring_data_new})





