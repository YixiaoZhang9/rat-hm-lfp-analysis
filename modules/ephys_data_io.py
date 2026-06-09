# this file includes functions used in load and store the ephys data

import os
import re
import pandas as pd
from modules import OpenEphys
from scipy.io import savemat

#--------------------------------------------------------------------

def find_path_by_suffix(base_path, suffix, search_dir=True, use_regex=False):

    """
    Search for a unique file or directory that ends with the given suffix,
    or matches a regex pattern if use_regex=True.

    Parameters:
        base_path (str): Root directory to search in.
        suffix (str or Pattern): Suffix string or regex pattern to match.
        search_dir (bool): Whether to search for directories (True) or files (False).
        use_regex (bool): If True, treat suffix as a regex pattern.

    Returns:
        str or None: The unique matching path, or None if not found.

    Raises:
        ValueError: If more than one match is found.
    """
    matched_paths = []

    if use_regex:
        pattern = re.compile(suffix) if isinstance(suffix, str) else suffix

    # Walk through the directory tree
    for root, dirs, files in os.walk(base_path):
        # Choose directories or files based on the search_dir flag
        targets = dirs if search_dir else files

        # Check each item in the current directory
        for name in targets:
            if (use_regex and pattern.search(name)) or (not use_regex and name.endswith(suffix)):
                matched_paths.append(os.path.join(root, name))

    # Handle the result based on the number of matches
    if len(matched_paths) == 0:
        return None  # No matches found
    elif len(matched_paths) > 1:
        raise ValueError(f"Multiple matching paths found ({len(matched_paths)}), expected only one.")
    else:
        return matched_paths[0]  # Return the unique match



def check_studyDay(csv_path, rat_number, target_date):
    """
    Check whether a given rat has a recording on a specific day
    
    Parameters:
    csv_path (str): file path of csv file 
    rat_number (int): rat number
    target_date (str or int): data to check，for example '20221003'

    Retrun:
    bool
    """
    # load csv file as a DataFrame
    df = pd.read_csv(csv_path, dtype=str)
    
    rat_column = f"Rat {rat_number}"

    # Check if region column exists
    if rat_column not in df.columns:
        raise ValueError(f"{rat_column} not in the colume")
    
    match_row = df[df[rat_column] == target_date]
    
    if not match_row.empty:
        return True
    else:
        return False

#--------------------------------------------------------------------

def get_selected_channel(csv_path: str, rat_number: int, target_region: str):
    """
    Read manually selected channel for a specific region of a rat.
    
    Parameters:
    csv_path (str): File path of the CSV file.
    rat_number: (int): Rat number (starting from 1).
    target_region (str): Column name indicating the target brain region.
    
    Returns:
    str: Channel number corresponding to the target region for the given rat.
    """
    # Load CSV file as a DataFrame 
    df = pd.read_csv(csv_path, dtype=str)

    # Check if region column exists
    if target_region not in df.columns:
        raise ValueError(f"{target_region}not in the colume")

    Target_row = df[(df["Rat_Number"] == "Rat "+str(rat_number))]

    # Get the channel string, rat n in the row of n-1
    channel_str = Target_row[target_region].values[0]

    return channel_str


#--------------------------------------------------------------------

def get_pre_post_suffixes(csv_path: str, rat_number: int):
    """
    Get the file suffixes for 'pre' and 'post' training sleep periods for a specific rat.
    
    Parameters:
        csv_path (str): File path of the CSV file.
        rat_number (int): Rat number (starting from 1).

    Returns:
        Tuple[List[str], List[str]]: Two lists containing the file suffixes for
                                     'pre' and 'post' stages, respectively.
    """
    # Load CSV file as a DataFrame 
    df = pd.read_csv(csv_path, dtype=str)

    # Check if the rat column exists
    rat_column = f"Rat {rat_number:}"
    if rat_column not in df.columns:
        raise ValueError(f"{rat_column} not in the colume")

    # The last column contains the file suffixes
    suffix_col = df.columns[-1]

    # Normalize the values in the rat column (strip whitespace, lowercase)
    col_normalized = df[rat_column].str.strip().str.lower()

    # Extract suffixes where phase is 'pre' or 'post'
    pre_list = df.loc[col_normalized == 'pre', suffix_col].dropna().tolist()
    post_list = df.loc[col_normalized == 'post', suffix_col].dropna().tolist()

    return pre_list, post_list


def read_recording(matched_dir, pattern):
    """
    Get the file suffixes for 'pre' and 'post' training sleep periods for a specific rat.

    Parameters:
        matched_dir (str) : File path of .continuous recording file.
        pattern (str): pattern of target file.

    Returns:
        List: ephys recordings
    """

    # find the matched recording

    # to make sure the matched file is correct,
    str_matched_file = find_path_by_suffix(matched_dir, pattern, search_dir=False,use_regex=True)
    # read recording
    Recording = OpenEphys.load(str_matched_file)
    return Recording


def get_folder_selected_channel_r5_8_20221005(csv_path, rat_number, target_region, idx_trial):
    """
        highly specified for Rat5_8 on 20221005
        Get the file folder and selected channel for a specific rat at a specific trial and on a specific region.

        Parameters:
            csv_path (str): File path of the CSV file.
            rat_number (int): rat number
            target_region (str): Column name indicating the target brain region.
             idx_trial (str):
        Returns:
            List: [Folder,selected_channel]
        """

    # Load CSV file as a DataFrame
    df = pd.read_csv(csv_path, dtype=str)

    Target_row = df[(df["Rat_Number"] == "Rat "+ str(rat_number)) & (df["Trial_Suffix"] == idx_trial)]

    Selected_channel = Target_row[target_region].values[0]
    File_Folder = Target_row["File_Folder"].values[0]


    return [File_Folder, Selected_channel]

def get_folder_selected_channel_r5_20221007(csv_path, target_region, idx_trial):
    """
        highly specified for Rat5 on 20221007
        Get the file folder and selected channel for a specific rat at a specific trial and on a specific region.

        Parameters:
            csv_path (str): File path of the CSV file.
            rat_number (int): rat number
            target_region (str): Column name indicating the target brain region.
             idx_trial (str):
        Returns:
            List: [Folder,selected_channel]
        """

    # Load CSV file as a DataFrame
    df = pd.read_csv(csv_path, dtype=str)

    Target_row = df[(df["Rat_Number"] == "Rat 5") & (df["Trial_Suffix"] == idx_trial)]

    Selected_channel = Target_row[target_region].values[0]
    File_Folder = Target_row["File_Folder"].values[0]


    return [File_Folder, Selected_channel]


def save_sleep_recording(path_recording, trial_suffix, channel, region_result_dir, sleep_period, filename_suffix=''):
    """
        highly specified for this dataset
       Load and save electrophysiological recording data for a specific trial and channel.

       This function locates the recording directory based on the trial suffix, reads the
       continuous recording data for the specified channel, and saves it as a .mat file
       in the designated results directory under the specified sleep period folder.

       Parameters:
           path_recording (str): Base path to the recording data folder.
           trial_suffix (str): Suffix or identifier for the specific trial (used to find the correct folder).
           channel (int or str): Channel number to read the recording from.
           region_result_dir (str): Base directory to save processed recordings for the brain region.
           sleep_period (str): Subfolder name indicating sleep period (e.g., 'presleep' or 'postsleep').
           filename_suffix (str, optional): Additional suffix to append to the saved filename (default is empty).

       Returns:
           None
       """

    matched_dir = find_path_by_suffix(path_recording, trial_suffix, search_dir=True)

    pattern = re.compile(rf"(?<![A-Z0-9])(?:CH)?{channel}\.continuous$")
    recording = read_recording(matched_dir, pattern)

    save_dir = os.path.join(region_result_dir, sleep_period)
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"chan{channel}{filename_suffix}.mat"
    savemat(os.path.join(save_dir, save_name), recording)


     
    
