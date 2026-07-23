"""View preprocessed LFP together with sleep-scoring states."""

# ruff: noqa: I001

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QApplication, QInputDialog
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ephys_signal_scoring_view import SignalPlotViewer
from modules.project_config import get_path


GROUPS = {
    "buildup_r1_4": {
        "path_key": "R1_8_root",
        "analysis_dir": "Rat_HM_Ephys_TD_Analysis_R1_8/R1-4",
    },
    "buildup_r5_8": {
        "path_key": "R1_8_root",
        "analysis_dir": "Rat_HM_Ephys_TD_Analysis_R1_8/R5-8",
    },
    "update_r9_12": {
        "path_key": "R9_16_root",
        "analysis_dir": "Rat_HM_Ephys_TD_Analysis_R9_16/R9-12",
    },
    "update_r13_16": {
        "path_key": "R9_16_root",
        "analysis_dir": "Rat_HM_Ephys_TD_Analysis_R9_16/R13-16",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="View an LFP .mat file with its matching sleep-scoring states."
    )
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--rat", type=int)
    parser.add_argument("--studyday", help="Study day, e.g. 20230905.")
    parser.add_argument(
        "--sleep-period", choices=["presleep", "postsleep"], help="Sleep session."
    )
    parser.add_argument(
        "--trial-suffix",
        help="Trial suffix used to match data and scoring files, e.g. 6 or 06.",
    )
    parser.add_argument("--fs", type=int, default=1000, help="LFP sampling rate in Hz.")
    parser.add_argument(
        "--window-sec", type=float, default=30.0, help="Initial viewer window length."
    )
    return parser.parse_args()


def analysis_root(group):
    group_config = GROUPS[group]
    return Path(get_path(group_config["path_key"])) / group_config["analysis_dir"]


def analysis_paths(root):
    return {
        "preprocessed_data": root / "PreprocessedData",
        "accelerometer_signal": root / "accelerometer_signal",
        "scoring": root / "Scoring",
        "automatic_scoring_new": [
            root / "Automatic_Scoring_new",
            root / "Automatic_Scroing_new",
        ],
    }


def list_mat_files(path):
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    return sorted(file for file in path.rglob("*.mat") if file.is_file())


def list_optional_mat_files(path):
    if not path.exists():
        return []
    return sorted(file for file in path.rglob("*.mat") if file.is_file())


def list_optional_mat_files_from_roots(paths):
    for path in paths:
        files = list_optional_mat_files(path)
        if files:
            return files
    return []


def list_dirs(path):
    if not path.exists():
        return []
    return sorted(file.name for file in path.iterdir() if file.is_dir())


def choose_option(label, options):
    if not options:
        raise FileNotFoundError(f"No options found for {label}.")

    selection, ok = QInputDialog.getItem(
        None,
        "Sleep scoring viewer",
        f"Choose {label}:",
        options,
        0,
        False,
    )
    if not ok:
        raise SystemExit("Selection cancelled.")
    return selection


def trial_suffix_from_name(path):
    match = re.search(r"_(\d+)\.mat$", path.name)
    if match:
        return match.group(1)

    match = re.search(r"(\d+)\.mat$", path.name)
    if match:
        return match.group(1)

    return None


def select_trial_file(files, trial_suffix):
    if not files:
        raise FileNotFoundError("No .mat data files found.")

    if trial_suffix is None:
        if len(files) == 1:
            return files[0]

        options = "\n".join(f"  - {file}" for file in files)
        raise ValueError(
            f"Multiple data files found. Pass --trial-suffix to choose one:\n{options}"
        )

    suffix = str(int(trial_suffix))
    matches = [
        file
        for file in files
        if trial_suffix_from_name(file) is not None
        and str(int(trial_suffix_from_name(file))) == suffix
    ]

    if len(matches) != 1:
        options = "\n".join(f"  - {file}" for file in files)
        raise ValueError(
            f"Expected one data file for trial suffix {trial_suffix}, "
            f"found {len(matches)}.\n{options}"
        )

    return matches[0]


def select_matching_file(files, trial_suffix, label):
    if not files:
        raise FileNotFoundError(f"No .mat {label} files found.")

    if len(files) == 1:
        return files[0]

    suffix = trial_suffix
    if suffix is None:
        options = "\n".join(f"  - {file}" for file in files)
        raise ValueError(
            f"Could not infer trial suffix for {label} match. "
            f"Choose a trial first or pass --trial-suffix.\n{options}"
        )

    suffix = str(int(suffix)).zfill(2)
    suffix_int = str(int(suffix))
    matches = [
        file
        for file in files
        if trial_suffix_from_name(file) is not None
        and str(int(trial_suffix_from_name(file))) == suffix_int
    ]

    if not matches:
        pattern = re.compile(rf"_{suffix}_")
        matches = [file for file in files if pattern.search(file.name)]

    if len(matches) != 1:
        options = "\n".join(f"  - {file}" for file in files)
        raise ValueError(
            f"Expected one {label} file for trial suffix {suffix}, "
            f"found {len(matches)}.\n{options}"
        )

    return matches[0]


def select_optional_matching_file(files, trial_suffix, label):
    if not files:
        print(f"Warning: no {label} files found.")
        return None

    try:
        return select_matching_file(files, trial_suffix, label)
    except ValueError as exc:
        print(f"Warning: {exc}")
        return None


def load_vector(mat_path, keys):
    mat = loadmat(mat_path)

    for key in keys:
        if key in mat:
            return np.asarray(mat[key]).squeeze()

    for key, value in mat.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value).squeeze()
        if np.issubdtype(array.dtype, np.number) and array.ndim == 1:
            print(f"Using variable {key!r} from {mat_path}.")
            return array

    raise KeyError(f"{mat_path} does not contain a numeric 1D vector.")


def scoring_to_lfp_rate(scoring, fs, target_len):
    scoring = np.asarray(scoring).squeeze()

    if len(scoring) == target_len:
        scoring_upsampled = scoring
    else:
        scoring_upsampled = np.repeat(scoring, fs)

    if len(scoring_upsampled) < target_len:
        pad_len = target_len - len(scoring_upsampled)
        scoring_upsampled = np.pad(scoring_upsampled, (0, pad_len), constant_values=0)

    return scoring_upsampled[:target_len]


def signal_to_lfp_rate(signal, target_len):
    signal = np.asarray(signal).squeeze()

    if len(signal) >= target_len:
        return signal[:target_len]

    return np.pad(signal, (0, target_len - len(signal)), constant_values=0)


def available_trials(trial_files):
    trials = []
    for file in trial_files:
        suffix = trial_suffix_from_name(file)
        if suffix is not None:
            trials.append((str(int(suffix)), file.name))
    return sorted(set(trials), key=lambda item: int(item[0]))


def resolve_group(args):
    if args.group is not None:
        return args.group
    return choose_option("group", list(GROUPS.keys()))


def resolve_selection(args, paths):
    hpc_root = paths["preprocessed_data"] / "HPC"

    rat = (
        str(args.rat)
        if args.rat is not None
        else choose_option("rat", list_dirs(hpc_root))
    )
    rat_root = hpc_root / rat

    studyday = args.studyday or choose_option("date", list_dirs(rat_root))
    day_root = rat_root / studyday

    sleep_period = args.sleep_period or choose_option("session", list_dirs(day_root))
    session_root = day_root / sleep_period

    hpc_files = list_mat_files(session_root)
    if args.trial_suffix is None:
        trials = available_trials(hpc_files)
        if not trials:
            raise FileNotFoundError(f"No trial-like .mat files found in {session_root}")
        trial_label = choose_option(
            "trial", [f"{suffix} ({name})" for suffix, name in trials]
        )
        trial_suffix = trial_label.split(" ", maxsplit=1)[0]
    else:
        trial_suffix = args.trial_suffix

    return rat, studyday, sleep_period, trial_suffix


def main():
    args = parse_args()
    app = QApplication.instance()
    app_created = app is None
    if app is None:
        app = QApplication(sys.argv)

    group = resolve_group(args)
    root = analysis_root(group)
    paths = analysis_paths(root)

    rat, studyday, sleep_period, trial_suffix = resolve_selection(args, paths)

    hpc_dir = paths["preprocessed_data"] / "HPC" / rat / studyday / sleep_period
    pl_dir = paths["preprocessed_data"] / "PL" / rat / studyday / sleep_period
    accelerometer_dir = paths["accelerometer_signal"] / rat / studyday / sleep_period
    scoring_dir = paths["scoring"] / rat / studyday / sleep_period
    automatic_scoring_dirs = [
        path / rat / studyday / sleep_period for path in paths["automatic_scoring_new"]
    ]

    hpc_file = select_trial_file(list_mat_files(hpc_dir), trial_suffix)
    pl_file = select_matching_file(list_mat_files(pl_dir), trial_suffix, "PL")
    accelerometer_file = select_optional_matching_file(
        list_optional_mat_files(accelerometer_dir), trial_suffix, "accelerometer"
    )
    scoring_file = select_matching_file(
        list_mat_files(scoring_dir), trial_suffix, "manual scoring"
    )
    automatic_scoring_file = select_optional_matching_file(
        list_optional_mat_files_from_roots(automatic_scoring_dirs),
        trial_suffix,
        "automatic scoring",
    )

    hpc_signal = load_vector(hpc_file, ["data"])
    target_len = len(hpc_signal)
    pl_signal = signal_to_lfp_rate(load_vector(pl_file, ["data"]), target_len)
    scoring = load_vector(scoring_file, ["states"])

    data_dict = {
        "HPC LFP": hpc_signal,
        "PL LFP": pl_signal,
    }

    if accelerometer_file is not None:
        accelerometer = load_vector(accelerometer_file, ["movement_index", "data"])
        data_dict["Accelerometer movement"] = signal_to_lfp_rate(
            accelerometer, target_len
        )

    data_dict["Manual scoring"] = scoring_to_lfp_rate(scoring, args.fs, target_len)

    if automatic_scoring_file is not None:
        automatic_scoring = load_vector(
            automatic_scoring_file,
            ["states", "automatic_scoring", "scoring", "predicted_states"],
        )
        data_dict["Automatic scoring"] = scoring_to_lfp_rate(
            automatic_scoring, args.fs, target_len
        )

    print(f"HPC: {hpc_file}")
    print(f"PL: {pl_file}")
    print(f"Accelerometer: {accelerometer_file}")
    print(f"Manual scoring: {scoring_file}")
    print(f"Automatic scoring: {automatic_scoring_file}")
    print(f"Samples: signal={target_len}, scoring={len(scoring)} seconds")

    window = SignalPlotViewer(data_dict, args.fs, window_sec=args.window_sec)
    window.setWindowTitle(
        f"{group} Rat {rat} {studyday} {sleep_period} "
        f"trial {trial_suffix} sleep scoring"
    )
    window.show()

    if app_created:
        app.exec()


if __name__ == "__main__":
    main()
