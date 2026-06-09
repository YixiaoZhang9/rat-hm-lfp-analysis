"""Project-level path configuration with backward-compatible defaults."""

import os

DEFAULT_PATHS = {
    "RAT_HM_GL14_ROOT": "/media/yixiao/GL14_RAT_FA/",
    "RAT_HM_DATA4_ROOT": "/media/yixiao/Data4/",
    "RAT_HM_RIPPLE_MARKING_ROOT": (
        "/mnt/genzel/Rat/HM/Rat_HM_Ephys_TD/Rat_HM_Ripple_Detection/Ripple_Marking"
    ),
    "RAT_HM_RIPPLE_TRAINING_DATA": (
        "/home/yixiao/PycharmProjects/Rat_HM/pipeline/ripple_detection/training_data"
    ),
}


def get_path(name, default=None):
    """Return a configured project path, preserving legacy defaults."""
    if default is None:
        default = DEFAULT_PATHS[name]
    return os.environ.get(name, default)
