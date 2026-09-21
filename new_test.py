from modules.spindle_detector import (
    ARSpindleDetector,
    SpindleDetectorConfig,
)
from task_loader import TaskLoader

loader = TaskLoader(
    "/home/mdadmin/Desktop/amirali/rat-hm-lfp-analysis/tasks_manifest.csv"
)

# Optional filtering
loader = loader.filter(
    rat="1",
)

tasks = loader.to_tasks()

config = SpindleDetectorConfig(
    input_fs=1000.0,
    target_fs=128.0,
)

detector = ARSpindleDetector(config)

events, ar_timeseries = detector.detect_tasks(
    tasks,
    signal_variable="LFP",   # <-- change this to your .mat variable
    channel=0,
    channel_axis="auto",
    return_ar_timeseries=False,
)

events.to_csv(
    "spindle_events.csv",
    index=False,
)
