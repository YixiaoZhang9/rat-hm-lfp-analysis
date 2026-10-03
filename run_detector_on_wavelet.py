import logging
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from tqdm import tqdm

from detector import detect_events
from modules.ephys_preprocessing import bandpass_filter
from task_loader import TaskLoader

ANALYSIS_ROOTS = [
    Path("/mnt/genzel/Rat/HM/Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_R1_8"),
    Path("/mnt/genzel/Rat/HM/Rat_HM_Ephys_TD/Rat_HM_Ephys_TD_Analysis_R9_16"),
]
SUFFIX = Path("postsleep/wavelet_amp_1_ampcore_3")
FS = 1000
TARGET_FS = 128
R_A = 0.70
R_B = 0.80
SPINDLE_BAND = (9.0, 20.0)
START_COL = "spindle_start_time_s"
END_COL = "spindle_end_time_s"
MANIFEST_PATH = Path(__file__).resolve().parent / "tasks_manifest.csv"
COMPARISON_CSV = Path(__file__).resolve().parent / "wavelet_detector_comparison.csv"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_id(value):
    if value is None or str(value).strip().lower() in ("", "nan"):
        return ""
    try:
        return str(int(float(value)))
    except ValueError:
        return str(value).strip()


def build_task_lookup():
    """Index manifest tasks by recording metadata, like the AR analysis script."""
    logger.info("Loading task manifest from %s", MANIFEST_PATH)
    tasks = TaskLoader(str(MANIFEST_PATH)).to_tasks()
    lookup = {}
    for task in tasks:
        match = re.match(
            r"chan(\d+)(?:_(\d+))?\.mat$",
            Path(str(task.get("data_path", ""))).name,
            re.IGNORECASE,
        )
        if not match:
            continue
        key = (
            str(task.get("rat", "")).strip(),
            str(task.get("region", "")).strip(),
            str(task.get("date", "")).strip(),
            normalize_id(match.group(2) or ""),
            normalize_id(match.group(1)),
        )
        lookup[key] = task
    logger.info("Built manifest lookup with %d recordings", len(lookup))
    return lookup


def get_signal_segment(start_time_s, end_time_s, event, task_lookup):
    """Find an event's recording in the manifest and return its raw signal segment."""
    key = (
        str(event["rat_number"]).strip(),
        str(event["region"]).strip(),
        str(event["date"]).strip(),
        normalize_id(event.get("trial", "")),
        normalize_id(event.get("channel", "")),
    )
    task = task_lookup.get(key)
    if task is None:
        logger.warning("No manifest recording matches event metadata: %s", key)
        raise LookupError(f"No manifest recording matches event metadata: {key}")

    data_path = task["data_path"]
    logger.debug("Loading signal from %s", data_path)
    signal = np.asarray(loadmat(data_path)["data"]).squeeze()
    start_sample = int(round(float(start_time_s) * FS))
    end_sample = int(round(float(end_time_s) * FS))
    if start_sample < 0 or end_sample < start_sample or end_sample > len(signal):
        raise ValueError(
            f"Invalid segment [{start_time_s}, {end_time_s}] for {data_path} "
            f"({len(signal) / FS:.3f}s signal)"
        )
    return signal[start_sample:end_sample]


def load_all_spindle_events() -> pd.DataFrame:
    file_pattern = re.compile(r"chan(\d+)(?:_(\d+))?_spindles_wavelet\.csv$")
    rows = []

    for analysis_root in tqdm(ANALYSIS_ROOTS, desc="Analysis folders"):
        if not analysis_root.exists():
            logger.warning("Analysis folder does not exist: %s", analysis_root)
            continue

        for rat_group in analysis_root.iterdir():
            if not rat_group.is_dir():
                continue
            spindle_root = rat_group / "Spindle_detection_results"
            if not spindle_root.exists():
                continue

            for region_dir in spindle_root.iterdir():
                if not region_dir.is_dir():
                    continue
                for animal_dir in region_dir.iterdir():
                    if not animal_dir.is_dir():
                        continue
                    for date_dir in animal_dir.iterdir():
                        if not date_dir.is_dir():
                            continue
                        csv_folder = date_dir / SUFFIX
                        if not csv_folder.exists():
                            continue

                        for csv_file in csv_folder.glob("*.csv"):
                            try:
                                events = pd.read_csv(csv_file)
                            except Exception as exc:
                                logger.warning("Could not read %s: %s", csv_file, exc)
                                continue
                            if events.empty:
                                continue

                            match = file_pattern.match(csv_file.name)
                            events["region"] = region_dir.name
                            events["rat_number"] = animal_dir.name
                            events["date"] = date_dir.name
                            events["file"] = csv_file.name
                            events["channel"] = match.group(1) if match else None
                            events["trial"] = (
                                match.group(2) if match and match.group(2) else ""
                            )
                            events[START_COL] = events["spindle_start_index"] / FS
                            events[END_COL] = events["spindle_end_index"] / FS
                            rows.append(events)

    if not rows:
        logger.warning("No wavelet events found")
        return pd.DataFrame()

    events = pd.concat(rows, ignore_index=True)
    logger.info("Loaded %d wavelet events", len(events))
    return events


def main():
    events = load_all_spindle_events()
    task_lookup = build_task_lookup()
    comparison_rows = []

    for event_id, event in tqdm(
        events.iterrows(), total=len(events), desc="Wavelet events"
    ):
        start_s = float(event[START_COL].item())
        end_s = float(event[END_COL].item())
        base_row = {
            "event_id": event_id,
            "rat_number": event["rat_number"],
            "region": event["region"],
            "date": event["date"],
            "file": event["file"],
            "channel": event["channel"],
            "trial": event["trial"],
            "wavelet_start_s": start_s,
            "wavelet_end_s": end_s,
        }

        try:
            segment = get_signal_segment(start_s, end_s, event, task_lookup)
        except (LookupError, ValueError, OSError, KeyError) as exc:
            logger.warning("Skipping event %.3f-%.3fs: %s", start_s, end_s, exc)
            comparison_rows.append({**base_row, "status": f"skipped: {exc}"})
            continue

        if len(segment) < FS:
            logger.warning(
                "Skipping short event segment %.3f-%.3fs (%d samples)",
                start_s,
                end_s,
                len(segment),
            )
            comparison_rows.append({**base_row, "status": "skipped: under 1 second"})
            continue

        filtered_segment = bandpass_filter(
            segment.astype(float), lowcut=cast(Any, 0.1), highcut=100, fs=FS
        )
        signal_128 = resample_poly(filtered_segment, TARGET_FS, FS)
        detected = detect_events(signal_128, TARGET_FS, R_A, R_B)
        in_band = [
            detected_event
            for detected_event in detected
            if SPINDLE_BAND[0] <= detected_event["frequency"] <= SPINDLE_BAND[1]
        ]
        if not in_band:
            status = "no_detection" if not detected else "no_in_band_detection"
            comparison_rows.append({**base_row, "status": status})
            continue

        for detected_event in in_band:
            detected_start = start_s + detected_event["t1"]
            detected_end = start_s + detected_event["t2"]
            overlap_s = max(
                0.0, min(end_s, detected_end) - max(start_s, detected_start)
            )
            union_s = max(end_s, detected_end) - min(start_s, detected_start)
            comparison_rows.append(
                {
                    **base_row,
                    "detector_start_s": detected_start,
                    "detector_end_s": detected_end,
                    "overlap_s": overlap_s,
                    "iou": overlap_s / union_s if union_s > 0 else 0.0,
                    "detector_peak_r": detected_event["r"],
                    "detector_peak_frequency_hz": detected_event["frequency"],
                    "status": "detected",
                }
            )

    pd.DataFrame(comparison_rows).to_csv(COMPARISON_CSV, index=False)
    logger.info(
        "Saved comparison for %d wavelet events to %s", len(events), COMPARISON_CSV
    )


if __name__ == "__main__":
    main()
