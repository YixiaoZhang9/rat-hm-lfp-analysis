import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from tqdm import tqdm

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from detector import convert_events_to_array, detect_events
from task_loader import TaskLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FS = 1000
TARGET_FS = 128

RA = 0.70
RB = 0.80

SPINDLE_LOW = 9.0
SPINDLE_HIGH = 20.0


OUTPUT_DIR = Path("results")

SPINDLES_OUT_CSV = (
    OUTPUT_DIR /
    "all_detected_spindles_per_region.csv"
)

FAILED_OUT_CSV = (
    OUTPUT_DIR /
    "extraction_failed_files.csv"
)


MAX_WORKERS = max(
    1,
    os.cpu_count() - 2
)

CHECKPOINT_EVERY = 25


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

run_stamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

log_path = (
    LOG_DIR /
    f"extract_spindles_{run_stamp}.log"
)


logger = logging.getLogger(
    "extract_spindles"
)

logger.setLevel(
    logging.DEBUG
)

logger.handlers.clear()


console_handler = logging.StreamHandler(
    sys.stdout
)

console_handler.setLevel(
    logging.INFO
)

file_handler = logging.FileHandler(
    log_path,
    mode="w"
)

file_handler.setLevel(
    logging.DEBUG
)


logger.addHandler(console_handler)
logger.addHandler(file_handler)



# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_signal(signal):
    """
    Only resampling.

    1000 Hz -> 128 Hz
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    return resample_poly(
        signal,
        TARGET_FS,
        FS
    )



def select_spindles(events):
    """
    Keep only spindle-frequency events.

    detect_events() remains
    frequency independent.
    """

    selected = []

    for e in events:

        if (
            SPINDLE_LOW
            <= e["frequency"]
            <= SPINDLE_HIGH
        ):
            selected.append(e)

    return selected



# ---------------------------------------------------------------------------
# NREM intervals
# ---------------------------------------------------------------------------

def get_nrem_intervals(scoring_path):

    states = loadmat(scoring_path)[
        "states"
    ].squeeze()


    nrem_mask = (
        states == 3
    ).astype(int)


    diff = np.diff(
        np.concatenate(
            (
                [0],
                nrem_mask,
                [0]
            )
        )
    )


    starts = np.where(
        diff == 1
    )[0]

    ends = np.where(
        diff == -1
    )[0]


    if len(starts) == 0:
        return np.empty(
            (0,2)
        )


    return np.column_stack(
        (
            starts,
            ends
        )
    )


# ---------------------------------------------------------------------------
# Failure class
# ---------------------------------------------------------------------------

class TaskFailure(Exception):

    def __init__(
        self,
        reason: str,
        detail: str = ""
    ):

        self.reason = reason
        self.detail = detail

        super().__init__(
            f"{reason}: {detail}"
            if detail
            else reason
        )



# ---------------------------------------------------------------------------
# Extract spindles from one file
# ---------------------------------------------------------------------------

def extract_spindles_for_file(
    task: Dict
) -> pd.DataFrame:


    data_path = task["data_path"]
    scoring_path = task["scoring_path"]


    if not os.path.exists(data_path):
        raise TaskFailure(
            "data file not found",
            data_path
        )


    if not os.path.exists(scoring_path):
        raise TaskFailure(
            "scoring file not found",
            scoring_path
        )


    # ------------------------------------------------------------
    # Load LFP
    # ------------------------------------------------------------

    try:

        raw_signal = loadmat(
            data_path
        )["data"].squeeze()

    except Exception as e:

        raise TaskFailure(
            "data read error",
            str(e)
        )



    # ------------------------------------------------------------
    # Load NREM
    # ------------------------------------------------------------

    try:

        nrem_intervals = get_nrem_intervals(
            scoring_path
        )

    except Exception as e:

        raise TaskFailure(
            "scoring read error",
            str(e)
        )



    if len(nrem_intervals) == 0:

        raise TaskFailure(
            "no NREM epochs"
        )



    all_spindles = []



    # ------------------------------------------------------------
    # Process every NREM block
    # ------------------------------------------------------------

    for start, end in nrem_intervals:


        # NO BUFFER
        start_sample = int(
            start * FS
        )

        end_sample = int(
            end * FS
        )


        segment = raw_signal[
            start_sample:end_sample
        ]


        if len(segment) < FS:

            continue



        # --------------------------------------------------------
        # 1000 Hz -> 128 Hz
        # --------------------------------------------------------

        segment = preprocess_signal(
            segment
        )



        # --------------------------------------------------------
        # Detect ALL AR events
        # --------------------------------------------------------

        events = detect_events(

            signal=segment,

            fs=TARGET_FS,

            r_a=RA,

            r_b=RB,
        )



        # --------------------------------------------------------
        # Keep spindle frequency only
        # --------------------------------------------------------

        events = select_spindles(
            events
        )



        spindles = convert_events_to_array(
            events
        )



        if len(spindles) == 0:

            continue



        # --------------------------------------------------------
        # Convert local NREM time -> global time
        # --------------------------------------------------------

        spindles[:,0:3] += (
            start_sample / FS
        )



        # --------------------------------------------------------
        # Keep strictly inside NREM
        # --------------------------------------------------------

        keep = (

            (spindles[:,0] >= start)

            &

            (spindles[:,2] <= end)

        )


        valid = spindles[keep]


        if len(valid) > 0:

            all_spindles.append(
                valid
            )



    # ------------------------------------------------------------
    # No events
    # ------------------------------------------------------------

    if not all_spindles:

        return pd.DataFrame()



    final_spindles = np.vstack(
        all_spindles
    )



    df = pd.DataFrame(

        final_spindles,

        columns=[

            "Start_s",

            "Peak_s",

            "End_s",

            "Duration_s",

            "Max_R",

            "Peak_Freq_Hz",

        ]

    )



    # Metadata

    df.insert(
        0,
        "File",
        task["file_name"]
    )

    df.insert(
        0,
        "Date",
        task["date"]
    )

    df.insert(
        0,
        "Region",
        task["region"]
    )

    df.insert(
        0,
        "Rat",
        task["rat"]
    )

    df.insert(
        0,
        "Cohort",
        task["cohort"]
    )


    df["Threshold_RB"] = RB
    df["Threshold_RA"] = RA


    df["Spindle_Band_Low"] = SPINDLE_LOW
    df["Spindle_Band_High"] = SPINDLE_HIGH


    return df


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker_process(
    task: Dict
) -> Dict:

    t0 = time.time()

    try:

        df = extract_spindles_for_file(
            task
        )


        return {

            "status": "OK",

            "task": task,

            "df": df,

            "elapsed": time.time() - t0,

        }



    except TaskFailure as e:


        return {

            "status": "SKIP",

            "task": task,

            "reason": e.reason,

            "detail": e.detail,

            "elapsed": time.time() - t0,

        }



    except Exception as e:


        return {

            "status": "ERROR",

            "task": task,

            "reason": "unexpected error",

            "detail": str(e),

            "elapsed": time.time() - t0,

        }



# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def run_extraction(
    tasks: List[Dict]
):


    if not tasks:

        logger.error(
            "No tasks found"
        )

        return



    logger.info(
        f"Running {len(tasks)} files"
    )



    all_dfs = []

    failures = []



    pbar = tqdm(

        total=len(tasks),

        desc="Extracting spindles",

        unit="file",

        dynamic_ncols=True

    )



    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:


        future_to_task = {


            executor.submit(

                worker_process,

                task

            ): task


            for task in tasks

        }



        for i, future in enumerate(

            as_completed(
                future_to_task
            ),

            start=1

        ):


            result = future.result()


            task = result["task"]


            pbar.set_postfix(

                {

                    "Rat": task["rat"],

                    "Region": task["region"],

                    "Status": result["status"],

                }

            )


            pbar.update(1)



            if result["status"] == "OK":


                df = result["df"]


                if not df.empty:

                    all_dfs.append(
                        df
                    )



                logger.info(

                    f"{task['file_name']} -> "
                    f"{len(df)} spindles"

                )



            else:


                failures.append(

                    {

                        **task,

                        "reason": result.get(
                            "reason",
                            ""
                        ),

                        "detail": result.get(
                            "detail",
                            ""
                        ),

                    }

                )



            # checkpoint

            if (

                i % CHECKPOINT_EVERY == 0

                or i == len(tasks)

            ):


                if all_dfs:


                    pd.concat(

                        all_dfs,

                        ignore_index=True

                    ).to_csv(

                        SPINDLES_OUT_CSV,

                        index=False

                    )



                if failures:


                    pd.DataFrame(
                        failures
                    ).to_csv(

                        FAILED_OUT_CSV,

                        index=False

                    )



    pbar.close()



    # ------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------

    if not all_dfs:


        logger.warning(
            "No spindles detected"
        )

        return



    master_df = pd.concat(

        all_dfs,

        ignore_index=True

    )



    master_df.to_csv(

        SPINDLES_OUT_CSV,

        index=False

    )



    logger.info(

        f"Saved {len(master_df)} spindles"

    )



# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":


    loader = TaskLoader(

        "/home/mdadmin/Desktop/amirali/rat-hm-lfp-analysis/tasks_manifest.csv"

    )


    tasks = loader.to_tasks()


    run_extraction(
        tasks
    )
