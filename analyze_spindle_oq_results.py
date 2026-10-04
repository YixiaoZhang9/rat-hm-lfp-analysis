"""Save random wavelet-spindle examples grouped by rat and OQ range.

For each rat:
    - Select 20 random spindle events from each OQ range.
    - Extract the corresponding raw LFP.
    - Save one waveform plot per event.

OQ = r_max from wavelet-detected spindle events.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm.gui import tqdm

ROOT = Path(__file__).resolve().parent

RESULTS_CSV = (
    ROOT
    / "results_ar_calibration"
    / "wavelet_spindles_ar_calibration.csv"
)

MANIFEST_CSV = ROOT / "tasks_manifest.csv"

OUTPUT_DIR = (
    ROOT
    / "results_ar_calibration"
    / "oq_analysis"
)

SAMPLES_PER_RAT_RANGE = 20
RANDOM_SEED = 42

# OQ ranges used in the notebook.
OQ_RANGES = [
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.95),
]

FS = 1000
PLOT_PADDING_SEC = 0.5


def normalize_id(value):
    if pd.isna(value) or str(value).strip() in ("", "nan"):
        return ""

    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def task_key(rat, region, date, trial, channel):
    return (
        str(rat).strip(),
        str(region).strip(),
        str(date).strip(),
        normalize_id(trial),
        normalize_id(channel),
    )


def recording_lookup(manifest_path):
    """Build lookup from spindle identifiers to raw MAT recordings."""

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    tasks = pd.read_csv(manifest_path)

    lookup = {}

    for _, row in tasks.iterrows():

        match = re.match(
            r"chan(\d+)(?:_(\d+))?\.mat$",
            Path(str(row["data_path"])).name,
            re.IGNORECASE,
        )

        if not match:
            continue

        channel = match.group(1)
        trial = match.group(2) or ""

        key = task_key(
            row["rat"],
            row["region"],
            row["date"],
            trial,
            channel,
        )

        lookup[key] = Path(str(row["data_path"]))

    return lookup


def plot_event(row, signal, output_path):
    """Plot and save one spindle waveform."""

    start = int(
        round(
            float(row["spindle_start_time_s"]) * FS
        )
    )

    end = int(
        round(
            float(row["spindle_end_time_s"]) * FS
        )
    )

    padding = int(PLOT_PADDING_SEC * FS)

    left = max(0, start - padding)
    right = min(len(signal), end + padding)

    if right <= left:
        return False

    times = np.arange(left, right) / FS

    fig, ax = plt.subplots(figsize=(9, 3))

    ax.plot(
        times,
        signal[left:right],
        linewidth=0.7,
        color="black",
    )

    ax.axvspan(
        start / FS,
        end / FS,
        color="tab:orange",
        alpha=0.25,
        label="Wavelet spindle",
    )

    oq = float(row["r_max"])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LFP")

    ax.set_title(
        f"Rat {int(row['rat_number'])} | "
        f"OQ = {oq:.3f} | "
        f"{row['oq_range']}"
    )

    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return True


def main():

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Event results CSV not found: {RESULTS_CSV}"
        )

    df = pd.read_csv(RESULTS_CSV)

    required = {
        "rat_number",
        "region",
        "date",
        "channel",
        "trial",
        "r_max",
        "spindle_start_time_s",
        "spindle_end_time_s",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Results CSV is missing required columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Prepare OQ
    # ---------------------------------------------------------

    df["oq"] = pd.to_numeric(
        df["r_max"],
        errors="coerce",
    )

    valid = df.dropna(
        subset=[
            "oq",
            "spindle_start_time_s",
            "spindle_end_time_s",
        ]
    ).copy()

    # ---------------------------------------------------------
    # Build recording lookup
    # ---------------------------------------------------------

    lookup = recording_lookup(MANIFEST_CSV)

    # Cache recordings so the same MAT file is not loaded repeatedly.
    signal_cache = {}

    total_expected = (
        valid["rat_number"].nunique()
        * len(OQ_RANGES)
        * SAMPLES_PER_RAT_RANGE
    )

    total_saved = 0
    total_missing = 0

    # ---------------------------------------------------------
    # Select and save samples
    # ---------------------------------------------------------

    for rat in tqdm(
        sorted(valid["rat_number"].unique()),
        desc="Processing rats",
    ):

        rat_df = valid[
            valid["rat_number"] == rat
        ].copy()

        rat_dir = OUTPUT_DIR / f"rat_{int(rat):02d}"

        for low, high in OQ_RANGES:

            range_df = rat_df[
                (rat_df["oq"] >= low)
                & (rat_df["oq"] < high)
            ].copy()

            oq_label = f"{low:.2f}_{high:.2f}"

            output_dir = (
                rat_dir
                / f"oq_{oq_label}"
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            if len(range_df) < SAMPLES_PER_RAT_RANGE:

                print(
                    f"Rat {int(rat)} | "
                    f"{low:.2f}-{high:.2f}: "
                    f"only {len(range_df)} events available "
                    f"(need {SAMPLES_PER_RAT_RANGE})"
                )

                continue

            # Reproducible random selection.
            samples = range_df.sample(
                n=SAMPLES_PER_RAT_RANGE,
                random_state=RANDOM_SEED,
            )

            print(
                f"Rat {int(rat)} | "
                f"OQ {low:.2f}-{high:.2f}: "
                f"saving {len(samples)} plots"
            )

            for sample_number, (_, row) in enumerate(
                samples.iterrows(),
                start=1,
            ):

                key = task_key(
                    row["rat_number"],
                    row["region"],
                    row["date"],
                    row["trial"],
                    row["channel"],
                )

                data_path = lookup.get(key)

                if data_path is None:
                    print(
                        f"  Recording not found: {key}"
                    )
                    total_missing += 1
                    continue

                # Load and cache recording.
                data_path_str = str(data_path)

                if data_path_str not in signal_cache:

                    try:
                        signal_cache[data_path_str] = np.asarray(
                            loadmat(data_path)["data"]
                        ).squeeze()

                    except Exception as exc:
                        print(
                            f"  Could not load {data_path}: {exc}"
                        )
                        total_missing += 1
                        continue

                signal = signal_cache[data_path_str]

                safe_date = str(row["date"])

                oq = float(row["oq"])

                output_path = (
                    output_dir
                    / (
                        f"sample_{sample_number:02d}"
                        f"_oq_{oq:.3f}"
                        f"_date_{safe_date}"
                        f".png"
                    )
                )

                row = row.copy()
                row["oq_range"] = (
                    f"{low:.2f}-{high:.2f}"
                )

                saved = plot_event(
                    row,
                    signal,
                    output_path,
                )

                if saved:
                    total_saved += 1

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Expected plots: {total_expected}")
    print(f"Saved plots:    {total_saved}")
    print(f"Missing plots:  {total_missing}")
    print(f"Output folder:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
