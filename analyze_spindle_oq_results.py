"""Save random wavelet-spindle examples grouped by rat and OQ range.

For each rat, pick random spindle events from each OQ range, extract the
matching raw LFP, and save one waveform plot per event.

OQ = r_max from the wavelet-detected spindle events.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_ar_calibration"

RESULTS_CSV = RESULTS_DIR / "wavelet_spindles_ar_calibration.csv"
MANIFEST_CSV = ROOT / "tasks_manifest.csv"
OUTPUT_DIR = RESULTS_DIR / "oq_analysis"

SAMPLES_PER_RAT_RANGE = 20
RANDOM_SEED = 42
FS = 1000  # Hz
PLOT_PADDING_SEC = 0.5

OQ_RANGES = [
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.95),
]

REQUIRED_COLUMNS = {
    "rat_number",
    "region",
    "date",
    "channel",
    "trial",
    "r_max",
    "spindle_start_time_s",
    "spindle_end_time_s",
}

FILENAME_PATTERN = re.compile(r"chan(\d+)(?:_(\d+))?\.mat$", re.IGNORECASE)


def normalize_id(value):
    """Turn 3, 3.0, '3' -> '3'; NaN/empty -> ''."""
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


def build_recording_lookup(manifest_path):
    """Map (rat, region, date, trial, channel) -> raw MAT file path."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    lookup = {}
    for row in pd.read_csv(manifest_path).itertuples(index=False):
        data_path = Path(str(row.data_path))
        match = FILENAME_PATTERN.match(data_path.name)
        if not match:
            continue
        channel, trial = match.group(1), match.group(2) or ""
        lookup[task_key(row.rat, row.region, row.date, trial, channel)] = data_path
    return lookup


def load_signal(path, cache):
    """Load the 'data' array from a MAT file, caching by path."""
    if path not in cache:
        cache[path] = np.asarray(loadmat(path)["data"]).squeeze()
    return cache[path]


def plot_event(row, oq, oq_range_label, signal, output_path):
    """Plot one spindle with padding; return False if the window is empty."""
    start = int(round(row["spindle_start_time_s"] * FS))
    end = int(round(row["spindle_end_time_s"] * FS))
    padding = int(PLOT_PADDING_SEC * FS)

    left = max(0, start - padding)
    right = min(len(signal), end + padding)
    if right <= left:
        return False

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(left, right) / FS, signal[left:right], lw=0.7, color="black")
    ax.axvspan(start / FS, end / FS, color="tab:orange", alpha=0.25,
               label="Wavelet spindle")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LFP")
    ax.set_title(f"Rat {int(row['rat_number'])} | OQ = {oq:.3f} | {oq_range_label}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def load_events(csv_path):
    """Read the spindle CSV and keep rows with a usable OQ and timing."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Event results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Results CSV is missing columns: {sorted(missing)}")

    df["oq"] = pd.to_numeric(df["r_max"], errors="coerce")
    return df.dropna(
        subset=["oq", "spindle_start_time_s", "spindle_end_time_s"]
    ).copy()


def main():
    events = load_events(RESULTS_CSV)
    lookup = build_recording_lookup(MANIFEST_CSV)
    signal_cache = {}

    rats = sorted(events["rat_number"].unique())
    total_expected = len(rats) * len(OQ_RANGES) * SAMPLES_PER_RAT_RANGE
    total_saved = 0
    total_missing = 0

    for rat in tqdm(rats, desc="Processing rats"):
        rat_events = events[events["rat_number"] == rat]

        for low, high in OQ_RANGES:
            label = f"{low:.2f}-{high:.2f}"
            in_range = rat_events[(rat_events["oq"] >= low) & (rat_events["oq"] < high)]

            if len(in_range) < SAMPLES_PER_RAT_RANGE:
                print(f"Rat {int(rat)} | {label}: only {len(in_range)} events "
                      f"available (need {SAMPLES_PER_RAT_RANGE})")
                continue

            samples = in_range.sample(n=SAMPLES_PER_RAT_RANGE, random_state=RANDOM_SEED)
            out_dir = OUTPUT_DIR / f"rat_{int(rat):02d}" / f"oq_{low:.2f}_{high:.2f}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Rat {int(rat)} | OQ {label}: saving {len(samples)} plots")

            for n, (_, row) in enumerate(samples.iterrows(), start=1):
                key = task_key(row["rat_number"], row["region"], row["date"],
                               row["trial"], row["channel"])
                data_path = lookup.get(key)
                if data_path is None:
                    print(f"  Recording not found: {key}")
                    total_missing += 1
                    continue

                try:
                    signal = load_signal(str(data_path), signal_cache)
                except Exception as exc:
                    print(f"  Could not load {data_path}: {exc}")
                    total_missing += 1
                    continue

                oq = float(row["oq"])
                out_path = out_dir / f"sample_{n:02d}_oq_{oq:.3f}_date_{row['date']}.png"
                if plot_event(row, oq, label, signal, out_path):
                    total_saved += 1

    print(f"\n{'=' * 60}\nDONE\n{'=' * 60}")
    print(f"Expected plots: {total_expected}")
    print(f"Saved plots:    {total_saved}")
    print(f"Missing plots:  {total_missing}")
    print(f"Output folder:  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
