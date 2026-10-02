"""Summarize event-level spindle oscillation quality (OQ = peak AR R).

Reads results_ar_calibration/wavelet_spindles_ar_calibration.csv produced by
compute_spindle_ar_r_values.py. Writes per-rat OQ-bin samples (and raw-LFP
example plots when recordings can be located), plus daily summaries and plots.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "results_ar_calibration" / "wavelet_spindles_ar_calibration.csv"
MANIFEST_CSV = ROOT / "tasks_manifest.csv"
OUTPUT_DIR = ROOT / "results_ar_calibration" / "oq_analysis"
SAMPLES_PER_RAT_BIN = 20
RANDOM_SEED = 42
# Adjacent intervals: [0.00, 0.50), ..., [0.95, 1.00].
OQ_EDGES = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]
FS = 1000
PLOT_PADDING_SEC = 0.5


def normalize_id(value) -> str:
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


def recording_lookup(manifest_path: Path):
    if not manifest_path.exists():
        print(
            f"Manifest not found ({manifest_path}); example waveform plots will be skipped."
        )
        return {}
    tasks = pd.read_csv(manifest_path)
    lookup = {}
    for _, row in tasks.iterrows():
        match = re.match(
            r"chan(\d+)(?:_(\d+))?\.mat$", Path(str(row["data_path"])).name, re.I
        )
        if not match:
            continue
        key = task_key(
            row["rat"], row["region"], row["date"], match.group(2) or "", match.group(1)
        )
        lookup[key] = Path(str(row["data_path"]))
    return lookup


def plot_event(row, data_path: Path, output_path: Path):
    try:
        signal = np.asarray(loadmat(data_path)["data"]).squeeze()
        start = int(round(float(row["spindle_start_time_s"]) * FS))
        end = int(round(float(row["spindle_end_time_s"]) * FS))
        left = max(0, start - int(PLOT_PADDING_SEC * FS))
        right = min(len(signal), end + int(PLOT_PADDING_SEC * FS))
        if right <= left:
            return False
        times = np.arange(left, right) / FS
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(times, signal[left:right], linewidth=0.7, color="black")
        ax.axvspan(
            start / FS,
            end / FS,
            color="tab:orange",
            alpha=0.25,
            label="Wavelet spindle",
        )
        ax.set(
            xlabel="Time (s)",
            ylabel="LFP",
            title=(
                f"Rat {row['rat_number']} | {row['date']} | OQ={row['oq']:.3f} | "
                f"{row['oq_bin']}"
            ),
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"Could not plot {data_path}: {exc}")
        return False


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Event results CSV not found: {RESULTS_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)
    required = {
        "rat_number",
        "date",
        "r_max",
        "spindle_start_time_s",
        "spindle_end_time_s",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Results CSV is missing required columns: {sorted(missing)}")

    df["oq"] = pd.to_numeric(df["r_max"], errors="coerce")
    df["date"] = df["date"].astype(str)
    valid = df.dropna(subset=["oq"]).copy()
    if "in_band_ratio" in valid:
        valid = valid[pd.to_numeric(valid["in_band_ratio"], errors="coerce") > 0]

    # Use chronological dates when parseable, with lexical order as a fallback.
    def date_sort_key(value):
        parsed = pd.to_datetime(value, errors="coerce")
        return (pd.isna(parsed), parsed if not pd.isna(parsed) else value)

    for rat, indexes in valid.groupby("rat_number").groups.items():
        dates = sorted(valid.loc[indexes, "date"].unique(), key=date_sort_key)
        day_map = {date: day for day, date in enumerate(dates, start=1)}
        valid.loc[indexes, "training_day"] = valid.loc[indexes, "date"].map(day_map)

    bins = pd.IntervalIndex.from_breaks(OQ_EDGES, closed="left")
    labels = [
        f"{interval.left:.2f}-{min(interval.right, 1.0):.2f}" for interval in bins
    ]
    valid["oq_bin"] = pd.cut(valid["oq"], bins=OQ_EDGES, labels=labels, right=False)

    rng = np.random.default_rng(RANDOM_SEED)
    sampled_parts = []
    for (_, _), group in valid.dropna(subset=["oq_bin"]).groupby(
        ["rat_number", "oq_bin"], observed=True
    ):
        count = min(SAMPLES_PER_RAT_BIN, len(group))
        chosen_positions = rng.choice(len(group), size=count, replace=False)
        sampled_parts.append(group.iloc[chosen_positions].copy())
    samples = (
        pd.concat(sampled_parts, ignore_index=True)
        if sampled_parts
        else valid.iloc[0:0].copy()
    )
    sample_csv = OUTPUT_DIR / "oq_range_samples.csv"
    samples.to_csv(sample_csv, index=False)

    lookup = recording_lookup(MANIFEST_CSV)
    if not samples.empty:
        plot_dir = OUTPUT_DIR / "example_waveforms"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = []
        for sample_num, (_, row) in enumerate(samples.iterrows(), start=1):
            key = task_key(
                row["rat_number"],
                row.get("region", ""),
                row["date"],
                row.get("trial", ""),
                row.get("channel", ""),
            )
            data_path = lookup.get(key)
            if data_path is None:
                plot_paths.append("")
                continue
            safe_bin = str(row["oq_bin"]).replace(".", "p").replace("-", "_")
            file_name = (
                f"rat_{row['rat_number']}_bin_{safe_bin}_event_{sample_num:04d}.png"
            )
            out_path = plot_dir / file_name
            plot_event(row, data_path, out_path)
            plot_paths.append(str(out_path.relative_to(OUTPUT_DIR)))
        samples["example_plot"] = plot_paths
        samples.to_csv(sample_csv, index=False)

    daily = valid.groupby(["rat_number", "date", "training_day"], as_index=False).agg(
        n_spindles=("oq", "size"), mean_oq=("oq", "mean"), median_oq=("oq", "median")
    )
    daily.to_csv(OUTPUT_DIR / "oq_by_training_day.csv", index=False)

    if not daily.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for rat, group in daily.groupby("rat_number"):
            group = group.sort_values("training_day")
            ax.plot(
                group["training_day"], group["mean_oq"], marker="o", label=f"Rat {rat}"
            )
        ax.set(
            xlabel="Training day (date order)",
            ylabel="Mean OQ (peak R)",
            title="Mean spindle oscillation quality across training days",
        )
        ax.set_xticks(sorted(daily["training_day"].dropna().unique()))
        ax.legend(title="Rat", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "mean_oq_by_training_day.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        for rat, group in daily.groupby("rat_number"):
            group = group.sort_values("training_day")
            ax.plot(
                group["training_day"],
                group["median_oq"],
                marker="o",
                label=f"Rat {rat}",
            )
        ax.set(
            xlabel="Training day (date order)",
            ylabel="Median OQ (peak R)",
            title="Median spindle oscillation quality across training days",
        )
        ax.set_xticks(sorted(daily["training_day"].dropna().unique()))
        ax.legend(title="Rat", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "median_oq_by_training_day.png", dpi=160)
        plt.close(fig)

    print(f"Valid spindle OQ values: {len(valid)}")
    print(f"Sampled events: {len(samples)}; saved {sample_csv}")
    print(f"Daily summary and plots saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
