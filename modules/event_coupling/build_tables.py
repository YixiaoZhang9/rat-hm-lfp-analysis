# build_tables.py
# Loader for the detection results. Reads only the folders it needs, caches per
# unit, can resume, and retries when the drive drops.
#
# Usage (after the config cell defines ROOT, SPINDLE_METHOD, PHASES, SESSION_MAP_CSV):
#
#     from build_tables import build_all
#     ripples, spindles, deltas = build_all(ROOT, SPINDLE_METHOD, PHASES,
#                                            SESSION_MAP_CSV, rebuild=False)
#
# Reads the exact folder per area/rat/date/phase, e.g.
#   ...\Spindle_detection_results\HPC\1\20220923\postsleep\wavelet_amp_1_ampcore_3
# One cache file per (sub-group, area, rat); a drive error skips that unit,
# run build_all again to resume.
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_GROUPS = {
    "Rat_HM_Ephys_TD_Analysis_R1_8": ["R1-4", "R5-8"],
    "Rat_HM_Ephys_TD_Analysis_R9_16": ["R9-12", "R13-16"],
}
FOLDER = {
    "ripple": "Ripple_detection_results",
    "spindle": "Spindle_detection_results",
    "delta": "Delta_detection_results",
}


def _retry(func, tries=5, wait=10):
    """Run func(); on a drive drop, wait and retry. Returns None if it keeps failing."""
    for attempt in range(tries):
        try:
            return func()
        except (FileNotFoundError, OSError) as e:
            print(f"      network blip ({attempt + 1}/{tries}), waiting {wait}s... {e}")
            time.sleep(wait)
    return None


def _read_csv_safe(path):
    """Read a CSV, falling back to latin1 if it is not valid UTF-8.

    Some Sept 2022 files contain bytes like 0xb0 (degree sign).
    """
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def files_for_date(phase_dir, wave, spindle_method):
    """The exact CSV files to read for one date/phase folder."""
    if not phase_dir.exists():
        return []
    if wave == "ripple":
        return list(phase_dir.glob("*ripples_threshold2.csv"))
    if wave == "delta":
        return list(phase_dir.rglob("*deltas.csv"))
    method_dir = None
    for d in phase_dir.iterdir():
        if d.is_dir() and d.name.lower() == spindle_method.lower():
            method_dir = d
            break
    return list(method_dir.glob("*.csv")) if method_dir else []


def read_unit(det, area, rat, wave, spindle_method, phases):
    """Read one (area, rat) unit by walking its date folders explicitly.

    Returns a DataFrame. Raises OSError if the drive can't be read after retries.
    """
    rat_dir = (det / str(rat)) if wave == "ripple" else (det / area / str(rat))
    dates = _retry(
        lambda: [
            d
            for d in rat_dir.iterdir()
            if d.is_dir() and re.fullmatch(r"\d{8}", d.name)
        ]
    )
    if dates is None:
        raise OSError(f"drive kept dropping while listing {rat_dir}")

    frames = []
    for date_dir in dates:
        for phase in phases:
            phase_dir = date_dir / phase
            files = _retry(lambda: files_for_date(phase_dir, wave, spindle_method))
            if files is None:
                raise OSError(f"drive kept dropping in {phase_dir}")
            for f in files:
                df = _retry(lambda f=f: _read_csv_safe(f))
                if df is None:
                    print("      unreadable after retries, skipping:", f.name)
                    continue
                if df.empty:
                    continue
                df["rat"] = rat
                df["area"] = "HPC" if wave == "ripple" else area
                df["date"] = date_dir.name
                df["phase"] = phase

                # channel + trial from the filename, e.g. "chan102_6".
                # early unsplit recordings are just "chan102": trial stays
                # None and is filled in later in build_wave
                m = re.search(r"chan(\d+)_(\d+)", f.stem)
                if m:
                    df["channel"] = int(m.group(1))
                    df["trial"] = int(m.group(2))
                else:
                    m2 = re.search(r"chan(\d+)", f.stem)
                    df["channel"] = int(m2.group(1)) if m2 else None
                    df["trial"] = None

                df["wave"] = wave
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def units_for_wave(root, wave):
    """Yield (sub, area, rat, detection_dir) for every unit to read."""
    root = Path(root)
    units = []
    for analysis, subgroups in ANALYSIS_GROUPS.items():
        for sub in subgroups:
            det = root / analysis / sub / FOLDER[wave]
            if not det.exists():
                print(f"  (skip, not found) {det}")
                continue
            if wave == "ripple":
                for d in det.iterdir():
                    if d.is_dir() and re.fullmatch(r"\d{1,2}", d.name):
                        units.append((sub, None, int(d.name), det))
            else:
                for area in ("HPC", "PL", "RSC"):
                    area_dir = det / area
                    if not area_dir.exists():
                        continue
                    for d in area_dir.iterdir():
                        if d.is_dir() and re.fullmatch(r"\d{1,2}", d.name):
                            units.append((sub, area, int(d.name), det))
    return units


def build_wave(root, wave, spindle_method, phases, cache_dir, rebuild):
    wave_cache = Path(cache_dir) / wave
    wave_cache.mkdir(parents=True, exist_ok=True)

    units = units_for_wave(root, wave)
    print(f"{wave}: {len(units)} units to gather")

    for sub, area, rat, det in sorted(units, key=lambda u: (u[0], u[1] or "", u[2])):
        tag = f"{sub}_rat{rat}" if area is None else f"{sub}_{area}_rat{rat}"
        cache_file = wave_cache / f"{tag}.pkl"
        if cache_file.exists() and not rebuild:
            continue
        label = f"{wave} {sub} rat {rat}" + ("" if area is None else f" {area}")
        print(f"  reading {label} ...")
        try:
            df = read_unit(det, area, rat, wave, spindle_method, phases)
        except OSError as e:
            print(f"     drive error, will retry next run: {e}")
            continue
        if df.empty:
            print("     0 rows (not cached, will retry next run)")
            continue
        df.to_pickle(cache_file)
        print(f"     -> {len(df)} rows saved")

    parts = []
    for pkl in sorted(wave_cache.glob("*.pkl")):
        d = pd.read_pickle(pkl)
        if not d.empty:
            parts.append(d)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
    out["cohort"] = np.where(out["rat"] <= 8, "rats1-8", "rats9-16")
    try:
        from .excluded_days import drop_excluded

        out = drop_excluded(out)
    except ImportError:
        print("  note: excluded_days.py not found, no days excluded")

    n_missing = out["trial"].isna().sum()
    if n_missing:
        days = out.loc[out["trial"].isna(), ["rat", "date"]].drop_duplicates()
        print(
            f"  {n_missing:,} rows had no trial number "
            f"({len(days)} rat-days) -> filled with 1"
        )
        out["trial"] = out["trial"].astype("float64").fillna(1).astype("int64")

    return out


def add_condition(df, session_map_csv):
    if df.empty:
        return df
    sessions = pd.read_csv(session_map_csv, parse_dates=["date"])[
        ["rat", "date", "condition"]
    ]
    df = df.merge(sessions, on=["rat", "date"], how="left")
    df["condition"] = df["condition"].fillna("other")
    df["group"] = np.where(
        df["condition"] == "HC",
        "HC",
        np.where(df["condition"].str.startswith("S"), "Training", "other"),
    )
    return df


def build_all(
    root, spindle_method, phases, session_map_csv, rebuild=False, cache_dir="wave_cache"
):
    out = {}
    for wave in ["ripple", "spindle", "delta"]:
        print(f"\n=== {wave} ===")
        df = build_wave(root, wave, spindle_method, phases, cache_dir, rebuild)
        df = add_condition(df, session_map_csv)
        out[wave] = df
        print(f"{wave} total: {len(df)} rows")
    r, s, d = out["ripple"], out["spindle"], out["delta"]
    print("\nripples :", r.shape)
    print("spindles:", s.shape)
    print("deltas  :", d.shape)
    for name, df in [("spindle", s), ("delta", d)]:
        if not df.empty:
            print(f"\n{name} events per brain area:")
            print(df["area"].value_counts())
    return r, s, d
