import csv
import logging
import os
import re
import sys
from pathlib import Path

# Adjust path if needed to reach your custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from modules.project_config import get_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Trial number at the end of the data filename, e.g. "chan51_7.mat" -> "7"
DATA_TRIAL_RE = re.compile(r"_(\d+)\.mat$")

# Trial number embedded in the scoring filename, e.g.
# "Rat_HM_Ephys_TD_Rat6_20221118_postsleep_07_FR-eegstates.mat" -> "07"
SCORING_TRIAL_RE = re.compile(r"_(\d+)_[A-Za-z]+-eegstates\.mat$")


def build_manifest(root_dirs, output_csv="tasks_manifest.csv"):
    manifest_records = []
    skipped_scoring_missing = 0
    skipped_unparseable_data = 0
    skipped_trial_no_match = 0
    ambiguous_matches = 0

    for root in root_dirs:
        root_path = Path(root)
        if not root_path.exists():
            logging.warning(f"Root path not found: {root_path}")
            continue

        logging.info(f"Scanning root: {root_path}")

        # Matches: {cohort}/PreprocessedData/{region}/{rat}/{date}/postsleep/{file.mat}
        pattern = "*/PreprocessedData/*/*/*/postsleep/*.mat"

        for data_path in root_path.glob(pattern):
            date = data_path.parents[1].name
            rat = data_path.parents[2].name
            region = data_path.parents[3].name
            cohort_dir = data_path.parents[5]
            cohort = cohort_dir.name

            # --- Extract trial number from the data filename ---
            data_match = DATA_TRIAL_RE.search(data_path.name)
            if not data_match:
                skipped_unparseable_data += 1
                logging.warning(f"Could not parse trial number from data file: {data_path}")
                continue
            data_trial = data_match.group(1)

            # --- Find all scoring files for this rat/date, any scorer suffix ---
            scoring_date_dir = cohort_dir / "Scoring" / rat / date / "postsleep"
            all_scoring_files = list(scoring_date_dir.glob("*-eegstates.mat"))

            if not all_scoring_files:
                skipped_scoring_missing += 1
                logging.warning(f"No scoring files found in: {scoring_date_dir}")
                continue

            # --- Match by trial number (zero-padding-insensitive) ---
            matches = []
            for f in all_scoring_files:
                m = SCORING_TRIAL_RE.search(f.name)
                if m and m.group(1).lstrip("0") == data_trial.lstrip("0"):
                    matches.append(f)

            if not matches:
                skipped_trial_no_match += 1
                logging.warning(
                    f"No scoring file matching trial '{data_trial}' for {data_path} "
                    f"(candidates in folder: {[f.name for f in all_scoring_files]})"
                )
                continue

            if len(matches) > 1:
                ambiguous_matches += 1
                logging.warning(
                    f"Multiple scoring files matched trial '{data_trial}' for {data_path}: "
                    f"{[f.name for f in matches]} -> using {matches[0].name}"
                )

            scoring_path = matches[0]

            manifest_records.append({
                "cohort": cohort,
                "rat": rat,
                "region": region,
                "date": date,
                "trial": data_trial,
                "data_path": str(data_path.resolve()),
                "scoring_path": str(scoring_path.resolve())
            })

    # Write to CSV
    if manifest_records:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_records[0].keys())
            writer.writeheader()
            writer.writerows(manifest_records)

        logging.info(f"Successfully generated manifest: {output_csv} ({len(manifest_records)} records)")
    else:
        logging.warning("No valid pairs found. Manifest not created.")

    logging.info(
        "Summary -- matched: %d, no scoring files in folder: %d, unparseable data filename: %d, "
        "no trial match: %d, ambiguous (multiple) matches: %d",
        len(manifest_records),
        skipped_scoring_missing,
        skipped_unparseable_data,
        skipped_trial_no_match,
        ambiguous_matches,
    )


if __name__ == "__main__":
    r1_8_root = get_path("R1_8_root")
    r9_16_root = get_path("R9_16_root")

    roots_to_scan = [r1_8_root, r9_16_root]

    build_manifest(roots_to_scan, output_csv="tasks_manifest.csv")
