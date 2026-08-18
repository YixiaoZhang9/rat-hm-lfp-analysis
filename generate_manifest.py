import csv
import logging
import os
import sys
from pathlib import Path

# Adjust path if needed to reach your custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from modules.project_config import get_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def build_manifest(root_dirs, output_csv="tasks_manifest.csv"):
    manifest_records = []
    skipped_scoring_missing = 0

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

            scoring_date_dir = cohort_dir / "Scoring" / rat / date / "postsleep"

            # Find the SW-eegstates.mat file
            scoring_files = list(scoring_date_dir.glob("*SW-eegstates.mat"))

            if not scoring_files:
                skipped_scoring_missing += 1
                continue

            scoring_path = scoring_files[0]

            manifest_records.append({
                "cohort": cohort,
                "rat": rat,
                "region": region,
                "date": date,
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

    if skipped_scoring_missing:
        logging.info(f"Skipped {skipped_scoring_missing} data files due to missing scoring files.")

if __name__ == "__main__":
    r1_8_root = get_path("R1_8_root")
    r9_16_root = get_path("R9_16_root")

    roots_to_scan = [r1_8_root, r9_16_root]

    build_manifest(roots_to_scan, output_csv="tasks_manifest.csv")
