from pathlib import Path

# Signal Processing Constants
FS = 1000
TARGET_FS = 128
N_SURROGATES = 4
SEGMENT_SEC = 10 * 60
TARGET_PCT_DIFF = 92.0

# System Constants
CHECKPOINT_EVERY = 25
OUTPUT_DIR = Path("results")
RAW_CSV = OUTPUT_DIR / "all_thresholds_raw.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary_thresholds_per_rat_region.csv"
LOG_DIR = OUTPUT_DIR / "logs"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
