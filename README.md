# Rat HM LFP Analysis

This repository contains analysis scripts for Rat HM local field potential
(LFP) recordings. The code loads OpenEphys recordings, preprocesses LFP data,
aligns sleep scoring, detects neural events, and supports visualization and
method comparison.

The main event types are:

- Hippocampal ripples
- Sleep spindles
- Delta waves

This is a research workflow repository. Most scripts are runnable analysis
entry points and still follow the original experiment-specific folder layout.

## Repository Structure

```text
modules/                    Reusable code for I/O, preprocessing, detection, features, and viewers
pipeline/                   Runnable analysis scripts grouped by workflow
pipeline/00_view_data/      Data inspection scripts
pipeline/01_load_and_store_ephys/
pipeline/02_preprocessing/
pipeline/ripple_detection/
pipeline/spindle_detection/
pipeline/delta_detection/
configs/                    Shared project metadata
archive/                    Archived local scripts
results/                    Local generated outputs, ignored by Git
```

## Installation

Create a Python environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Some workflows also require PyQt, TensorFlow, PyTorch, trained model
files, or local training data. Model files and data files are not stored in
Git.

## Configuration

Local data paths are read from `config.yaml`. This file is ignored by Git so
each collaborator can use their own local paths.

Create it from the example file:

```bash
cp config.example.yaml config.yaml
```

Then edit `config.yaml`:

```yaml
paths:
  R1_8_root: /path/to/Rat_HM_Ephys_TD
  R9_16_root: /path/to/Rat_HM_Ephys_TD
  R1_8_Raw_root: /path/to/Rat_HM_Ephys_TD
  ripple_marking_root: /path/to/Ripple_Marking
  ripple_training_data: /path/to/training_data
```

The scripts access these paths through `modules.project_config.get_path(...)`.
You can also set `RAT_HM_CONFIG` to use a different config file.

`configs/rat_groups.json` stores shared metadata for the four rat groups:

- `buildup_r1_4`
- `buildup_r5_8`
- `update_r9_12`
- `update_r13_16`

Most current scripts still define their own rat lists, dates, and special
cases internally. The JSON file is shared metadata for standardizing these
groups.

## Pipeline Overview

The typical data flow is:

```text
OpenEphys raw recordings
    -> RawData/*.mat
    -> Preprec_withartifacts/
    -> Scoring/
    -> PreprocessedData/
    -> Ripple_detection_results/
    -> Spindle_detection_results/
    -> Delta_detection_results/
```

Run scripts by rat group. For example, use the matching script or folder for
`buildup_r1_4`, `buildup_r5_8`, `update_r9_12`, or `update_r13_16`.

Recommended order:

1. Inspect data if needed:
   `pipeline/00_view_data/`
2. Load and store selected OpenEphys channels:
   `pipeline/01_load_and_store_ephys/`
3. Run basic preprocessing:
   `pipeline/02_preprocessing/<group>/01_basic_preprocessing.py`
4. Align sleep scoring and LFP duration:
   `pipeline/02_preprocessing/<group>/02_scoring_alignment.py`
5. Reduce artifacts:
   `pipeline/02_preprocessing/<group>/03_artifact_reduction.py`
6. Run event detection:
   `pipeline/ripple_detection/`,
   `pipeline/spindle_detection/`,
   `pipeline/delta_detection/`
7. Run visualization, validation, or method comparison scripts as needed.

There is no single workflow runner at the moment. The scripts are intended to
be run directly.

## Data and Model Policy

Do not commit raw data, processed data, generated results, training data, or
model checkpoints.

Ignored artifact types include:

- `*.mat`, `*.npz`, `*.csv`
- `*.pth`, `*.pkl`, `*.h5`
- `training_data/`, `trained_networks/`, `results/`

Store large files outside Git and document where collaborators can access
them.

## Development

Run these checks before committing changes:

```bash
python -m ruff format modules pipeline archive
python -m ruff check modules pipeline archive
python -m compileall -q modules pipeline archive
```

When adding shared logic, prefer adding reusable functions to `modules/` while
keeping existing pipeline entry scripts stable.
