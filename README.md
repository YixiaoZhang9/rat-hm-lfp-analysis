# Rat HM LFP Analysis

Code for loading, preprocessing, visualizing, and detecting events in rat LFP
recordings. The current workflows cover ripple, spindle, and delta detection,
manual ripple annotation, model-based ripple detection, and method comparison.

This repository is being organized for collaboration. The analysis scripts are
still close to the original research workflows, so most scripts expect the
project-specific folder layout used for the Rat HM data.

## Repository Layout

```text
modules/                    Reusable Python helpers for I/O, filtering, event detection, features, and viewers
pipeline/                   Runnable analysis scripts grouped by workflow
pipeline/00_view_data/      Raw/loaded data inspection scripts
pipeline/01_load_and_store_ephys/
pipeline/02_preprocessing/
pipeline/delta_detection/
pipeline/ripple_detection/
pipeline/spindle_detection/
ripple_manual_annotation/   Manual ripple annotation and comparison utilities
configs/                    Example local path and rat-group configuration
scripts/                    Thin command-line wrappers for running existing workflow scripts
matlab/                     MATLAB helpers, if separated from workflow folders later
results/                    Local generated outputs; ignored by Git
```

## Installation

Create and activate a Python environment, then install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

Some workflows also require MATLAB scripts and deep-learning model files. Model
checkpoints and training data are intentionally not tracked in Git.

## Local Data Paths

The original scripts used absolute paths such as `/media/yixiao/...`. Those
defaults are preserved for backward compatibility, but collaborators can
override them with environment variables:

```bash
export RAT_HM_GL14_ROOT=/path/to/Rat_HM_Ephys_TD_Analysis_New_root
export RAT_HM_DATA4_ROOT=/path/to/Rat_HM_Ephys_TD_Analysis_root
export RAT_HM_RIPPLE_MARKING_ROOT=/path/to/Ripple_Marking
export RAT_HM_RIPPLE_TRAINING_DATA=/path/to/training_data
```

See `configs/local_paths.example.json` for the expected path keys.

## Running Existing Workflows

The original scripts can still be run directly. A consolidated wrapper is also
provided for common rat-group workflows:

```bash
python scripts/run_workflow.py --list
python scripts/run_workflow.py ripple_threshold buildup_r1_4
python scripts/run_workflow.py spindle_threshold update_r13_16
python scripts/run_workflow.py delta buildup_r5_8
```

The wrapper delegates to the existing scripts; it does not change the analysis
logic.

## Data and Model Policy

Do not commit raw LFP data, derived result tables, training datasets, or model
checkpoints. Store these outside Git, for example in institutional storage,
OSF, Zenodo, Figshare, or GitHub Releases, and document how to obtain them.

Ignored artifact types include:

- `*.mat`, `*.npz`, large `*.csv`
- `*.pth`, `*.pkl`, `*.h5`
- `training_data/`, `trained_networks/`, `results/`

## Development

Formatting and import cleanup are configured in `pyproject.toml`:

```bash
python -m ruff format modules pipeline ripple_manual_annotation scripts
python -m ruff check modules pipeline ripple_manual_annotation scripts
python -m compileall -q modules pipeline ripple_manual_annotation scripts
```

Before opening a pull request, run the checks above and confirm that no private
data or large artifacts are staged.
