# Event coupling notebooks (Leticia)

These notebooks look at how delta waves, ripples and spindles are coupled during
postsleep NREM, and whether that coupling changes over training (HC vs Training,
goal-location blocks, days since the first recording). Each analysis has a
full-postsleep version and a first-hour-of-postsleep version.

The helper code is in `modules/event_coupling/`. Saved outputs were cleared before
committing, so the notebooks have to be run to see results.

## Notebooks

| Notebook | What it does |
|---|---|
| `Task2_delta_ripple_sequence` | Delta -> ripple and ripple -> delta sequences: per-day measures, pooled lag histograms and circular-shift significance, delta phase at ripple peaks (Hilbert), trends over time, block plots, ripple-anchored proportions normalized to HC |
| `Task2_ripple_spindle_sequence` | Ripple -> spindle sequences: lag histograms, circular-shift surrogate, HC vs Training |
| `Task2b_ripple_spindle_overlap_full` | Ripple-spindle overlap ("ripple inside a spindle"): per-day rate and ratio, per-day circular shift, trends over time, pooled significance |
| `Task2d_delta_spindle_full` | Delta -> spindle for all nine delta-area x spindle-area pairs: lag histograms, surrogates, delta phase at spindle peaks, HC vs Training, over-time plots with HC normalization (7 rats) |
| `Task2d_delta_spindle_16rats_full` | Same as above on all 16 rats, plots only (no surrogates) |
| `cross_regional_full` | Same wave type in two areas (PL-RSC, PL-HPC, RSC-HPC; delta and spindle): nearest-neighbour co-occurrence, wave similarity on an 8% sample of files, co-occurrence over time with permutation tests |
| `cross_regional_16rats_full` | Same on all 16 rats (no circular shift or wave similarity) |

Each one has a `_first_hour` twin. It uses only events in the first hour of
postsleep and NREM minutes within hour 1, and `cross_regional_first_hour` reads every
file for wave similarity. The 7-rat notebooks use rats 1, 3, 7, 8, 11, 12 and 13,
because those rats have ripples.

All over-time parts drop rat-days with less than 10 min of NREM (`MIN_NREM_MIN = 10`)
and list the removed days in red under each plot.

## Setup

1. Install the repo requirements (`pip install -r requirements.txt`).
2. Copy `config.example.yaml` to `config.yaml` and set these keys:

   | Key | What it points to |
   |---|---|
   | `R1_8_root`, `R9_16_root` | The `Rat_HM_Ephys_TD` folder. These notebooks expect both keys to be the same folder. |
   | `event_cache_root` | Folder with the cached event tables (`delta/`, `ripple/`, `spindle/`, `gl_sessions.pkl`) |
   | `session_map_csv` | `rat_sessions_HC_S1-5.csv` |
   | `day_grid_xlsx` | `Rats_1-16__HC_vs_Training_days__.xlsx` |

   The event cache and the two session files are data, so they're not in Git.
   Ask Leticia for a copy. If `event_cache_root` points to an empty folder, the
   first run rebuilds the cache from the `*_detection_results` folders on the
   drive. This is slow and needs the drive connected.
3. Open a notebook from this folder and run it from the top. The first cell finds
   the repo root by itself, so `modules` imports work without extra setup.

## Outputs

Figures, surrogate caches and checkpoints are written to
`results/leticia_event_coupling/<folder>/`, which Git ignores. To reuse finished
surrogates instead of recomputing them (1000 iterations can take hours), copy the
existing cache and checkpoint folders (`significance_cache*`, `task2d_*checkpoints`,
`spindle_surrogate_cache*`, `wave_similarity_cache_files*`, ...) into that folder.

The delta-phase cells and the optional checks at the end of
`cross_regional_first_hour` read `.mat` files from `PreprocessedData` / `Scoring`
on the drive. Everything else runs from the event cache.

## Notes

- Some markdown cells refer to files that are not in this repo
  (`PROJECT_DATA_REFERENCE.md`, `Corrected_Task2_OverTime.ipynb`,
  `Task2_coupling.ipynb`). These are earlier notes and notebooks that Leticia kept.
- In the cross-regional notebooks, the circular-shift sections are commented out
  on purpose. Wave similarity is also commented out in the 16-rat versions.
