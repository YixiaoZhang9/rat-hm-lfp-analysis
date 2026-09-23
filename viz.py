#!/usr/bin/env python3
"""
Spindle Alignment Inspector — final AR detector view.

Shows:

1. Raw LFP
2. 9–20 Hz band-passed LFP
3. AR R-value trajectory
4. AR peak-frequency trajectory

The AR R/frequency traces are computed using the same signal-processing
pipeline used by the final spindle detector:

    Raw LFP
        ↓
    0.1–100 Hz preprocessing filter
        ↓
    downsample to 128 Hz
        ↓
    1-second AR(8) windows
        ↓
    4-sample stride = 31.25 ms
        ↓
    select AR pole in 9–20 Hz with maximum R

Final region-specific thresholds:

    HPC: upper = 0.84, lower = 0.68
    PL:  upper = 0.82, lower = 0.67
    RSC: upper = 0.86, lower = 0.72

The GUI shows a 10-second window centered on the selected detected event.
"""

# =====================================================================
# Imports
# =====================================================================

import argparse
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import statsmodels.api as sm
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

# =====================================================================
# Import preprocessing modules
# =====================================================================

PROJECT_ROOT = os.environ.get(
    "AR_MODULES_ROOT",
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../.."
        )
    ),
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


MODULES_AVAILABLE = False
_IMPORT_ERROR = ""


try:

    from modules.ephys_preprocessing import bandpass_filter as ar_bandpass_filter
    from modules.ephys_preprocessing import downsampling as ar_downsampling

    MODULES_AVAILABLE = True

except ImportError as e:

    MODULES_AVAILABLE = False
    _IMPORT_ERROR = str(e)


# =====================================================================
# Configuration
# =====================================================================

DEFAULT_MANIFEST = os.environ.get(
    "SPINDLE_MANIFEST",
    "tasks_manifest.csv"
)

DEFAULT_AR_CSV = os.environ.get(
    "SPINDLE_AR_CSV",
    "results/all_detected_spindles_per_region.csv"
)


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------

FS = 1000.0

AR_TARGET_FS = 128


# ---------------------------------------------------------------------
# FINAL DETECTION BAND
# ---------------------------------------------------------------------

BP_LOW = 9.0
BP_HIGH = 20.0

AR_SPINDLE_BAND = (
    9.0,
    20.0
)


# ---------------------------------------------------------------------
# AR detector parameters
# ---------------------------------------------------------------------

AR_ORDER = 8

AR_WINDOW_SEC = 1.0

AR_STRIDE_SAMPLES = 4


# ---------------------------------------------------------------------
# Preprocessing used before AR fitting
# ---------------------------------------------------------------------

AR_PRE_LOWCUT = 0.1
AR_PRE_HIGHCUT = 100.0


# ---------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------

VIEW_WINDOW_SEC = 10.0

DEBOUNCE_MS = 300

STALE_TOLERANCE_SEC = 0.5


# ---------------------------------------------------------------------
# FINAL REGION-SPECIFIC THRESHOLDS
# ---------------------------------------------------------------------

REGION_THRESHOLDS = {

    "HPC": {
        "upper": 0.84,
        "lower": 0.68,
    },

    "PL": {
        "upper": 0.82,
        "lower": 0.67,
    },

    "RSC": {
        "upper": 0.86,
        "lower": 0.72,
    },

}

DEFAULT_THRESHOLDS = {
    "upper": 0.84,
    "lower": 0.68,
}


# ---------------------------------------------------------------------
# Manifest requirements
# ---------------------------------------------------------------------

REQUIRED_MANIFEST_COLS = {
    "rat",
    "region",
    "date",
    "data_path",
}


VERBOSE = False


# =====================================================================
# Utility
# =====================================================================

def vprint(*args, **kwargs):

    if VERBOSE:
        print(
            *args,
            flush=True,
            **kwargs
        )


# =====================================================================
# Sanitize signal
# =====================================================================

def _sanitize(x):

    arr = np.asarray(
        x,
        dtype=np.float64
    )

    if not np.all(np.isfinite(arr)):

        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    return arr


# =====================================================================
# AR Window Worker
# =====================================================================

class ARWindowWorker(QtCore.QThread):

    """
    Computes the AR R and frequency trajectory for the visible window.

    Every point corresponds to one 1-second AR(8) window.

    Windows are evaluated every 4 samples at 128 Hz:

        4 / 128 = 0.03125 s

    For each window:

        Burg AR(8)
            ↓
        AR poles
            ↓
        positive-frequency poles
            ↓
        keep 9–20 Hz poles
            ↓
        choose pole with maximum R

    The returned frequency is the frequency of the SAME pole
    that produced the R value.
    """

    finished_ok = QtCore.pyqtSignal(
        float,
        float,
        list,
        list,
        list,
    )

    failed = QtCore.pyqtSignal(str)


    def __init__(
        self,
        signal_128,
        view_start_s,
        view_end_s,
        parent=None
    ):

        super().__init__(parent)

        self.signal_128 = signal_128

        self.view_start_s = max(
            0.0,
            float(view_start_s)
        )

        self.view_end_s = float(
            view_end_s
        )

        self._is_cancelled = False


    def cancel(self):

        self._is_cancelled = True


    def run(self):

        try:

            n128 = len(
                self.signal_128
            )

            # ---------------------------------------------------------
            # 1-second window = 128 samples
            # ---------------------------------------------------------

            win = int(
                AR_WINDOW_SEC
                * AR_TARGET_FS
            )


            # ---------------------------------------------------------
            # Stride
            # ---------------------------------------------------------

            stride = max(
                1,
                int(AR_STRIDE_SAMPLES)
            )


            # ---------------------------------------------------------
            # Window starts
            #
            # We select windows whose CENTER falls inside
            # the visible GUI range.
            # ---------------------------------------------------------

            start_center = int(
                self.view_start_s
                * AR_TARGET_FS
                - win / 2
            )

            end_center = int(
                self.view_end_s
                * AR_TARGET_FS
                - win / 2
            )


            # ---------------------------------------------------------
            # Clamp
            # ---------------------------------------------------------

            lo = max(
                0,
                start_center
            )

            hi = min(
                n128 - win,
                end_center
            )


            if hi < lo:

                self.finished_ok.emit(
                    self.view_start_s,
                    self.view_end_s,
                    [],
                    [],
                    [],
                )

                return


            # ---------------------------------------------------------
            # Align to detector's global stride grid
            # ---------------------------------------------------------

            first = (
                (lo + stride - 1)
                // stride
            ) * stride


            starts = np.arange(
                first,
                hi + 1,
                stride
            )


            # ---------------------------------------------------------
            # Output arrays
            # ---------------------------------------------------------

            n_windows = len(
                starts
            )

            r_vals = np.zeros(
                n_windows,
                dtype=np.float64
            )

            freq_vals = np.full(
                n_windows,
                np.nan,
                dtype=np.float64
            )


            # ---------------------------------------------------------
            # Fit AR model for every window
            # ---------------------------------------------------------

            for k, s in enumerate(starts):

                if self._is_cancelled:
                    return


                window = self.signal_128[
                    s:s + win
                ]


                if len(window) < win:
                    break


                try:

                    # -------------------------------------------------
                    # Burg AR
                    # -------------------------------------------------

                    a, _ = sm.regression.linear_model.burg(
                        window,
                        order=AR_ORDER,
                        demean=False
                    )


                    # -------------------------------------------------
                    # AR polynomial roots
                    # -------------------------------------------------

                    poles = np.roots(
                        np.r_[1, -a]
                    )


                    # Keep positive-frequency poles
                    poles = poles[
                        np.imag(poles) > 0
                    ]


                    if len(poles) == 0:
                        continue


                    # -------------------------------------------------
                    # Pole frequency
                    # -------------------------------------------------

                    freqs = (
                        np.angle(poles)
                        * AR_TARGET_FS
                        / (2 * np.pi)
                    )


                    # -------------------------------------------------
                    # Pole magnitude R
                    # -------------------------------------------------

                    r = np.abs(
                        poles
                    )


                    # -------------------------------------------------
                    # Keep 9–20 Hz poles
                    # -------------------------------------------------

                    mask = (

                        (freqs >= AR_SPINDLE_BAND[0])

                        &

                        (freqs <= AR_SPINDLE_BAND[1])

                    )


                    if not np.any(mask):
                        continue


                    # -------------------------------------------------
                    # Among 9–20 Hz poles:
                    # select maximum R
                    # -------------------------------------------------

                    masked_indices = np.where(
                        mask
                    )[0]


                    best_idx = masked_indices[
                        np.argmax(
                            r[mask]
                        )
                    ]


                    # -------------------------------------------------
                    # Store R and frequency of same pole
                    # -------------------------------------------------

                    r_vals[k] = float(
                        r[best_idx]
                    )

                    freq_vals[k] = float(
                        freqs[best_idx]
                    )


                except Exception:

                    # One failed AR window should not crash the GUI
                    pass


            # ---------------------------------------------------------
            # Time = center of each AR window
            # ---------------------------------------------------------

            t_vals = (
                starts
                + win / 2.0
            ) / AR_TARGET_FS


            if len(t_vals) > 0:

                vprint(
                    f"[AR] "
                    f"n={n_windows} "
                    f"t=[{t_vals[0]:.3f}, "
                    f"{t_vals[-1]:.3f}]"
                )


            # ---------------------------------------------------------
            # Emit
            # ---------------------------------------------------------

            self.finished_ok.emit(

                self.view_start_s,

                self.view_end_s,

                t_vals.tolist(),

                r_vals.tolist(),

                freq_vals.tolist(),

            )


        except Exception as e:

            traceback.print_exc()

            self.failed.emit(
                f"{type(e).__name__}: {e}"
            )


# =====================================================================
# Signal Loader
# =====================================================================

class SignalLoader(QtCore.QThread):

    """
    Loads one MAT recording and prepares:

        raw signal
        9–20 Hz display filter
        128 Hz AR signal
        time vector
    """

    finished_ok = QtCore.pyqtSignal(
        object,
        object,
        object,
        object,
    )

    failed = QtCore.pyqtSignal(str)


    def __init__(
        self,
        data_path,
        parent=None
    ):

        super().__init__(parent)

        self.data_path = data_path


    def run(self):

        try:

            vprint(
                f"\n[Load] "
                f"{self.data_path}"
            )


            # ---------------------------------------------------------
            # Load MAT
            # ---------------------------------------------------------

            mat_data = loadmat(
                self.data_path
            )


            if "data" not in mat_data:

                keys = [
                    k
                    for k in mat_data.keys()
                    if not k.startswith("__")
                ]

                raise KeyError(
                    f"'.mat' file has no "
                    f"'data' variable "
                    f"(found: {keys})"
                )


            raw = mat_data[
                "data"
            ].squeeze().astype(
                np.float32
            )


            if raw.ndim != 1:

                raw = raw.reshape(-1)


            # ---------------------------------------------------------
            # Sanitize
            # ---------------------------------------------------------

            n_bad = int(
                np.count_nonzero(
                    ~np.isfinite(raw)
                )
            )


            if n_bad:

                vprint(
                    f"[Load] "
                    f"sanitizing "
                    f"{n_bad} "
                    f"non-finite samples"
                )


                raw = np.nan_to_num(
                    raw,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )


            # ---------------------------------------------------------
            # Display filter: FINAL 9–20 Hz
            # ---------------------------------------------------------

            filtered = (
                butter_bandpass_filter(
                    raw,
                    BP_LOW,
                    BP_HIGH,
                    FS
                )
                .astype(np.float32)
            )


            filtered = np.nan_to_num(
                filtered,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )


            # ---------------------------------------------------------
            # AR detector preprocessing
            #
            # Same 0.1–100 Hz preprocessing
            # used before downsampling.
            # ---------------------------------------------------------

            ar_filtered = (
                ar_bandpass_filter(
                    raw,
                    lowcut=AR_PRE_LOWCUT,
                    highcut=AR_PRE_HIGHCUT,
                    fs=FS
                )
            )


            ar_filtered = _sanitize(
                ar_filtered
            )


            # ---------------------------------------------------------
            # Downsample to 128 Hz
            # ---------------------------------------------------------

            signal_128 = _sanitize(
                ar_downsampling(
                    ar_filtered,
                    FS,
                    AR_TARGET_FS
                )
            )


            # ---------------------------------------------------------
            # Time vector
            # ---------------------------------------------------------

            time_128 = (
                np.arange(
                    len(signal_128),
                    dtype=np.float64
                )
                / AR_TARGET_FS
            )


            vprint(
                f"[Load] "
                f"raw len={len(raw)} "
                f"std={raw.std():.3f} | "
                f"signal_128 len="
                f"{len(signal_128)}"
            )


            self.finished_ok.emit(
                raw,
                filtered,
                signal_128,
                time_128,
            )


        except Exception as e:

            vprint(
                f"[Load] "
                f"exception: "
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

            self.failed.emit(
                f"{type(e).__name__}: {e}"
            )


# =====================================================================
# Helpers
# =====================================================================

def butter_bandpass_filter(
    data,
    lowcut,
    highcut,
    fs,
    order=4
):

    nyq = 0.5 * fs

    low = max(
        0.001,
        lowcut / nyq
    )

    high = min(
        0.999,
        highcut / nyq
    )

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        data
    )


# =====================================================================

def clean_str(val):

    if pd.isna(val):
        return ""

    s = str(val).strip()

    if s.endswith(".0"):
        s = s[:-2]

    return s


# =====================================================================

def parse_channel_trial(
    data_path
):

    filename = Path(
        str(data_path)
    ).name


    match = re.match(
        r"chan(\d+)(?:_(\d+))?\.mat$",
        filename,
        re.IGNORECASE
    )


    if not match:

        return "", ""


    chan = str(
        int(match.group(1))
    )


    trial = (
        match.group(2)
        if match.group(2)
        else ""
    )


    return chan, trial


# =====================================================================

def safe_float(
    val,
    default=np.nan
):

    try:

        f = float(val)

        return (
            default
            if np.isnan(f)
            else f
        )

    except (
        TypeError,
        ValueError
    ):

        return default


# =====================================================================

def fmt(
    val,
    spec=".4f",
    placeholder="N/A"
):

    f = safe_float(val)

    if np.isnan(f):
        return placeholder

    return format(
        f,
        spec
    )


# =====================================================================

def resolve_interval_columns(
    df,
    default_fs=1000.0
):

    if df.empty:
        return df, True


    cols_lower = {
        str(c).lower().strip(): c
        for c in df.columns
    }


    start_candidates = [

        "start_s",
        "start_time",
        "start_sec",
        "start_secs",
        "start",
        "start_time_s",
        "spindle_start",
        "onset_s",
        "onset_time",

    ]


    end_candidates = [

        "end_s",
        "end_time",
        "end_sec",
        "end_secs",
        "end",
        "end_time_s",
        "spindle_end",
        "offset_s",
        "offset_time",

    ]


    def find_column(cands):

        for c in cands:

            if c in cols_lower:
                return cols_lower[c]


        for c in cands:

            for (
                lower_name,
                orig_name
            ) in cols_lower.items():

                if c in lower_name:
                    return orig_name


        return None


    found_start = find_column(
        start_candidates
    )

    found_end = find_column(
        end_candidates
    )


    if found_start and found_end:

        starts = pd.to_numeric(
            df[found_start],
            errors="coerce"
        ).fillna(0.0).values


        ends = pd.to_numeric(
            df[found_end],
            errors="coerce"
        ).fillna(0.0).values


        df["Start_s"] = starts

        df["End_s"] = ends

        return df, True


    if "Start_s" not in df.columns:
        df["Start_s"] = 0.0


    if "End_s" not in df.columns:
        df["End_s"] = 0.0


    return df, False


# =====================================================================
# Main Window
# =====================================================================

class SpindleViewer(
    QtWidgets.QMainWindow
):


    def __init__(
        self,
        manifest_path,
        ar_csv_path
    ):

        super().__init__()


        self.setWindowTitle(
            "Spindle Alignment Inspector — Final AR Detector"
        )


        self.resize(
            1500,
            1100
        )


        # -------------------------------------------------------------
        # Paths
        # -------------------------------------------------------------

        self.manifest_path = (
            manifest_path
        )

        self.ar_csv_path = (
            ar_csv_path
        )


        # -------------------------------------------------------------
        # Data
        # -------------------------------------------------------------

        self.manifest_df = None

        self.ar_df = pd.DataFrame()

        self.current_ar_events = (
            pd.DataFrame()
        )

        self.current_ar_idx = -1


        # -------------------------------------------------------------
        # Signal cache
        # -------------------------------------------------------------

        self.raw_signal = None

        self.filtered_signal = None

        self.signal_128 = None

        self.time_vector = None

        self.data_len = 0

        self.data_duration = 0.0


        # -------------------------------------------------------------
        # Threads
        # -------------------------------------------------------------

        self._loader_thread = None

        self._ar_signal_thread = None

        self._pending_task = None


        # -------------------------------------------------------------
        # View update debounce
        # -------------------------------------------------------------

        self.view_update_timer = (
            QtCore.QTimer(self)
        )

        self.view_update_timer.setSingleShot(
            True
        )

        self.view_update_timer.timeout.connect(
            self._start_window_computation
        )


        # -------------------------------------------------------------
        # UI
        # -------------------------------------------------------------

        self.init_ui()

        self.load_data()


    # =================================================================
    # UI
    # =================================================================

    def init_ui(
        self
    ):

        main_widget = (
            QtWidgets.QWidget()
        )

        self.setCentralWidget(
            main_widget
        )


        root_layout = (
            QtWidgets.QVBoxLayout(
                main_widget
            )
        )


        # =============================================================
        # Dataset filtering
        # =============================================================

        filter_box = (
            QtWidgets.QGroupBox(
                "Dataset Filtering"
            )
        )


        filter_layout = (
            QtWidgets.QHBoxLayout()
        )


        # -------------------------------------------------------------
        # Rat
        # -------------------------------------------------------------

        filter_layout.addWidget(
            QtWidgets.QLabel(
                "Rat:"
            )
        )


        self.combo_rat = (
            QtWidgets.QComboBox()
        )


        self.combo_rat.currentIndexChanged.connect(
            self.on_rat_changed
        )


        filter_layout.addWidget(
            self.combo_rat
        )


        # -------------------------------------------------------------
        # Region
        # -------------------------------------------------------------

        filter_layout.addWidget(
            QtWidgets.QLabel(
                "Region:"
            )
        )


        self.combo_region = (
            QtWidgets.QComboBox()
        )


        self.combo_region.currentIndexChanged.connect(
            self.on_region_changed
        )


        filter_layout.addWidget(
            self.combo_region
        )


        # -------------------------------------------------------------
        # File
        # -------------------------------------------------------------

        filter_layout.addWidget(
            QtWidgets.QLabel(
                "File:"
            )
        )


        self.combo_files = (
            QtWidgets.QComboBox()
        )


        self.combo_files.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContents
        )


        self.combo_files.currentIndexChanged.connect(
            self.on_file_selected
        )


        filter_layout.addWidget(
            self.combo_files,
            stretch=2
        )


        filter_box.setLayout(
            filter_layout
        )


        root_layout.addWidget(
            filter_box
        )


        # =============================================================
        # Navigation
        # =============================================================

        nav_box = (
            QtWidgets.QGroupBox(
                "AR Detection Navigation"
            )
        )


        nav_layout = (
            QtWidgets.QHBoxLayout()
        )


        self.btn_prev_ar = (
            QtWidgets.QPushButton(
                "◀ Prev AR"
            )
        )


        self.btn_prev_ar.clicked.connect(
            self.on_prev_ar_event
        )


        nav_layout.addWidget(
            self.btn_prev_ar
        )


        self.lbl_ar_tracker = (
            QtWidgets.QLabel(
                "AR: 0 / 0"
            )
        )


        self.lbl_ar_tracker.setAlignment(
            QtCore.Qt.AlignCenter
        )


        self.lbl_ar_tracker.setMinimumWidth(
            110
        )


        nav_layout.addWidget(
            self.lbl_ar_tracker
        )


        self.btn_next_ar = (
            QtWidgets.QPushButton(
                "Next AR ▶"
            )
        )


        self.btn_next_ar.clicked.connect(
            self.on_next_ar_event
        )


        nav_layout.addWidget(
            self.btn_next_ar
        )


        # -------------------------------------------------------------
        # Legend
        # -------------------------------------------------------------

        nav_layout.addSpacing(
            30
        )


        legend_items = [

            (
                "— AR R trace",
                "#ffab40"
            ),

            (
                "— AR frequency",
                "#7e57c2"
            ),

            (
                "■ AR spindle",
                "#ff3333"
            ),

            (
                "--- T_upper",
                "#ff5252"
            ),

            (
                "--- T_lower",
                "#52a0ff"
            ),

        ]


        for text, color in legend_items:

            lbl = QtWidgets.QLabel(
                text
            )

            lbl.setStyleSheet(
                f"color: {color}; "
                f"font-weight: bold;"
            )

            nav_layout.addWidget(
                lbl
            )


        nav_layout.addStretch(
            1
        )


        nav_box.setLayout(
            nav_layout
        )


        root_layout.addWidget(
            nav_box
        )


        # =============================================================
        # Graphics
        # =============================================================

        pg.setConfigOptions(
            antialias=True
        )


        self.graphics_layout = (
            pg.GraphicsLayoutWidget()
        )


        root_layout.addWidget(
            self.graphics_layout,
            stretch=1
        )


        # =============================================================
        # Plot 1 — Raw
        # =============================================================

        self.p_raw = (
            self.graphics_layout.addPlot(
                row=0,
                col=0
            )
        )


        self.p_raw.showGrid(
            x=True,
            y=True,
            alpha=0.3
        )


        self.p_raw.setLabel(
            "left",
            "Amplitude (Raw)",
            units="uV"
        )


        self.p_raw.setLabel(
            "bottom",
            "Time",
            units="s"
        )


        self.curve_raw = (
            self.p_raw.plot(
                pen=pg.mkPen(
                    color="#dcdcdc",
                    width=1
                )
            )
        )


        # =============================================================
        # Plot 2 — 9–20 Hz filtered
        # =============================================================

        self.p_filt = (
            self.graphics_layout.addPlot(
                row=1,
                col=0
            )
        )


        self.p_filt.showGrid(
            x=True,
            y=True,
            alpha=0.3
        )


        self.p_filt.setLabel(
            "left",
            "Amplitude (9–20 Hz)",
            units="uV"
        )


        self.p_filt.setLabel(
            "bottom",
            "Time",
            units="s"
        )


        self.curve_filt = (
            self.p_filt.plot(
                pen=pg.mkPen(
                    color="#4db6ac",
                    width=1.2
                )
            )
        )


        # =============================================================
        # Plot 3 — R
        # =============================================================

        self.p_r = (
            self.graphics_layout.addPlot(
                row=2,
                col=0
            )
        )


        self.p_r.showGrid(
            x=True,
            y=True,
            alpha=0.3
        )


        self.p_r.setLabel(
            "left",
            "R"
        )


        self.p_r.setLabel(
            "bottom",
            "Time",
            units="s"
        )


        self.curve_r = (
            self.p_r.plot(
                pen=pg.mkPen(
                    color="#ffab40",
                    width=2
                ),
                connect="finite"
            )
        )


        # -------------------------------------------------------------
        # Upper threshold
        # -------------------------------------------------------------

        self.line_upper = (
            pg.InfiniteLine(
                angle=0,
                movable=False,
                pen=pg.mkPen(
                    color=(
                        255,
                        82,
                        82,
                        200
                    ),
                    width=1.5,
                    style=QtCore.Qt.DashLine
                ),
                label="T_upper",
                labelOpts={
                    "position": 0.05,
                    "color": (
                        255,
                        82,
                        82
                    ),
                    "fill": (
                        0,
                        0,
                        0,
                        120
                    ),
                    "movable": False
                }
            )
        )


        # -------------------------------------------------------------
        # Lower threshold
        # -------------------------------------------------------------

        self.line_lower = (
            pg.InfiniteLine(
                angle=0,
                movable=False,
                pen=pg.mkPen(
                    color=(
                        82,
                        160,
                        255,
                        200
                    ),
                    width=1.5,
                    style=QtCore.Qt.DashLine
                ),
                label="T_lower",
                labelOpts={
                    "position": 0.05,
                    "color": (
                        82,
                        160,
                        255
                    ),
                    "fill": (
                        0,
                        0,
                        0,
                        120
                    ),
                    "movable": False
                }
            )
        )


        self.line_upper.setZValue(
            15
        )

        self.line_lower.setZValue(
            15
        )


        self.p_r.addItem(
            self.line_upper
        )

        self.p_r.addItem(
            self.line_lower
        )


        self.p_r.setYRange(
            0.0,
            1.05,
            padding=0
        )


        # =============================================================
        # Plot 4 — Frequency
        # =============================================================

        self.p_freq = (
            self.graphics_layout.addPlot(
                row=3,
                col=0
            )
        )


        self.p_freq.showGrid(
            x=True,
            y=True,
            alpha=0.3
        )


        self.p_freq.setLabel(
            "left",
            "Peak frequency",
            units="Hz"
        )


        self.p_freq.setLabel(
            "bottom",
            "Time",
            units="s"
        )


        self.curve_freq = (
            self.p_freq.plot(
                pen=pg.mkPen(
                    color="#7e57c2",
                    width=2
                ),
                connect="finite"
            )
        )


        self.p_freq.setYRange(
            9.0,
            20.0,
            padding=0
        )


        # =============================================================
        # Link x axes
        # =============================================================

        self.p_filt.setXLink(
            self.p_raw
        )

        self.p_r.setXLink(
            self.p_raw
        )

        self.p_freq.setXLink(
            self.p_raw
        )


        self._span_plots = (
            self.p_raw,
            self.p_filt,
            self.p_r,
            self.p_freq,
        )


        self.region_items = []


        # =============================================================
        # Details panel
        # =============================================================

        self.details_panel = (
            QtWidgets.QTextEdit()
        )


        self.details_panel.setReadOnly(
            True
        )


        self.details_panel.setMaximumHeight(
            120
        )


        self.details_panel.setStyleSheet(
            "background-color: #1e1e1e; "
            "color: #00e676; "
            "font-family: Monospace; "
            "font-size: 12px;"
        )


        root_layout.addWidget(
            self.details_panel
        )


        # =============================================================
        # Status bar
        # =============================================================

        self.status_bar = (
            self.statusBar()
        )


        # =============================================================
        # Keyboard shortcuts
        # =============================================================

        QtWidgets.QShortcut(
            QtGui.QKeySequence(
                QtCore.Qt.Key_Left
            ),
            self,
            self.on_prev_ar_event
        )


        QtWidgets.QShortcut(
            QtGui.QKeySequence(
                QtCore.Qt.Key_Right
            ),
            self,
            self.on_next_ar_event
        )


        # =============================================================
        # X-range callback
        # =============================================================

        self.p_raw.sigXRangeChanged.connect(
            self._on_xrange_changed
        )


    # =================================================================
    # Thresholds
    # =================================================================

    def _current_thresholds(
        self
    ):

        region = (
            self.combo_region.currentData()
        )


        thr = REGION_THRESHOLDS.get(
            region,
            DEFAULT_THRESHOLDS
        )


        return (
            float(thr["upper"]),
            float(thr["lower"])
        )


    # =================================================================

    def _refresh_threshold_lines(
        self
    ):

        upper, lower = (
            self._current_thresholds()
        )


        self.line_upper.setValue(
            upper
        )


        self.line_lower.setValue(
            lower
        )


        self.p_r.setYRange(
            0.0,
            max(
                1.05,
                upper * 1.05
            ),
            padding=0
        )


    # =================================================================
    # Data loading
    # =================================================================

    def load_data(
        self
    ):

        if not MODULES_AVAILABLE:

            self.details_panel.setText(
                "Could not import "
                "ephys_preprocessing modules:\n"
                f"{_IMPORT_ERROR}"
            )

            return


        if not os.path.exists(
            self.manifest_path
        ):

            self.details_panel.setText(
                "Manifest not found:\n"
                f"{self.manifest_path}"
            )

            return


        # -------------------------------------------------------------
        # Manifest
        # -------------------------------------------------------------

        self.manifest_df = pd.read_csv(
            self.manifest_path
        )


        missing = (
            REQUIRED_MANIFEST_COLS
            -
            {
                c.lower()
                for c in self.manifest_df.columns
            }
        )


        if missing:

            self.details_panel.setText(
                "Manifest missing columns:\n"
                f"{sorted(missing)}"
            )

            self.manifest_df = None

            return


        # -------------------------------------------------------------
        # Normalize metadata
        # -------------------------------------------------------------

        rat_numeric = pd.to_numeric(
            self.manifest_df["rat"],
            errors="coerce"
        )


        self.manifest_df = (
            self.manifest_df.loc[
                rat_numeric.notna()
            ].copy()
        )


        self.manifest_df["Rat"] = (
            rat_numeric.loc[
                rat_numeric.notna()
            ].astype(int)
        )


        self.manifest_df["Region"] = (
            self.manifest_df["region"]
            .astype(str)
            .str.strip()
        )


        self.manifest_df["Date"] = (
            self.manifest_df["date"]
            .apply(clean_str)
        )


        parsed = (
            self.manifest_df["data_path"]
            .apply(parse_channel_trial)
        )


        self.manifest_df["channel"] = (
            parsed.apply(
                lambda x: x[0]
            )
        )


        self.manifest_df["trial"] = (
            parsed.apply(
                lambda x: clean_str(x[1])
            )
        )


        self.manifest_df["File"] = (
            self.manifest_df["data_path"]
            .apply(
                lambda p:
                Path(str(p)).name
            )
        )


        # -------------------------------------------------------------
        # AR event CSV
        # -------------------------------------------------------------

        if os.path.exists(
            self.ar_csv_path
        ):

            self.ar_df = pd.read_csv(
                self.ar_csv_path
            )


            self.ar_df, _ = (
                resolve_interval_columns(
                    self.ar_df,
                    FS
                )
            )


            if "Rat" in self.ar_df.columns:

                self.ar_df["Rat"] = (
                    pd.to_numeric(
                        self.ar_df["Rat"],
                        errors="coerce"
                    )
                    .fillna(-1)
                    .astype(int)
                )


            if "Region" in self.ar_df.columns:

                self.ar_df["Region"] = (
                    self.ar_df["Region"]
                    .astype(str)
                    .str.strip()
                )


            if "Date" in self.ar_df.columns:

                self.ar_df["Date"] = (
                    self.ar_df["Date"]
                    .apply(clean_str)
                )


            if "File" in self.ar_df.columns:

                self.ar_df["File"] = (
                    self.ar_df["File"]
                    .apply(
                        lambda p:
                        Path(str(p)).name
                    )
                )


        else:

            self.details_panel.setText(
                "AR CSV not found:\n"
                f"{self.ar_csv_path}"
            )


        # -------------------------------------------------------------
        # Populate selectors
        # -------------------------------------------------------------

        self.populate_rat_selector()


    # =================================================================
    # Rat selector
    # =================================================================

    def populate_rat_selector(
        self
    ):

        rats = sorted(
            self.manifest_df[
                "Rat"
            ].unique().tolist()
        )


        self.combo_rat.blockSignals(
            True
        )


        self.combo_rat.clear()


        for r in rats:

            self.combo_rat.addItem(
                str(r),
                userData=r
            )


        self.combo_rat.blockSignals(
            False
        )


        if rats:

            self.on_rat_changed(
                0
            )


    # =================================================================
    # Rat changed
    # =================================================================

    def on_rat_changed(
        self,
        index
    ):

        if (
            index < 0
            or self.manifest_df is None
        ):
            return


        rat = (
            self.combo_rat.currentData()
        )


        sub = self.manifest_df[
            self.manifest_df["Rat"] == rat
        ]


        regions = sorted(
            sub["Region"]
            .unique()
            .tolist()
        )


        self.combo_region.blockSignals(
            True
        )


        self.combo_region.clear()


        for reg in regions:

            self.combo_region.addItem(
                reg,
                userData=reg
            )


        self.combo_region.blockSignals(
            False
        )


        if regions:

            self.on_region_changed(
                0
            )


    # =================================================================
    # Region changed
    # =================================================================

    def on_region_changed(
        self,
        index
    ):

        if (
            index < 0
            or self.manifest_df is None
        ):
            return


        rat = (
            self.combo_rat.currentData()
        )


        region = (
            self.combo_region.currentData()
        )


        matched = self.manifest_df[

            (self.manifest_df["Rat"] == rat)

            &

            (self.manifest_df["Region"] == region)

        ]


        self.combo_files.blockSignals(
            True
        )


        self.combo_files.clear()


        for orig_idx, row in (
            matched.iterrows()
        ):

            lbl = (

                f"{row['Date']} | "
                f"ch:{row['channel']} "
                f"tr:{row['trial']} "
                f"-> {row['File']}"

            )


            self.combo_files.addItem(
                lbl,
                userData=orig_idx
            )


        self.combo_files.blockSignals(
            False
        )


        self._refresh_threshold_lines()


        if self.combo_files.count() > 0:

            self.on_file_selected(
                0
            )


    # =================================================================
    # File selected
    # =================================================================

    def on_file_selected(
        self,
        index
    ):

        if (
            index < 0
            or self.manifest_df is None
            or self.combo_files.count() == 0
        ):

            return


        manifest_idx = (
            self.combo_files.itemData(
                index
            )
        )


        task = (
            self.manifest_df.loc[
                manifest_idx
            ]
        )


        data_path = (
            task["data_path"]
        )


        if not os.path.exists(
            data_path
        ):

            self.details_panel.setText(
                f"File missing:\n"
                f"{data_path}"
            )

            return


        self._pending_task = task


        self.combo_files.setEnabled(
            False
        )


        self.status_bar.showMessage(
            f"Loading "
            f"{Path(data_path).name} ..."
        )


        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.WaitCursor
        )


        # -------------------------------------------------------------
        # Stop old loader
        # -------------------------------------------------------------

        if (
            self._loader_thread is not None
            and self._loader_thread.isRunning()
        ):

            self._loader_thread.quit()

            self._loader_thread.wait()


        # -------------------------------------------------------------
        # Start new loader
        # -------------------------------------------------------------

        self._loader_thread = (
            SignalLoader(
                data_path,
                parent=self
            )
        )


        self._loader_thread.finished_ok.connect(
            self._on_signal_loaded
        )


        self._loader_thread.failed.connect(
            self._on_signal_load_failed
        )


        self._loader_thread.start()


    # =================================================================
    # Signal loading failed
    # =================================================================

    def _on_signal_load_failed(
        self,
        msg
    ):

        QtWidgets.QApplication.restoreOverrideCursor()

        self.combo_files.setEnabled(
            True
        )

        self.status_bar.clearMessage()

        self.details_panel.setText(
            f"Signal error:\n{msg}"
        )


    # =================================================================
    # Signal loaded
    # =================================================================

    def _on_signal_loaded(
        self,
        raw,
        filtered,
        signal_128,
        time_128
    ):

        QtWidgets.QApplication.restoreOverrideCursor()

        self.combo_files.setEnabled(
            True
        )

        self.status_bar.clearMessage()


        task = (
            self._pending_task
        )


        if task is None:
            return


        # -------------------------------------------------------------
        # Cache signals
        # -------------------------------------------------------------

        self.raw_signal = raw

        self.filtered_signal = filtered

        self.signal_128 = signal_128

        self.time_vector = (
            np.arange(
                len(raw),
                dtype=np.float32
            )
            / FS
        )


        self.data_len = len(
            raw
        )


        self.data_duration = (
            self.data_len / FS
        )


        # -------------------------------------------------------------
        # Filter events for current file
        # -------------------------------------------------------------

        file_name = task["File"]


        if not self.ar_df.empty:

            if "File" in self.ar_df.columns:

                ar_sub = (
                    self.ar_df[
                        self.ar_df["File"]
                        == file_name
                    ]
                )

            else:

                ar_sub = self.ar_df[

                    (self.ar_df["Rat"]
                     == task["Rat"])

                    &

                    (self.ar_df["Region"]
                     == task["Region"])

                    &

                    (self.ar_df["Date"]
                     == task["Date"])

                ]


            self.current_ar_events = (

                ar_sub
                .sort_values("Start_s")
                .reset_index(drop=True)

                if not ar_sub.empty

                else pd.DataFrame()

            )

        else:

            self.current_ar_events = (
                pd.DataFrame()
            )


        # -------------------------------------------------------------
        # Draw full signal
        # -------------------------------------------------------------

        self.curve_raw.setData(
            self.time_vector,
            raw
        )


        self.curve_filt.setData(
            self.time_vector,
            filtered
        )


        self.curve_r.clear()

        self.curve_freq.clear()


        # -------------------------------------------------------------
        # Thresholds
        # -------------------------------------------------------------

        self._refresh_threshold_lines()


        # -------------------------------------------------------------
        # Event spans
        # -------------------------------------------------------------

        self.draw_spans()


        # -------------------------------------------------------------
        # Start at first event
        # -------------------------------------------------------------

        self.current_ar_idx = (
            0
            if len(
                self.current_ar_events
            ) > 0
            else -1
        )


        self.lbl_ar_tracker.setText(

            f"AR: "
            f"{max(0, self.current_ar_idx + 1)}"
            f" / "
            f"{len(self.current_ar_events)}"

        )


        if self.current_ar_idx >= 0:

            self.jump_to_ar_event()


        else:

            self.p_raw.setXRange(
                0,
                min(
                    VIEW_WINDOW_SEC,
                    self.data_duration
                ),
                padding=0
            )


            self.details_panel.setText(

                f"File: {file_name} | "
                f"Length: "
                f"{self.data_duration:.1f}s | "
                f"No AR events."

            )


            self.view_update_timer.start(
                DEBOUNCE_MS
            )


    # =================================================================
    # X range changed
    # =================================================================

    def _on_xrange_changed(
        self,
        _,
        range_tuple
    ):

        if self.raw_signal is not None:

            self.view_update_timer.start(
                DEBOUNCE_MS
            )


    # =================================================================
    # AR computation
    # =================================================================

    def _start_window_computation(
        self
    ):

        if (
            not MODULES_AVAILABLE
            or self.signal_128 is None
        ):

            return


        # -------------------------------------------------------------
        # Cancel previous calculation
        # -------------------------------------------------------------

        if (
            self._ar_signal_thread is not None
            and self._ar_signal_thread.isRunning()
        ):

            self._ar_signal_thread.cancel()

            self._ar_signal_thread.wait()


        # -------------------------------------------------------------
        # Current view
        # -------------------------------------------------------------

        view = (
            self.p_raw.viewRange()[0]
        )


        start_s = float(
            view[0]
        )


        end_s = float(
            view[1]
        )


        self.curve_r.clear()

        self.curve_freq.clear()


        # -------------------------------------------------------------
        # Start AR worker
        # -------------------------------------------------------------

        self._ar_signal_thread = (
            ARWindowWorker(

                self.signal_128,

                start_s,

                end_s,

                parent=self

            )
        )


        self._ar_signal_thread.finished_ok.connect(
            self._on_window_computation_ready
        )


        self._ar_signal_thread.failed.connect(
            self._on_window_computation_failed
        )


        self._ar_signal_thread.start()


    # =================================================================
    # AR computation ready
    # =================================================================

    def _on_window_computation_ready(
        self,
        requested_start,
        requested_end,
        t_vals,
        r_vals,
        freq_vals
    ):

        # -------------------------------------------------------------
        # Make sure result corresponds to current view
        # -------------------------------------------------------------

        current_view = (
            self.p_raw.viewRange()[0]
        )


        cur_lo = float(
            current_view[0]
        )


        cur_hi = float(
            current_view[1]
        )


        if (

            abs(
                cur_lo
                - requested_start
            )
            > STALE_TOLERANCE_SEC

            or

            abs(
                cur_hi
                - requested_end
            )
            > STALE_TOLERANCE_SEC

        ):

            return


        if len(t_vals) == 0:
            return


        # -------------------------------------------------------------
        # Convert
        # -------------------------------------------------------------

        t_arr = np.asarray(
            t_vals,
            dtype=np.float64
        )


        r_arr = np.asarray(
            r_vals,
            dtype=np.float64
        )


        freq_arr = np.asarray(
            freq_vals,
            dtype=np.float64
        )


        # -------------------------------------------------------------
        # R = 0 means no in-band pole
        # Show these as gaps
        # -------------------------------------------------------------

        r_plot = np.where(
            r_arr > 0.0,
            r_arr,
            np.nan
        )


        # -------------------------------------------------------------
        # Draw R
        # -------------------------------------------------------------

        self.curve_r.setData(
            t_arr,
            r_plot
        )


        # -------------------------------------------------------------
        # Draw frequency
        # -------------------------------------------------------------

        self.curve_freq.setData(
            t_arr,
            freq_arr
        )


        # -------------------------------------------------------------
        # R axis
        # -------------------------------------------------------------

        upper, lower = (
            self._current_thresholds()
        )


        finite_r = (
            r_plot[
                np.isfinite(r_plot)
            ]
        )


        data_max = (

            float(
                np.nanmax(
                    finite_r
                )
            )

            if finite_r.size

            else 0.0

        )


        ymax = max(

            1.05,

            upper * 1.05,

            data_max * 1.05

        )


        self.p_r.setYRange(
            0.0,
            ymax,
            padding=0
        )


        # -------------------------------------------------------------
        # Frequency axis
        # -------------------------------------------------------------

        self.p_freq.setYRange(
            9.0,
            20.0,
            padding=0
        )


    # =================================================================
    # AR computation failed
    # =================================================================

    def _on_window_computation_failed(
        self,
        msg
    ):

        self.status_bar.showMessage(
            f"AR failed: {msg}",
            12000
        )


    # =================================================================
    # Draw detected spindle spans
    # =================================================================

    def draw_spans(
        self
    ):

        # -------------------------------------------------------------
        # Remove previous spans
        # -------------------------------------------------------------

        for triple in (
            self.region_items
        ):

            for plot, item in zip(
                self._span_plots,
                triple
            ):

                try:

                    plot.removeItem(
                        item
                    )

                except Exception:

                    pass


        self.region_items.clear()


        # -------------------------------------------------------------
        # Nothing to draw
        # -------------------------------------------------------------

        if self.current_ar_events.empty:
            return


        # -------------------------------------------------------------
        # Draw every detected event
        # -------------------------------------------------------------

        for _, row in (
            self.current_ar_events.iterrows()
        ):

            s = safe_float(
                row["Start_s"],
                0.0
            )


            e = safe_float(
                row["End_s"],
                0.0
            )


            triple = tuple(

                pg.LinearRegionItem(

                    [
                        s,
                        e
                    ],

                    movable=False,

                    brush=QtGui.QColor(
                        255,
                        50,
                        50,
                        80
                    ),

                    pen=pg.mkPen(
                        color=(
                            255,
                            50,
                            50,
                            230
                        ),
                        width=1.5
                    )

                )

                for _ in range(
                    len(self._span_plots)
                )

            )


            for item, plot in zip(
                triple,
                self._span_plots
            ):

                item.setZValue(
                    10
                )

                plot.addItem(
                    item
                )


            self.region_items.append(
                triple
            )


    # =================================================================
    # Navigation
    # =================================================================

    def on_prev_ar_event(
        self
    ):

        if len(
            self.current_ar_events
        ) == 0:

            return


        self.current_ar_idx = (

            self.current_ar_idx - 1

        ) % len(
            self.current_ar_events
        )


        self.jump_to_ar_event()


    # =================================================================

    def on_next_ar_event(
        self
    ):

        if len(
            self.current_ar_events
        ) == 0:

            return


        self.current_ar_idx = (

            self.current_ar_idx + 1

        ) % len(
            self.current_ar_events
        )


        self.jump_to_ar_event()


    # =================================================================
    # Centered 10-second view
    # =================================================================

    def _centered_view_range(
        self,
        start_s,
        end_s
    ):

        center = (
            0.5
            * (
                start_s
                + end_s
            )
        )


        half = (
            VIEW_WINDOW_SEC
            / 2.0
        )


        vs = (
            center
            - half
        )


        ve = (
            center
            + half
        )


        # -------------------------------------------------------------
        # Beginning of recording
        # -------------------------------------------------------------

        if vs < 0.0:

            vs = 0.0

            ve = min(
                VIEW_WINDOW_SEC,
                self.data_duration
            )


        # -------------------------------------------------------------
        # End of recording
        # -------------------------------------------------------------

        if ve > self.data_duration:

            ve = (
                self.data_duration
            )

            vs = max(
                0.0,
                ve - VIEW_WINDOW_SEC
            )


        return vs, ve


    # =================================================================
    # Jump to selected AR event
    # =================================================================

    def jump_to_ar_event(
        self
    ):

        if (

            self.current_ar_idx < 0

            or

            self.current_ar_idx
            >= len(
                self.current_ar_events
            )

        ):

            return


        # -------------------------------------------------------------
        # Selected event
        # -------------------------------------------------------------

        event = (
            self.current_ar_events.iloc[
                self.current_ar_idx
            ]
        )


        start_s = safe_float(
            event["Start_s"],
            0.0
        )


        end_s = safe_float(
            event["End_s"],
            0.0
        )


        # -------------------------------------------------------------
        # 10-second centered view
        # -------------------------------------------------------------

        vs, ve = (
            self._centered_view_range(
                start_s,
                end_s
            )
        )


        self.p_raw.setXRange(
            vs,
            ve,
            padding=0
        )


        # -------------------------------------------------------------
        # R axis
        # -------------------------------------------------------------

        upper, lower = (
            self._current_thresholds()
        )


        self.p_r.setYRange(
            0.0,
            max(
                1.05,
                upper * 1.05
            ),
            padding=0
        )


        # -------------------------------------------------------------
        # Frequency axis
        # -------------------------------------------------------------

        self.p_freq.setYRange(
            9.0,
            20.0,
            padding=0
        )


        # -------------------------------------------------------------
        # Start AR computation
        # -------------------------------------------------------------

        self.view_update_timer.start(
            DEBOUNCE_MS
        )


        # -------------------------------------------------------------
        # Event information
        # -------------------------------------------------------------

        r_val = event.get(
            "Max_R",
            np.nan
        )


        freq_val = event.get(
            "Peak_Freq_Hz",
            np.nan
        )


        dur = (
            end_s
            - start_s
        )


        region = (
            self.combo_region.currentData()
        )


        threshold = (
            REGION_THRESHOLDS.get(
                region,
                DEFAULT_THRESHOLDS
            )
        )


        # -------------------------------------------------------------
        # Details
        # -------------------------------------------------------------

        self.details_panel.setText(

            "\n".join([

                (
                    f"[AR #{self.current_ar_idx + 1}"
                    f"/{len(self.current_ar_events)}]"
                ),

                (
                    f"Region = {region}"
                ),

                (
                    f"Interval = "
                    f"[{start_s:.3f}s, "
                    f"{end_s:.3f}s]"
                ),

                (
                    f"Duration = "
                    f"{dur:.3f}s"
                ),

                (
                    f"CSV Max R = "
                    f"{fmt(r_val)}"
                ),

                (
                    f"CSV Peak Freq = "
                    f"{fmt(freq_val, '.2f')} Hz"
                ),

                (
                    f"Thresholds = "
                    f"upper {threshold['upper']:.2f} | "
                    f"lower {threshold['lower']:.2f}"
                ),

                (
                    f"View = "
                    f"[{vs:.3f}s, "
                    f"{ve:.3f}s]"
                ),

            ])

        )


        # -------------------------------------------------------------
        # Update tracker
        # -------------------------------------------------------------

        self.lbl_ar_tracker.setText(

            f"AR: "
            f"{self.current_ar_idx + 1}"
            f" / "
            f"{len(self.current_ar_events)}"

        )


# =====================================================================
# Command-line arguments
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Spindle Alignment Inspector "
            "(final AR detector)"
        )
    )


    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST
    )


    parser.add_argument(
        "--ar-csv",
        default=DEFAULT_AR_CSV
    )


    parser.add_argument(
        "--debug",
        action="store_true"
    )


    args, _ = (
        parser.parse_known_args()
    )


    return args


# =====================================================================
# Main
# =====================================================================

def main():

    global VERBOSE


    args = parse_args()


    VERBOSE = args.debug


    if VERBOSE:

        print(
            "=" * 70
        )

        print(
            "Spindle Alignment Inspector "
            "— Final AR Detector"
        )

        print(
            f"manifest = "
            f"{args.manifest}"
        )

        print(
            f"ar_csv = "
            f"{args.ar_csv}"
        )

        print(
            f"MODULES_AVAILABLE = "
            f"{MODULES_AVAILABLE}"
        )

        print(
            f"AR_ORDER = "
            f"{AR_ORDER}"
        )

        print(
            f"AR_WINDOW_SEC = "
            f"{AR_WINDOW_SEC}"
        )

        print(
            f"AR_STRIDE_SAMPLES = "
            f"{AR_STRIDE_SAMPLES}"
        )

        print(
            f"AR_BAND = "
            f"{AR_SPINDLE_BAND}"
        )

        print(
            f"AR_PRE_FILTER = "
            f"({AR_PRE_LOWCUT}, "
            f"{AR_PRE_HIGHCUT})"
        )

        print(
            f"VIEW_WINDOW_SEC = "
            f"{VIEW_WINDOW_SEC}"
        )

        print(
            "REGION_THRESHOLDS = "
            f"{REGION_THRESHOLDS}"
        )

        print(
            "=" * 70
        )


    # -------------------------------------------------------------
    # Qt application
    # -------------------------------------------------------------

    app = (
        QtWidgets.QApplication(
            sys.argv
        )
    )


    app.setStyle(
        "Fusion"
    )


    # -------------------------------------------------------------
    # Dark palette
    # -------------------------------------------------------------

    palette = (
        QtGui.QPalette()
    )


    palette.setColor(
        QtGui.QPalette.Window,
        QtGui.QColor(
            40,
            40,
            40
        )
    )


    palette.setColor(
        QtGui.QPalette.WindowText,
        QtCore.Qt.white
    )


    palette.setColor(
        QtGui.QPalette.Base,
        QtGui.QColor(
            25,
            25,
            25
        )
    )


    palette.setColor(
        QtGui.QPalette.AlternateBase,
        QtGui.QColor(
            40,
            40,
            40
        )
    )


    palette.setColor(
        QtGui.QPalette.Text,
        QtCore.Qt.white
    )


    palette.setColor(
        QtGui.QPalette.Button,
        QtGui.QColor(
            50,
            50,
            50
        )
    )


    palette.setColor(
        QtGui.QPalette.ButtonText,
        QtCore.Qt.white
    )


    palette.setColor(
        QtGui.QPalette.Highlight,
        QtGui.QColor(
            0,
            188,
            212
        )
    )


    app.setPalette(
        palette
    )


    # -------------------------------------------------------------
    # Viewer
    # -------------------------------------------------------------

    viewer = (
        SpindleViewer(
            args.manifest,
            args.ar_csv
        )
    )


    viewer.show()


    sys.exit(
        app.exec_()
    )


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":

    main()
