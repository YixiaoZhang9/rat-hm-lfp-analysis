#!/usr/bin/env python3
"""
Spindle Alignment Inspector (AR vs Wavelet)

Page through detected spindle events from two detectors overlaid on the raw
LFP, band-passed LFP, and a continuous AR R-value trace dynamically computed
for the active view window.

Architecture notes:
  * AR trace is computed on a background QThread (ARWindowWorker) for the
    currently visible X range plus a small padding window. The worker echoes
    back the range it computed for; the UI thread discards any result whose
    requested range no longer matches the current view (stale-result
    rejection).
  * The R value recorded per AR window is the magnitude of the pole whose
    frequency is closest to the spindle band center (12.5 Hz). This gives a
    dense, continuous trace. The strict in-band presence is tracked
    separately as a diagnostic.
  * Raw / filtered / downsampled signals are sanitized at every stage so a
    single NaN cannot poison every Burg window. r_values is initialized with
    NaN so failures are visible as gaps rather than a misleading flat line.
  * Region-specific upper / lower detection thresholds are drawn as dashed
    horizontal reference lines on the R trace (T_upper, T_lower). They update
    whenever the Region combo changes and the Y-axis is never allowed to crop
    below the upper threshold.
  * Navigation is time-synced: when the user jumps to the next AR event, the
    Wavelet tracker jumps to whichever of its events is nearest in time to
    the new view center (and vice versa). Both trackers therefore always
    describe the same time window on screen.
"""

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

# --------------------------------------------------------------------------- #
# Import Preprocessing Modules
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.environ.get(
    "AR_MODULES_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
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

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_MANIFEST = os.environ.get("SPINDLE_MANIFEST", "tasks_manifest.csv")
DEFAULT_AR_CSV = os.environ.get(
    "SPINDLE_AR_CSV", "results/all_detected_spindles_per_region.csv"
)
DEFAULT_WAVELET_CSV = os.environ.get(
    "SPINDLE_WAVELET_CSV",
    "results_ar_calibration/wavelet_spindles_with_ar_dynamics.csv",
)

FS = 1000.0
BP_LOW = 10.0
BP_HIGH = 15.0
VIEW_WINDOW_SEC = 10.0

AR_TARGET_FS = 128
AR_ORDER = 8
AR_SPINDLE_BAND = (BP_LOW, BP_HIGH)
AR_BAND_CENTER = 0.5 * (AR_SPINDLE_BAND[0] + AR_SPINDLE_BAND[1])
AR_WINDOW_SEC = 1.0

# Compute-window padding so the final 1 s AR window of the visible range has
# enough data to complete.
AR_PAD_SEC = AR_WINDOW_SEC

# Debounce delay before starting a new AR computation after a view change.
DEBOUNCE_MS = 300

# How much mismatch (in seconds) between the requested range and the current
# view is tolerated before a result is considered stale.
STALE_TOLERANCE_SEC = 0.5

REQUIRED_MANIFEST_COLS = {"rat", "region", "date", "data_path"}

# Upper / lower AR pole-magnitude thresholds used for spindle detection,
# per brain region. Drawn as horizontal reference lines on the R trace and
# re-applied automatically when the Region combo changes.
REGION_THRESHOLDS = {
    "HPC": {"upper": 0.85, "lower": 0.40},
    "PL":  {"upper": 0.85, "lower": 0.40},
    "RSC": {"upper": 0.90, "lower": 0.40},
}

# Fallback thresholds for regions not listed above.
DEFAULT_THRESHOLDS = {"upper": 0.85, "lower": 0.40}

# Populated by parse_args() before main() runs.
VERBOSE = False


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs, flush=True)


def _sanitize(x):
    """Return a float64 numpy array free of NaN/Inf."""
    arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# --------------------------------------------------------------------------- #
# AR Window Worker
# --------------------------------------------------------------------------- #
class ARWindowWorker(QtCore.QThread):
    """
    Computes the continuous R-value trace for a given time slice.

    Emits:
      finished_ok(requested_start_s, requested_end_s, t_vals, r_vals)
      failed(message)

    The requested range is echoed back so the UI can detect stale results.
    Lists are used for the payload to guarantee thread-safe delivery across
    the PyQt5 event loop.
    """

    finished_ok = QtCore.pyqtSignal(float, float, list, list)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, raw_signal, start_s, end_s, parent=None):
        super().__init__(parent)
        self.raw_signal = raw_signal
        self.start_s = max(0.0, float(start_s))
        self.end_s = float(end_s)
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        vprint("\n[AR] worker entered "
               f"range=[{self.start_s:.3f}, {self.end_s:.3f}] "
               f"raw_len={len(self.raw_signal)}")
        try:
            s_idx = int(self.start_s * FS)
            e_idx = min(
                len(self.raw_signal),
                int((self.end_s + AR_PAD_SEC) * FS),
            )

            if s_idx >= e_idx:
                self._emit_empty()
                return

            signal_segment = _sanitize(self.raw_signal[s_idx:e_idx])
            if len(signal_segment) == 0:
                self._emit_empty()
                return

            filtered = ar_bandpass_filter(
                signal_segment, lowcut=0.1, highcut=100, fs=FS
            )
            filtered = _sanitize(filtered)

            signal_128 = ar_downsampling(filtered, FS, AR_TARGET_FS)
            signal_128 = _sanitize(signal_128)

            window_samples = int(AR_WINDOW_SEC * AR_TARGET_FS)
            if len(signal_128) < window_samples:
                self._emit_empty()
                return

            total_windows = len(signal_128) - window_samples + 1
            r_values = np.full(total_windows, np.nan, dtype=np.float64)
            n_in_band = 0
            n_failed = 0
            first_error = None

            for i in range(total_windows):
                if self._is_cancelled:
                    return

                window = signal_128[i : i + window_samples]
                if not np.all(np.isfinite(window)) or np.all(window == 0.0):
                    n_failed += 1
                    continue

                try:
                    a, _ = sm.regression.linear_model.burg(
                        window, order=AR_ORDER, demean=True
                    )
                    poles = np.roots(np.r_[1, -a])
                    poles = poles[np.imag(poles) > 0]

                    if len(poles) == 0:
                        n_failed += 1
                        continue

                    freqs = np.angle(poles) * AR_TARGET_FS / (2 * np.pi)
                    r_vals = np.abs(poles)

                    # Continuous trace: R of the pole closest to band center.
                    idx = int(np.argmin(np.abs(freqs - AR_BAND_CENTER)))
                    r_values[i] = float(r_vals[idx])

                    # Diagnostic: did *any* pole fall inside the band?
                    if np.any(
                        (freqs >= AR_SPINDLE_BAND[0])
                        & (freqs <= AR_SPINDLE_BAND[1])
                    ):
                        n_in_band += 1
                except Exception as exc:
                    n_failed += 1
                    if first_error is None:
                        first_error = f"{type(exc).__name__}: {exc}"

            if self._is_cancelled:
                return

            n_finite = int(np.isfinite(r_values).sum())
            if n_finite == 0:
                msg = (
                    f"All {total_windows} AR windows failed. "
                    f"First error: {first_error or 'no poles found'}. "
                    f"Likely cause: NaN/Inf in source signal or statsmodels "
                    f"burg signature mismatch."
                )
                vprint(f"[AR] {msg}")
                self.failed.emit(msg)
                return

            t_vals = self.start_s + np.arange(len(r_values)) / AR_TARGET_FS
            vprint(
                f"[AR] emit n={len(r_values)} "
                f"finite={n_finite}/{total_windows} "
                f"in_band={n_in_band} "
                f"range=[{t_vals[0]:.3f}, {t_vals[-1]:.3f}]"
            )
            self.finished_ok.emit(
                self.start_s,
                self.end_s,
                t_vals.tolist(),
                r_values.tolist(),
            )
        except Exception as e:
            vprint(f"[AR] exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _emit_empty(self):
        self.finished_ok.emit(self.start_s, self.end_s, [], [])


# --------------------------------------------------------------------------- #
# Signal Loader Thread
# --------------------------------------------------------------------------- #
class SignalLoader(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, data_path, parent=None):
        super().__init__(parent)
        self.data_path = data_path

    def run(self):
        vprint(f"\n[Load] {self.data_path}")
        try:
            mat_data = loadmat(self.data_path)
            if "data" not in mat_data:
                keys = [k for k in mat_data.keys() if not k.startswith("__")]
                raise KeyError(
                    f"'.mat' file has no 'data' variable (found: {keys})"
                )

            raw = mat_data["data"].squeeze().astype(np.float32)
            if raw.ndim != 1:
                raw = raw.reshape(-1)

            n_bad = int(np.count_nonzero(~np.isfinite(raw)))
            if n_bad:
                vprint(f"[Load] sanitizing {n_bad} non-finite sample(s)")
                raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

            filtered = butter_bandpass_filter(raw, BP_LOW, BP_HIGH, FS)
            filtered = np.nan_to_num(
                filtered.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            vprint(f"[Load] raw len={len(raw)} std={raw.std():.3f}")
            self.finished_ok.emit(raw, filtered)
        except Exception as e:
            vprint(f"[Load] exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def clean_str(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_channel(val):
    s = clean_str(val).lower()
    m = re.search(r"(\d+)", s)
    return str(int(m.group(1))) if m else s


def parse_channel_trial(data_path):
    filename = Path(str(data_path)).name
    match = re.match(r"chan(\d+)(?:_(\d+))?\.mat$", filename, re.IGNORECASE)
    if not match:
        return "", ""
    chan = str(int(match.group(1)))
    trial = match.group(2) if match.group(2) else ""
    return chan, trial


def safe_float(val, default=np.nan):
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def fmt(val, spec=".4f", placeholder="N/A"):
    f = safe_float(val)
    return placeholder if np.isnan(f) else format(f, spec)


def resolve_interval_columns(df, default_fs=1000.0):
    if df.empty:
        return df, True

    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    start_candidates = [
        "start_s", "start_time", "start_sec", "start_secs", "start",
        "start_time_s", "spindle_start", "onset_s", "onset_time",
        "start_sample", "start_idx", "start_sample_idx", "start_pts",
    ]
    end_candidates = [
        "end_s", "end_time", "end_sec", "end_secs", "end",
        "end_time_s", "spindle_end", "offset_s", "offset_time",
        "end_sample", "end_idx", "end_sample_idx", "end_pts",
    ]

    def find_column(candidates):
        for cand in candidates:
            if cand in cols_lower:
                return cols_lower[cand]
        for cand in candidates:
            for lower_name, orig_name in cols_lower.items():
                if cand in lower_name:
                    return orig_name
        return None

    found_start = find_column(start_candidates)
    found_end = find_column(end_candidates)

    if found_start and found_end:
        starts = pd.to_numeric(df[found_start], errors="coerce").fillna(0.0).values
        ends = pd.to_numeric(df[found_end], errors="coerce").fillna(0.0).values
        durations = ends - starts
        median_dur = np.nanmedian(durations) if len(durations) > 0 else 0
        lower = found_start.lower()
        looks_like_samples = (
            median_dur > 20.0
            or "sample" in lower
            or "idx" in lower
            or "index" in lower
            or "pts" in lower
        )
        if looks_like_samples:
            df["Start_s"] = starts / default_fs
            df["End_s"] = ends / default_fs
        else:
            df["Start_s"] = starts
            df["End_s"] = ends
        return df, True

    if "Start_s" not in df.columns:
        df["Start_s"] = 0.0
    if "End_s" not in df.columns:
        df["End_s"] = 0.0
    return df, False


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class SpindleViewer(QtWidgets.QMainWindow):
    def __init__(self, manifest_path, ar_csv_path, wavelet_csv_path):
        super().__init__()
        self.setWindowTitle("Spindle Alignment Inspector (AR vs Wavelet)")
        self.resize(1500, 950)

        self.manifest_path = manifest_path
        self.ar_csv_path = ar_csv_path
        self.wavelet_csv_path = wavelet_csv_path

        self.manifest_df = None
        self.ar_df = pd.DataFrame()
        self.wavelet_df = pd.DataFrame()

        self.current_ar_events = pd.DataFrame()
        self.current_wavelet_events = pd.DataFrame()
        self.current_ar_idx = -1
        self.current_wav_idx = -1

        self.raw_signal = None
        self.filtered_signal = None
        self.time_vector = None
        self.data_len = 0
        self.data_duration = 0.0

        self._loader_thread = None
        self._ar_signal_thread = None
        self._pending_task = None

        # Suppresses syncing while we ourselves are moving the other tracker,
        # to prevent runaway cross-recursion between jump handlers.
        self._syncing_navigation = False

        self.view_update_timer = QtCore.QTimer(self)
        self.view_update_timer.setSingleShot(True)
        self.view_update_timer.timeout.connect(self._start_window_computation)

        self.spinner_timer = QtCore.QTimer(self)
        self.spinner_timer.timeout.connect(self._update_spinner)
        self.spinner_frames = ["|", "/", "-", "\\"]
        self.spinner_idx = 0

        self.init_ui()
        self.load_data()

    # --------------------------------------------------------------------- #
    # UI
    # --------------------------------------------------------------------- #
    def init_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QtWidgets.QVBoxLayout(main_widget)

        # --- Filters ---------------------------------------------------- #
        filter_box = QtWidgets.QGroupBox("Dataset Filtering")
        filter_layout = QtWidgets.QHBoxLayout()

        filter_layout.addWidget(QtWidgets.QLabel("Rat:"))
        self.combo_rat = QtWidgets.QComboBox()
        self.combo_rat.currentIndexChanged.connect(self.on_rat_changed)
        filter_layout.addWidget(self.combo_rat)

        filter_layout.addWidget(QtWidgets.QLabel("Region:"))
        self.combo_region = QtWidgets.QComboBox()
        self.combo_region.currentIndexChanged.connect(self.on_region_changed)
        filter_layout.addWidget(self.combo_region)

        filter_layout.addWidget(QtWidgets.QLabel("Recording / File:"))
        self.combo_files = QtWidgets.QComboBox()
        self.combo_files.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo_files.currentIndexChanged.connect(self.on_file_selected)
        filter_layout.addWidget(self.combo_files, stretch=2)

        filter_box.setLayout(filter_layout)
        root_layout.addWidget(filter_box)

        # --- Navigation ------------------------------------------------- #
        nav_box = QtWidgets.QGroupBox("Detection Navigation")
        nav_layout = QtWidgets.QHBoxLayout()

        self.btn_prev_ar = QtWidgets.QPushButton("◀ Prev AR")
        self.btn_prev_ar.clicked.connect(self.on_prev_ar_event)
        nav_layout.addWidget(self.btn_prev_ar)

        self.lbl_ar_tracker = QtWidgets.QLabel("AR: 0 / 0")
        self.lbl_ar_tracker.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_ar_tracker.setMinimumWidth(90)
        nav_layout.addWidget(self.lbl_ar_tracker)

        self.btn_next_ar = QtWidgets.QPushButton("Next AR ▶")
        self.btn_next_ar.clicked.connect(self.on_next_ar_event)
        nav_layout.addWidget(self.btn_next_ar)

        nav_layout.addSpacing(30)

        self.btn_prev_wav = QtWidgets.QPushButton("◀ Prev WAV")
        self.btn_prev_wav.clicked.connect(self.on_prev_wav_event)
        nav_layout.addWidget(self.btn_prev_wav)

        self.lbl_wav_tracker = QtWidgets.QLabel("WAV: 0 / 0")
        self.lbl_wav_tracker.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_wav_tracker.setMinimumWidth(90)
        nav_layout.addWidget(self.lbl_wav_tracker)

        self.btn_next_wav = QtWidgets.QPushButton("Next WAV ▶")
        self.btn_next_wav.clicked.connect(self.on_next_wav_event)
        nav_layout.addWidget(self.btn_next_wav)

        nav_layout.addSpacing(30)

        for text, color in [
            ("■ AR Spindle Event", "#ff3333"),
            ("■ Wavelet Spindle Event", "#00e5ff"),
            ("— AR Pole R Value", "#ffab40"),
            ("--- T_upper", "#ff5252"),
            ("--- T_lower", "#52a0ff"),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            nav_layout.addWidget(lbl)

        nav_layout.addStretch(1)
        nav_box.setLayout(nav_layout)
        root_layout.addWidget(nav_box)

        # --- Plots ------------------------------------------------------ #
        pg.setConfigOptions(antialias=True)
        self.graphics_layout = pg.GraphicsLayoutWidget()
        root_layout.addWidget(self.graphics_layout, stretch=1)

        self.p_raw = self.graphics_layout.addPlot(row=0, col=0)
        self.p_raw.showGrid(x=True, y=True, alpha=0.3)
        self.p_raw.setLabel("left", "Amplitude (Raw LFP)", units="uV")
        self.p_raw.setLabel("bottom", "Time", units="s")
        self.curve_raw = self.p_raw.plot(pen=pg.mkPen(color="#dcdcdc", width=1))

        self.p_filt = self.graphics_layout.addPlot(row=1, col=0)
        self.p_filt.showGrid(x=True, y=True, alpha=0.3)
        self.p_filt.setLabel("left", "Amplitude (10-15 Hz)", units="uV")
        self.p_filt.setLabel("bottom", "Time", units="s")
        self.curve_filt = self.p_filt.plot(pen=pg.mkPen(color="#4db6ac", width=1.2))

        self.p_r = self.graphics_layout.addPlot(row=2, col=0)
        self.p_r.showGrid(x=True, y=True, alpha=0.3)
        self.p_r.setLabel("left", "R")
        self.p_r.setLabel("bottom", "Time", units="s")
        self.curve_r_continuous = self.p_r.plot(
            pen=pg.mkPen(color="#ffab40", width=2),
            connect="finite",
        )

        # --- Threshold reference lines --------------------------------- #
        self.line_upper = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(
                color=(255, 82, 82, 200),
                width=1.5,
                style=QtCore.Qt.DashLine,
            ),
            label="T_upper",
            labelOpts={
                "position": 0.05,
                "color": (255, 82, 82),
                "fill": (0, 0, 0, 120),
                "movable": False,
            },
        )
        self.line_lower = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(
                color=(82, 160, 255, 200),
                width=1.5,
                style=QtCore.Qt.DashLine,
            ),
            label="T_lower",
            labelOpts={
                "position": 0.05,
                "color": (82, 160, 255),
                "fill": (0, 0, 0, 120),
                "movable": False,
            },
        )
        self.line_upper.setZValue(15)
        self.line_lower.setZValue(15)
        self.p_r.addItem(self.line_upper)
        self.p_r.addItem(self.line_lower)

        # Loading indicator for the R plot.
        self.loading_text = pg.TextItem("", color=(255, 171, 64), anchor=(0.5, 0.5))
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.loading_text.setFont(font)
        self.p_r.addItem(self.loading_text)
        self.loading_text.hide()

        self.p_r.setYRange(0.0, 1.05, padding=0)

        # Link X axes so navigation on p_raw moves all three.
        self.p_filt.setXLink(self.p_raw)
        self.p_r.setXLink(self.p_raw)
        self._span_plots = (self.p_raw, self.p_filt, self.p_r)
        self.region_items = []

        # --- Details panel --------------------------------------------- #
        self.details_panel = QtWidgets.QTextEdit()
        self.details_panel.setReadOnly(True)
        self.details_panel.setMaximumHeight(90)
        self.details_panel.setStyleSheet(
            "background-color: #1e1e1e; color: #00e676; "
            "font-family: Monospace; font-size: 12px;"
        )
        root_layout.addWidget(self.details_panel)

        self.status_bar = self.statusBar()

        # --- Shortcuts -------------------------------------------------- #
        QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Left), self, self.on_prev_ar_event
        )
        QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Right), self, self.on_next_ar_event
        )
        QtWidgets.QShortcut(
            QtGui.QKeySequence("Shift+W"), self, self.on_next_wav_event
        )
        QtWidgets.QShortcut(
            QtGui.QKeySequence("Shift+Q"), self, self.on_prev_wav_event
        )

        self.p_raw.sigXRangeChanged.connect(self._on_xrange_changed)

    # --------------------------------------------------------------------- #
    # Threshold lines
    # --------------------------------------------------------------------- #
    def _current_thresholds(self):
        """Return (upper, lower) thresholds for the currently selected region."""
        region = self.combo_region.currentData()
        thr = REGION_THRESHOLDS.get(region, DEFAULT_THRESHOLDS)
        return float(thr["upper"]), float(thr["lower"])

    def _refresh_threshold_lines(self):
        """Set the upper/lower threshold lines based on the current region."""
        upper, lower = self._current_thresholds()

        self.line_upper.setValue(upper)
        self.line_lower.setValue(lower)

        # Rescale Y so the upper threshold is always visible, but preserve
        # any already-drawn R data. If the trace is present, take the max of
        # the two so nothing gets cropped.
        ymax = max(1.05, upper * 1.05)
        self.p_r.setYRange(0.0, ymax, padding=0)

        vprint(
            f"[UI] thresholds set: T_upper={upper:.3f} "
            f"T_lower={lower:.3f} (region={self.combo_region.currentData()})"
        )

    # --------------------------------------------------------------------- #
    # Data loading
    # --------------------------------------------------------------------- #
    def load_data(self):
        if not MODULES_AVAILABLE:
            self.details_panel.setText(
                f"Error: Could not import ephys_preprocessing modules.\n"
                f"Details: {_IMPORT_ERROR}\n"
                f"Ensure AR_MODULES_ROOT is set correctly."
            )
            return

        if not os.path.exists(self.manifest_path):
            self.details_panel.setText(
                f"Manifest not found: {self.manifest_path}\n"
                f"Pass --manifest, or set SPINDLE_MANIFEST, to point at your "
                f"tasks CSV."
            )
            return

        try:
            self.manifest_df = pd.read_csv(self.manifest_path)
        except Exception as e:
            self.details_panel.setText(f"Failed to read manifest CSV: {e}")
            return

        missing = REQUIRED_MANIFEST_COLS - {
            c.lower() for c in self.manifest_df.columns
        }
        if missing:
            self.details_panel.setText(
                f"Manifest is missing required column(s): {sorted(missing)}. "
                f"Expected at least: {sorted(REQUIRED_MANIFEST_COLS)}"
            )
            self.manifest_df = None
            return

        rat_numeric = pd.to_numeric(self.manifest_df["rat"], errors="coerce")
        bad = int(rat_numeric.isna().sum())
        if bad:
            self.status_bar.showMessage(
                f"Warning: dropped {bad} manifest row(s) with a "
                f"non-numeric 'rat' value.",
                8000,
            )
        self.manifest_df = self.manifest_df.loc[rat_numeric.notna()].copy()
        self.manifest_df["Rat"] = rat_numeric.loc[rat_numeric.notna()].astype(int)
        self.manifest_df["Region"] = (
            self.manifest_df["region"].astype(str).str.strip()
        )
        self.manifest_df["Date"] = self.manifest_df["date"].apply(clean_str)

        parsed = self.manifest_df["data_path"].apply(parse_channel_trial)
        self.manifest_df["channel"] = parsed.apply(lambda x: x[0])
        self.manifest_df["trial"] = parsed.apply(lambda x: clean_str(x[1]))
        self.manifest_df["File"] = self.manifest_df["data_path"].apply(
            lambda p: Path(str(p)).name
        )

        if self.manifest_df.empty:
            self.details_panel.setText("Manifest loaded but contains no usable rows.")
            return

        self.ar_df = self._load_detection_csv(self.ar_csv_path, kind="AR")
        self.wavelet_df = self._load_detection_csv(
            self.wavelet_csv_path, kind="Wavelet", is_wavelet=True
        )

        self.populate_rat_selector()

    def _load_detection_csv(self, path, kind, is_wavelet=False):
        if not path or not os.path.exists(path):
            self.status_bar.showMessage(
                f"{kind} detections CSV not found ({path}); continuing without it.",
                8000,
            )
            return pd.DataFrame()

        try:
            df = pd.read_csv(path)
        except Exception as e:
            self.status_bar.showMessage(
                f"Failed to read {kind} CSV ({path}): {e}", 8000
            )
            return pd.DataFrame()

        if is_wavelet:
            rename_map = {}
            for col in df.columns:
                clow = col.lower().strip()
                if clow in ("rat", "rat_number", "rat_id"):
                    rename_map[col] = "Rat"
                elif clow in ("region", "brain_region", "area"):
                    rename_map[col] = "Region"
                elif clow in ("date", "session_date", "day"):
                    rename_map[col] = "Date"
                elif clow in ("channel", "chan", "channel_id"):
                    rename_map[col] = "channel"
                elif clow in ("trial", "trial_id", "recording"):
                    rename_map[col] = "trial"
            df = df.rename(columns=rename_map)

            if "Rat" in df.columns:
                df["Rat"] = (
                    pd.to_numeric(df["Rat"], errors="coerce").fillna(-1).astype(int)
                )
            if "Region" in df.columns:
                df["Region"] = df["Region"].astype(str).str.strip()
            if "Date" in df.columns:
                df["Date"] = df["Date"].apply(clean_str)
            df["channel"] = (
                df["channel"].apply(normalize_channel)
                if "channel" in df.columns
                else ""
            )
            df["trial"] = (
                df["trial"].apply(clean_str) if "trial" in df.columns else ""
            )
        else:
            if "Rat" in df.columns:
                df["Rat"] = (
                    pd.to_numeric(df["Rat"], errors="coerce").fillna(-1).astype(int)
                )
            if "Region" in df.columns:
                df["Region"] = df["Region"].astype(str).str.strip()
            if "Date" in df.columns:
                df["Date"] = df["Date"].apply(clean_str)
            if "File" in df.columns:
                df["File"] = df["File"].apply(lambda p: Path(str(p)).name)

        df, resolved = resolve_interval_columns(df, FS)
        if not df.empty and not resolved:
            self.status_bar.showMessage(
                f"Warning: couldn't find start/end time columns in {kind} CSV; "
                f"events from it will not be positioned correctly.",
                10000,
            )
        return df

    # --------------------------------------------------------------------- #
    # Filter combo handlers
    # --------------------------------------------------------------------- #
    def populate_rat_selector(self):
        rats = sorted(self.manifest_df["Rat"].unique().tolist())
        self.combo_rat.blockSignals(True)
        self.combo_rat.clear()
        for r in rats:
            self.combo_rat.addItem(str(r), userData=r)
        self.combo_rat.blockSignals(False)
        if rats:
            self.on_rat_changed(0)

    def on_rat_changed(self, index):
        if index < 0 or self.manifest_df is None:
            return
        selected_rat = self.combo_rat.currentData()
        sub = self.manifest_df[self.manifest_df["Rat"] == selected_rat]
        regions = sorted(sub["Region"].unique().tolist())

        self.combo_region.blockSignals(True)
        self.combo_region.clear()
        for reg in regions:
            self.combo_region.addItem(reg, userData=reg)
        self.combo_region.blockSignals(False)

        if regions:
            self.on_region_changed(0)
        else:
            self.clear_plots()

    def on_region_changed(self, index):
        if index < 0 or self.manifest_df is None:
            return
        selected_rat = self.combo_rat.currentData()
        selected_region = self.combo_region.currentData()
        matched = self.manifest_df[
            (self.manifest_df["Rat"] == selected_rat)
            & (self.manifest_df["Region"] == selected_region)
        ]

        self.combo_files.blockSignals(True)
        self.combo_files.clear()
        for orig_idx, row in matched.iterrows():
            lbl = (
                f"{row['Date']} | ch:{row['channel']} "
                f"tr:{row['trial']} -> {row['File']}"
            )
            self.combo_files.addItem(lbl, userData=orig_idx)
        self.combo_files.blockSignals(False)

        # Reposition the threshold lines for this region *before* loading the
        # file, so they're correct when the first trace is drawn.
        self._refresh_threshold_lines()

        if self.combo_files.count() > 0:
            self.on_file_selected(0)
        else:
            self.clear_plots()

    def clear_plots(self):
        self.curve_raw.clear()
        self.curve_filt.clear()
        self.curve_r_continuous.clear()
        self._remove_span_items()
        self.current_ar_events = pd.DataFrame()
        self.current_wavelet_events = pd.DataFrame()
        self.lbl_ar_tracker.setText("AR: 0 / 0")
        self.lbl_wav_tracker.setText("WAV: 0 / 0")
        self.details_panel.setText(
            "No recordings match current Rat and Region filters."
        )

        if self._ar_signal_thread is not None and self._ar_signal_thread.isRunning():
            self._ar_signal_thread.cancel()
            self._ar_signal_thread.wait()

        self.view_update_timer.stop()
        self.spinner_timer.stop()
        self.loading_text.hide()

    def _remove_span_items(self):
        for triple in self.region_items:
            for plot, item in zip(self._span_plots, triple):
                try:
                    plot.removeItem(item)
                except Exception:
                    pass
        self.region_items.clear()

    # --------------------------------------------------------------------- #
    # File loading
    # --------------------------------------------------------------------- #
    def on_file_selected(self, index):
        if index < 0 or self.manifest_df is None or self.combo_files.count() == 0:
            return

        manifest_idx = self.combo_files.itemData(index)
        if manifest_idx is None or manifest_idx not in self.manifest_df.index:
            return

        task = self.manifest_df.loc[manifest_idx]
        data_path = task["data_path"]

        if not os.path.exists(data_path):
            self.details_panel.setText(f"File missing on disk: {data_path}")
            return

        self._pending_task = task
        self.combo_files.setEnabled(False)
        self.status_bar.showMessage(f"Loading {Path(data_path).name} ...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait()

        self._loader_thread = SignalLoader(data_path, parent=self)
        self._loader_thread.finished_ok.connect(self._on_signal_loaded)
        self._loader_thread.failed.connect(self._on_signal_load_failed)
        self._loader_thread.start()

    def _on_signal_load_failed(self, message):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.combo_files.setEnabled(True)
        self.status_bar.clearMessage()
        self.details_panel.setText(f"Signal processing error: {message}")

    def _on_signal_loaded(self, raw_signal, filtered_signal):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.combo_files.setEnabled(True)
        self.status_bar.clearMessage()

        task = self._pending_task
        if task is None:
            return

        self.raw_signal = raw_signal
        self.filtered_signal = filtered_signal
        self.data_len = len(self.raw_signal)
        self.data_duration = self.data_len / FS
        self.time_vector = np.arange(self.data_len, dtype=np.float32) / FS

        self._select_events_for_task(task)

        self.curve_raw.setData(self.time_vector, self.raw_signal)
        self.curve_filt.setData(self.time_vector, self.filtered_signal)
        self.curve_r_continuous.clear()

        # Make sure threshold lines reflect the current region and the Y
        # axis accommodates the upper threshold.
        self._refresh_threshold_lines()

        self.draw_spans()

        self.current_ar_idx = 0 if len(self.current_ar_events) > 0 else -1
        self.current_wav_idx = 0 if len(self.current_wavelet_events) > 0 else -1

        self.lbl_ar_tracker.setText(
            f"AR: {max(0, self.current_ar_idx + 1)} / {len(self.current_ar_events)}"
        )
        self.lbl_wav_tracker.setText(
            f"WAV: {max(0, self.current_wav_idx + 1)} / "
            f"{len(self.current_wavelet_events)}"
        )

        # Time-sync: if AR has detections, jump there and let the sync logic
        # move the WAV tracker to the nearest-in-time wavelet event. If AR
        # has none but WAV does, do the reverse.
        if self.current_ar_idx >= 0:
            self.jump_to_ar_event(sync=True)
        elif self.current_wav_idx >= 0:
            self.jump_to_wav_event(sync=True)
        else:
            initial_end = min(VIEW_WINDOW_SEC, self.data_duration)
            self.p_raw.setXRange(0, initial_end, padding=0)
            self.details_panel.setText(
                f"File: {task['File']} | Length: {self.data_duration:.1f}s | "
                f"No spindle detections found for this recording."
            )
            self.view_update_timer.start(DEBOUNCE_MS)

    def _select_events_for_task(self, task):
        file_name = task["File"]

        # AR events: prefer exact file match, fall back to (Rat, Region, Date).
        if not self.ar_df.empty:
            if "File" in self.ar_df.columns and (
                self.ar_df["File"] == file_name
            ).any():
                ar_sub = self.ar_df[self.ar_df["File"] == file_name]
            else:
                ar_sub = self.ar_df[
                    (self.ar_df["Rat"] == task["Rat"])
                    & (self.ar_df["Region"] == task["Region"])
                    & (self.ar_df["Date"] == task["Date"])
                ]
            self.current_ar_events = (
                ar_sub.sort_values("Start_s").reset_index(drop=True)
                if not ar_sub.empty
                else pd.DataFrame()
            )
        else:
            self.current_ar_events = pd.DataFrame()

        # Wavelet events: filter by (Rat, Region, Date), then narrow by
        # channel and trial when available.
        if not self.wavelet_df.empty:
            w = self.wavelet_df
            wav_sub = w[
                (w["Rat"] == task["Rat"])
                & (w["Region"] == task["Region"])
                & (w["Date"] == task["Date"])
            ]

            if not wav_sub.empty and task["channel"]:
                chan_match = wav_sub[wav_sub["channel"] == task["channel"]]
                if not chan_match.empty:
                    wav_sub = chan_match

            if not wav_sub.empty and task["trial"]:
                trial_match = wav_sub[wav_sub["trial"] == task["trial"]]
                if not trial_match.empty:
                    wav_sub = trial_match

            self.current_wavelet_events = (
                wav_sub.sort_values("Start_s").reset_index(drop=True)
                if not wav_sub.empty
                else pd.DataFrame()
            )
        else:
            self.current_wavelet_events = pd.DataFrame()

    # --------------------------------------------------------------------- #
    # View-change debounce
    # --------------------------------------------------------------------- #
    def _on_xrange_changed(self, _, range_tuple):
        if self.raw_signal is not None:
            self.view_update_timer.start(DEBOUNCE_MS)

    def _update_spinner(self):
        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)
        frame = self.spinner_frames[self.spinner_idx]
        self.loading_text.setText(f"Calculating R-values {frame}")

        view_range = self.p_r.viewRange()
        cx = 0.5 * (view_range[0][0] + view_range[0][1])
        cy = 0.5 * (view_range[1][0] + view_range[1][1])
        if view_range[1][0] == 0.0 and view_range[1][1] == 1.0:
            cy = 0.5
        self.loading_text.setPos(cx, cy)

    # --------------------------------------------------------------------- #
    # AR computation
    # --------------------------------------------------------------------- #
    def _start_window_computation(self):
        if not MODULES_AVAILABLE or self.raw_signal is None:
            return

        if self._ar_signal_thread is not None and self._ar_signal_thread.isRunning():
            self._ar_signal_thread.cancel()
            self._ar_signal_thread.wait()

        view = self.p_raw.viewRange()[0]
        start_s, end_s = float(view[0]), float(view[1])
        vprint(f"[UI] start AR for range=[{start_s:.3f}, {end_s:.3f}]")

        self.loading_text.show()
        self.spinner_timer.start(100)
        self.curve_r_continuous.clear()

        self._ar_signal_thread = ARWindowWorker(
            self.raw_signal, start_s, end_s, parent=self
        )
        self._ar_signal_thread.finished_ok.connect(self._on_window_computation_ready)
        self._ar_signal_thread.failed.connect(self._on_window_computation_failed)
        self._ar_signal_thread.start()

    def _on_window_computation_ready(
        self, requested_start, requested_end, t_vals, r_vals
    ):
        self.spinner_timer.stop()
        self.loading_text.hide()

        current_view = self.p_raw.viewRange()[0]
        cur_lo, cur_hi = float(current_view[0]), float(current_view[1])

        # Stale-result rejection: the view has moved while we were computing.
        if (
            abs(cur_lo - requested_start) > STALE_TOLERANCE_SEC
            or abs(cur_hi - requested_end) > STALE_TOLERANCE_SEC
        ):
            vprint(
                f"[UI] stale result for [{requested_start:.3f}, {requested_end:.3f}] "
                f"(current view [{cur_lo:.3f}, {cur_hi:.3f}]) -> discarded"
            )
            return

        if len(t_vals) == 0 or len(t_vals) != len(r_vals):
            return

        t_arr = np.asarray(t_vals, dtype=np.float64)
        r_arr = np.asarray(r_vals, dtype=np.float64)

        # Safety net: snap view to data if they don't overlap.
        data_lo, data_hi = float(t_arr[0]), float(t_arr[-1])
        if cur_hi < data_lo or cur_lo > data_hi:
            vprint(
                f"[UI] view/data mismatch; snapping X view to "
                f"[{data_lo:.3f}, {data_hi:.3f}]"
            )
            self.p_raw.setXRange(data_lo, data_hi, padding=0)

        self.curve_r_continuous.setData(t_arr, r_arr)

        # Y-range: never crop below the region's upper threshold line.
        upper, _ = self._current_thresholds()
        finite = r_arr[np.isfinite(r_arr)]
        data_max = float(np.nanmax(finite)) if finite.size else 0.0
        ymax = max(1.05, upper * 1.05, data_max * 1.05)
        self.p_r.setYRange(0.0, ymax, padding=0)
        vprint(
            f"[UI] drew AR trace n={len(r_arr)} finite={finite.size} "
            f"ymax={ymax:.3f} upper_thr={upper:.3f}"
        )

    def _on_window_computation_failed(self, msg):
        self.spinner_timer.stop()
        self.loading_text.hide()
        self.status_bar.showMessage(f"AR computation failed: {msg}", 12000)

    # --------------------------------------------------------------------- #
    # Event spans
    # --------------------------------------------------------------------- #
    def draw_spans(self):
        self._remove_span_items()

        def add_spans(df, brush_rgba, pen_rgba, width, z):
            for _, row in df.iterrows():
                s = safe_float(row["Start_s"], 0.0)
                e = safe_float(row["End_s"], 0.0)
                triple = tuple(
                    pg.LinearRegionItem(
                        [s, e],
                        movable=False,
                        brush=QtGui.QColor(*brush_rgba),
                        pen=pg.mkPen(color=pen_rgba, width=width),
                    )
                    for _ in range(3)
                )
                for item, plot in zip(triple, self._span_plots):
                    item.setZValue(z)
                    plot.addItem(item)
                self.region_items.append(triple)

        if not self.current_wavelet_events.empty:
            add_spans(
                self.current_wavelet_events,
                brush_rgba=(0, 229, 255, 90),
                pen_rgba=(0, 229, 255, 230),
                width=1.8,
                z=5,
            )

        if not self.current_ar_events.empty:
            add_spans(
                self.current_ar_events,
                brush_rgba=(255, 50, 50, 80),
                pen_rgba=(255, 50, 50, 230),
                width=1.5,
                z=10,
            )

    # --------------------------------------------------------------------- #
    # Nearest-event helpers (for time-sync)
    # --------------------------------------------------------------------- #
    @staticmethod
    def _nearest_index(df, center_s):
        """Index of the row in df whose [Start_s, End_s] is nearest to center_s.

        Distance is measured to the interval (0 if center_s is inside the
        interval). Returns -1 if df is empty.
        """
        if df.empty:
            return -1

        starts = df["Start_s"].to_numpy(dtype=np.float64)
        ends = df["End_s"].to_numpy(dtype=np.float64)
        # Distance from center to the interval [start, end]:
        # 0 if inside, else distance to nearest endpoint.
        dist = np.where(
            center_s < starts, starts - center_s,
            np.where(center_s > ends, center_s - ends, 0.0),
        )
        return int(np.argmin(dist))

    def _sync_other_tracker_to(self, center_s, source):
        """Move the other detector's current index to its nearest event.

        source is 'ar' or 'wav'. Guards against recursion via
        self._syncing_navigation.
        """
        if self._syncing_navigation:
            return

        self._syncing_navigation = True
        try:
            if source == "ar":
                if not self.current_wavelet_events.empty:
                    new_idx = self._nearest_index(
                        self.current_wavelet_events, center_s
                    )
                    if new_idx >= 0:
                        self.current_wav_idx = new_idx
                        self.lbl_wav_tracker.setText(
                            f"WAV: {new_idx + 1} / "
                            f"{len(self.current_wavelet_events)}"
                        )
            else:
                if not self.current_ar_events.empty:
                    new_idx = self._nearest_index(
                        self.current_ar_events, center_s
                    )
                    if new_idx >= 0:
                        self.current_ar_idx = new_idx
                        self.lbl_ar_tracker.setText(
                            f"AR: {new_idx + 1} / "
                            f"{len(self.current_ar_events)}"
                        )
        finally:
            self._syncing_navigation = False

    # --------------------------------------------------------------------- #
    # Navigation
    # --------------------------------------------------------------------- #
    def on_prev_ar_event(self):
        if len(self.current_ar_events) == 0:
            return
        self.current_ar_idx = (self.current_ar_idx - 1) % len(self.current_ar_events)
        self.jump_to_ar_event(sync=True)

    def on_next_ar_event(self):
        if len(self.current_ar_events) == 0:
            return
        self.current_ar_idx = (self.current_ar_idx + 1) % len(self.current_ar_events)
        self.jump_to_ar_event(sync=True)

    def on_prev_wav_event(self):
        if len(self.current_wavelet_events) == 0:
            return
        self.current_wav_idx = (self.current_wav_idx - 1) % len(
            self.current_wavelet_events
        )
        self.jump_to_wav_event(sync=True)

    def on_next_wav_event(self):
        if len(self.current_wavelet_events) == 0:
            return
        self.current_wav_idx = (self.current_wav_idx + 1) % len(
            self.current_wavelet_events
        )
        self.jump_to_wav_event(sync=True)

    def _centered_view_range(self, start_s, end_s):
        center = 0.5 * (start_s + end_s)
        half = VIEW_WINDOW_SEC / 2.0
        view_start = center - half
        view_end = center + half

        if view_start < 0.0:
            view_start = 0.0
            view_end = min(VIEW_WINDOW_SEC, self.data_duration)
        if view_end > self.data_duration:
            view_end = self.data_duration
            view_start = max(0.0, view_end - VIEW_WINDOW_SEC)

        return view_start, view_end

    def jump_to_ar_event(self, sync=False):
        if self.current_ar_idx < 0 or self.current_ar_idx >= len(
            self.current_ar_events
        ):
            return

        event = self.current_ar_events.iloc[self.current_ar_idx]
        start_s = safe_float(event["Start_s"], 0.0)
        end_s = safe_float(event["End_s"], 0.0)
        center_s = 0.5 * (start_s + end_s)

        view_start, view_end = self._centered_view_range(start_s, end_s)
        self.p_raw.setXRange(view_start, view_end, padding=0)

        upper, _ = self._current_thresholds()
        self.p_r.setYRange(0.0, max(1.05, upper * 1.05), padding=0)

        self.view_update_timer.start(DEBOUNCE_MS)

        if sync:
            self._sync_other_tracker_to(center_s, source="ar")

        self._update_details_panel_ar(start_s, end_s)

    def jump_to_wav_event(self, sync=False):
        if self.current_wav_idx < 0 or self.current_wav_idx >= len(
            self.current_wavelet_events
        ):
            return

        event = self.current_wavelet_events.iloc[self.current_wav_idx]
        start_s = safe_float(event["Start_s"], 0.0)
        end_s = safe_float(event["End_s"], 0.0)
        center_s = 0.5 * (start_s + end_s)

        view_start, view_end = self._centered_view_range(start_s, end_s)
        self.p_raw.setXRange(view_start, view_end, padding=0)

        upper, _ = self._current_thresholds()
        self.p_r.setYRange(0.0, max(1.05, upper * 1.05), padding=0)

        self.view_update_timer.start(DEBOUNCE_MS)

        if sync:
            self._sync_other_tracker_to(center_s, source="wav")

        self._update_details_panel_wav(start_s, end_s)

    # --------------------------------------------------------------------- #
    # Details panel content
    # --------------------------------------------------------------------- #
    def _update_details_panel_ar(self, start_s, end_s):
        """Compose the details panel from the current AR + nearest WAV."""
        ar_event = self.current_ar_events.iloc[self.current_ar_idx]
        r_val = ar_event.get("Max_R", np.nan)
        freq_val = ar_event.get("Peak_Freq_Hz", np.nan)

        # Find wavelet events overlapping the focused AR window.
        overlapping = pd.DataFrame()
        if not self.current_wavelet_events.empty:
            w = self.current_wavelet_events
            overlapping = w[(w["Start_s"] <= end_s) & (w["End_s"] >= start_s)]

        # Also identify the nearest wavelet event by center distance.
        nearest_txt = "N/A"
        if not self.current_wavelet_events.empty:
            center_s = 0.5 * (start_s + end_s)
            w_idx = self._nearest_index(self.current_wavelet_events, center_s)
            w_row = self.current_wavelet_events.iloc[w_idx]
            ws, we = safe_float(w_row["Start_s"], 0.0), safe_float(w_row["End_s"], 0.0)
            nearest_txt = (
                f"#{w_idx + 1} [{ws:.3f}s - {we:.3f}s] "
                f"(Δcenter={abs(0.5*(ws+we) - center_s):.3f}s)"
            )

        upper, lower = self._current_thresholds()
        region = self.combo_region.currentData()

        align_desc = (
            f"ALIGNED: {len(overlapping)} wavelet event(s) overlap this AR window"
            if len(overlapping) > 0
            else "ISOLATED: no overlapping wavelet event"
        )

        lines = [
            f"[AR #{self.current_ar_idx + 1}/{len(self.current_ar_events)}] "
            f"region={region}  T_upper={upper:.2f} T_lower={lower:.2f}",
            f"  AR interval: [{start_s:.3f}s - {end_s:.3f}s] "
            f"(Duration: {end_s - start_s:.3f}s)  "
            f"CSV Max R={fmt(r_val)}  Peak Freq={fmt(freq_val, '.2f')} Hz",
            f"  Nearest WAV event: {nearest_txt}",
            f"  Alignment: {align_desc}",
        ]
        self.details_panel.setText("\n".join(lines))
        self.lbl_ar_tracker.setText(
            f"AR: {self.current_ar_idx + 1} / {len(self.current_ar_events)}"
        )

    def _update_details_panel_wav(self, start_s, end_s):
        """Compose the details panel from the current WAV + nearest AR."""
        ar_event = None
        nearest_txt = "N/A"
        if not self.current_ar_events.empty:
            center_s = 0.5 * (start_s + end_s)
            a_idx = self._nearest_index(self.current_ar_events, center_s)
            a_row = self.current_ar_events.iloc[a_idx]
            as_, ae = safe_float(a_row["Start_s"], 0.0), safe_float(a_row["End_s"], 0.0)
            nearest_txt = (
                f"#{a_idx + 1} [{as_:.3f}s - {ae:.3f}s] "
                f"(Δcenter={abs(0.5*(as_+ae) - center_s):.3f}s)"
            )
            ar_event = a_row

        overlapping = pd.DataFrame()
        if not self.current_ar_events.empty:
            a = self.current_ar_events
            overlapping = a[(a["Start_s"] <= end_s) & (a["End_s"] >= start_s)]

        align_desc = (
            f"ALIGNED: {len(overlapping)} AR event(s) overlap this WAV window"
            if len(overlapping) > 0
            else "ISOLATED: no overlapping AR event"
        )

        upper, lower = self._current_thresholds()
        region = self.combo_region.currentData()

        # If the nearest AR event has CSV metadata, surface it.
        ar_meta = ""
        if ar_event is not None:
            ar_meta = (
                f"  Nearest AR: Max R={fmt(ar_event.get('Max_R', np.nan))}  "
                f"Peak Freq={fmt(ar_event.get('Peak_Freq_Hz', np.nan), '.2f')} Hz"
            )

        lines = [
            f"[WAV #{self.current_wav_idx + 1}/"
            f"{len(self.current_wavelet_events)}] "
            f"region={region}  T_upper={upper:.2f} T_lower={lower:.2f}",
            f"  WAV interval: [{start_s:.3f}s - {end_s:.3f}s] "
            f"(Duration: {end_s - start_s:.3f}s)",
            f"  Nearest AR event: {nearest_txt}",
            f"  Alignment: {align_desc}",
        ]
        if ar_meta:
            lines.append(ar_meta)

        self.details_panel.setText("\n".join(lines))
        self.lbl_wav_tracker.setText(
            f"WAV: {self.current_wav_idx + 1} / "
            f"{len(self.current_wavelet_events)}"
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Spindle Alignment Inspector (AR vs Wavelet)"
    )
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST,
        help="Path to tasks_manifest.csv",
    )
    parser.add_argument(
        "--ar-csv", default=DEFAULT_AR_CSV,
        help="Path to AR-detected spindles CSV",
    )
    parser.add_argument(
        "--wavelet-csv", default=DEFAULT_WAVELET_CSV,
        help="Path to wavelet-detected spindles CSV",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print verbose diagnostic output to the terminal.",
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    global VERBOSE

    args = parse_args()
    VERBOSE = args.debug

    if VERBOSE:
        print("=" * 70)
        print("Spindle Alignment Inspector — DEBUG")
        print("=" * 70)
        print(f"manifest    = {args.manifest}")
        print(f"ar_csv      = {args.ar_csv}")
        print(f"wavelet_csv = {args.wavelet_csv}")
        print(f"MODULES_AVAILABLE = {MODULES_AVAILABLE}")
        if not MODULES_AVAILABLE:
            print(f"IMPORT_ERROR = {_IMPORT_ERROR}")
        print("=" * 70)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(40, 40, 40))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(40, 40, 40))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(50, 50, 50))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 188, 212))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(dark_palette)

    def excepthook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = excepthook

    viewer = SpindleViewer(args.manifest, args.ar_csv, args.wavelet_csv)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
