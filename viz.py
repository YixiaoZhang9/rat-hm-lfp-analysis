#!/usr/bin/env python3
"""
Spindle Alignment Inspector (AR vs Wavelet)

Page through detected spindle events from two detectors overlaid on the raw
LFP, band-passed LFP, and a continuous AR R-value trace dynamically computed
for the active view window.

The AR trace is computed with the exact same pipeline that the extraction
script uses via find_spindles_lfp() so the live trace and the CSV events agree:

    1. bandpass_filter(raw, lowcut=0.1, highcut=100, fs=1000)
    2. downsampling(filtered, 1000, 128)
    3. sliding 1.0 s window @ 128 Hz, stride_samples=4
    4. burg(order=8, demean=False), poles with imag>0
    5. per-window R = max |pole| over poles with frequency in [10, 15] Hz,
       or 0.0 if no in-band pole
    6. time axis = (start_sample + window_samples/2) / 128
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
# Configuration — must mirror find_spindles_lfp() defaults and the extraction
# script's call so the live trace reproduces the CSV events.
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
AR_WINDOW_SEC = 1.0
AR_STRIDE_SAMPLES = 4              # matches extraction script's STRIDE
AR_PRE_LOWCUT = 0.1
AR_PRE_HIGHCUT = 100.0

# Extra padding beyond the visible window so the final AR window can complete.
# In samples of the downsampled signal.
AR_PAD_DOWNSAMPLED = int(AR_WINDOW_SEC * AR_TARGET_FS)

DEBOUNCE_MS = 300
STALE_TOLERANCE_SEC = 0.5

REQUIRED_MANIFEST_COLS = {"rat", "region", "date", "data_path"}

REGION_THRESHOLDS = {
    "HPC": {"upper": 0.85, "lower": 0.40},
    "PL":  {"upper": 0.85, "lower": 0.40},
    "RSC": {"upper": 0.90, "lower": 0.40},
}
DEFAULT_THRESHOLDS = {"upper": 0.85, "lower": 0.40}

VERBOSE = False


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs, flush=True)


def _sanitize(x):
    arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# --------------------------------------------------------------------------- #
# AR Window Worker — mirrors find_spindles_lfp internals exactly
# --------------------------------------------------------------------------- #
class ARWindowWorker(QtCore.QThread):
    """
    Computes the continuous R trace for the given time slice, using the same
    pipeline as find_spindles_lfp() so the trace reproduces the detector's
    per-window R values.

    Emits: finished_ok(requested_start_s, requested_end_s, t_vals, r_vals)
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
        vprint(
            f"\n[AR] range=[{self.start_s:.3f}, {self.end_s:.3f}] "
            f"raw_len={len(self.raw_signal)}"
        )
        try:
            # Pad on both sides so windows whose CENTER falls within the
            # visible range have all their samples available.
            half_win_sec = AR_WINDOW_SEC / 2.0
            s_idx = max(0, int((self.start_s - half_win_sec) * FS))
            e_idx = min(
                len(self.raw_signal), int((self.end_s + half_win_sec) * FS)
            )

            if s_idx >= e_idx:
                self._emit_empty()
                return

            segment_raw = _sanitize(self.raw_signal[s_idx:e_idx])
            if len(segment_raw) == 0:
                self._emit_empty()
                return

            # ---- Stage 1: same pre-filter as find_spindles_lfp -------- #
            try:
                filtered = ar_bandpass_filter(
                    segment_raw,
                    lowcut=AR_PRE_LOWCUT,
                    highcut=AR_PRE_HIGHCUT,
                    fs=FS,
                )
            except TypeError:
                filtered = ar_bandpass_filter(
                    segment_raw, AR_PRE_LOWCUT, AR_PRE_HIGHCUT, FS
                )
            filtered = _sanitize(filtered)

            # ---- Stage 2: same downsample ----------------------------- #
            signal_128 = _sanitize(ar_downsampling(filtered, FS, AR_TARGET_FS))

            window_samples = int(AR_WINDOW_SEC * AR_TARGET_FS)
            if len(signal_128) < window_samples:
                self._emit_empty()
                return

            total_windows = len(signal_128) - window_samples + 1
            starts = np.arange(0, total_windows, AR_STRIDE_SAMPLES)
            n_windows = len(starts)

            r_values = np.zeros(n_windows, dtype=np.float64)  # detector uses 0.0 for no-in-band
            n_in_band = 0
            n_failed = 0
            first_error = None

            for k, i in enumerate(starts):
                if self._is_cancelled:
                    return
                window = signal_128[i : i + window_samples]
                if not np.all(np.isfinite(window)) or np.all(window == 0.0):
                    n_failed += 1
                    continue

                try:
                    a, _ = sm.regression.linear_model.burg(
                        window, order=AR_ORDER, demean=False  # matches detector
                    )
                    poles = np.roots(np.r_[1, -a])
                    poles = poles[np.imag(poles) > 0]

                    if len(poles) > 0:
                        freqs = np.angle(poles) * AR_TARGET_FS / (2 * np.pi)
                        r_vals = np.abs(poles)
                        mask = (
                            (freqs >= AR_SPINDLE_BAND[0])
                            & (freqs <= AR_SPINDLE_BAND[1])
                        )
                        if np.any(mask):
                            r_values[k] = float(np.max(r_vals[mask]))
                            n_in_band += 1
                        # else: leave 0.0, same as _fit_window
                except Exception as exc:
                    n_failed += 1
                    if first_error is None:
                        first_error = f"{type(exc).__name__}: {exc}"

            if self._is_cancelled:
                return

            # ---- Stage 3: time axis = window CENTER in absolute time --- #
            # We computed windows on signal_128 starting at index 0, which
            # corresponds to raw sample s_idx. The window's start sample in
            # the raw timeline is s_idx + i * (FS/AR_TARGET_FS), and its
            # center adds half the window length.
            downsample_ratio = FS / AR_TARGET_FS
            half_win_raw = window_samples * downsample_ratio / 2.0
            t_vals = (s_idx + starts * downsample_ratio + half_win_raw) / FS

            vprint(
                f"[AR] emit n={n_windows} in_band={n_in_band} "
                f"fail={n_failed} "
                f"t=[{t_vals[0]:.3f},{t_vals[-1]:.3f}]"
            )

            if n_in_band == 0 and n_failed == n_windows:
                self.failed.emit(
                    f"All {n_windows} AR windows failed. "
                    f"First error: {first_error or 'no poles'}"
                )
                return

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
                filtered.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
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
            ("— AR R (in-band max)", "#ffab40"),
            ("--- T_upper", "#ff5252"),
            ("--- T_lower", "#52a0ff"),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            nav_layout.addWidget(lbl)

        nav_layout.addStretch(1)
        nav_box.setLayout(nav_layout)
        root_layout.addWidget(nav_box)

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

        # The detector returns 0.0 when no in-band pole exists; use a marker
        # (not a line) so the zeros don't dominate the plot.
        self.curve_r_continuous = self.p_r.plot(
            pen=pg.mkPen(color="#ffab40", width=2),
            connect="finite",
        )

        # Threshold reference lines
        self.line_upper = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(color=(255, 82, 82, 200), width=1.5,
                         style=QtCore.Qt.DashLine),
            label="T_upper",
            labelOpts={"position": 0.05, "color": (255, 82, 82),
                       "fill": (0, 0, 0, 120), "movable": False},
        )
        self.line_lower = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(color=(82, 160, 255, 200), width=1.5,
                         style=QtCore.Qt.DashLine),
            label="T_lower",
            labelOpts={"position": 0.05, "color": (82, 160, 255),
                       "fill": (0, 0, 0, 120), "movable": False},
        )
        self.line_upper.setZValue(15)
        self.line_lower.setZValue(15)
        self.p_r.addItem(self.line_upper)
        self.p_r.addItem(self.line_lower)

        self.loading_text = pg.TextItem("", color=(255, 171, 64), anchor=(0.5, 0.5))
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.loading_text.setFont(font)
        self.p_r.addItem(self.loading_text)
        self.loading_text.hide()

        self.p_r.setYRange(0.0, 1.05, padding=0)

        self.p_filt.setXLink(self.p_raw)
        self.p_r.setXLink(self.p_raw)
        self._span_plots = (self.p_raw, self.p_filt, self.p_r)
        self.region_items = []

        self.details_panel = QtWidgets.QTextEdit()
        self.details_panel.setReadOnly(True)
        self.details_panel.setMaximumHeight(90)
        self.details_panel.setStyleSheet(
            "background-color: #1e1e1e; color: #00e676; "
            "font-family: Monospace; font-size: 12px;"
        )
        root_layout.addWidget(self.details_panel)

        self.status_bar = self.statusBar()

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
    # Thresholds
    # --------------------------------------------------------------------- #
    def _current_thresholds(self):
        region = self.combo_region.currentData()
        thr = REGION_THRESHOLDS.get(region, DEFAULT_THRESHOLDS)
        return float(thr["upper"]), float(thr["lower"])

    def _refresh_threshold_lines(self):
        upper, lower = self._current_thresholds()
        self.line_upper.setValue(upper)
        self.line_lower.setValue(lower)
        ymax = max(1.05, upper * 1.05)
        self.p_r.setYRange(0.0, ymax, padding=0)
        vprint(f"[UI] T_upper={upper:.3f} T_lower={lower:.3f}")

    # --------------------------------------------------------------------- #
    # Data loading
    # --------------------------------------------------------------------- #
    def load_data(self):
        if not MODULES_AVAILABLE:
            self.details_panel.setText(
                f"Error: Could not import ephys_preprocessing modules.\n"
                f"Details: {_IMPORT_ERROR}"
            )
            return

        if not os.path.exists(self.manifest_path):
            self.details_panel.setText(
                f"Manifest not found: {self.manifest_path}"
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
                f"Manifest is missing required column(s): {sorted(missing)}."
            )
            self.manifest_df = None
            return

        rat_numeric = pd.to_numeric(self.manifest_df["rat"], errors="coerce")
        bad = int(rat_numeric.isna().sum())
        if bad:
            self.status_bar.showMessage(
                f"Warning: dropped {bad} manifest row(s) with a non-numeric 'rat'.",
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
                f"Warning: couldn't find start/end columns in {kind} CSV.", 10000
            )
        return df

    # --------------------------------------------------------------------- #
    # Combos
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

        if self.current_ar_idx >= 0:
            self.jump_to_ar_event(sync=True)
        elif self.current_wav_idx >= 0:
            self.jump_to_wav_event(sync=True)
        else:
            self.p_raw.setXRange(
                0, min(VIEW_WINDOW_SEC, self.data_duration), padding=0
            )
            self.details_panel.setText(
                f"File: {task['File']} | Length: {self.data_duration:.1f}s | "
                f"No spindle detections found."
            )
            self.view_update_timer.start(DEBOUNCE_MS)

    def _select_events_for_task(self, task):
        file_name = task["File"]

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
    # View change
    # --------------------------------------------------------------------- #
    def _on_xrange_changed(self, _, range_tuple):
        if self.raw_signal is not None:
            self.view_update_timer.start(DEBOUNCE_MS)

    def _update_spinner(self):
        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)
        frame = self.spinner_frames[self.spinner_idx]
        self.loading_text.setText(f"Calculating R {frame}")
        view_range = self.p_r.viewRange()
        cx = 0.5 * (view_range[0][0] + view_range[0][1])
        cy = 0.5 * (view_range[1][0] + view_range[1][1])
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

        if (
            abs(cur_lo - requested_start) > STALE_TOLERANCE_SEC
            or abs(cur_hi - requested_end) > STALE_TOLERANCE_SEC
        ):
            return

        if len(t_vals) == 0 or len(t_vals) != len(r_vals):
            return

        t_arr = np.asarray(t_vals, dtype=np.float64)
        r_arr = np.asarray(r_vals, dtype=np.float64)

        # Detector returns 0.0 when no in-band pole exists. Convert those to
        # NaN so connect="finite" draws gaps instead of dropping to zero.
        r_plot = np.where(r_arr > 0.0, r_arr, np.nan)

        data_lo, data_hi = float(t_arr[0]), float(t_arr[-1])
        if cur_hi < data_lo or cur_lo > data_hi:
            self.p_raw.setXRange(data_lo, data_hi, padding=0)

        self.curve_r_continuous.setData(t_arr, r_plot)

        upper, _ = self._current_thresholds()
        finite = r_plot[np.isfinite(r_plot)]
        data_max = float(np.nanmax(finite)) if finite.size else 0.0
        ymax = max(1.05, upper * 1.05, data_max * 1.05)
        self.p_r.setYRange(0.0, ymax, padding=0)

    def _on_window_computation_failed(self, msg):
        self.spinner_timer.stop()
        self.loading_text.hide()
        self.status_bar.showMessage(f"AR computation failed: {msg}", 12000)

    # --------------------------------------------------------------------- #
    # Spans
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
                width=1.8, z=5,
            )
        if not self.current_ar_events.empty:
            add_spans(
                self.current_ar_events,
                brush_rgba=(255, 50, 50, 80),
                pen_rgba=(255, 50, 50, 230),
                width=1.5, z=10,
            )

    # --------------------------------------------------------------------- #
    # Nearest-event helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _nearest_index(df, center_s):
        if df.empty:
            return -1
        starts = df["Start_s"].to_numpy(dtype=np.float64)
        ends = df["End_s"].to_numpy(dtype=np.float64)
        dist = np.where(
            center_s < starts, starts - center_s,
            np.where(center_s > ends, center_s - ends, 0.0),
        )
        return int(np.argmin(dist))

    def _sync_other_tracker_to(self, center_s, source):
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
    # Details panel
    # --------------------------------------------------------------------- #
    def _update_details_panel_ar(self, start_s, end_s):
        ar_event = self.current_ar_events.iloc[self.current_ar_idx]
        r_val = ar_event.get("Max_R", np.nan)
        freq_val = ar_event.get("Peak_Freq_Hz", np.nan)
        center_s = 0.5 * (start_s + end_s)

        overlapping = pd.DataFrame()
        if not self.current_wavelet_events.empty:
            w = self.current_wavelet_events
            overlapping = w[(w["Start_s"] <= end_s) & (w["End_s"] >= start_s)]

        nearest_txt = "N/A"
        if not self.current_wavelet_events.empty:
            w_idx = self._nearest_index(self.current_wavelet_events, center_s)
            w_row = self.current_wavelet_events.iloc[w_idx]
            ws = safe_float(w_row["Start_s"], 0.0)
            we = safe_float(w_row["End_s"], 0.0)
            nearest_txt = (
                f"#{w_idx + 1} [{ws:.3f}s - {we:.3f}s] "
                f"(Δcenter={abs(0.5*(ws+we) - center_s):.3f}s)"
            )

        upper, lower = self._current_thresholds()
        region = self.combo_region.currentData()
        align_desc = (
            f"ALIGNED: {len(overlapping)} wavelet event(s) overlap"
            if len(overlapping) > 0
            else "ISOLATED: no overlapping wavelet event"
        )

        self.details_panel.setText("\n".join([
            f"[AR #{self.current_ar_idx + 1}/{len(self.current_ar_events)}] "
            f"region={region}  T_upper={upper:.2f} T_lower={lower:.2f}",
            f"  AR interval: [{start_s:.3f}s - {end_s:.3f}s] "
            f"(Duration: {end_s - start_s:.3f}s)  "
            f"CSV Max R={fmt(r_val)}  Peak Freq={fmt(freq_val, '.2f')} Hz",
            f"  Nearest WAV: {nearest_txt}",
            f"  Alignment: {align_desc}",
        ]))
        self.lbl_ar_tracker.setText(
            f"AR: {self.current_ar_idx + 1} / {len(self.current_ar_events)}"
        )

    def _update_details_panel_wav(self, start_s, end_s):
        center_s = 0.5 * (start_s + end_s)

        nearest_txt = "N/A"
        ar_event = None
        if not self.current_ar_events.empty:
            a_idx = self._nearest_index(self.current_ar_events, center_s)
            a_row = self.current_ar_events.iloc[a_idx]
            as_ = safe_float(a_row["Start_s"], 0.0)
            ae = safe_float(a_row["End_s"], 0.0)
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
            f"ALIGNED: {len(overlapping)} AR event(s) overlap"
            if len(overlapping) > 0
            else "ISOLATED: no overlapping AR event"
        )

        upper, lower = self._current_thresholds()
        region = self.combo_region.currentData()

        lines = [
            f"[WAV #{self.current_wav_idx + 1}/"
            f"{len(self.current_wavelet_events)}] "
            f"region={region}  T_upper={upper:.2f} T_lower={lower:.2f}",
            f"  WAV interval: [{start_s:.3f}s - {end_s:.3f}s] "
            f"(Duration: {end_s - start_s:.3f}s)",
            f"  Nearest AR: {nearest_txt}",
            f"  Alignment: {align_desc}",
        ]
        if ar_event is not None:
            lines.append(
                f"  Nearest AR: Max R={fmt(ar_event.get('Max_R', np.nan))}  "
                f"Peak Freq={fmt(ar_event.get('Peak_Freq_Hz', np.nan), '.2f')} Hz"
            )
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
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--ar-csv", default=DEFAULT_AR_CSV)
    parser.add_argument("--wavelet-csv", default=DEFAULT_WAVELET_CSV)
    parser.add_argument("--debug", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def main():
    global VERBOSE
    args = parse_args()
    VERBOSE = args.debug

    if VERBOSE:
        print("=" * 70)
        print("Spindle Alignment Inspector — DEBUG")
        print(f"manifest    = {args.manifest}")
        print(f"ar_csv      = {args.ar_csv}")
        print(f"wavelet_csv = {args.wavelet_csv}")
        print(f"MODULES_AVAILABLE = {MODULES_AVAILABLE}")
        print(f"AR_STRIDE_SAMPLES = {AR_STRIDE_SAMPLES}")
        print(f"AR_ORDER = {AR_ORDER}  AR_WINDOW_SEC = {AR_WINDOW_SEC}")
        print(f"AR_PRE_FILTER = ({AR_PRE_LOWCUT}, {AR_PRE_HIGHCUT})")
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
