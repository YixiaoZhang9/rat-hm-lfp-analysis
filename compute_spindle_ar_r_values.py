#!/usr/bin/env python3
"""
Spindle Alignment Inspector (AR vs Wavelet)

Lets you page through detected spindle events from two detectors (an
autoregressive/"AR" detector and a wavelet detector) overlaid on the raw and
band-passed LFP trace for a given rat / region / recording.

Usage:
    python spindle_viewer.py \
        --manifest /path/to/tasks_manifest.csv \
        --ar-csv /path/to/all_detected_spindles_per_region.csv \
        --wavelet-csv /path/to/wavelet_spindles_with_ar_dynamics.csv

Paths can also be supplied via environment variables (SPINDLE_MANIFEST,
SPINDLE_AR_CSV, SPINDLE_WAVELET_CSV) so the script isn't tied to one
machine's directory layout. If nothing is supplied, the tool still starts
and lets you know what's missing instead of crashing on launch.
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
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

# --------------------------------------------------------------------------- #
# Optional: on-the-fly AR "R" profile computation for AR-detected events.
#
# The AR detections CSV only carries a single Max_R summary value per event
# (whatever triggered the detector), not a within-event trend. To show a real
# R(t) curve for AR events - the same way the wavelet CSV's r_profile_*
# columns already give us one for wavelet events - we call the *actual*
# functions from your own AR calibration pipeline (compute_spindle_ar_r_values.py)
# live, on the raw trace, rather than approximating.
#
# This assumes spindle_viewer.py lives at the same depth in your repo that
# compute_spindle_ar_r_values.py does (it uses the identical
# sys.path.append(.../"../..")) trick to find the `modules` package). If your
# layout differs, adjust AR_MODULES_ROOT below, or set the AR_MODULES_ROOT
# environment variable to the folder containing `modules/`.
# --------------------------------------------------------------------------- #
_AR_LIVE_IMPORT_ERROR = None
try:
    AR_MODULES_ROOT = os.environ.get(
        "AR_MODULES_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
    )
    if AR_MODULES_ROOT not in sys.path:
        sys.path.append(AR_MODULES_ROOT)
    from modules.ephys_preprocessing import bandpass_filter as ar_bandpass_filter
    from modules.ephys_preprocessing import downsampling as ar_downsampling
    from modules.find_spindles_lfp_o_quality import _fit_window as ar_fit_window

    AR_LIVE_ANALYSIS_AVAILABLE = True
except Exception as _e:  # pragma: no cover - depends on the caller's own repo layout
    AR_LIVE_ANALYSIS_AVAILABLE = False
    _AR_LIVE_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

# --------------------------------------------------------------------------- #
# Paths & Default Configuration
# --------------------------------------------------------------------------- #
DEFAULT_MANIFEST = os.environ.get("SPINDLE_MANIFEST", "tasks_manifest.csv")
DEFAULT_AR_CSV = os.environ.get("SPINDLE_AR_CSV", "results/all_detected_spindles_per_region.csv")
DEFAULT_WAVELET_CSV = os.environ.get(
    "SPINDLE_WAVELET_CSV", "results_ar_calibration/wavelet_spindles_with_ar_dynamics.csv"
)

FS = 1000.0
BP_LOW = 10.0
BP_HIGH = 15.0
VIEW_WINDOW_SEC = 10.0  # total width of the plot view shown at one time

# Same parameters compute_spindle_ar_r_values.py uses to fit the AR pole
# magnitude (R) inside a spindle window.
AR_TARGET_FS = 128
AR_ORDER = 8
AR_SPINDLE_BAND = (BP_LOW, BP_HIGH)
AR_WINDOW_SEC = 1.0
AR_STRIDE_SAMPLES = 4

REQUIRED_MANIFEST_COLS = {"rat", "region", "date", "data_path"}

# The wavelet CSV carries a within-event R trend sampled at these relative
# positions (r_profile_0% .. r_profile_100%). We use these to draw a real
# up/down R curve across each event's [Start_s, End_s] span, rather than a
# single summary point.
PROFILE_PERCENTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PROFILE_COLS = [f"r_profile_{p}%" for p in PROFILE_PERCENTS]


def build_profile_curve(events_df: pd.DataFrame):
    """Builds a single (x, y) pair tracing the R profile across every event in
    events_df, in time order, with NaN gaps between events so the line does
    not connect across the silence between spindles."""
    if events_df.empty or not all(c in events_df.columns for c in PROFILE_COLS):
        return np.array([]), np.array([])

    xs, ys = [], []
    for _, row in events_df.iterrows():
        s = safe_float(row.get("Start_s"))
        e = safe_float(row.get("End_s"))
        if np.isnan(s) or np.isnan(e) or e <= s:
            continue
        t = np.linspace(s, e, len(PROFILE_COLS))
        vals = [safe_float(row.get(c)) for c in PROFILE_COLS]
        xs.extend(t.tolist())
        ys.extend(vals)
        xs.append(np.nan)
        ys.append(np.nan)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def analyze_spindle_r_dynamics(
    signal_128: np.ndarray,
    start_s: float,
    end_s: float,
    window_sec: float = AR_WINDOW_SEC,
    stride_samples: int = AR_STRIDE_SAMPLES,
    target_fs: int = AR_TARGET_FS,
    spindle_band=AR_SPINDLE_BAND,
    ar_order: int = AR_ORDER,
) -> dict:
    """Ported from compute_spindle_ar_r_values.py: slides a 1.0s AR window
    across [start_s, end_s], calling the real _fit_window from
    modules.find_spindles_lfp_o_quality, and returns the same
    peak/min/mean/start/end/profile/in_band_ratio metrics that script
    produces - computed live here instead of read from a pre-built CSV.

    r_max/r_min/r_mean are NaN (not 0.0) when no sliding window resolved an
    in-band AR pole, matching the calibration script's fixed handling: a
    0.0 R-value is indistinguishable from "no pole found" otherwise.
    """
    win_samples = int(window_sec * target_fs)
    half_win = win_samples // 2

    c_start = int(round(start_s * target_fs))
    c_end = int(round(end_s * target_fs))

    centers = np.arange(c_start, max(c_start + 1, c_end + 1), stride_samples)
    r_series, f_series = [], []

    for c in centers:
        w_start = c - half_win
        w_end = w_start + win_samples
        if w_start < 0 or w_end > len(signal_128):
            continue
        r_val, f_val = ar_fit_window(signal_128[w_start:w_end], ar_order, target_fs, spindle_band)
        r_series.append(r_val)
        f_series.append(f_val)

    if not r_series:
        return {
            "r_max": np.nan, "r_min": np.nan, "r_mean": np.nan,
            "r_start": np.nan, "r_end": np.nan, "r_peak_freq": np.nan,
            "r_profile": [np.nan] * len(PROFILE_COLS), "in_band_ratio": 0.0,
            "n_windows": 0,
        }

    r_arr = np.array(r_series)
    f_arr = np.array(f_series)
    n_windows = len(r_arr)

    valid_mask = r_arr > 0
    in_band_ratio = float(np.mean(valid_mask))

    if not np.any(valid_mask):
        return {
            "r_max": np.nan, "r_min": np.nan, "r_mean": np.nan,
            "r_start": np.nan, "r_end": np.nan, "r_peak_freq": np.nan,
            "r_profile": [np.nan] * len(PROFILE_COLS), "in_band_ratio": in_band_ratio,
            "n_windows": n_windows,
        }

    valid_r = r_arr[valid_mask]
    peak_idx = int(np.argmax(r_arr))

    if len(r_arr) >= 2:
        x_norm = np.linspace(0, 1, len(r_arr))
        try:
            from scipy.interpolate import interp1d
            interpolator = interp1d(x_norm, r_arr, kind="linear", bounds_error=False, fill_value="extrapolate")
            profile = interpolator(np.linspace(0, 1, len(PROFILE_COLS))).tolist()
        except Exception:
            profile = np.interp(np.linspace(0, 1, len(PROFILE_COLS)), x_norm, r_arr).tolist()
    else:
        profile = [float(r_arr[0])] * len(PROFILE_COLS)

    return {
        "r_max": float(r_arr[peak_idx]),
        "r_min": float(np.min(valid_r)),
        "r_mean": float(np.mean(valid_r)),
        "r_start": float(r_arr[0]),
        "r_end": float(r_arr[-1]),
        "r_peak_freq": float(f_arr[peak_idx]),
        "r_profile": profile,
        "in_band_ratio": in_band_ratio,
        "n_windows": n_windows,
    }


class ARProfileWorker(QtCore.QThread):
    """Computes a live R(t) profile for a batch of AR events on a background
    thread, reusing the already-loaded raw signal (no re-reading from disk)."""

    finished_ok = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, raw_signal, events, parent=None):
        super().__init__(parent)
        self.raw_signal = raw_signal
        self.events = events  # list of (idx, start_s, end_s)

    def run(self):
        try:
            filtered = ar_bandpass_filter(self.raw_signal, lowcut=0.1, highcut=100, fs=FS)
            signal_128 = ar_downsampling(filtered, FS, AR_TARGET_FS)
            results = {}
            for idx, start_s, end_s in self.events:
                results[idx] = analyze_spindle_r_dynamics(signal_128, start_s, end_s)
            self.finished_ok.emit(results)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")



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
    if m:
        return str(int(m.group(1)))
    return s


def parse_channel_trial(data_path: str):
    filename = Path(str(data_path)).name
    match = re.match(r"chan(\d+)(?:_(\d+))?\.mat$", filename, re.IGNORECASE)
    if not match:
        return "", ""
    chan = str(int(match.group(1)))
    trial = match.group(2) if match.group(2) else ""
    return chan, trial


def safe_float(val, default=np.nan):
    """Coerce a value to float, never raising, for safe display formatting."""
    try:
        f = float(val)
        if np.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def fmt(val, spec=".4f", placeholder="N/A"):
    """Format a possibly-missing/non-numeric value without ever crashing the UI."""
    f = safe_float(val)
    if np.isnan(f):
        return placeholder
    return format(f, spec)


def resolve_interval_columns(df: pd.DataFrame, default_fs: float = 1000.0):
    """
    Scans for possible column names indicating start/end times or sample indices
    and converts them to seconds (Start_s and End_s).

    Returns (df, resolved) where `resolved` is False if no usable start/end
    columns could be found (Start_s/End_s were filled with 0.0 as a
    fallback) so callers can warn the user instead of silently drawing
    zero-length/zero-position regions.
    """
    if df.empty:
        return df, True

    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    start_candidates = [
        "start_s", "start_time", "start_sec", "start_secs", "start",
        "start_time_s", "spindle_start", "onset_s", "onset_time",
        "start_sample", "start_idx", "start_sample_idx", "start_pts"
    ]
    end_candidates = [
        "end_s", "end_time", "end_sec", "end_secs", "end",
        "end_time_s", "spindle_end", "offset_s", "offset_time",
        "end_sample", "end_idx", "end_sample_idx", "end_pts"
    ]

    def find_column(candidates):
        # Pass 1: exact header match (case-insensitive), in priority order.
        for cand in candidates:
            if cand in cols_lower:
                return cols_lower[cand]
        # Pass 2: substring match, still in priority order, so a specific
        # candidate like "start_time" is preferred over a generic one like
        # "start" even when both would technically match. This lets headers
        # like "spindle_start_time_s" resolve correctly without accidentally
        # grabbing an unrelated column such as "nrem_bout_start_index" that
        # merely happens to contain "start".
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

        if (
            median_dur > 20.0
            or "sample" in found_start.lower()
            or "idx" in found_start.lower()
            or "index" in found_start.lower()
        ):
            df["Start_s"] = starts / default_fs
            df["End_s"] = ends / default_fs
        else:
            df["Start_s"] = starts
            df["End_s"] = ends
        return df, True

    # Fallback: nothing usable found.
    if "start_s" not in df.columns:
        df["Start_s"] = 0.0
    if "end_s" not in df.columns:
        df["End_s"] = 0.0
    return df, False


class SignalLoader(QtCore.QThread):
    """Loads a .mat trace and band-pass filters it off the GUI thread so large
    recordings don't freeze the interface while filtfilt runs."""

    finished_ok = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, data_path, parent=None):
        super().__init__(parent)
        self.data_path = data_path

    def run(self):
        try:
            mat_data = loadmat(self.data_path)
            if "data" not in mat_data:
                raise KeyError(
                    f"'.mat' file has no 'data' variable (found: "
                    f"{[k for k in mat_data.keys() if not k.startswith('__')]})"
                )
            raw = mat_data["data"].squeeze().astype(np.float32)
            if raw.ndim != 1:
                raw = raw.reshape(-1)
            filtered = butter_bandpass_filter(raw, BP_LOW, BP_HIGH, FS).astype(np.float32)
            self.finished_ok.emit(raw, filtered)
        except Exception as e:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(f"{type(e).__name__}: {e}")


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

        self._loader_thread = None
        self._ar_profile_thread = None
        self._pending_task = None  # manifest row awaiting async load completion

        self.init_ui()
        self.load_data()

    # ------------------------------------------------------------------ UI ---
    def init_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QtWidgets.QVBoxLayout(main_widget)

        # Dataset Filtering
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

        # Navigation Bar
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
        lbl_ar_legend = QtWidgets.QLabel("■ AR Spindle")
        lbl_ar_legend.setStyleSheet("color: #ff3333; font-weight: bold;")
        nav_layout.addWidget(lbl_ar_legend)

        lbl_wav_legend = QtWidgets.QLabel("■ Wavelet Spindle")
        lbl_wav_legend.setStyleSheet("color: #00e5ff; font-weight: bold;")
        nav_layout.addWidget(lbl_wav_legend)

        lbl_rprofile_legend = QtWidgets.QLabel("— R profile (per wavelet event)")
        lbl_rprofile_legend.setStyleSheet("color: #26c6da; font-weight: bold;")
        nav_layout.addWidget(lbl_rprofile_legend)

        lbl_maxr_legend = QtWidgets.QLabel("● Max R (per AR event)")
        lbl_maxr_legend.setStyleSheet("color: #ff5722; font-weight: bold;")
        nav_layout.addWidget(lbl_maxr_legend)

        lbl_ar_profile_legend = QtWidgets.QLabel("— AR R profile (live-computed)")
        lbl_ar_profile_legend.setStyleSheet("color: #ffab40; font-weight: bold;")
        nav_layout.addWidget(lbl_ar_profile_legend)

        nav_layout.addStretch(1)
        nav_box.setLayout(nav_layout)
        root_layout.addWidget(nav_box)

        # Viewports
        pg.setConfigOptions(antialias=False)
        self.graphics_layout = pg.GraphicsLayoutWidget()
        root_layout.addWidget(self.graphics_layout, stretch=1)

        self.p_raw = self.graphics_layout.addPlot(row=0, col=0)
        self.p_raw.showGrid(x=True, y=True, alpha=0.3)
        self.p_raw.setLabel("left", "Raw LFP", units="uV")
        self.curve_raw = self.p_raw.plot(pen=pg.mkPen(color="#dcdcdc", width=1))

        self.p_filt = self.graphics_layout.addPlot(row=1, col=0)
        self.p_filt.showGrid(x=True, y=True, alpha=0.3)
        self.p_filt.setLabel("left", "10-15 Hz", units="uV")
        self.curve_filt = self.p_filt.plot(pen=pg.mkPen(color="#4db6ac", width=1.2))

        self.p_r = self.graphics_layout.addPlot(row=2, col=0)
        self.p_r.showGrid(x=True, y=True, alpha=0.3)
        self.p_r.setLabel("left", "R value")
        self.p_r.setLabel("bottom", "Time", units="s")
        # Continuous R trend traced across each wavelet event's own duration,
        # from its r_profile_0%..100% columns. This is real per-event data
        # from the wavelet CSV, not an estimate.
        self.curve_r_profile = self.p_r.plot(
            pen=pg.mkPen(color="#26c6da", width=2),
            connect="finite",
        )
        # One summary point (Max_R) per AR-detected event, straight from the
        # detections CSV (whatever value actually triggered that detection).
        self.curve_r = self.p_r.plot(
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush="#ff5722",
            symbolPen=pg.mkPen(color="w", width=0.5),
        )
        # Live-computed R(t) trend across each AR event's duration, built by
        # calling the real AR-fitting pipeline (analyze_spindle_r_dynamics /
        # _fit_window) on the raw trace - not read from any CSV, since the AR
        # detections CSV never stored a within-event profile to begin with.
        self.curve_ar_r_profile = self.p_r.plot(
            pen=pg.mkPen(color="#ffab40", width=2),
            connect="finite",
        )

        self.p_filt.setXLink(self.p_raw)
        self.p_r.setXLink(self.p_raw)
        # Plots that each region-span triple maps onto, in the same order
        # draw_spans/clear_plots create the triples in. Keeping this explicit
        # avoids relying on item.getViewBox(), which can raise once an item
        # has already been detached from its view.
        self._span_plots = (self.p_raw, self.p_filt, self.p_r)
        self.region_items = []

        # Info Box
        self.details_panel = QtWidgets.QTextEdit()
        self.details_panel.setReadOnly(True)
        self.details_panel.setMaximumHeight(90)
        self.details_panel.setStyleSheet(
            "background-color: #1e1e1e; color: #00e676; font-family: Monospace; font-size: 11px;"
        )
        root_layout.addWidget(self.details_panel)

        self.status_bar = self.statusBar()

        # Shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, self.on_prev_ar_event)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, self.on_next_ar_event)
        QtWidgets.QShortcut(QtGui.QKeySequence("W"), self, self.on_next_wav_event)
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, self.on_prev_wav_event)

    # -------------------------------------------------------------- Loading ---
    def load_data(self):
        if not os.path.exists(self.manifest_path):
            self.details_panel.setText(
                f"Manifest not found: {self.manifest_path}\n"
                f"Pass --manifest, or set SPINDLE_MANIFEST, to point at your tasks CSV."
            )
            return

        try:
            self.manifest_df = pd.read_csv(self.manifest_path)
        except Exception as e:
            self.details_panel.setText(f"Failed to read manifest CSV: {e}")
            return

        missing = REQUIRED_MANIFEST_COLS - set(c.lower() for c in self.manifest_df.columns)
        if missing:
            self.details_panel.setText(
                f"Manifest is missing required column(s): {sorted(missing)}. "
                f"Expected at least: {sorted(REQUIRED_MANIFEST_COLS)}"
            )
            self.manifest_df = None
            return

        rat_numeric = pd.to_numeric(self.manifest_df["rat"], errors="coerce")
        bad_rats = int(rat_numeric.isna().sum())
        if bad_rats:
            self.status_bar.showMessage(
                f"Warning: dropped {bad_rats} manifest row(s) with a non-numeric 'rat' value.", 8000
            )
        self.manifest_df = self.manifest_df.loc[rat_numeric.notna()].copy()
        self.manifest_df["Rat"] = rat_numeric.loc[rat_numeric.notna()].astype(int)
        self.manifest_df["Region"] = self.manifest_df["region"].astype(str).str.strip()
        self.manifest_df["Date"] = self.manifest_df["date"].apply(clean_str)

        parsed = self.manifest_df["data_path"].apply(parse_channel_trial)
        self.manifest_df["channel"] = parsed.apply(lambda x: x[0])
        self.manifest_df["trial"] = parsed.apply(lambda x: clean_str(x[1]))
        self.manifest_df["File"] = self.manifest_df["data_path"].apply(lambda p: Path(str(p)).name)

        if self.manifest_df.empty:
            self.details_panel.setText("Manifest loaded but contains no usable rows.")
            return

        # AR Loading
        self.ar_df = self._load_detection_csv(self.ar_csv_path, kind="AR")

        # Wavelet Loading
        self.wavelet_df = self._load_detection_csv(self.wavelet_csv_path, kind="Wavelet", is_wavelet=True)

        self.populate_rat_selector()

    def _load_detection_csv(self, path, kind, is_wavelet=False):
        if not path or not os.path.exists(path):
            self.status_bar.showMessage(f"{kind} detections CSV not found ({path}); continuing without it.", 8000)
            return pd.DataFrame()

        try:
            df = pd.read_csv(path)
        except Exception as e:
            self.status_bar.showMessage(f"Failed to read {kind} CSV ({path}): {e}", 8000)
            return pd.DataFrame()

        if is_wavelet:
            rename_map = {}
            for col in df.columns:
                clow = col.lower().strip()
                if clow in ["rat", "rat_number", "rat_id"]:
                    rename_map[col] = "Rat"
                elif clow in ["region", "brain_region", "area"]:
                    rename_map[col] = "Region"
                elif clow in ["date", "session_date", "day"]:
                    rename_map[col] = "Date"
                elif clow in ["channel", "chan", "channel_id"]:
                    rename_map[col] = "channel"
                elif clow in ["trial", "trial_id", "recording"]:
                    rename_map[col] = "trial"
            df = df.rename(columns=rename_map)

            if "Rat" in df.columns:
                df["Rat"] = pd.to_numeric(df["Rat"], errors="coerce").fillna(-1).astype(int)
            if "Region" in df.columns:
                df["Region"] = df["Region"].astype(str).str.strip()
            if "Date" in df.columns:
                df["Date"] = df["Date"].apply(clean_str)
            df["channel"] = df["channel"].apply(normalize_channel) if "channel" in df.columns else ""
            df["trial"] = df["trial"].apply(clean_str) if "trial" in df.columns else ""
        else:
            if "Rat" in df.columns:
                df["Rat"] = pd.to_numeric(df["Rat"], errors="coerce").fillna(-1).astype(int)
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
                f"events from it will not be positioned correctly.", 10000
            )
        return df

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
            lbl = f"{row['Date']} | ch:{row['channel']} tr:{row['trial']} -> {row['File']}"
            self.combo_files.addItem(lbl, userData=orig_idx)
        self.combo_files.blockSignals(False)

        if self.combo_files.count() > 0:
            self.on_file_selected(0)
        else:
            self.clear_plots()

    def clear_plots(self):
        self.curve_raw.clear()
        self.curve_filt.clear()
        self.curve_r.clear()
        self.curve_r_profile.clear()
        self.curve_ar_r_profile.clear()
        self._remove_span_items()
        self.current_ar_events = pd.DataFrame()
        self.current_wavelet_events = pd.DataFrame()
        self.lbl_ar_tracker.setText("AR: 0 / 0")
        self.lbl_wav_tracker.setText("WAV: 0 / 0")
        self.details_panel.setText("No recordings match current Rat and Region filters.")

    def _remove_span_items(self):
        for triple in self.region_items:
            for plot, item in zip(self._span_plots, triple):
                try:
                    plot.removeItem(item)
                except Exception:
                    pass
        self.region_items.clear()

    # ---------------------------------------------------------- File select ---
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

        # Kick off async load/filter so the UI doesn't freeze on large files.
        self._pending_task = task
        self.combo_files.setEnabled(False)
        self.status_bar.showMessage(f"Loading {Path(data_path).name} ...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait()

        self._loader_thread = SignalLoader(data_path)
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
        self.time_vector = np.arange(self.data_len, dtype=np.float32) / FS

        file_name = task["File"]

        # Filter AR detections
        if not self.ar_df.empty:
            if "File" in self.ar_df.columns and (self.ar_df["File"] == file_name).any():
                ar_sub = self.ar_df[self.ar_df["File"] == file_name]
            else:
                ar_sub = self.ar_df[
                    (self.ar_df["Rat"] == task["Rat"])
                    & (self.ar_df["Region"] == task["Region"])
                    & (self.ar_df["Date"] == task["Date"])
                ]
            self.current_ar_events = (
                ar_sub.sort_values("Start_s").reset_index(drop=True) if not ar_sub.empty else pd.DataFrame()
            )
        else:
            self.current_ar_events = pd.DataFrame()

        # Filter Wavelet detections with graceful fallbacks
        if not self.wavelet_df.empty:
            w_df = self.wavelet_df
            wav_sub = w_df[
                (w_df["Rat"] == task["Rat"])
                & (w_df["Region"] == task["Region"])
                & (w_df["Date"] == task["Date"])
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
                wav_sub.sort_values("Start_s").reset_index(drop=True) if not wav_sub.empty else pd.DataFrame()
            )
        else:
            self.current_wavelet_events = pd.DataFrame()

        self.curve_raw.setData(self.time_vector, self.raw_signal)
        self.curve_filt.setData(self.time_vector, self.filtered_signal)

        if not self.current_ar_events.empty and "Max_R" in self.current_ar_events.columns:
            x_pts = (
                self.current_ar_events["Peak_s"].values
                if "Peak_s" in self.current_ar_events.columns
                else self.current_ar_events["Start_s"].values
            )
            y_pts = pd.to_numeric(self.current_ar_events["Max_R"], errors="coerce").values
            self.curve_r.setData(x_pts, y_pts)
        else:
            self.curve_r.clear()

        x_prof, y_prof = build_profile_curve(self.current_wavelet_events)
        if len(x_prof):
            self.curve_r_profile.setData(x_prof, y_prof)
        else:
            self.curve_r_profile.clear()

        self.curve_ar_r_profile.clear()
        self._start_ar_profile_computation()

        self.draw_spans()

        self.current_ar_idx = 0 if len(self.current_ar_events) > 0 else -1
        self.current_wav_idx = 0 if len(self.current_wavelet_events) > 0 else -1

        self.lbl_ar_tracker.setText(
            f"AR: {max(0, self.current_ar_idx + 1)} / {len(self.current_ar_events)}"
        )
        self.lbl_wav_tracker.setText(
            f"WAV: {max(0, self.current_wav_idx + 1)} / {len(self.current_wavelet_events)}"
        )

        if self.current_ar_idx >= 0:
            self.jump_to_ar_event()
        elif self.current_wav_idx >= 0:
            self.jump_to_wav_event()
        else:
            self.p_raw.setXRange(0, min(VIEW_WINDOW_SEC, self.data_len / FS), padding=0)
            self.details_panel.setText(
                f"File: {file_name} | Length: {self.data_len/FS:.1f}s | "
                f"No spindle detections found for this recording."
            )

    # ------------------------------------------------------ Live AR R profile ---
    def _start_ar_profile_computation(self):
        if not AR_LIVE_ANALYSIS_AVAILABLE:
            if _AR_LIVE_IMPORT_ERROR:
                self.status_bar.showMessage(
                    f"AR R-profile modules not importable ({_AR_LIVE_IMPORT_ERROR}); "
                    f"showing Max_R points only for AR events. Set AR_MODULES_ROOT "
                    f"if your repo layout differs.",
                    12000,
                )
            return

        if self.current_ar_events.empty or self.raw_signal is None:
            return

        events = []
        for idx, row in self.current_ar_events.iterrows():
            s, e = safe_float(row.get("Start_s")), safe_float(row.get("End_s"))
            if not (np.isnan(s) or np.isnan(e)) and e > s:
                events.append((idx, s, e))
        if not events:
            return

        if self._ar_profile_thread is not None and self._ar_profile_thread.isRunning():
            self._ar_profile_thread.quit()
            self._ar_profile_thread.wait()

        self.status_bar.showMessage(f"Computing live AR R profiles for {len(events)} event(s) ...")
        self._ar_profile_thread = ARProfileWorker(self.raw_signal, events)
        self._ar_profile_thread.finished_ok.connect(self._on_ar_profiles_ready)
        self._ar_profile_thread.failed.connect(self._on_ar_profiles_failed)
        self._ar_profile_thread.start()

    def _on_ar_profiles_failed(self, message):
        self.status_bar.showMessage(f"AR R-profile computation failed: {message}", 12000)

    def _on_ar_profiles_ready(self, results):
        self.status_bar.clearMessage()
        for idx, metrics in results.items():
            if idx not in self.current_ar_events.index:
                continue
            self.current_ar_events.at[idx, "r_max"] = metrics["r_max"]
            self.current_ar_events.at[idx, "r_min"] = metrics["r_min"]
            self.current_ar_events.at[idx, "r_mean"] = metrics["r_mean"]
            self.current_ar_events.at[idx, "r_start"] = metrics["r_start"]
            self.current_ar_events.at[idx, "r_end"] = metrics["r_end"]
            self.current_ar_events.at[idx, "r_peak_freq_live"] = metrics["r_peak_freq"]
            self.current_ar_events.at[idx, "in_band_ratio"] = metrics["in_band_ratio"]
            self.current_ar_events.at[idx, "n_windows"] = metrics["n_windows"]
            for col, val in zip(PROFILE_COLS, metrics["r_profile"]):
                self.current_ar_events.at[idx, col] = val

        x_prof, y_prof = build_profile_curve(self.current_ar_events)
        if len(x_prof):
            self.curve_ar_r_profile.setData(x_prof, y_prof)

        # Refresh the focused event's panel now that live metrics exist.
        if self.current_ar_idx >= 0:
            self.jump_to_ar_event()



    # -------------------------------------------------------------- Drawing ---
    def draw_spans(self):
        self._remove_span_items()

        # 1. Overlay Wavelet detections (High visibility Cyan, zValue=5)
        if not self.current_wavelet_events.empty:
            for _, row in self.current_wavelet_events.iterrows():
                s, e = safe_float(row["Start_s"], 0.0), safe_float(row["End_s"], 0.0)
                triple = tuple(
                    pg.LinearRegionItem(
                        [s, e], movable=False,
                        brush=QtGui.QColor(0, 229, 255, 90),
                        pen=pg.mkPen(color=(0, 229, 255, 230), width=1.8, style=QtCore.Qt.DashLine),
                    )
                    for _ in range(3)
                )
                for r_item, plot in zip(triple, self._span_plots):
                    r_item.setZValue(5)
                    plot.addItem(r_item)
                self.region_items.append(triple)

        # 2. Overlay AR detections (Red/Orange, zValue=10)
        if not self.current_ar_events.empty:
            for _, row in self.current_ar_events.iterrows():
                s, e = safe_float(row["Start_s"], 0.0), safe_float(row["End_s"], 0.0)
                triple = tuple(
                    pg.LinearRegionItem(
                        [s, e], movable=False,
                        brush=QtGui.QColor(255, 50, 50, 80),
                        pen=pg.mkPen(color=(255, 50, 50, 230), width=1.5),
                    )
                    for _ in range(3)
                )
                for r_item, plot in zip(triple, self._span_plots):
                    r_item.setZValue(10)
                    plot.addItem(r_item)
                self.region_items.append(triple)

    # ----------------------------------------------------------- Navigation ---
    def on_prev_ar_event(self):
        if len(self.current_ar_events) == 0:
            return
        self.current_ar_idx = (self.current_ar_idx - 1) % len(self.current_ar_events)
        self.jump_to_ar_event()

    def on_next_ar_event(self):
        if len(self.current_ar_events) == 0:
            return
        self.current_ar_idx = (self.current_ar_idx + 1) % len(self.current_ar_events)
        self.jump_to_ar_event()

    def on_prev_wav_event(self):
        if len(self.current_wavelet_events) == 0:
            return
        self.current_wav_idx = (self.current_wav_idx - 1) % len(self.current_wavelet_events)
        self.jump_to_wav_event()

    def on_next_wav_event(self):
        if len(self.current_wavelet_events) == 0:
            return
        self.current_wav_idx = (self.current_wav_idx + 1) % len(self.current_wavelet_events)
        self.jump_to_wav_event()

    def _centered_view_range(self, start_s, end_s):
        """Returns a fixed VIEW_WINDOW_SEC-wide range centered on [start_s,
        end_s], clamped to the recording's bounds."""
        total_duration = self.data_len / FS
        center = (start_s + end_s) / 2.0
        half = VIEW_WINDOW_SEC / 2.0
        view_start = center - half
        view_end = center + half
        if view_start < 0.0:
            view_start = 0.0
            view_end = min(VIEW_WINDOW_SEC, total_duration)
        if view_end > total_duration:
            view_end = total_duration
            view_start = max(0.0, view_end - VIEW_WINDOW_SEC)
        return view_start, view_end

    def jump_to_ar_event(self):
        if self.current_ar_idx < 0 or self.current_ar_idx >= len(self.current_ar_events):
            return

        event = self.current_ar_events.iloc[self.current_ar_idx]
        start_s = safe_float(event["Start_s"], 0.0)
        end_s = safe_float(event["End_s"], 0.0)

        view_start, view_end = self._centered_view_range(start_s, end_s)

        self.p_raw.setXRange(view_start, view_end, padding=0)
        self.p_r.setYRange(0.0, 1.05, padding=0)

        overlapping = pd.DataFrame()
        if not self.current_wavelet_events.empty:
            w = self.current_wavelet_events
            overlapping = w[(w["Start_s"] <= end_s) & (w["End_s"] >= start_s)]

        align_desc = (
            f"ALIGNED: Overlaps with {len(overlapping)} wavelet detection(s)"
            if len(overlapping) > 0
            else "ISOLATED: No overlapping wavelet event"
        )

        r_val = event.get("Max_R", np.nan)
        freq_val = event.get("Peak_Freq_Hz", np.nan)
        dur = end_s - start_s

        lines = [
            f"[Focus: AR Spindle #{self.current_ar_idx + 1}/{len(self.current_ar_events)}] "
            f"Interval: [{start_s:.3f}s - {end_s:.3f}s] (Duration: {dur:.3f}s)",
            f"CSV Detection: Max R = {fmt(r_val)} | Peak Freq = {fmt(freq_val, '.2f')} Hz",
        ]

        live_max = event.get("r_max", np.nan)
        if not np.isnan(safe_float(live_max)):
            lines.append(
                f"Live AR fit: R max={fmt(live_max)} min={fmt(event.get('r_min', np.nan))} "
                f"mean={fmt(event.get('r_mean', np.nan))} | "
                f"in_band_ratio={fmt(event.get('in_band_ratio', np.nan), '.2f')} "
                f"({int(safe_float(event.get('n_windows', 0), 0))} windows)"
            )
        elif AR_LIVE_ANALYSIS_AVAILABLE:
            lines.append("Live AR fit: computing...")

        lines.append(f"Alignment Status: {align_desc}")
        self.details_panel.setText("\n".join(lines))
        self.lbl_ar_tracker.setText(f"AR: {self.current_ar_idx + 1} / {len(self.current_ar_events)}")

    def jump_to_wav_event(self):
        if self.current_wav_idx < 0 or self.current_wav_idx >= len(self.current_wavelet_events):
            return

        event = self.current_wavelet_events.iloc[self.current_wav_idx]
        start_s = safe_float(event["Start_s"], 0.0)
        end_s = safe_float(event["End_s"], 0.0)

        view_start, view_end = self._centered_view_range(start_s, end_s)

        self.p_raw.setXRange(view_start, view_end, padding=0)
        self.p_r.setYRange(0.0, 1.05, padding=0)

        overlapping = pd.DataFrame()
        if not self.current_ar_events.empty:
            a = self.current_ar_events
            overlapping = a[(a["Start_s"] <= end_s) & (a["End_s"] >= start_s)]

        align_desc = (
            f"ALIGNED: Overlaps with {len(overlapping)} AR detection(s)"
            if len(overlapping) > 0
            else "ISOLATED: No overlapping AR event"
        )

        dur = end_s - start_s
        info = (
            f"[Focus: Wavelet Spindle #{self.current_wav_idx + 1}/{len(self.current_wavelet_events)}] "
            f"Interval: [{start_s:.3f}s - {end_s:.3f}s] (Duration: {dur:.3f}s)\n"
            f"Alignment Status: {align_desc}"
        )
        self.details_panel.setText(info)
        self.lbl_wav_tracker.setText(f"WAV: {self.current_wav_idx + 1} / {len(self.current_wavelet_events)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Spindle Alignment Inspector (AR vs Wavelet)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Path to tasks_manifest.csv")
    parser.add_argument("--ar-csv", default=DEFAULT_AR_CSV, help="Path to AR-detected spindles CSV")
    parser.add_argument("--wavelet-csv", default=DEFAULT_WAVELET_CSV, help="Path to wavelet-detected spindles CSV")
    # Ignore unknown args so this still works fine under Qt's own arg parsing.
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

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
        # Keep unexpected exceptions from silently killing the Qt event loop.
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = excepthook

    viewer = SpindleViewer(args.manifest, args.ar_csv, args.wavelet_csv)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
