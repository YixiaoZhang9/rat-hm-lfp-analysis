#!/usr/bin/env python3
"""
Spindle Alignment Inspector (AR vs Wavelet)

Lets you page through detected spindle events from two detectors overlaid on
the raw LFP, band-passed LFP, and a continuous AR R-value trace dynamically
computed for the active view window.
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
# Import AR analysis modules
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
    from modules.find_spindles_lfp_o_quality import fit_ar_on_prepared_signal

    AR_LIVE_ANALYSIS_AVAILABLE = True
except Exception as _e:
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
VIEW_WINDOW_SEC = 10.0

AR_TARGET_FS = 128
AR_ORDER = 8
AR_SPINDLE_BAND = (BP_LOW, BP_HIGH)
AR_WINDOW_SEC = 1.0
AR_STRIDE_SAMPLES = 4

REQUIRED_MANIFEST_COLS = {"rat", "region", "date", "data_path"}


class ARWindowWorker(QtCore.QThread):
    """Computes the continuous R-value trace only for the given time slice."""
    finished_ok = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, raw_signal, start_s, end_s, parent=None):
        super().__init__(parent)
        self.raw_signal = raw_signal
        self.start_s = start_s
        self.end_s = end_s

    def run(self):
        try:
            # Add padding to ensure edge windows calculate correctly
            pad = AR_WINDOW_SEC
            s_idx = max(0, int((self.start_s - pad) * FS))
            e_idx = min(len(self.raw_signal), int((self.end_s + pad) * FS))
            chunk = self.raw_signal[s_idx:e_idx]

            if len(chunk) == 0:
                self.finished_ok.emit(np.array([]), np.array([]))
                return

            filtered = ar_bandpass_filter(chunk, lowcut=0.1, highcut=100, fs=FS)
            signal_128 = ar_downsampling(filtered, FS, AR_TARGET_FS)

            # Force n_jobs=1: multiprocessing is slower for tiny array chunks
            # due to serialization overhead.
            r_vals, _, starts = fit_ar_on_prepared_signal(
                signal_128,
                target_fs=AR_TARGET_FS,
                ar_order=AR_ORDER,
                window_sec=AR_WINDOW_SEC,
                stride_samples=AR_STRIDE_SAMPLES,
                spindle_band=AR_SPINDLE_BAND,
                n_jobs=1,
                verbose=False
            )

            chunk_start_t = s_idx / FS
            half_win = int(AR_WINDOW_SEC * AR_TARGET_FS) / 2.0
            t_vals = chunk_start_t + (starts + half_win) / AR_TARGET_FS

            self.finished_ok.emit(t_vals, r_vals)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


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
    try:
        f = float(val)
        if np.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def fmt(val, spec=".4f", placeholder="N/A"):
    f = safe_float(val)
    if np.isnan(f):
        return placeholder
    return format(f, spec)


def resolve_interval_columns(df: pd.DataFrame, default_fs: float = 1000.0):
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

        if (median_dur > 20.0 or "sample" in found_start.lower() or "idx" in found_start.lower() or "index" in found_start.lower()):
            df["Start_s"] = starts / default_fs
            df["End_s"] = ends / default_fs
        else:
            df["Start_s"] = starts
            df["End_s"] = ends
        return df, True

    if "start_s" not in df.columns:
        df["Start_s"] = 0.0
    if "end_s" not in df.columns:
        df["End_s"] = 0.0
    return df, False


class SignalLoader(QtCore.QThread):
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
        except Exception as e:
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
        self._ar_signal_thread = None
        self._pending_task = None

        # Debouncer for window panning
        self.view_update_timer = QtCore.QTimer()
        self.view_update_timer.setSingleShot(True)
        self.view_update_timer.timeout.connect(self._start_window_computation)

        # Spinner animation
        self.spinner_timer = QtCore.QTimer()
        self.spinner_timer.timeout.connect(self._update_spinner)
        self.spinner_frames = ["|", "/", "-", "\\"]
        self.spinner_idx = 0

        self.init_ui()
        self.load_data()

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
        lbl_ar_legend = QtWidgets.QLabel("■ AR Spindle Event")
        lbl_ar_legend.setStyleSheet("color: #ff3333; font-weight: bold;")
        nav_layout.addWidget(lbl_ar_legend)

        lbl_wav_legend = QtWidgets.QLabel("■ Wavelet Spindle Event")
        lbl_wav_legend.setStyleSheet("color: #00e5ff; font-weight: bold;")
        nav_layout.addWidget(lbl_wav_legend)

        lbl_continuous_legend = QtWidgets.QLabel("— Live AR R-Value")
        lbl_continuous_legend.setStyleSheet("color: #ffab40; font-weight: bold;")
        nav_layout.addWidget(lbl_continuous_legend)

        nav_layout.addStretch(1)
        nav_box.setLayout(nav_layout)
        root_layout.addWidget(nav_box)

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

        self.curve_r_continuous = self.p_r.plot(
            pen=pg.mkPen(color="#ffab40", width=2),
            connect="finite",
        )

        # Loading Spinner Text
        self.loading_text = pg.TextItem("", color=(255, 171, 64), anchor=(0.5, 0.5))
        font = QtGui.QFont()
        font.setBold(True)
        self.loading_text.setFont(font)
        self.p_r.addItem(self.loading_text)
        self.loading_text.hide()

        self.p_filt.setXLink(self.p_raw)
        self.p_r.setXLink(self.p_raw)
        self._span_plots = (self.p_raw, self.p_filt, self.p_r)
        self.region_items = []

        self.details_panel = QtWidgets.QTextEdit()
        self.details_panel.setReadOnly(True)
        self.details_panel.setMaximumHeight(70)
        self.details_panel.setStyleSheet(
            "background-color: #1e1e1e; color: #00e676; font-family: Monospace; font-size: 12px;"
        )
        root_layout.addWidget(self.details_panel)

        self.status_bar = self.statusBar()

        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, self.on_prev_ar_event)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, self.on_next_ar_event)
        QtWidgets.QShortcut(QtGui.QKeySequence("W"), self, self.on_next_wav_event)
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, self.on_prev_wav_event)

        # Connect view change to debounce timer
        self.p_raw.sigXRangeChanged.connect(self._on_xrange_changed)

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

        self.ar_df = self._load_detection_csv(self.ar_csv_path, kind="AR")
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
        self.curve_r_continuous.clear()
        self._remove_span_items()
        self.current_ar_events = pd.DataFrame()
        self.current_wavelet_events = pd.DataFrame()
        self.lbl_ar_tracker.setText("AR: 0 / 0")
        self.lbl_wav_tracker.setText("WAV: 0 / 0")
        self.details_panel.setText("No recordings match current Rat and Region filters.")

        if self._ar_signal_thread is not None and self._ar_signal_thread.isRunning():
            self._ar_signal_thread.quit()
            self._ar_signal_thread.wait()

    def _remove_span_items(self):
        for triple in self.region_items:
            for plot, item in zip(self._span_plots, triple):
                try:
                    plot.removeItem(item)
                except Exception:
                    pass
        self.region_items.clear()

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
        self.curve_r_continuous.clear()

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

    def _on_xrange_changed(self, _, range_tuple):
        """Fires repeatedly during panning. Debounce to prevent computation spam."""
        if self.raw_signal is not None:
            self.view_update_timer.start(300)

    def _update_spinner(self):
        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)
        frame = self.spinner_frames[self.spinner_idx]
        self.loading_text.setText(f"Calculating R-values {frame}")

        # Center the text dynamically
        view_range = self.p_r.viewRange()
        cx = (view_range[0][0] + view_range[0][1]) / 2.0
        cy = 0.5
        self.loading_text.setPos(cx, cy)

    def _start_window_computation(self):
        if not AR_LIVE_ANALYSIS_AVAILABLE or self.raw_signal is None:
            if _AR_LIVE_IMPORT_ERROR:
                self.status_bar.showMessage(f"AR R-profile modules not importable: {_AR_LIVE_IMPORT_ERROR}", 12000)
            return

        if self._ar_signal_thread is not None and self._ar_signal_thread.isRunning():
            self._ar_signal_thread.quit()
            self._ar_signal_thread.wait()

        # Extract current view range
        view_range = self.p_raw.viewRange()[0]
        start_s, end_s = view_range[0], view_range[1]

        # Launch spinner
        self.loading_text.show()
        self.spinner_timer.start(100)
        self.curve_r_continuous.clear() # Clear out of bounds traces

        self._ar_signal_thread = ARWindowWorker(self.raw_signal, start_s, end_s)
        self._ar_signal_thread.finished_ok.connect(self._on_window_computation_ready)
        self._ar_signal_thread.failed.connect(self._on_window_computation_failed)
        self._ar_signal_thread.start()

    def _on_window_computation_ready(self, t_vals, r_vals):
        self.spinner_timer.stop()
        self.loading_text.hide()
        self.curve_r_continuous.setData(t_vals, r_vals)

    def _on_window_computation_failed(self, msg):
        self.spinner_timer.stop()
        self.loading_text.hide()
        self.status_bar.showMessage(f"AR computation failed: {msg}", 12000)

    def draw_spans(self):
        self._remove_span_items()

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
            f"Alignment Status: {align_desc}"
        ]

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
        lines = [
            f"[Focus: Wavelet Spindle #{self.current_wav_idx + 1}/{len(self.current_wavelet_events)}] "
            f"Interval: [{start_s:.3f}s - {end_s:.3f}s] (Duration: {dur:.3f}s)",
            f"Alignment Status: {align_desc}"
        ]

        self.details_panel.setText("\n".join(lines))
        self.lbl_wav_tracker.setText(f"WAV: {self.current_wav_idx + 1} / {len(self.current_wavelet_events)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Spindle Alignment Inspector (AR vs Wavelet)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Path to tasks_manifest.csv")
    parser.add_argument("--ar-csv", default=DEFAULT_AR_CSV, help="Path to AR-detected spindles CSV")
    parser.add_argument("--wavelet-csv", default=DEFAULT_WAVELET_CSV, help="Path to wavelet-detected spindles CSV")
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
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = excepthook

    viewer = SpindleViewer(args.manifest, args.ar_csv, args.wavelet_csv)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
