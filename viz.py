import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

# --------------------------------------------------------------------------- #
# Paths & Default Configuration
# --------------------------------------------------------------------------- #
DEFAULT_MANIFEST = "/home/mdadmin/Desktop/amirali/rat-hm-lfp-analysis/tasks_manifest.csv"
DEFAULT_AR_CSV = "results/all_detected_spindles_per_region.csv"
DEFAULT_WAVELET_CSV = (
    "/home/mdadmin/Desktop/amirali/rat-hm-lfp-analysis/results_ar_calibration/"
    "wavelet_spindles_with_ar_dynamics.csv"
)

FS = 1000.0
BP_LOW = 10.0
BP_HIGH = 15.0
VIEW_PADDING_SEC = 2.5


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


def resolve_interval_columns(df: pd.DataFrame, default_fs: float = 1000.0) -> pd.DataFrame:
    """
    Scans for possible column names indicating start/end times or sample indices
    and converts them to seconds (Start_s and End_s).
    """
    if df.empty:
        return df

    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    # Priority-ordered possible headers
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

    found_start = None
    found_end = None

    for cand in start_candidates:
        if cand in cols_lower:
            found_start = cols_lower[cand]
            break

    for cand in end_candidates:
        if cand in cols_lower:
            found_end = cols_lower[cand]
            break

    if found_start and found_end:
        starts = pd.to_numeric(df[found_start], errors="coerce").fillna(0.0).values
        ends = pd.to_numeric(df[found_end], errors="coerce").fillna(0.0).values

        # Detect if coordinates are raw sample indices or seconds
        # Typical recording durations in seconds are rarely > 86400, while sample indices
        # will quickly exceed thousands.
        durations = ends - starts
        median_dur = np.nanmedian(durations) if len(durations) > 0 else 0

        if median_dur > 20.0 or ("sample" in found_start.lower()) or ("idx" in found_start.lower()):
            df["Start_s"] = starts / default_fs
            df["End_s"] = ends / default_fs
        else:
            df["Start_s"] = starts
            df["End_s"] = ends
    else:
        # Fallback: check if duration and peak/start are available
        if "start_s" not in df.columns:
            df["Start_s"] = 0.0
        if "end_s" not in df.columns:
            df["End_s"] = 0.0

    return df


class SpindleViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spindle Alignment Inspector (AR vs Wavelet)")
        self.resize(1500, 950)

        self.manifest_df = None
        self.ar_df = None
        self.wavelet_df = None

        self.current_ar_events = pd.DataFrame()
        self.current_wavelet_events = pd.DataFrame()
        self.current_ar_idx = -1
        self.current_wav_idx = -1

        self.raw_signal = None
        self.filtered_signal = None
        self.time_vector = None
        self.data_len = 0

        self.init_ui()
        self.load_data()

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
        self.p_r.setLabel("left", "Max R")
        self.p_r.setLabel("bottom", "Time", units="s")
        self.curve_r = self.p_r.plot(
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush="#ff5722",
            symbolPen=pg.mkPen(color="w", width=0.5),
        )

        self.p_filt.setXLink(self.p_raw)
        self.p_r.setXLink(self.p_raw)
        self.region_items = []

        # Info Box
        self.details_panel = QtWidgets.QTextEdit()
        self.details_panel.setReadOnly(True)
        self.details_panel.setMaximumHeight(90)
        self.details_panel.setStyleSheet(
            "background-color: #1e1e1e; color: #00e676; font-family: Monospace; font-size: 11px;"
        )
        root_layout.addWidget(self.details_panel)

        # Shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, self.on_prev_ar_event)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, self.on_next_ar_event)
        QtWidgets.QShortcut(QtGui.QKeySequence("W"), self, self.on_next_wav_event)
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, self.on_prev_wav_event)

    def load_data(self):
        if not os.path.exists(DEFAULT_MANIFEST):
            self.details_panel.setText(f"Manifest not found: {DEFAULT_MANIFEST}")
            return

        self.manifest_df = pd.read_csv(DEFAULT_MANIFEST)
        self.manifest_df["Rat"] = self.manifest_df["rat"].astype(int)
        self.manifest_df["Region"] = self.manifest_df["region"].astype(str).str.strip()
        self.manifest_df["Date"] = self.manifest_df["date"].apply(clean_str)

        parsed = self.manifest_df["data_path"].apply(parse_channel_trial)
        self.manifest_df["channel"] = parsed.apply(lambda x: x[0])
        self.manifest_df["trial"] = parsed.apply(lambda x: clean_str(x[1]))
        self.manifest_df["File"] = self.manifest_df["data_path"].apply(lambda p: Path(p).name)

        # AR Loading
        if os.path.exists(DEFAULT_AR_CSV):
            self.ar_df = pd.read_csv(DEFAULT_AR_CSV)
            self.ar_df["Rat"] = self.ar_df["Rat"].astype(int)
            self.ar_df["Region"] = self.ar_df["Region"].astype(str).str.strip()
            self.ar_df["Date"] = self.ar_df["Date"].apply(clean_str)
            if "File" in self.ar_df.columns:
                self.ar_df["File"] = self.ar_df["File"].apply(lambda p: Path(str(p)).name)
            self.ar_df = resolve_interval_columns(self.ar_df, FS)
        else:
            self.ar_df = pd.DataFrame()

        # Wavelet Loading
        if os.path.exists(DEFAULT_WAVELET_CSV):
            self.wavelet_df = pd.read_csv(DEFAULT_WAVELET_CSV)

            # Normalize headers
            rename_map = {}
            for col in self.wavelet_df.columns:
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

            self.wavelet_df = self.wavelet_df.rename(columns=rename_map)

            if "Rat" in self.wavelet_df.columns:
                self.wavelet_df["Rat"] = pd.to_numeric(self.wavelet_df["Rat"], errors="coerce").fillna(0).astype(int)
            if "Region" in self.wavelet_df.columns:
                self.wavelet_df["Region"] = self.wavelet_df["Region"].astype(str).str.strip()
            if "Date" in self.wavelet_df.columns:
                self.wavelet_df["Date"] = self.wavelet_df["Date"].apply(clean_str)
            if "channel" in self.wavelet_df.columns:
                self.wavelet_df["channel"] = self.wavelet_df["channel"].apply(normalize_channel)
            else:
                self.wavelet_df["channel"] = ""
            if "trial" in self.wavelet_df.columns:
                self.wavelet_df["trial"] = self.wavelet_df["trial"].apply(clean_str)
            else:
                self.wavelet_df["trial"] = ""

            self.wavelet_df = resolve_interval_columns(self.wavelet_df, FS)
        else:
            self.wavelet_df = pd.DataFrame()

        self.populate_rat_selector()

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
        for r in self.region_items:
            for item in r:
                item.getViewBox().removeItem(item)
        self.region_items.clear()
        self.current_ar_events = pd.DataFrame()
        self.current_wavelet_events = pd.DataFrame()
        self.lbl_ar_tracker.setText("AR: 0 / 0")
        self.lbl_wav_tracker.setText("WAV: 0 / 0")
        self.details_panel.setText("No recordings match current Rat and Region filters.")

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

        try:
            mat_data = loadmat(data_path)
            self.raw_signal = mat_data["data"].squeeze().astype(np.float32)
            self.data_len = len(self.raw_signal)
            self.time_vector = np.arange(self.data_len, dtype=np.float32) / FS
            self.filtered_signal = butter_bandpass_filter(
                self.raw_signal, BP_LOW, BP_HIGH, FS
            )
        except Exception as e:
            self.details_panel.setText(f"Signal processing error: {str(e)}")
            return

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

            if not ar_sub.empty and "Start_s" in ar_sub.columns:
                self.current_ar_events = ar_sub.sort_values("Start_s").reset_index(drop=True)
            else:
                self.current_ar_events = pd.DataFrame()
        else:
            self.current_ar_events = pd.DataFrame()

        # Filter Wavelet detections with graceful fallbacks
        if not self.wavelet_df.empty:
            w_df = self.wavelet_df

            # Tier 1: Rat + Region + Date
            wav_sub = w_df[
                (w_df["Rat"] == task["Rat"])
                & (w_df["Region"] == task["Region"])
                & (w_df["Date"] == task["Date"])
            ]

            # Tier 2: Restrict by Channel if matches exist
            if not wav_sub.empty and task["channel"]:
                chan_match = wav_sub[wav_sub["channel"] == task["channel"]]
                if not chan_match.empty:
                    wav_sub = chan_match

            # Tier 3: Restrict by Trial if present and non-empty
            if not wav_sub.empty and task["trial"]:
                trial_match = wav_sub[wav_sub["trial"] == task["trial"]]
                if not trial_match.empty:
                    wav_sub = trial_match

            if not wav_sub.empty and "Start_s" in wav_sub.columns:
                self.current_wavelet_events = wav_sub.sort_values("Start_s").reset_index(drop=True)
            else:
                self.current_wavelet_events = pd.DataFrame()
        else:
            self.current_wavelet_events = pd.DataFrame()

        self.curve_raw.setData(self.time_vector, self.raw_signal)
        self.curve_filt.setData(self.time_vector, self.filtered_signal)

        # Plot R-values
        if not self.current_ar_events.empty and "Max_R" in self.current_ar_events.columns:
            x_pts = (
                self.current_ar_events["Peak_s"].values
                if "Peak_s" in self.current_ar_events.columns
                else self.current_ar_events["Start_s"].values
            )
            y_pts = self.current_ar_events["Max_R"].values
            self.curve_r.setData(x_pts, y_pts)
        else:
            self.curve_r.clear()

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
            self.p_raw.setXRange(0, min(30.0, self.data_len / FS), padding=0)
            self.details_panel.setText(
                f"File: {file_name} | Length: {self.data_len/FS:.1f}s | "
                f"No spindle detections found for this recording."
            )

    def draw_spans(self):
        for r in self.region_items:
            self.p_raw.removeItem(r[0])
            self.p_filt.removeItem(r[1])
            self.p_r.removeItem(r[2])
        self.region_items.clear()

        # 1. Overlay Wavelet detections (High visibility Cyan, zValue=5)
        if not self.current_wavelet_events.empty:
            for _, row in self.current_wavelet_events.iterrows():
                s, e = float(row["Start_s"]), float(row["End_s"])
                r1 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(0, 229, 255, 90),
                                         pen=pg.mkPen(color=(0, 229, 255, 230), width=1.8, style=QtCore.Qt.DashLine))
                r2 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(0, 229, 255, 90),
                                         pen=pg.mkPen(color=(0, 229, 255, 230), width=1.8, style=QtCore.Qt.DashLine))
                r3 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(0, 229, 255, 90),
                                         pen=pg.mkPen(color=(0, 229, 255, 230), width=1.8, style=QtCore.Qt.DashLine))
                for r_item in (r1, r2, r3):
                    r_item.setZValue(5)
                self.p_raw.addItem(r1)
                self.p_filt.addItem(r2)
                self.p_r.addItem(r3)
                self.region_items.append((r1, r2, r3))

        # 2. Overlay AR detections (Red/Orange, zValue=10)
        if not self.current_ar_events.empty:
            for _, row in self.current_ar_events.iterrows():
                s, e = float(row["Start_s"]), float(row["End_s"])
                r1 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(255, 50, 50, 80),
                                         pen=pg.mkPen(color=(255, 50, 50, 230), width=1.5))
                r2 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(255, 50, 50, 80),
                                         pen=pg.mkPen(color=(255, 50, 50, 230), width=1.5))
                r3 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(255, 50, 50, 80),
                                         pen=pg.mkPen(color=(255, 50, 50, 230), width=1.5))
                for r_item in (r1, r2, r3):
                    r_item.setZValue(10)
                self.p_raw.addItem(r1)
                self.p_filt.addItem(r2)
                self.p_r.addItem(r3)
                self.region_items.append((r1, r2, r3))

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

    def jump_to_ar_event(self):
        if self.current_ar_idx < 0 or self.current_ar_idx >= len(self.current_ar_events):
            return

        event = self.current_ar_events.iloc[self.current_ar_idx]
        start_s = float(event["Start_s"])
        end_s = float(event["End_s"])

        view_start = max(0.0, start_s - VIEW_PADDING_SEC)
        view_end = min(self.data_len / FS, end_s + VIEW_PADDING_SEC)

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

        info = (
            f"[Focus: AR Spindle #{self.current_ar_idx + 1}/{len(self.current_ar_events)}] "
            f"Interval: [{start_s:.3f}s - {end_s:.3f}s] (Duration: {dur:.3f}s)\n"
            f"AR Metrics: Max R = {r_val:.4f} | Peak Freq = {freq_val:.2f} Hz\n"
            f"Alignment Status: {align_desc}"
        )
        self.details_panel.setText(info)
        self.lbl_ar_tracker.setText(f"AR: {self.current_ar_idx + 1} / {len(self.current_ar_events)}")

    def jump_to_wav_event(self):
        if self.current_wav_idx < 0 or self.current_wav_idx >= len(self.current_wavelet_events):
            return

        event = self.current_wavelet_events.iloc[self.current_wav_idx]
        start_s = float(event["Start_s"])
        end_s = float(event["End_s"])

        view_start = max(0.0, start_s - VIEW_PADDING_SEC)
        view_end = min(self.data_len / FS, end_s + VIEW_PADDING_SEC)

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


def main():
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

    viewer = SpindleViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
