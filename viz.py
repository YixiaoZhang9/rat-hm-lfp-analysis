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
# Default Configuration
# --------------------------------------------------------------------------- #
DEFAULT_MANIFEST = "/home/mdadmin/Desktop/amirali/rat-hm-lfp-analysis/tasks_manifest.csv"
DEFAULT_AR_CSV = "results/all_detected_spindles_per_region.csv"
DEFAULT_WAVELET_CSV = (
    "/home/mdadmin/Desktop/amirali/rat-hm-lfp-analysis/results_ar_calibration/"
    "wavelet_spindles_with_ar_dynamics.csv"
)

FS = 1000
BP_LOW = 10.0
BP_HIGH = 15.0
VIEW_PADDING_SEC = 2.5


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def parse_channel_trial(data_path: str):
    filename = Path(str(data_path)).name
    match = re.match(r"chan(\d+)(?:_(\d+))?\.mat$", filename, re.IGNORECASE)
    if not match:
        return "", ""
    chan = match.group(1)
    trial = match.group(2) if match.group(2) else ""
    return chan, trial


def standardize_interval_columns(df: pd.DataFrame, default_fs: float = 1000.0) -> pd.DataFrame:
    """Detects and normalizes start/end column headers into Start_s and End_s."""
    if df.empty:
        return df

    cols = {c.lower(): c for c in df.columns}
    start_col = None
    end_col = None
    is_sample_based = False

    # Priority lookup for start column names
    candidates_start = [
        "start_s", "start_time", "start_sec", "start_secs", "start",
        "spindle_start", "start_time_s", "start_sample", "start_idx"
    ]
    candidates_end = [
        "end_s", "end_time", "end_sec", "end_secs", "end",
        "spindle_end", "end_time_s", "end_sample", "end_idx"
    ]

    for cand in candidates_start:
        if cand in cols:
            start_col = cols[cand]
            if "sample" in cand or "idx" in cand:
                is_sample_based = True
            break

    for cand in candidates_end:
        if cand in cols:
            end_col = cols[cand]
            break

    if start_col and end_col:
        scale = 1.0 / default_fs if is_sample_based else 1.0
        df["Start_s"] = df[start_col].astype(float) * scale
        df["End_s"] = df[end_col].astype(float) * scale
    else:
        # Fallback empty interval setup if neither pattern resolved
        if "Start_s" not in df.columns:
            df["Start_s"] = 0.0
        if "End_s" not in df.columns:
            df["End_s"] = 0.0

    return df


# --------------------------------------------------------------------------- #
# GUI Application
# --------------------------------------------------------------------------- #
class SpindleViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spindle Alignment Inspector (AR vs Wavelet)")
        self.resize(1450, 950)

        self.manifest_df = None
        self.ar_df = None
        self.wavelet_df = None

        self.current_ar_events = pd.DataFrame()
        self.current_wavelet_events = pd.DataFrame()
        self.current_event_idx = -1
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

        # ---------------- Filter & Selection Controls ---------------- #
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

        # ---------------- AR Navigation Bar ---------------- #
        nav_box = QtWidgets.QGroupBox("AR Navigation")
        nav_layout = QtWidgets.QHBoxLayout()

        self.btn_prev = QtWidgets.QPushButton("◀ Previous (AR)")
        self.btn_prev.clicked.connect(self.on_prev_event)
        nav_layout.addWidget(self.btn_prev)

        self.lbl_event_tracker = QtWidgets.QLabel("Event: 0 / 0")
        self.lbl_event_tracker.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_event_tracker.setMinimumWidth(110)
        nav_layout.addWidget(self.lbl_event_tracker)

        self.btn_next = QtWidgets.QPushButton("Next (AR) ▶")
        self.btn_next.clicked.connect(self.on_next_event)
        nav_layout.addWidget(self.btn_next)

        nav_layout.addSpacing(30)
        lbl_ar_legend = QtWidgets.QLabel("■ AR Spindle")
        lbl_ar_legend.setStyleSheet("color: #ff3333; font-weight: bold;")
        nav_layout.addWidget(lbl_ar_legend)

        lbl_wav_legend = QtWidgets.QLabel("■ Wavelet Spindle")
        lbl_wav_legend.setStyleSheet("color: #00bcd4; font-weight: bold;")
        nav_layout.addWidget(lbl_wav_legend)

        nav_layout.addStretch(1)
        nav_box.setLayout(nav_layout)
        root_layout.addWidget(nav_box)

        # ---------------- Signal Viewports ---------------- #
        pg.setConfigOptions(antialias=False)
        self.graphics_layout = pg.GraphicsLayoutWidget()
        root_layout.addWidget(self.graphics_layout, stretch=1)

        # Row 1: Unfiltered LFP
        self.p_raw = self.graphics_layout.addPlot(row=0, col=0)
        self.p_raw.showGrid(x=True, y=True, alpha=0.3)
        self.p_raw.setLabel("left", "Raw LFP", units="uV")
        self.curve_raw = self.p_raw.plot(pen=pg.mkPen(color="#dcdcdc", width=1))

        # Row 2: 10-15 Hz Filtered LFP
        self.p_filt = self.graphics_layout.addPlot(row=1, col=0)
        self.p_filt.showGrid(x=True, y=True, alpha=0.3)
        self.p_filt.setLabel("left", "10-15 Hz", units="uV")
        self.curve_filt = self.p_filt.plot(pen=pg.mkPen(color="#4db6ac", width=1.2))

        # Row 3: AR Pole Metric (R-value)
        self.p_r = self.graphics_layout.addPlot(row=2, col=0)
        self.p_r.showGrid(x=True, y=True, alpha=0.3)
        self.p_r.setLabel("left", "Max R")
        self.p_r.setLabel("bottom", "Time", units="s")
        self.curve_r = self.p_r.plot(
            pen=None,
            symbol="o",
            symbolSize=8,
            symbolBrush="#ff5722",
            symbolPen=pg.mkPen(color="w", width=0.5),
        )

        self.p_filt.setXLink(self.p_raw)
        self.p_r.setXLink(self.p_raw)
        self.region_items = []

        # ---------------- Detailed Inspection Panel ---------------- #
        self.details_panel = QtWidgets.QTextEdit()
        self.details_panel.setReadOnly(True)
        self.details_panel.setMaximumHeight(85)
        self.details_panel.setStyleSheet(
            "background-color: #1e1e1e; color: #00e676; font-family: Monospace; font-size: 11px;"
        )
        root_layout.addWidget(self.details_panel)

        # Shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, self.on_prev_event)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, self.on_next_event)

    def load_data(self):
        if not os.path.exists(DEFAULT_MANIFEST):
            self.details_panel.setText(f"Manifest not found: {DEFAULT_MANIFEST}")
            return

        self.manifest_df = pd.read_csv(DEFAULT_MANIFEST)
        self.manifest_df["Rat"] = self.manifest_df["rat"].astype(int)
        self.manifest_df["Region"] = self.manifest_df["region"].astype(str)
        self.manifest_df["Date"] = (
            self.manifest_df["date"].astype(str).str.replace(".0", "", regex=False)
        )

        parsed = self.manifest_df["data_path"].apply(parse_channel_trial)
        self.manifest_df["channel"] = parsed.apply(lambda x: x[0])
        self.manifest_df["trial"] = parsed.apply(lambda x: x[1])

        # AR Loading & Standardization
        if os.path.exists(DEFAULT_AR_CSV):
            self.ar_df = pd.read_csv(DEFAULT_AR_CSV)
            self.ar_df["Rat"] = self.ar_df["Rat"].astype(int)
            self.ar_df["Region"] = self.ar_df["Region"].astype(str)
            self.ar_df["Date"] = (
                self.ar_df["Date"].astype(str).str.replace(".0", "", regex=False)
            )
            self.ar_df = standardize_interval_columns(self.ar_df, FS)
        else:
            self.ar_df = pd.DataFrame()

        # Wavelet Loading & Standardization
        if os.path.exists(DEFAULT_WAVELET_CSV):
            self.wavelet_df = pd.read_csv(DEFAULT_WAVELET_CSV)
            self.wavelet_df = self.wavelet_df.rename(
                columns={
                    "rat_number": "Rat",
                    "region": "Region",
                    "date": "Date",
                }
            )
            self.wavelet_df["Rat"] = self.wavelet_df["Rat"].astype(int)
            self.wavelet_df["Region"] = self.wavelet_df["Region"].astype(str)
            self.wavelet_df["Date"] = (
                self.wavelet_df["Date"].astype(str).str.replace(".0", "", regex=False)
            )
            if "channel" in self.wavelet_df.columns:
                self.wavelet_df["channel"] = (
                    self.wavelet_df["channel"].astype(str).str.replace(".0", "", regex=False)
                )
            if "trial" in self.wavelet_df.columns:
                self.wavelet_df["trial"] = (
                    self.wavelet_df["trial"].fillna("").astype(str).str.replace(".0", "", regex=False)
                )
            self.wavelet_df = standardize_interval_columns(self.wavelet_df, FS)
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
            lbl = f"{row['Date']} | ch:{row['channel']} tr:{row['trial']} -> {Path(row['data_path']).name}"
            self.combo_files.addItem(lbl, userData=orig_idx)
        self.combo_files.blockSignals(False)

        if self.combo_files.count() > 0:
            self.on_file_selected(0)

    def on_file_selected(self, index):
        if index < 0 or self.manifest_df is None:
            return

        manifest_idx = self.combo_files.itemData(index)
        task = self.manifest_df.iloc[manifest_idx]
        data_path = task["data_path"]

        if not os.path.exists(data_path):
            self.details_panel.setText(f"LFP MAT file not found at: {data_path}")
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
            self.details_panel.setText(f"File load/filter failed: {str(e)}")
            return

        file_name = Path(data_path).name

        # Query AR detections for this file
        if not self.ar_df.empty:
            if "File" in self.ar_df.columns:
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

        # Query Wavelet detections for this file
        if not self.wavelet_df.empty:
            wav_sub = self.wavelet_df[
                (self.wavelet_df["Rat"] == task["Rat"])
                & (self.wavelet_df["Region"] == task["Region"])
                & (self.wavelet_df["Date"] == task["Date"])
            ]
            if "channel" in wav_sub.columns and task["channel"]:
                wav_sub = wav_sub[wav_sub["channel"] == task["channel"]]
            if "trial" in wav_sub.columns and task["trial"]:
                wav_sub = wav_sub[wav_sub["trial"] == task["trial"]]

            if not wav_sub.empty and "Start_s" in wav_sub.columns:
                self.current_wavelet_events = wav_sub.sort_values("Start_s").reset_index(drop=True)
            else:
                self.current_wavelet_events = pd.DataFrame()
        else:
            self.current_wavelet_events = pd.DataFrame()

        # Plot waveforms
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

        if len(self.current_ar_events) > 0:
            self.current_event_idx = 0
            self.jump_to_current_ar_event()
        else:
            self.current_event_idx = -1
            self.lbl_event_tracker.setText("AR: 0 / 0")
            self.p_raw.setXRange(0, min(30, self.data_len / FS), padding=0)
            self.details_panel.setText(
                f"File: {file_name} | Length: {self.data_len/FS:.1f}s | "
                f"AR Spindles: 0 | Wavelet Spindles: {len(self.current_wavelet_events)}"
            )

    def draw_spans(self):
        for r in self.region_items:
            self.p_raw.removeItem(r[0])
            self.p_filt.removeItem(r[1])
            self.p_r.removeItem(r[2])
        self.region_items.clear()

        # Overlay AR detections (Red/Orange)
        if not self.current_ar_events.empty:
            for _, row in self.current_ar_events.iterrows():
                s, e = float(row["Start_s"]), float(row["End_s"])
                r1 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(255, 60, 60, 50),
                                         pen=pg.mkPen(color=(255, 50, 50, 180), width=1.2))
                r2 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(255, 60, 60, 50),
                                         pen=pg.mkPen(color=(255, 50, 50, 180), width=1.2))
                r3 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(255, 60, 60, 50),
                                         pen=pg.mkPen(color=(255, 50, 50, 180), width=1.2))
                self.p_raw.addItem(r1)
                self.p_filt.addItem(r2)
                self.p_r.addItem(r3)
                self.region_items.append((r1, r2, r3))

        # Overlay Wavelet detections (Cyan)
        if not self.current_wavelet_events.empty:
            for _, row in self.current_wavelet_events.iterrows():
                s, e = float(row["Start_s"]), float(row["End_s"])
                r1 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(0, 200, 230, 45),
                                         pen=pg.mkPen(color=(0, 200, 230, 180), width=1.2, style=QtCore.Qt.DashLine))
                r2 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(0, 200, 230, 45),
                                         pen=pg.mkPen(color=(0, 200, 230, 180), width=1.2, style=QtCore.Qt.DashLine))
                r3 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(0, 200, 230, 45),
                                         pen=pg.mkPen(color=(0, 200, 230, 180), width=1.2, style=QtCore.Qt.DashLine))
                self.p_raw.addItem(r1)
                self.p_filt.addItem(r2)
                self.p_r.addItem(r3)
                self.region_items.append((r1, r2, r3))

    def on_prev_event(self):
        if len(self.current_ar_events) == 0:
            return
        self.current_event_idx = (self.current_event_idx - 1) % len(self.current_ar_events)
        self.jump_to_current_ar_event()

    def on_next_event(self):
        if len(self.current_ar_events) == 0:
            return
        self.current_event_idx = (self.current_event_idx + 1) % len(self.current_ar_events)
        self.jump_to_current_ar_event()

    def jump_to_current_ar_event(self):
        if self.current_event_idx < 0 or self.current_event_idx >= len(self.current_ar_events):
            return

        event = self.current_ar_events.iloc[self.current_event_idx]
        start_s = float(event["Start_s"])
        end_s = float(event["End_s"])

        # Center camera around detection bounds
        view_start = max(0.0, start_s - VIEW_PADDING_SEC)
        view_end = min(self.data_len / FS, end_s + VIEW_PADDING_SEC)

        self.p_raw.setXRange(view_start, view_end, padding=0)
        self.p_r.setYRange(0.0, 1.05, padding=0)

        # Compute overlap with wavelet detections
        overlapping = pd.DataFrame()
        if not self.current_wavelet_events.empty:
            w = self.current_wavelet_events
            overlapping = w[(w["Start_s"] <= end_s) & (w["End_s"] >= start_s)]

        align_desc = (
            f"ALIGNED: {len(overlapping)} wavelet detection(s) in interval"
            if len(overlapping) > 0
            else "ISOLATED: No overlapping wavelet event"
        )

        r_val = event.get("Max_R", np.nan)
        freq_val = event.get("Peak_Freq_Hz", np.nan)
        dur = end_s - start_s

        info = (
            f"Event #{self.current_event_idx + 1}/{len(self.current_ar_events)} | "
            f"Interval: [{start_s:.3f}s - {end_s:.3f}s] (Duration: {dur:.3f}s)\n"
            f"AR Metrics: Max R = {r_val:.4f} | Peak Freq = {freq_val:.2f} Hz\n"
            f"Validation: {align_desc}"
        )
        self.details_panel.setText(info)
        self.lbl_event_tracker.setText(f"AR: {self.current_event_idx + 1} / {len(self.current_ar_events)}")


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
