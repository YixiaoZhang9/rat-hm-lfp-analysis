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

FS = 1000
BP_LOW = 10.0
BP_HIGH = 15.0
VIEW_PADDING_SEC = 2.5


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def parse_channel_trial(data_path: str):
    """Extracts channel and trial matching the manifest layout."""
    filename = Path(str(data_path)).name
    match = re.match(r"chan(\d+)(?:_(\d+))?\.mat$", filename, re.IGNORECASE)
    if not match:
        return "", ""
    chan = match.group(1)
    trial = match.group(2) if match.group(2) else ""
    return chan, trial


# --------------------------------------------------------------------------- #
# GUI Implementation
# --------------------------------------------------------------------------- #
class SpindleViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AR vs. Wavelet Spindle Alignment Inspector")
        self.resize(1350, 900)

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
        layout = QtWidgets.QVBoxLayout(main_widget)

        # Control Panel
        control_panel = QtWidgets.QHBoxLayout()

        control_panel.addWidget(QtWidgets.QLabel("File/Recording:"))
        self.combo_files = QtWidgets.QComboBox()
        self.combo_files.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo_files.currentIndexChanged.connect(self.on_file_selected)
        control_panel.addWidget(self.combo_files, stretch=2)

        self.btn_prev = QtWidgets.QPushButton("◀ Previous (AR)")
        self.btn_prev.clicked.connect(self.on_prev_event)
        control_panel.addWidget(self.btn_prev)

        self.lbl_event_tracker = QtWidgets.QLabel("Event: 0 / 0")
        self.lbl_event_tracker.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_event_tracker.setMinimumWidth(100)
        control_panel.addWidget(self.lbl_event_tracker)

        self.btn_next = QtWidgets.QPushButton("Next (AR) ▶")
        self.btn_next.clicked.connect(self.on_next_event)
        control_panel.addWidget(self.btn_next)

        control_panel.addSpacing(20)
        lbl_ar_legend = QtWidgets.QLabel("■ AR Spindle")
        lbl_ar_legend.setStyleSheet("color: #ff3333; font-weight: bold;")
        control_panel.addWidget(lbl_ar_legend)

        lbl_wav_legend = QtWidgets.QLabel("■ Wavelet Spindle")
        lbl_wav_legend.setStyleSheet("color: #00bcd4; font-weight: bold;")
        control_panel.addWidget(lbl_wav_legend)

        control_panel.addStretch(1)
        layout.addLayout(control_panel)

        # Metadata banner
        self.lbl_metadata = QtWidgets.QLabel("Ready.")
        self.lbl_metadata.setStyleSheet("color: #aaaaaa; padding-bottom: 4px;")
        layout.addWidget(self.lbl_metadata)

        # Plot Grid
        pg.setConfigOptions(antialias=False)
        self.graphics_layout = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_layout, stretch=1)

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

        # Keyboard shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, self.on_prev_event)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, self.on_next_event)

    def load_data(self):
        try:
            if not os.path.exists(DEFAULT_MANIFEST):
                self.lbl_metadata.setText(f"Manifest not found: {DEFAULT_MANIFEST}")
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

            if os.path.exists(DEFAULT_AR_CSV):
                self.ar_df = pd.read_csv(DEFAULT_AR_CSV)
                self.ar_df["Rat"] = self.ar_df["Rat"].astype(int)
                self.ar_df["Region"] = self.ar_df["Region"].astype(str)
                self.ar_df["Date"] = (
                    self.ar_df["Date"].astype(str).str.replace(".0", "", regex=False)
                )
            else:
                self.ar_df = pd.DataFrame()

            if os.path.exists(DEFAULT_WAVELET_CSV):
                self.wavelet_df = pd.read_csv(DEFAULT_WAVELET_CSV)
                self.wavelet_df = self.wavelet_df.rename(
                    columns={
                        "rat_number": "Rat",
                        "region": "Region",
                        "date": "Date",
                        "start_time": "Start_s",
                        "end_time": "End_s",
                        "start_sec": "Start_s",
                        "end_sec": "End_s",
                        "start": "Start_s",
                        "end": "End_s",
                    }
                )
                self.wavelet_df["Rat"] = self.wavelet_df["Rat"].astype(int)
                self.wavelet_df["Region"] = self.wavelet_df["Region"].astype(str)
                self.wavelet_df["Date"] = (
                    self.wavelet_df["Date"].astype(str).str.replace(".0", "", regex=False)
                )
                self.wavelet_df["channel"] = (
                    self.wavelet_df["channel"].astype(str).str.replace(".0", "", regex=False)
                )
                self.wavelet_df["trial"] = (
                    self.wavelet_df["trial"]
                    .fillna("")
                    .astype(str)
                    .str.replace(".0", "", regex=False)
                )
            else:
                self.wavelet_df = pd.DataFrame()

            self.combo_files.blockSignals(True)
            self.combo_files.clear()
            for idx, row in self.manifest_df.iterrows():
                label = (
                    f"Rat {row['Rat']} | {row['Region']} | {row['Date']} | "
                    f"ch:{row['channel']} tr:{row['trial']} -> {Path(row['data_path']).name}"
                )
                self.combo_files.addItem(label, userData=idx)
            self.combo_files.blockSignals(False)

            if self.combo_files.count() > 0:
                self.on_file_selected(0)

        except Exception as e:
            self.lbl_metadata.setText(f"Load failed: {str(e)}")

    def on_file_selected(self, index):
        if index < 0 or self.manifest_df is None:
            return

        manifest_idx = self.combo_files.itemData(index)
        task = self.manifest_df.iloc[manifest_idx]

        data_path = task["data_path"]
        if not os.path.exists(data_path):
            self.lbl_metadata.setText(f"File missing: {data_path}")
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
            self.lbl_metadata.setText(f"Signal read/filter error: {str(e)}")
            return

        file_name = Path(data_path).name
        if not self.ar_df.empty:
            if "File" in self.ar_df.columns:
                self.current_ar_events = self.ar_df[
                    self.ar_df["File"] == file_name
                ].sort_values("Start_s").reset_index(drop=True)
            else:
                self.current_ar_events = self.ar_df[
                    (self.ar_df["Rat"] == task["Rat"])
                    & (self.ar_df["Region"] == task["Region"])
                    & (self.ar_df["Date"] == task["Date"])
                ].sort_values("Start_s").reset_index(drop=True)
        else:
            self.current_ar_events = pd.DataFrame()

        if not self.wavelet_df.empty:
            self.current_wavelet_events = self.wavelet_df[
                (self.wavelet_df["Rat"] == task["Rat"])
                & (self.wavelet_df["Region"] == task["Region"])
                & (self.wavelet_df["Date"] == task["Date"])
                & (self.wavelet_df["channel"] == task["channel"])
                & (self.wavelet_df["trial"] == task["trial"])
            ].sort_values("Start_s").reset_index(drop=True)
        else:
            self.current_wavelet_events = pd.DataFrame()

        self.curve_raw.setData(self.time_vector, self.raw_signal)
        self.curve_filt.setData(self.time_vector, self.filtered_signal)

        if not self.current_ar_events.empty and "Max_R" in self.current_ar_events:
            x_pts = (
                self.current_ar_events["Peak_s"].values
                if "Peak_s" in self.current_ar_events
                else self.current_ar_events["Start_s"].values
            )
            y_pts = self.current_ar_events["Max_R"].values
            self.curve_r.setData(x_pts, y_pts)
        else:
            self.curve_r.clear()

        self.draw_detection_spans()

        if len(self.current_ar_events) > 0:
            self.current_event_idx = 0
            self.jump_to_current_ar_event()
        else:
            self.current_event_idx = -1
            self.lbl_event_tracker.setText("AR: 0 / 0")
            self.p_raw.setXRange(0, min(30, self.data_len / FS), padding=0)

        self.update_status_bar()

    def draw_detection_spans(self):
        for r in self.region_items:
            self.p_raw.removeItem(r[0])
            self.p_filt.removeItem(r[1])
            self.p_r.removeItem(r[2])
        self.region_items.clear()

        # AR spans (Red)
        if not self.current_ar_events.empty:
            for _, row in self.current_ar_events.iterrows():
                s, e = float(row["Start_s"]), float(row["End_s"])
                r1 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(255, 60, 60, 50),
                                         pen=pg.mkPen(color=(255, 50, 50, 200), width=1.2))
                r2 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(255, 60, 60, 50),
                                         pen=pg.mkPen(color=(255, 50, 50, 200), width=1.2))
                r3 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(255, 60, 60, 50),
                                         pen=pg.mkPen(color=(255, 50, 50, 200), width=1.2))
                self.p_raw.addItem(r1)
                self.p_filt.addItem(r2)
                self.p_r.addItem(r3)
                self.region_items.append((r1, r2, r3))

        # Wavelet spans (Cyan)
        if not self.current_wavelet_events.empty:
            for _, row in self.current_wavelet_events.iterrows():
                s, e = float(row["Start_s"]), float(row["End_s"])
                r1 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(0, 200, 230, 45),
                                         pen=pg.mkPen(color=(0, 200, 230, 200), width=1.2, style=QtCore.Qt.DashLine))
                r2 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(0, 200, 230, 45),
                                         pen=pg.mkPen(color=(0, 200, 230, 200), width=1.2, style=QtCore.Qt.DashLine))
                r3 = pg.LinearRegionItem([s, e], movable=False,
                                         brush=QtGui.QColor(0, 200, 230, 45),
                                         pen=pg.mkPen(color=(0, 200, 230, 200), width=1.2, style=QtCore.Qt.DashLine))
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

        view_start = max(0.0, start_s - VIEW_PADDING_SEC)
        view_end = min(self.data_len / FS, end_s + VIEW_PADDING_SEC)

        self.p_raw.setXRange(view_start, view_end, padding=0)
        self.p_r.setYRange(0.0, 1.05, padding=0)

        overlapping_wav = []
        if not self.current_wavelet_events.empty:
            w = self.current_wavelet_events
            overlapping_wav = w[(w["Start_s"] <= end_s) & (w["End_s"] >= start_s)]

        overlap_str = (
            f"Aligned with {len(overlapping_wav)} wavelet event(s)"
            if len(overlapping_wav) > 0
            else "No overlapping wavelet event"
        )

        dur = end_s - start_s
        r_val = event.get("Max_R", np.nan)
        freq_val = event.get("Peak_Freq_Hz", np.nan)

        self.lbl_metadata.setText(
            f"AR Event #{self.current_event_idx + 1} | Time: [{start_s:.2f}s - {end_s:.2f}s] "
            f"(Dur: {dur:.2f}s) | Max R: {r_val:.3f} | Peak Freq: {freq_val:.1f} Hz | {overlap_str}"
        )
        self.lbl_event_tracker.setText(
            f"AR: {self.current_event_idx + 1} / {len(self.current_ar_events)}"
        )

    def update_status_bar(self):
        n_ar = len(self.current_ar_events)
        n_wav = len(self.current_wavelet_events)
        total_time = self.data_len / FS
        self.setWindowTitle(
            f"AR vs Wavelet Spindle Alignment Inspector — "
            f"AR Spindles: {n_ar} | Wavelet Spindles: {n_wav} | Length: {total_time:.1f}s"
        )


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(55, 55, 55))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(dark_palette)

    viewer = SpindleViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
