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

FS = 1000  # Target sampling rate (Hz)
BP_LOW = 10.0
BP_HIGH = 15.0
VIEW_PADDING_SEC = 2.5  # Window padding added to both sides of a detected spindle


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

        # In-memory datasets
        self.manifest_df = None
        self.ar_df = None
        self.wavelet_df = None

        # State tracking
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

        # ---------------- Control Bar ---------------- #
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

        # Legend/Color reference labels
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

        # ---------------- Plot Grid ---------------- #
        pg.setConfigOptions(antialias=False)  # Performance optimization for large signals
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

        # Link X-axes together for synchronous panning and zooming
        self.p_filt.setXLink(self.p_raw)
        self.p_r.setXLink(self.p_raw)

        # Visual ROI spans (LinearRegions)
        self.region_items = []

    # ----------------------------------------------------------------------- #
    # Data Ingestion
    # ----------------------------------------------------------------------- #
    def load_data(self):
        try:
            # 1. Manifest
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

            # 2. AR Spindles
            if os.path.exists(DEFAULT_AR_CSV):
                self.ar_df = pd.read_csv(DEFAULT_AR_CSV)
                self.ar_df["Rat"] = self.ar_df["Rat"].astype(int)
                self.ar_df["Region"] = self.ar_df["Region"].astype(str)
                self.ar_df["Date"] = (
                    self.ar_df["Date"].astype(str).str.replace(".0", "", regex=False)
                )
            else:
                self.ar_df = pd.DataFrame()

            # 3. Wavelet Spindles
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
                    self.wavelet_df["Date"]
                    .astype(str)
                    .str.replace(".0", "", regex=False)
                )
                self.wavelet_df["channel"] = (
                    self.wavelet_df["channel"]
                    .astype(str)
                    .str.replace(".0", "", regex=False)
                )
                self.wavelet_df["trial"] = (
                    self.wavelet_df["trial"]
                    .fillna("")
                    .astype(str)
                    .str.replace(".0", "", regex=False)
                )
            else:
                self.wavelet_df = pd.DataFrame()

            # Populate combo box with valid manifest entries
            self.combo_files.clear()
            for idx, row in self.manifest_df.iterrows():
                label = (
                    f"Rat {row['Rat']} | {row['Region']} | {row['Date']} | "
                    f"ch:{row['channel']} tr:{row['trial']} -> {Path(row['data_path']).name}"
                )
                self.combo_files.addItem(label, userData=idx)

        except Exception as e:
            self.lbl_metadata.setText(f"Load failed: {str(e)}")

    # ----------------------------------------------------------------------- #
    # File & Event Loading
    # ----------------------------------------------------------------------- #
    def on_file_selected(self, index):
        if index < 0 or self.manifest_df is None:
            return

        manifest_idx = self.combo_files.itemData(index)
        task = self.manifest_df.iloc[manifest_idx]

        data_path = task["data_path"]
        if not os.path.exists(data_path):
            self.lbl_metadata.setText(f"File missing: {data_path}")
            return

        # Load LFP MAT file
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

        # Filter AR detections
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

        # Filter Wavelet detections
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

        # Plot entire trace
        self.curve_raw.setData(self.time_vector, self.raw_signal)
        self.curve_filt.setData(self.time_vector, self.filtered_signal)

        # Plot AR events R metric
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

        # Clear and draw region spans
        self.draw_detection_spans()

        # Reset navigation index to first AR event
        if len(self.current_ar_events) > 0:
            self.current_event_idx = 0
            self.jump_to_current_ar_event()
        else:
            self.current_event_idx = -1
            self.lbl_event_tracker.setText("AR: 0 / 0")
            self.p_raw.setXRange(0, min(30, self.data_len / FS), padding=0)

        self.update_status_bar()

    def draw_detection_spans(self):
        # Remove old intervals
        for r in self.region_items:
            self.p_raw.removeItem(r[0])
            self.p_filt.removeItem(r[1])
        self.region_items.clear()

        # 1. AR Intervals (Red with transparency)
        if not self.current_ar_events.empty:
            for _, row in self.current_ar_events.iterrows():
                s, e = float(row["Start_s"]), float(row["End_s"])
                r1 = pg.LinearRegionItem(
                    values=[s, e],
                    orientation="vertical",
                    brush=QtGui.QColor(255, 60, 60, 50),
                    pen=pg.mkPen(color=(255, 50, 50, 200), width=1.2),
                    movable=False,
                )
                r2 = pg.LinearRegionItem(
                    values=[s, e],
                    orientation="vertical",
                    brush=QtGui.QColor(255, 60, 60, 50),
                    pen=pg.mkPen(color=(255, 50, 50, 200), width=1.2),
                    movable=False,
                )
                self.p_raw.addItem(r1)
                self.p_filt.addItem(r2)
                self.region_items.append((r1, r2))

        # 2. Wavelet Intervals (Cyan with transparency)
        if not self.current_wavelet_events.empty:
            for _, row in self.current_wavelet_events.iterrows():
                s, e = float(row["Start_s"]), float(row["End_s"])
                r1 = pg.LinearRegionItem(
                    values=[s, e],
                    orientation="vertical",
                    brush=QtGui.QColor(0, 200, 230, 45),
                    pen=pg.mkPen(color=(0, 200, 230, 200), width=1.2, style=QtCore.Qt.DashLine),
                    movable=False,
                )
                r2 = pg.LinearRegionItem(
                    values=[s, e],
                    orientation="vertical",
                    brush=QtGui.QColor(0, 200, 230, 45),
                    pen=pg.mkPen(color=(0, 200, 230, 200), width=1.2, style=QtCore.Qt.DashLine),
                    movable=False,
                )
                self.p_raw.addItem(r1)
                self.p_filt.addItem(r2)
                self.region_items.append((r1, r2))

    # ----------------------------------------------------------------------- #
    # AR-Guided Navigation
    # ----------------------------------------------------------------------- #
    def on_prev_event(self):
        if len(self.current_ar_events) == 0:
            return
        self.current_event_idx = (self.current_event_idx - 1) % len(
            self.current_ar_events
        )
        self.jump_to_current_ar_event()

    def on_next_event(self):
        if len(self.current_ar_events) == 0:
            return
        self.current_event_idx = (self.current_event_idx + 1) % len(
            self.current_ar_events
        )
        self.jump_to_current_ar_event()

    def jump_to_current_ar_event(self):
        if self.current_event_idx < 0 or self.current_event_idx >= len(
            self.current_ar_events
        ):
            return

        event = self.current_ar_events.iloc[self.current_event_idx]
        start_s = float(event["Start_s"])
        end_s = float(event["End_s"])

        # Center the display with configured temporal padding
        view_start = max(0.0, start_s - VIEW_PADDING_SEC)
        view_end = min(self.data_len / FS, end_s + VIEW_PADDING_SEC)

        self.p_raw.setXRange(view_start, view_end, padding=0)
        self.p_r.setYRange(0.0, 1.05, padding=0)

        # Check for temporal overlap with wavelet detections
        overlapping_wav = []
        if not self.current_wavelet_events.empty:
            w = self.current_wavelet_events
            overlapping_wav = w[
                (w["Start_s"] <= end_s) & (w["End_s"] >= start_s)
            ]

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

    # Dark theme configuration
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(22, 22, 22))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Below is a standalone PyQt5 desktop application script using `pyqtgraph` (which provides performant signal rendering and fast multi-axis interactions in Qt5).

### Key Features
1. **Three-Row Synchronized Display**:
   - **Row 1**: Raw/unfiltered LFP signal.
   - **Row 2**: Zero-phase bandpass filtered signal ($10-15\text{ Hz}$).
   - **Row 3**: AR pole radius ($R$-values) track with upper/lower threshold reference lines.
   - All horizontal time axes are locked and zoom/pan in lockstep.
2. **Dual-Method Visual Distinction**:
   - **AR Detections**: Highlighted with an orange span across all rows and labeled markers.
   - **Wavelet Detections**: Highlighted with a semi-transparent cyan span across all rows for direct visual overlap and alignment comparison.
3. **AR-Driven Event Navigation**:
   - Step through events with **Previous** / **Next** buttons, direct jump index input, or the Left/Right arrow keys.
   - Centers the viewing window symmetrically around the target AR detection with a configurable time buffer.

---

### Implementation Script (`spindle_inspector_gui.py`)

```python
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# Configure PyQtGraph visuals
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOption("antialias", True)


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def compute_sliding_ar_r_values(signal: np.ndarray, fs: float = 1000, target_fs: float = 128,
                                ar_order: int = 8, window_sec: float = 1.0, stride_samples: int = 4,
                                band: tuple = (10, 15)) -> tuple:
    """
    Computes pole radius R across a local time window using an AR model.
    Falls back to a downsampled decimation if resample isn't needed.
    """
    decim = max(1, int(fs / target_fs))
    sig_sub = signal[::decim]
    eff_fs = fs / decim
    win_len = int(window_sec * eff_fs)

    if len(sig_sub) < win_len:
        return np.array([]), np.array([])

    n_windows = (len(sig_sub) - win_len) // stride_samples + 1
    r_vals = np.zeros(n_windows)
    time_pts = np.zeros(n_windows)

    # Pre-calculate frequency band limits in radians
    w_min = 2 * np.pi * band[0] / eff_fs
    w_max = 2 * np.pi * band[1] / eff_fs

    for i in range(n_windows):
        start_idx = i * stride_samples
        seg = sig_sub[start_idx : start_idx + win_len]
        center_orig_sample = (start_idx + win_len // 2) * decim
        time_pts[i] = center_orig_sample / fs

        # Biased autocorrelation-based Levinson-Durbin AR fitting
        seg = seg - np.mean(seg)
        r = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
        if r[0] <= 1e-12:
            r_vals[i] = 0.0
            continue

        r = r / r[0]
        # Levinson-Durbin recursion
        a = np.zeros(ar_order + 1)
        a[0] = 1.0
        e = 1.0
        for m in range(1, ar_order + 1):
            km = 0.0
            for j in range(m):
                km += a[j] * r[m - j]
            km = -km / e if abs(e) > 1e-12 else 0.0
            a_new = a.copy()
            for j in range(1, m):
                a_new[j] = a[j] + km * a[m - j]
            a_new[m] = km
            a = a_new
            e *= 1.0 - km * km

        # Root analysis for pole radii in frequency range
        roots = np.roots(a)
        angles = np.abs(np.angle(roots))
        radii = np.abs(roots)
        in_band = (angles >= w_min) & (angles <= w_max)

        if np.any(in_band):
            r_vals[i] = np.max(radii[in_band])
        else:
            r_vals[i] = 0.0

    return time_pts, r_vals


class SpindleInspector(QtWidgets.QMainWindow):
    def __init__(self, ar_csv: str, wavelet_csv: str, manifest_csv: str):
        super().__init__()
        self.setWindowTitle("LFP Spindle Inspector: AR vs Wavelet")
        self.resize(1400, 850)

        self.ar_csv_path = Path(ar_csv)
        self.wavelet_csv_path = Path(wavelet_csv)
        self.manifest_csv_path = Path(manifest_csv)

        self.current_idx = 0
        self.fs = 1000
        self.window_view_sec = 6.0  # Total time window centered on detection

        self.cached_file_name = None
        self.raw_signal = None
        self.filtered_signal = None
        self.r_times = None
        self.r_values = None

        self._load_datasets()
        self._init_ui()
        self._setup_key_shortcuts()

        if len(self.ar_df) > 0:
            self.jump_to_index(0)

    def _load_datasets(self):
        # Load AR detections
        self.ar_df = pd.read_csv(self.ar_csv_path)
        self.ar_df["Rat"] = self.ar_df["Rat"].astype(int)
        self.ar_df["Region"] = self.ar_df["Region"].astype(str)
        self.ar_df["Date"] = self.ar_df["Date"].astype(str).str.replace(".0", "", regex=False)

        # Load Wavelet detections
        self.wavelet_df = pd.read_csv(self.wavelet_csv_path)
        self.wavelet_df = self.wavelet_df.rename(columns={
            "rat_number": "Rat",
            "region": "Region",
            "date": "Date",
        })
        self.wavelet_df["Rat"] = self.wavelet_df["Rat"].astype(int)
        self.wavelet_df["Region"] = self.wavelet_df["Region"].astype(str)
        self.wavelet_df["Date"] = self.wavelet_df["Date"].astype(str).str.replace(".0", "", regex=False)

        # Harmonize column names for start/end in wavelet
        if "start_time" in self.wavelet_df.columns:
            self.wavelet_df = self.wavelet_df.rename(columns={"start_time": "Start_s", "end_time": "End_s"})
        elif "Start" in self.wavelet_df.columns:
            self.wavelet_df = self.wavelet_df.rename(columns={"Start": "Start_s", "End": "End_s"})

        # Load Manifest
        self.manifest = pd.read_csv(self.manifest_csv_path)
        self.manifest["Rat"] = self.manifest["rat"].astype(int)
        self.manifest["Region"] = self.manifest["region"].astype(str)
        self.manifest["Date"] = self.manifest["date"].astype(str).str.replace(".0", "", regex=False)

    def _init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Navigation & Status Panel
        nav_box = QtWidgets.QGroupBox("Detection Navigation (Base: AR Method)")
        nav_layout = QtWidgets.QHBoxLayout()

        self.btn_prev = QtWidgets.QPushButton("◀ Previous")
        self.btn_prev.clicked.connect(self.prev_spindle)
        self.btn_next = QtWidgets.QPushButton("Next ▶")
        self.btn_next.clicked.connect(self.next_spindle)

        self.spin_idx = QtWidgets.QSpinBox()
        self.spin_idx.setRange(0, max(0, len(self.ar_df) - 1))
        self.spin_idx.valueChanged.connect(self.jump_to_index)

        self.lbl_count = QtWidgets.QLabel(f"/ {len(self.ar_df)}")
        self.lbl_meta = QtWidgets.QLabel("Recording Info: -")
        self.lbl_meta.setStyleSheet("font-weight: bold; color: #2C3E50;")

        # Visual legend
        legend_layout = QtWidgets.QHBoxLayout()
        legend_ar = QtWidgets.QLabel("■ AR Detections")
        legend_ar.setStyleSheet("color: rgba(220, 100, 0, 1.0); font-weight: bold;")
        legend_wav = QtWidgets.QLabel("■ Wavelet Detections")
        legend_wav.setStyleSheet("color: rgba(0, 130, 200, 1.0); font-weight: bold;")
        legend_layout.addWidget(legend_ar)
        legend_layout.addWidget(legend_wav)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(QtWidgets.QLabel("Event Index:"))
        nav_layout.addWidget(self.spin_idx)
        nav_layout.addWidget(self.lbl_count)
        nav_layout.addSpacing(20)
        nav_layout.addLayout(legend_layout)
        nav_layout.addSpacing(20)
        nav_layout.addWidget(self.lbl_meta)
        nav_layout.addStretch()

        nav_box.setLayout(nav_layout)
        main_layout.addWidget(nav_box)

        # PyQtGraph 3-Row Layout
        self.win = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.win)

        # Plot 1: Unfiltered
        self.p1 = self.win.addPlot(row=0, col=0)
        self.p1.showGrid(x=True, y=True, alpha=0.3)
        self.p1.setLabel("left", "Raw LFP", units="a.u.")
        self.p1.setLabel("bottom", "Time", units="s")

        # Plot 2: 10-15 Hz Filtered
        self.p2 = self.win.addPlot(row=1, col=0)
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        self.p2.setLabel("left", "10-15 Hz", units="a.u.")
        self.p2.setLabel("bottom", "Time", units="s")
        self.p2.setXLink(self.p1)

        # Plot 3: Pole Radius (R)
        self.p3 = self.win.addPlot(row=2, col=0)
        self.p3.showGrid(x=True, y=True, alpha=0.3)
        self.p3.setLabel("left", "AR Radius (R)")
        self.p3.setLabel("bottom", "Time", units="s")
        self.p3.setXLink(self.p1)
        self.p3.setYRange(0.0, 1.05)

        # Plot curves
        self.curve_raw = self.p1.plot(pen=pg.mkPen("#333333", width=1))
        self.curve_filt = self.p2.plot(pen=pg.mkPen("#1f77b4", width=1))
        self.curve_r = self.p3.plot(pen=pg.mkPen("#2ca02c", width=1.5))

        # Threshold lines on R plot
        self.line_upper = pg.InfiniteLine(angle=0, pen=pg.mkPen("r", style=QtCore.Qt.DashLine, width=1))
        self.line_lower = pg.InfiniteLine(angle=0, pen=pg.mkPen("k", style=QtCore.Qt.DashLine, width=1))
        self.p3.addItem(self.line_upper)
        self.p3.addItem(self.line_lower)

        # Regions container for dynamic spans
        self.active_regions = []

    def _setup_key_shortcuts(self):
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self, self.prev_spindle)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, self.next_spindle)

    def prev_spindle(self):
        if self.current_idx > 0:
            self.spin_idx.setValue(self.current_idx - 1)

    def next_spindle(self):
        if self.current_idx < len(self.ar_df) - 1:
            self.spin_idx.setValue(self.current_idx + 1)

    def jump_to_index(self, idx: int):
        if idx < 0 or idx >= len(self.ar_df):
            return
        self.current_idx = idx
        if self.spin_idx.value() != idx:
            self.spin_idx.blockSignals(True)
            self.spin_idx.setValue(idx)
            self.spin_idx.blockSignals(False)

        self._render_current_detection()

    def _clear_event_spans(self):
        for item in self.active_regions:
            plot_parent = item.parentWidget() if hasattr(item, "parentWidget") else None
            for p in (self.p1, self.p2, self.p3):
                try:
                    p.removeItem(item)
                except Exception:
                    pass
        self.active_regions.clear()

    def _render_current_detection(self):
        event = self.ar_df.iloc[self.current_idx]
        file_name = event["File"]
        start_s = float(event["Start_s"])
        end_s = float(event["End_s"])
        peak_s = float(event.get("Peak_s", (start_s + end_s) / 2.0))

        # Check if new file needs to be read
        if file_name != self.cached_file_name:
            matched = self.manifest[self.manifest["data_path"].str.endswith(file_name)]
            if matched.empty:
                self.lbl_meta.setText(f"Error: File '{file_name}' not found in manifest.")
                return

            task_row = matched.iloc[0]
            mat_path = task_row["data_path"]

            try:
                mat_data = loadmat(mat_path)["data"].squeeze()
            except Exception as e:
                self.lbl_meta.setText(f"Read error for {mat_path}: {e}")
                return

            self.cached_file_name = file_name
            self.raw_signal = mat_data
            self.filtered_signal = bandpass_filter(self.raw_signal, 10.0, 15.0, self.fs)

            # Compute R track across whole signal
            self.r_times, self.r_values = compute_sliding_ar_r_values(
                self.raw_signal,
                fs=self.fs,
                target_fs=128,
                ar_order=8,
                window_sec=1.0,
                stride_samples=4,
                band=(10, 15)
            )

            # Update curves
            t_axis = np.arange(len(self.raw_signal)) / self.fs
            self.curve_raw.setData(t_axis, self.raw_signal)
            self.curve_filt.setData(t_axis, self.filtered_signal)
            self.curve_r.setData(self.r_times, self.r_values)

        # Update metadata display
        upper_th = float(event.get("Threshold_Upper", 0.85))
        lower_th = float(event.get("Threshold_Lower", 0.40))
        self.line_upper.setValue(upper_th)
        self.line_lower.setValue(lower_th)

        self.lbl_meta.setText(
            f"Rat: {event['Rat']} | Region: {event['Region']} | Date: {event['Date']} | "
            f"File: {file_name} | Duration: {event['Duration_s']:.2f}s | Peak: {peak_s:.2f}s"
        )

        # Redraw highlight intervals across all 3 rows
        self._clear_event_spans()

        # 1. Overlay Wavelet detections for this recording
        file_wav = self.wavelet_df[
            (self.wavelet_df["Rat"] == int(event["Rat"])) &
            (self.wavelet_df["Region"] == str(event["Region"])) &
            (self.wavelet_df["Date"] == str(event["Date"]))
        ]

        wav_color = QtGui.QColor(0, 180, 240, 60)
        for _, wav_row in file_wav.iterrows():
            w_start, w_end = float(wav_row["Start_s"]), float(wav_row["End_s"])
            for p in (self.p1, self.p2, self.p3):
                lr = pg.LinearRegionItem([w_start, w_end], movable=False, brush=pg.mkBrush(wav_color), pen=pg.mkPen(0, 130, 200, 100))
                p.addItem(lr)
                self.active_regions.append(lr)

        # 2. Overlay AR detections for this recording (Current event in darker orange)
        file_ar = self.ar_df[self.ar_df["File"] == file_name]
        for idx, ar_row in file_ar.iterrows():
            a_start, a_end = float(ar_row["Start_s"]), float(ar_row["End_s"])
            is_current = (idx == self.current_idx)
            brush_color = QtGui.QColor(255, 140, 0, 120) if is_current else QtGui.QColor(255, 180, 50, 50)
            pen_color = QtGui.QColor(200, 80, 0, 200) if is_current else QtGui.QColor(200, 120, 0, 80)

            for p in (self.p1, self.p2, self.p3):
                lr = pg.LinearRegionItem([a_start, a_end], movable=False, brush=pg.mkBrush(brush_color), pen=pg.mkPen(pen_color))
                p.addItem(lr)
                self.active_regions.append(lr)

        # Center zoom around active detection
        half_win = self.window_view_sec / 2.0
        self.p1.setXRange(peak_s - half_win, peak_s + half_win, padding=0)


if __name__ == "__main__":
    AR_CSV = "results/all_detected_spindles_per_region.csv"
    WAVELET_CSV = "/home/mdadmin/Desktop/amirali/rat-hm-lfp-analysis/results_ar_calibration/wavelet_spindles_with_ar_dynamics.csv"
    MANIFEST_CSV = "/home/mdadmin/Desktop/amirali/rat-hm-lfp-analysis/tasks_manifest.csv"

    app = QtWidgets.QApplication(sys.argv)
    viewer = SpindleInspector(AR_CSV, WAVELET_CSV, MANIFEST_CSV)
    viewer.show()
    sys.exit(app.exec_())
