import os
from itertools import groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import butter, filtfilt


# ===============================
# ----- SIGNAL FILTER FUNCTION -----
# ===============================
def filter_lfp(signal, fs, band):
    """Bandpass filter for LFP"""
    nyq = 0.5 * fs
    low, high = band[0] / nyq, band[1] / nyq
    b, a = butter(3, [low, high], btype="band")
    return filtfilt(b, a, signal)


# ===============================
# ----- LOAD EVENTS -----
# ===============================
def load_events(csv_path):
    """Load ripple events from CSV"""
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return []
    # CSV assumed to have columns: ripple_start, ripple_end
    return list(zip(df["start_time"].values, df["end_time"].values))


# ===============================
# ----- RIPPLE VIEWER CLASS -----
# ===============================
class RippleViewer(QWidget):
    """
    Visualize LFP + 100-250Hz band + sleep stage + ripple annotations.
    Supports multiple annotators dynamically.
    """

    def __init__(self, lfp, scoring, fs, events_dict):
        super().__init__()

        self.lfp = lfp
        self.scoring = scoring  # 1 value per second
        self.fs = fs

        self.events_dict = events_dict  # {"annotator1": [...], "annotator2": [...]}
        self.filtered = filter_lfp(lfp, fs, [100, 250])
        self.window_sec = 10

        # Automatically assign colors to annotators
        default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        self.annotator_keys = list(self.events_dict.keys())
        self.colors = {
            key: default_colors[i % len(default_colors)]
            for i, key in enumerate(self.annotator_keys)
        }

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Ripple Viewer")
        layout = QVBoxLayout()
        self.fig = plt.Figure(figsize=(15, 7))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax_raw = self.fig.add_subplot(211)
        self.ax_filtered = self.fig.add_subplot(212, sharex=self.ax_raw)

        # Controls
        control_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int(len(self.lfp) / self.fs - self.window_sec))
        self.slider.valueChanged.connect(self.update_plot)
        control_layout.addWidget(QLabel("Time (s)"))
        control_layout.addWidget(self.slider)

        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        control_layout.addWidget(self.zoom_in_btn)
        control_layout.addWidget(self.zoom_out_btn)
        layout.addLayout(control_layout)

        self.setLayout(layout)
        self.add_legends()
        self.update_plot()

    # Legends
    def add_legends(self):
        ripple_legend = [
            Line2D([0], [0], color=self.colors[k], lw=3, label=k)
            for k in self.annotator_keys
        ]
        sleep_legend = [
            Line2D([0], [0], color="#8090C0", lw=6, alpha=0.3, label="NREM"),
            Line2D([0], [0], color="#A0C080", lw=6, alpha=0.3, label="Wake"),
            Line2D([0], [0], color="#C080A0", lw=6, alpha=0.3, label="REM"),
            Line2D([0], [0], color="#B0B0B0", lw=6, alpha=0.3, label="Intermediate"),
        ]
        self.fig.legend(
            handles=ripple_legend,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            title="Annotators",
            fontsize=11,
        )
        self.fig.legend(
            handles=sleep_legend,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.75),
            title="Sleep Stage",
            fontsize=11,
        )

    # Zoom
    def zoom_in(self):
        if self.window_sec > 2:
            self.window_sec //= 2
            self.update_slider()

    def zoom_out(self):
        max_len = len(self.lfp) / self.fs
        self.window_sec = min(self.window_sec * 2, max_len)
        self.update_slider()

    def update_slider(self):
        self.slider.setMaximum(max(int(len(self.lfp) / self.fs - self.window_sec), 0))
        self.update_plot()

    # Main plot update
    def update_plot(self):
        t0 = self.slider.value()
        start = int(t0 * self.fs)
        end = int((t0 + self.window_sec) * self.fs)
        t = np.arange(start, end) / self.fs

        self.ax_raw.clear()
        self.ax_filtered.clear()

        # Signals
        self.ax_raw.plot(t, self.lfp[start:end], color="#5B5F97", linewidth=1.2)
        self.ax_raw.set_ylabel("Signal", fontsize=13)
        self.ax_filtered.plot(
            t, self.filtered[start:end], color="#C06C84", linewidth=1.2
        )
        self.ax_filtered.set_ylabel("Band-pass", fontsize=13)

        # Sleep stage
        self.plot_sleep_background(start, end)
        # Ripples
        self.plot_ripples(start, end)

        self.ax_filtered.set_xlim(t0, t0 + self.window_sec)
        self.ax_filtered.set_xlabel("Time (s)", fontsize=13)
        self.ax_filtered.set_xticks(np.arange(t0, t0 + self.window_sec + 1, 1))

        self.ax_raw.set_title(f"Time: {t0:.2f}s", fontsize=15)
        self.ax_raw.grid(True)
        self.ax_filtered.grid(True)

        self.canvas.draw()

    # Sleep background
    def plot_sleep_background(self, start, end):
        scoring_full = np.repeat(self.scoring, self.fs)
        colors = {1: "#A0C080", 3: "#8090C0", 4: "#B0B0B0", 5: "#C080A0"}
        for state, color in colors.items():
            mask = scoring_full[start:end] == state
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]
            for _, g in groupby(enumerate(idx), lambda x: x[0] - x[1]):
                group = list(map(itemgetter(1), g))
                s = (start + group[0]) / self.fs
                e = (start + group[-1]) / self.fs
                self.ax_raw.axvspan(s, e, color=color, alpha=0.1)
                self.ax_filtered.axvspan(s, e, color=color, alpha=0.1)

    # Ripple bars
    def plot_ripples(self, start, end):
        y_min = np.min(self.lfp[start:end])
        y_max = np.max(self.lfp[start:end])
        height = y_max - y_min
        for i, key in enumerate(self.annotator_keys):
            events = self.events_dict[key]
            self._draw_bars(
                events,
                start / self.fs,
                end / self.fs,
                y_max + (0.05 + 0.07 * i) * height,
                self.colors[key],
            )
        self.ax_raw.set_ylim(y_min, y_max + 0.3 * height)

    def _draw_bars(self, events, start, end, y, color):
        for s, e in events:
            if e < start or s > end:
                continue
            self.ax_raw.hlines(y=y, xmin=s, xmax=e, colors=color, linewidth=5)
