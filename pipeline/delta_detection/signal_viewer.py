import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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
from scipy.signal import butter, sosfiltfilt


def filter_lfp(lfp, fs, band):
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, lfp)


class DeltaViewer(QWidget):
    """
    Final clean ripple visualization tool (publication-style).

    Features:
    - Raw + filtered signal (separate panels)
    - Ripple detection (3 methods → top short bars)
    - Sleep stage background (light colors)
    - Zoom + slider navigation
    """

    def __init__(self, lfp, scoring, fs, event_sets=None):

        super().__init__()

        self.lfp = lfp
        self.scoring = scoring
        self.fs = fs

        self.event_sets = event_sets if event_sets is not None else {}

        self.method_colors = {
            "Threshold": "#1f77b4",
        }

        self.default_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        self.filtered = filter_lfp(lfp, fs, [0.5, 4])

        self.window_sec = 30

        self.init_ui()

    # =========================
    # UI SETUP
    # =========================
    def init_ui(self):

        self.setWindowTitle("Ripple Viewer")

        layout = QVBoxLayout()

        self.fig = Figure(figsize=(15, 7))
        self.canvas = FigureCanvas(self.fig)

        self.ax_raw = self.fig.add_subplot(211)
        self.ax_filtered = self.fig.add_subplot(212, sharex=self.ax_raw)

        layout.addWidget(self.canvas)

        # --- Controls ---
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

    # =========================
    # STATIC LEGENDS
    # =========================
    def add_legends(self):

        delta_legend = []

        for i, method in enumerate(self.event_sets.keys()):
            color = self.method_colors.get(
                method, self.default_colors[i % len(self.default_colors)]
            )

            delta_legend.append(Line2D([0], [0], color=color, lw=3, label=method))

        sleep_legend = [
            Line2D([0], [0], color="#8090C0", lw=6, alpha=0.3, label="NREM"),
            Line2D([0], [0], color="#A0C080", lw=6, alpha=0.3, label="Wake"),
            Line2D([0], [0], color="#C080A0", lw=6, alpha=0.3, label="REM"),
            Line2D([0], [0], color="#B0B0B0", lw=6, alpha=0.3, label="Intermediate"),
        ]

        self.fig.legend(
            handles=delta_legend,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            title="Delta Detection",
            fontsize=11,
        )

        self.fig.legend(
            handles=sleep_legend,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.75),
            title="Sleep Stage",
            fontsize=11,
        )

    # =========================
    # ZOOM
    # =========================
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

    # =========================
    # MAIN PLOT
    # =========================
    def update_plot(self):

        t0 = self.slider.value()

        start = int(t0 * self.fs)
        end = int((t0 + self.window_sec) * self.fs)

        t = np.arange(start, end) / self.fs

        self.ax_raw.clear()
        self.ax_filtered.clear()

        # --- Plot signals ---
        self.ax_raw.plot(t, self.lfp[start:end], color="#5B5F97", linewidth=1.2)
        self.ax_raw.set_ylabel("0.3_30_Hz_filtered_signal", fontsize=13)

        self.ax_filtered.plot(
            t, self.filtered[start:end], color="#C06C84", linewidth=1.2
        )
        self.ax_filtered.set_ylabel("0.5_4_Hz_filtered_signal", fontsize=13)

        # --- Sleep background ---
        self.plot_sleep_background(start, end)

        # --- Ripple bars ---
        self.plot_deltas(start, end)

        # --- Axis ---
        self.ax_filtered.set_xlim(t0, t0 + self.window_sec)
        self.ax_filtered.set_xlabel("Time (s)", fontsize=13)
        self.ax_filtered.set_xticks(np.arange(t0, t0 + self.window_sec + 1, 1))

        self.ax_raw.set_title(f"Time: {t0:.2f}s", fontsize=15)

        self.ax_raw.grid(True)
        self.ax_filtered.grid(True)

        self.canvas.draw()

    # =========================
    # SLEEP BACKGROUND
    # =========================
    def plot_sleep_background(self, start, end):

        scoring_full = np.repeat(self.scoring, self.fs)

        colors = {1: "#A0C080", 3: "#8090C0", 4: "#B0B0B0", 5: "#C080A0"}

        for state, color in colors.items():
            mask = scoring_full[start:end] == state
            if not np.any(mask):
                continue

            idx = np.where(mask)[0]

            from itertools import groupby
            from operator import itemgetter

            for _, g in groupby(enumerate(idx), lambda x: x[0] - x[1]):
                group = list(map(itemgetter(1), g))

                s = (start + group[0]) / self.fs
                e = (start + group[-1]) / self.fs

                self.ax_raw.axvspan(s, e, color=color, alpha=0.1)
                self.ax_filtered.axvspan(s, e, color=color, alpha=0.1)

    # =========================
    # RIPPLE SHORT BARS
    # =========================
    def plot_ripples(self, start, end):

        y_min = np.min(self.lfp[start:end])
        y_max = np.max(self.lfp[start:end])
        height = y_max - y_min

        # three layers

        self.ax_raw.set_ylim(y_min, y_max + 0.3 * height)

    def _draw_bars(self, events, start, end, y, color):

        for s, e in events:
            if e < start or s > end:
                continue

            self.ax_raw.hlines(
                y=y, xmin=s / self.fs, xmax=e / self.fs, colors=color, linewidth=5
            )

    def plot_deltas(self, start, end):

        y_min = np.min(self.lfp[start:end])
        y_max = np.max(self.lfp[start:end])
        height = y_max - y_min

        spacing = 0.07 * height
        base_y = y_max + 0.05 * height

        for i, (method, events) in enumerate(self.event_sets.items()):
            color = self.method_colors.get(
                method, self.default_colors[i % len(self.default_colors)]
            )

            y = base_y + i * spacing

            self._draw_bars(events, start, end, y, color)

        extra_height = max(len(self.event_sets), 1) * spacing
        self.ax_raw.set_ylim(y_min, y_max + extra_height + 0.15 * height)
