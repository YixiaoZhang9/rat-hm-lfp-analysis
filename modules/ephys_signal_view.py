import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class SignalPlotViewer(QWidget):
    def __init__(self, data_dict, fs, window_sec=5):
        super().__init__()

        self.data_dict = data_dict
        self.fs = fs
        self.window_sec = window_sec

        self.channel_names = list(data_dict.keys())
        self.num_channels = len(self.channel_names)
        self.total_len = len(next(iter(data_dict.values())))
        self.win_len = int(window_sec * fs)
        self.start_idx = 0
        # Create a full time array in seconds
        self.time = np.arange(self.total_len) / fs

        # Event regions
        self.event_regions = []  # store LinearRegionItem objects
        self.event_intervals = []  # store event sample indices

        # Amplitude scaling
        self.amplitude_scale = 1.0

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Multichannel Signal Viewer")
        self.resize(1200, 800)

        # PyQtGraph background/foreground
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Plot widget configuration
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)
        self.plot.setMenuEnabled(True)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.enableAutoRange("y", False)
        self.plot.setLabel("bottom", "Time", units="s")

        axis = self.plot.getAxis("bottom")
        axis.enableAutoSIPrefix(False)
        axis.setStyle(tickTextOffset=10, autoReduceTextSpace=False)
        axis.setTickSpacing(major=1, minor=0.5)

        # Curves for each channel
        self.curves = []
        self.offset_step = 10
        self.offsets = np.arange(self.num_channels) * self.offset_step

        self.colors = [
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

        for i in range(self.num_channels):
            color = self.colors[i % len(self.colors)]
            # Enable downsampling and clipToView for performance
            curve = self.plot.plot(
                pen=pg.mkPen(color, width=1.5), downsample=10, clipToView=True
            )
            self.curves.append(curve)

        # Y-axis ticks with channel names
        ticks = [
            (offset, name) for offset, name in zip(self.offsets, self.channel_names)
        ]
        axis = self.plot.getAxis("left")
        axis.setTicks([ticks])

        main_layout.addWidget(self.plot)

        # Slider configuration
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.total_len - self.win_len)
        self.slider.setSingleStep(int(self.win_len / 10))
        # Update plot only on slider release to avoid UI freeze
        self.slider.sliderReleased.connect(self.update_plot)
        slider_layout.addWidget(self.slider)
        main_layout.addLayout(slider_layout)

        # Control buttons
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(30)

        button_style = """
            QPushButton {
                min-width: 150px;
                max-width: 150px;
                padding: 5px;
                font-weight: bold;
            }
        """

        # Navigation buttons
        self.btn_left = QPushButton("◀")
        self.btn_right = QPushButton("▶")

        # Time label
        self.label_time = QLabel("Time: 0.00 s")
        self.label_time.setAlignment(Qt.AlignCenter)
        self.label_time.setStyleSheet(
            """
            QLabel {
                color: black;
                font-weight: bold;
                font-size: 12pt;
                min-width: 60px;
                background: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 3px;
            }
        """
        )

        # Zoom buttons
        self.btn_zoom_in_time = QPushButton("+ Time")
        self.btn_zoom_out_time = QPushButton("- Time")
        self.btn_zoom_in_amp = QPushButton("+ Amp")
        self.btn_zoom_out_amp = QPushButton("- Amp")

        # Apply style and connect signals
        for btn in [
            self.btn_left,
            self.btn_right,
            self.btn_zoom_in_time,
            self.btn_zoom_out_time,
            self.btn_zoom_in_amp,
            self.btn_zoom_out_amp,
        ]:
            btn.setStyleSheet(button_style)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btn_left.clicked.connect(self.move_left)
        self.btn_right.clicked.connect(self.move_right)
        self.btn_zoom_in_time.clicked.connect(self.zoom_in_time)
        self.btn_zoom_out_time.clicked.connect(self.zoom_out_time)
        self.btn_zoom_in_amp.clicked.connect(self.zoom_in_amplitude)
        self.btn_zoom_out_amp.clicked.connect(self.zoom_out_amplitude)

        # Add controls to layout
        controls_layout.addWidget(self.btn_left)
        controls_layout.addWidget(self.label_time)
        controls_layout.addWidget(self.btn_right)
        controls_layout.addSpacing(50)
        controls_layout.addWidget(self.btn_zoom_in_time)
        controls_layout.addWidget(self.btn_zoom_out_time)
        controls_layout.addSpacing(50)
        controls_layout.addWidget(self.btn_zoom_in_amp)
        controls_layout.addWidget(self.btn_zoom_out_amp)

        main_layout.addLayout(controls_layout)

        # Initial plot update
        self.update_plot()

    def update_plot(self):
        """Update the plot with the current window"""

        self.start_idx = self.slider.value()
        end_idx = min(self.start_idx + self.win_len, self.total_len)
        t_window = self.time[self.start_idx : end_idx]
        self.label_time.setText(f"Time: {t_window[0]:.2f} - {t_window[-1]:.2f} s")
        for i, ch in enumerate(self.channel_names):
            data = self.data_dict[ch][self.start_idx : end_idx]
            ptp = np.ptp(data) or 1
            scaled_data = (
                data / ptp
            ) * self.offset_step * 0.8 * self.amplitude_scale + self.offsets[i]
            self.curves[i].setData(t_window, scaled_data)

        self.plot.setXRange(t_window[0], t_window[-1], padding=0)
        self.plot.setYRange(
            -self.offset_step, self.offsets[-1] + self.offset_step, padding=0.1
        )

        # Update event regions safely
        self.update_event_regions()

    def move_left(self):
        new_val = max(self.slider.value() - int(self.win_len / 10), 0)
        self.slider.setValue(new_val)
        self.update_plot()

    def move_right(self):
        new_val = min(
            self.slider.value() + int(self.win_len / 10), self.total_len - self.win_len
        )
        self.slider.setValue(new_val)
        self.update_plot()

    def zoom_in_time(self):
        new_win_len = max(int(self.win_len * 0.8), int(self.fs))
        if new_win_len != self.win_len:
            self.win_len = new_win_len
            self.slider.setMaximum(self.total_len - self.win_len)
            if self.slider.value() > self.total_len - self.win_len:
                self.slider.setValue(self.total_len - self.win_len)
            self.update_plot()

    def zoom_out_time(self):
        new_win_len = min(int(self.win_len * 1.25), self.total_len)
        if new_win_len != self.win_len:
            self.win_len = new_win_len
            self.slider.setMaximum(self.total_len - self.win_len)
            self.update_plot()

    def zoom_in_amplitude(self):
        self.amplitude_scale *= 1.2
        self.update_plot()

    def zoom_out_amplitude(self):
        self.amplitude_scale /= 1.2
        self.update_plot()

    def set_event_intervals(self, event_intervals):
        """
        Set event intervals.
        event_intervals: list of tuples [(start_idx, end_idx), ...]
                         or numpy array of shape (N, 2)
        """
        # Convert to integer tuples
        if isinstance(event_intervals, np.ndarray):
            event_intervals = [tuple(e) for e in event_intervals]

        self.event_intervals = event_intervals

    def update_event_regions(self):
        """
        Draw only events that are currently visible in the window.
        Safe to call even if event_intervals is empty or not set.
        """
        # Remove old regions
        for region in getattr(self, "event_regions", []):
            self.plot.removeItem(region)
        self.event_regions = []

        # Exit if no events
        if not hasattr(self, "event_intervals") or len(self.event_intervals) == 0:
            return

        win_start = self.start_idx
        win_end = min(self.start_idx + self.win_len, self.total_len)

        for start, end in self.event_intervals:
            if end < win_start or start > win_end:
                continue
            t_start = self.time[max(start, win_start)]
            t_end = self.time[min(end, win_end)]
            region = pg.LinearRegionItem(
                values=(t_start, t_end), brush=pg.mkBrush(255, 0, 0, 50), movable=False
            )
            region.setZValue(-10)
            self.plot.addItem(region)
            self.event_regions.append(region)
