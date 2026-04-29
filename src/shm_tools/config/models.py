"""ShmConfig dataclass — single source of truth for all pipeline settings."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ShmConfig:
    """Configuration for all shm_tools pipelines.

    All field names mirror the original CONFIG dict keys for easy migration.

    Parameters
    ----------
    data_format : str
        ``"wide"`` for sensor-logger CSVs, ``"long"`` for Abaqus output.
    healthy_csv : str or None
        Path to the healthy-condition CSV.  ``None`` to skip.
    damaged_csv : str or None
        Path to the damaged-condition CSV.  ``None`` to skip.
    fs : float
        Sampling frequency in Hz.
    start : float or None
        Start of time window (seconds).  ``None`` = beginning of record.
    end : float or None
        End of time window (seconds).  ``None`` = end of record.
    sensor_prefix : str
        Accelerometer column prefix in wide format (e.g. ``"BS"``).
    sensor_plot : list[int] or None
        Sensor numbers to include.  ``None`` = all detected.
    directions : list[str] or None
        Acceleration directions in long format (e.g. ``["AT2"]``).
        ``None`` = all found.
    node_plot : list[int] or None
        Node labels in long format.  ``None`` = all nodes.
    node_plot_by_x : list[float] or None
        Select nodes closest to these X-coordinates (metres).
        Ignored if ``node_plot`` is already set.
    x_tolerance : float
        Warn if nearest node exceeds this distance from target X (metres).
    method : str
        Spectral method: ``"fft"``, ``"welch"``, or ``"fdd"``.
    welch_nperseg : int or None
        Segment length for Welch/FDD CPSD.  ``None`` = auto.
    welch_overlap : float
        Overlap fraction for Welch/FDD CPSD (0.0 to 0.9).
    apply_window : bool
        Apply Hanning window before FFT (ignored for Welch/FDD).
    n_modes : int
        Number of modes to extract.
    peak_prominence : float
        Minimum peak prominence as fraction of spectrum maximum.
    peak_min_distance_hz : float
        Minimum separation between detected peaks in Hz.
    freq_min : float or None
        Lower frequency search bound (Hz).
    freq_max : float or None
        Upper frequency search bound (Hz).
    di_modes : list[int] or None
        1-based mode indices for damage-indicator plots.  ``None`` = all.
    plot_selection : list[str] or None
        Indicators to plot: ``"displacement"``, ``"slope"``, ``"curvature"``.
        ``None`` = all three.
    plot_comparison : bool
        Produce healthy-vs-damaged comparison figures.
    plot_di : bool
        Produce absolute damage-indicator figures.
    plot_type : str
        Plot style: ``"line"``, ``"bar"``, or ``"points"``.
    overlay : bool
        Overlay healthy and damaged on the same axes.
    output_dir : str
        Directory for saved figures (active when ``--save`` is used).
    fig_width : float
        Figure width in inches.
    fig_height : float
        Figure height per subplot in inches.
    dpi : int
        Figure resolution in dots per inch.
    line_width : float
        Line width for all plots.
    color_healthy : str
        Hex colour for the healthy condition.
    color_damaged : str
        Hex colour for the damaged condition.
    color_di : str
        Hex colour for damage-indicator bars/lines.
    label_healthy : str
        Legend label for the healthy condition.
    label_damaged : str
        Legend label for the damaged condition.
    """

    # ── Data source ──────────────────────────────────────────────────────────
    data_format:  str           = "wide"
    healthy_csv: Optional[str] = None
    damaged_csv: Optional[str] = None

    # ── Time window ──────────────────────────────────────────────────────────
    fs: float = 200.0
    start: Optional[float] = None
    end:   Optional[float] = None

    # ── Wide-format ──────────────────────────────────────────────────────────
    sensor_prefix: str                 = "BS"
    sensor_plot:   Optional[list[int]] = None

    # ── Long-format ──────────────────────────────────────────────────────────
    directions:     Optional[list[str]]   = None
    node_plot:      Optional[list[int]]   = None
    node_plot_by_x: Optional[list[float]] = None
    x_tolerance:    float                 = 0.01

    # ── Column-name candidates ───────────────────────────────────────────────
    timecol_candidates: list[str] = field(default_factory=lambda: [
        "step time", "time", "time step", "timestep", "steptime",
        "timestamp", "elapsed time", "elapsed", "t", "time (s)", "times", "",
    ])
    nodecol_candidates: list[str] = field(default_factory=lambda: [
        "node label", "node", "nodelabel", "nodal label",
        "node id", "nodeid", "element", "label", "",
    ])
    xcol_candidates: list[str] = field(default_factory=lambda: [
        "x", "x-coord", "xcoord", "x_coord", "x coordinate", "x (m)", "",
    ])
    accelcol_candidates: dict[str, list[str]] = field(default_factory=lambda: {
        "AT1": ["at-at1", "at1", "a1", "accel1", "acc1", "ax"],
        "AT2": ["at-at2", "at2", "a2", "accel2", "acc2", "ay"],
        "AT3": ["at-at3", "at3", "a3", "accel3", "acc3", "az"],
    })

    # ── Signal processing ────────────────────────────────────────────────────
    method:          str            = "fdd"
    welch_nperseg:   Optional[int]  = 1024
    welch_overlap:   float          = 0.5
    apply_window:    bool           = True
    n_modes:         int            = 3
    peak_prominence: float          = 0.05
    peak_min_distance_hz: float     = 0.5
    freq_min:        Optional[float] = None
    freq_max:        Optional[float] = None

    # ── Damage indicators ────────────────────────────────────────────────────
    di_modes: Optional[list[int]] = None

    # ── Plot selection ───────────────────────────────────────────────────────
    plot_selection:  Optional[list[str]] = None
    plot_comparison: bool  = True
    plot_di:         bool  = True
    plot_type:       str   = "line"
    overlay:         bool  = False

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str   = "output/plots"
    fig_width:  float = 12.0
    fig_height: float = 4.0
    dpi:        int   = 150
    line_width: float = 1.0

    # ── Colours & labels ─────────────────────────────────────────────────────
    color_healthy: str = "#1a6faf"
    color_damaged: str = "#c0392b"
    color_di:      str = "#2ecc71"
    label_healthy: str = "Healthy"
    label_damaged: str = "Damaged"
