"""
shm_tools — Structural Health Monitoring signal processing library.

Quick start
-----------
>>> from shm_tools import ShmConfig, run_analysis
>>> cfg = ShmConfig(method="fdd", data_format="wide", fs=200.0)
>>> results = run_analysis("data/healthy.csv", "data/damaged.csv", cfg, save=False)

Public API
----------
Configuration
    ShmConfig

High-level
    run_analysis

Loaders
    load_wide
    load_long
    resolve_nodes_by_x

Processing
    compute_spectrum
    find_peak_freqs
    extract_mode_shapes

FDD
    compute_fdd

Indicators
    damage_indicators
    spatial_diff

Plotting
    plot_time_history
    plot_spectrum
    plot_spectrum_comparison
    plot_mode_shapes
    plot_damage_indicators
    plot_damage_indicator_diff
"""

from shm_tools.config.models import ShmConfig
from shm_tools.api import run_analysis
from shm_tools.loaders.wide import load_wide
from shm_tools.loaders.long import load_long, resolve_nodes_by_x
from shm_tools.processing.spectra import compute_spectrum, extract_mode_shapes
from shm_tools.processing.peaks import find_peak_freqs
from shm_tools.fdd.core import compute_fdd
from shm_tools.indicators.damage import damage_indicators, spatial_diff
from shm_tools.plotting.time_history import plot_time_history
from shm_tools.plotting.spectra import plot_spectrum, plot_spectrum_comparison
from shm_tools.plotting.modal import (
    plot_mode_shapes,
    plot_damage_indicators,
    plot_damage_indicator_diff,
)

__all__ = [
    # config
    "ShmConfig",
    # high-level
    "run_analysis",
    # loaders
    "load_wide",
    "load_long",
    "resolve_nodes_by_x",
    # processing
    "compute_spectrum",
    "extract_mode_shapes",
    "find_peak_freqs",
    # fdd
    "compute_fdd",
    # indicators
    "damage_indicators",
    "spatial_diff",
    # plotting
    "plot_time_history",
    "plot_spectrum",
    "plot_spectrum_comparison",
    "plot_mode_shapes",
    "plot_damage_indicators",
    "plot_damage_indicator_diff",
]