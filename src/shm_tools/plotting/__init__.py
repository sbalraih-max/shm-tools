"""Plotting subpackage — time history, spectra, and modal figures."""

from shm_tools.plotting.time_history import plot_time_history
from shm_tools.plotting.spectra import plot_spectrum, plot_spectrum_comparison
from shm_tools.plotting.modal import (
    plot_mode_shapes,
    plot_damage_indicators,
    plot_damage_indicator_diff,
)
from shm_tools.plotting.utils import save_or_show

__all__ = [
    "plot_time_history",
    "plot_spectrum",
    "plot_spectrum_comparison",
    "plot_mode_shapes",
    "plot_damage_indicators",
    "plot_damage_indicator_diff",
    "save_or_show",
]