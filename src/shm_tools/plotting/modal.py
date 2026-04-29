"""Mode-shape and damage-indicator plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from shm_tools.plotting.utils import save_or_show


def plot_mode_shapes(
    coords: NDArray[np.floating],
    healthy_shapes: NDArray[np.floating],
    damaged_shapes: NDArray[np.floating],
    mode_labels: list[str] | None = None,
    title: str = "Mode Shape Comparison",
    xlabel: str = "Position (m)",
    ylabel: str = "Normalised Amplitude",
    overlay: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot healthy vs damaged mode shapes for each mode.

    Parameters
    ----------
    coords : NDArray[np.floating]
        Sensor spatial coordinates in metres, shape ``(n_sensors,)``.
    healthy_shapes : NDArray[np.floating]
        Mode-shape matrix for healthy state, shape ``(n_sensors, n_modes)``.
    damaged_shapes : NDArray[np.floating]
        Mode-shape matrix for damaged state, shape ``(n_sensors, n_modes)``.
    mode_labels : list[str] or None, optional
        Labels for each mode. Auto-generated if ``None``.
    title : str, optional
        Figure title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    overlay : bool, optional
        If ``True`` (default), healthy and damaged overlay on one axis per
        mode.
    save_path : str or Path or None, optional
        Save path or ``None`` to display.
    dpi : int, optional
        Resolution when saving.

    Returns
    -------
    plt.Figure
    """
    n_modes = healthy_shapes.shape[1]
    if mode_labels is None:
        mode_labels = [f"Mode {i+1}" for i in range(n_modes)]

    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 4))
    if n_modes == 1:
        axes = [axes]

    for m, ax in enumerate(axes):
        ax.plot(
            coords,
            healthy_shapes[:, m],
            color="steelblue",
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Healthy",
        )
        ax.plot(
            coords,
            damaged_shapes[:, m],
            color="tomato",
            linewidth=1.5,
            linestyle="--",
            marker="s",
            markersize=4,
            label="Damaged",
        )
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(mode_labels[m], fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel if m == 0 else "", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_or_show(fig, save_path=save_path, dpi=dpi)
    return fig


def plot_damage_indicators(
    coords: NDArray[np.floating],
    di_result: dict[str, dict[str, NDArray[np.floating]]],
    mode_index: int = 0,
    indicators: list[str] | None = None,
    title: str = "Damage Indicators",
    xlabel: str = "Position (m)",
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot damage indicator differences along the structure.

    Parameters
    ----------
    coords : NDArray[np.floating]
        Sensor spatial coordinates in metres, shape ``(n_sensors,)``.
    di_result : dict
        Output of :func:`shm_tools.indicators.damage.damage_indicators`.
        Keys are ``"MD"``, ``"MS"``, ``"MC"``.
    mode_index : int, optional
        Which mode column to plot, by default 0 (first mode).
    indicators : list[str] or None, optional
        Subset of indicators to plot, e.g. ``["MD", "MC"]``.
        Defaults to all three: ``["MD", "MS", "MC"]``.
    title : str, optional
        Figure title.
    xlabel : str, optional
        X-axis label.
    save_path : str or Path or None, optional
        Save path or ``None`` to display.
    dpi : int, optional
        Resolution when saving.

    Returns
    -------
    plt.Figure
    """
    if indicators is None:
        indicators = ["MD", "MS", "MC"]

    indicator_labels = {
        "MD": "Modal Displacement DI",
        "MS": "Modal Slope DI",
        "MC": "Modal Curvature DI",
    }
    colors = {
        "MD": "steelblue",
        "MS": "darkorange",
        "MC": "mediumseagreen",
    }

    n = len(indicators)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ind in zip(axes, indicators):
        diff    = di_result[ind]["diff"][:, mode_index]
        healthy = di_result[ind]["healthy"][:, mode_index]
        damaged = di_result[ind]["damaged"][:, mode_index]

        # healthy and damaged overlaid
        ax.plot(coords, healthy,
                color="steelblue", linewidth=1.2,
                marker="o", markersize=4, label="Healthy")
        ax.plot(coords, damaged,
                color="tomato", linewidth=1.2, linestyle="--",
                marker="s", markersize=4, label="Damaged")

        # fill area between curves to highlight DI magnitude
        ax.fill_between(
            coords, healthy, damaged,
            alpha=0.15,
            color=colors.get(ind, "grey"),
        )

        ax.set_title(indicator_labels.get(ind, ind), fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_or_show(fig, save_path=save_path, dpi=dpi)
    return fig


def plot_damage_indicator_diff(
    coords: NDArray[np.floating],
    di_result: dict[str, dict[str, NDArray[np.floating]]],
    mode_index: int = 0,
    indicators: list[str] | None = None,
    title: str = "Damage Indicator — Absolute Difference",
    xlabel: str = "Position (m)",
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot the absolute difference (DI) for each indicator as a bar chart.

    This is the primary damage localisation plot — a spike in the bar
    chart at a given sensor position indicates likely damage at that location.

    Parameters
    ----------
    coords : NDArray[np.floating]
        Sensor spatial coordinates in metres, shape ``(n_sensors,)``.
    di_result : dict
        Output of :func:`shm_tools.indicators.damage.damage_indicators`.
    mode_index : int, optional
        Which mode column to plot, by default 0.
    indicators : list[str] or None, optional
        Subset of indicators to plot. Defaults to ``["MD", "MS", "MC"]``.
    title : str, optional
        Figure title.
    xlabel : str, optional
        X-axis label.
    save_path : str or Path or None, optional
        Save path or ``None`` to display.
    dpi : int, optional
        Resolution when saving.

    Returns
    -------
    plt.Figure
    """
    if indicators is None:
        indicators = ["MD", "MS", "MC"]

    indicator_labels = {
        "MD": "Modal Displacement |ΔMD|",
        "MS": "Modal Slope |ΔMS|",
        "MC": "Modal Curvature |ΔMC|",
    }
    colors = {
        "MD": "steelblue",
        "MS": "darkorange",
        "MC": "mediumseagreen",
    }

    n = len(indicators)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    bar_width = (coords[-1] - coords[0]) / (len(coords) * 1.5) if len(coords) > 1 else 0.05

    for ax, ind in zip(axes, indicators):
        diff = di_result[ind]["diff"][:, mode_index]
        ax.bar(
            coords,
            diff,
            width=bar_width,
            color=colors.get(ind, "grey"),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_title(indicator_labels.get(ind, ind), fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("|Δ|", fontsize=9)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_or_show(fig, save_path=save_path, dpi=dpi)
    return fig