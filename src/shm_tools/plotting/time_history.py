"""Acceleration time-history plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from shm_tools.plotting.utils import save_or_show, _make_fig_axes


def plot_time_history(
    time: NDArray[np.floating],
    signals: NDArray[np.floating],
    labels: list[str],
    title: str = "Acceleration – Time History",
    xlabel: str = "Time (s)",
    ylabel: str = "Acceleration (m/s²)",
    overlay: bool = False,
    healthy_signals: NDArray[np.floating] | None = None,
    healthy_label_suffix: str = " (Healthy)",
    damaged_label_suffix: str = " (Damaged)",
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot acceleration time histories for one or two states.

    Supports overlay mode (all channels on one axis) and stacked mode
    (one subplot per channel).  Optionally overlays a healthy baseline
    against a damaged/current state.

    Parameters
    ----------
    time : NDArray[np.floating]
        1-D time axis in seconds, shape ``(n_samples,)``.
    signals : NDArray[np.floating]
        2-D array of shape ``(n_samples, n_channels)``.  Plotted as the
        primary (or damaged) state.
    labels : list[str]
        Channel labels, length ``n_channels``.
    title : str, optional
        Figure title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    overlay : bool, optional
        If ``True``, all channels share one axis.  If ``False`` (default),
        each channel gets its own subplot.
    healthy_signals : NDArray[np.floating] or None, optional
        If provided, plotted alongside *signals* as the healthy baseline.
        Must have the same shape as *signals*.
    healthy_label_suffix : str, optional
        Suffix appended to each label for the healthy state.
    damaged_label_suffix : str, optional
        Suffix appended to each label for the damaged/current state.
    save_path : str or Path or None, optional
        If provided, saves the figure instead of displaying it.
    dpi : int, optional
        Resolution when saving, by default 150.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    n_channels = signals.shape[1]
    compare = healthy_signals is not None

    fig, axes = _make_fig_axes(
        n_panels=n_channels,
        overlay=overlay,
        figsize_single=(12.0, 5.0),
        figsize_stack=(12.0, 3.0),
    )

    colors_damaged = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors_healthy = ["grey"] * n_channels

    for ch in range(n_channels):
        ax = axes[0] if overlay else axes[ch]
        lbl_d = labels[ch] + (damaged_label_suffix if compare else "")
        color_d = colors_damaged[ch % len(colors_damaged)]

        if compare:
            ax.plot(
                time,
                healthy_signals[:, ch],
                color=colors_healthy[ch % len(colors_healthy)],
                linewidth=0.8,
                alpha=0.7,
                label=labels[ch] + healthy_label_suffix,
            )

        ax.plot(
            time,
            signals[:, ch],
            color=color_d,
            linewidth=0.8,
            label=lbl_d,
        )

        if not overlay:
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_title(f"{title} — {labels[ch]}", fontsize=10)
            ax.grid(True, linewidth=0.4, alpha=0.5)

    if overlay:
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[0].set_title(title)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, linewidth=0.4, alpha=0.5)
    else:
        axes[-1].set_xlabel(xlabel)
        fig.suptitle(title, fontsize=12, y=1.01)

    fig.tight_layout()
    save_or_show(fig, save_path=save_path, dpi=dpi)
    return fig