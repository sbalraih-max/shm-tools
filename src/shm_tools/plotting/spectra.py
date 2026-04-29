"""Spectral plot helpers — FFT, Welch PSD, and FDD singular value plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from shm_tools.plotting.utils import save_or_show, _make_fig_axes


def plot_spectrum(
    freqs: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
    peak_freqs: NDArray[np.floating] | None = None,
    label: str = "Spectrum",
    title: str = "Frequency Spectrum",
    xlabel: str = "Frequency (Hz)",
    ylabel: str = "Amplitude",
    method: str = "fft",
    save_path: str | Path | None = None,
    dpi: int = 150,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a single frequency spectrum with optional peak markers.

    Parameters
    ----------
    freqs : NDArray[np.floating]
        Frequency axis in Hz, shape ``(M,)``.
    amplitudes : NDArray[np.floating]
        Spectral amplitude or PSD, shape ``(M,)``.
    peak_freqs : NDArray[np.floating] or None, optional
        Peak frequencies to mark with vertical dashed lines.
    label : str, optional
        Legend label for the spectrum curve.
    title : str, optional
        Figure title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    method : str, optional
        Method name shown in the title suffix (``"fft"``, ``"welch"``,
        ``"fdd"``).
    save_path : str or Path or None, optional
        If provided, saves the figure instead of displaying it.
    dpi : int, optional
        Resolution when saving, by default 150.
    ax : plt.Axes or None, optional
        If provided, draws onto this existing axis (useful for multi-panel
        figures). A new figure is created if ``None``.

    Returns
    -------
    plt.Figure
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    ax.plot(freqs, amplitudes, linewidth=0.9, label=label)

    if peak_freqs is not None and len(peak_freqs) > 0:
        for i, pf in enumerate(peak_freqs):
            ax.axvline(
                pf,
                color="red",
                linestyle="--",
                linewidth=0.9,
                alpha=0.8,
                label=f"Peak {i+1}: {pf:.2f} Hz",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} [{method.upper()}]")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    if created_fig:
        save_or_show(fig, save_path=save_path, dpi=dpi)

    return fig


def plot_spectrum_comparison(
    freqs_healthy: NDArray[np.floating],
    amplitudes_healthy: NDArray[np.floating],
    freqs_damaged: NDArray[np.floating],
    amplitudes_damaged: NDArray[np.floating],
    peak_freqs_healthy: NDArray[np.floating] | None = None,
    peak_freqs_damaged: NDArray[np.floating] | None = None,
    method: str = "fft",
    overlay: bool = True,
    title: str = "Spectrum Comparison",
    ylabel: str = "Amplitude",
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot healthy vs damaged spectra side-by-side or overlaid.

    Parameters
    ----------
    freqs_healthy : NDArray[np.floating]
        Frequency axis for healthy state.
    amplitudes_healthy : NDArray[np.floating]
        Amplitude/PSD for healthy state.
    freqs_damaged : NDArray[np.floating]
        Frequency axis for damaged state.
    amplitudes_damaged : NDArray[np.floating]
        Amplitude/PSD for damaged state.
    peak_freqs_healthy : NDArray[np.floating] or None, optional
        Peak markers for healthy state.
    peak_freqs_damaged : NDArray[np.floating] or None, optional
        Peak markers for damaged state.
    method : str, optional
        Method name for title label.
    overlay : bool, optional
        If ``True`` (default), both spectra share one axis.
        If ``False``, shown in stacked subplots.
    title : str, optional
        Figure title.
    ylabel : str, optional
        Y-axis label.
    save_path : str or Path or None, optional
        Save path or ``None`` to display.
    dpi : int, optional
        Resolution when saving.

    Returns
    -------
    plt.Figure
    """
    if overlay:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(freqs_healthy, amplitudes_healthy,
                linewidth=0.9, color="steelblue", label="Healthy")
        ax.plot(freqs_damaged, amplitudes_damaged,
                linewidth=0.9, color="tomato", label="Damaged", alpha=0.85)
        _add_peak_vlines(ax, peak_freqs_healthy, color="steelblue",
                         label_prefix="H")
        _add_peak_vlines(ax, peak_freqs_damaged, color="tomato",
                         label_prefix="D")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} [{method.upper()}]")
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        fig.tight_layout()
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax1.plot(freqs_healthy, amplitudes_healthy,
                 linewidth=0.9, color="steelblue")
        _add_peak_vlines(ax1, peak_freqs_healthy, color="red",
                         label_prefix="f")
        ax1.set_ylabel(ylabel)
        ax1.set_title(f"Healthy [{method.upper()}]")
        ax1.legend(fontsize=8)
        ax1.grid(True, linewidth=0.4, alpha=0.5)

        ax2.plot(freqs_damaged, amplitudes_damaged,
                 linewidth=0.9, color="tomato")
        _add_peak_vlines(ax2, peak_freqs_damaged, color="red",
                         label_prefix="f")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel(ylabel)
        ax2.set_title(f"Damaged [{method.upper()}]")
        ax2.legend(fontsize=8)
        ax2.grid(True, linewidth=0.4, alpha=0.5)

        fig.suptitle(title, fontsize=12)
        fig.tight_layout()

    save_or_show(fig, save_path=save_path, dpi=dpi)
    return fig


def _add_peak_vlines(
    ax: plt.Axes,
    peak_freqs: NDArray[np.floating] | None,
    color: str = "red",
    label_prefix: str = "f",
) -> None:
    """Add vertical dashed lines at peak frequencies to an existing axis.

    Parameters
    ----------
    ax : plt.Axes
        Target axis.
    peak_freqs : NDArray[np.floating] or None
        Peak frequencies in Hz. Does nothing if ``None`` or empty.
    color : str, optional
        Line colour, by default ``"red"``.
    label_prefix : str, optional
        Prefix for the legend label, e.g. ``"H"`` → ``"H1: 3.50 Hz"``.
    """
    if peak_freqs is None or len(peak_freqs) == 0:
        return
    for i, pf in enumerate(peak_freqs):
        ax.axvline(
            pf,
            color=color,
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
            label=f"{label_prefix}{i+1}: {pf:.2f} Hz",
        )