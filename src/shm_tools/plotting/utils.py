"""Shared plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_or_show(
    fig: plt.Figure,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> None:
    """Save a figure to disk or display it interactively.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to save or show.
    save_path : str or Path or None, optional
        If provided, saves the figure to this path (including filename and
        extension, e.g. ``output/spectrum.png``).  The parent directory is
        created automatically if it does not exist.
        If ``None``, calls ``plt.show()`` instead.
    dpi : int, optional
        Resolution in dots per inch used when saving, by default 150.
    """
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[shm_tools] Figure saved → {out.resolve()}")
        plt.close(fig)
    else:
        plt.show()


def _make_fig_axes(
    n_panels: int,
    overlay: bool,
    figsize_single: tuple[float, float] = (10.0, 4.0),
    figsize_stack: tuple[float, float] = (10.0, 3.0),
    sharex: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create a figure with either one shared axis (overlay) or stacked axes.

    Parameters
    ----------
    n_panels : int
        Number of data series / panels needed.
    overlay : bool
        If ``True``, all series share a single axis (overlay mode).
        If ``False``, each series gets its own subplot (stacked mode).
    figsize_single : tuple[float, float]
        Figure size when ``overlay=True``.
    figsize_stack : tuple[float, float]
        Per-panel height when ``overlay=False``; total height is
        ``n_panels * figsize_stack[1]``.
    sharex : bool
        Whether stacked subplots share the x-axis, by default ``True``.

    Returns
    -------
    fig : plt.Figure
    axes : list[plt.Axes]
        Length-1 list in overlay mode; length-``n_panels`` in stacked mode.
    """
    if overlay or n_panels == 1:
        fig, ax = plt.subplots(figsize=figsize_single)
        return fig, [ax]

    total_height = figsize_stack[1] * n_panels
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(figsize_stack[0], total_height),
        sharex=sharex,
    )
    return fig, list(axes)