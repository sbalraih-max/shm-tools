"""High-level orchestration API for shm_tools.

This module provides ``run_analysis()`` — a single entry point that
wires loaders → processing → indicators → plotting into one call,
matching the workflow of the original scripts while keeping every
step individually importable for custom pipelines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from shm_tools.config.models import ShmConfig
from shm_tools.loaders.wide import load_wide
from shm_tools.loaders.long import load_long
from shm_tools.processing.spectra import compute_spectrum, extract_mode_shapes
from shm_tools.processing.peaks import find_peak_freqs
from shm_tools.fdd.core import compute_fdd
from shm_tools.indicators.damage import damage_indicators
from shm_tools.plotting.time_history import plot_time_history
from shm_tools.plotting.spectra import plot_spectrum_comparison
from shm_tools.plotting.modal import plot_mode_shapes, plot_damage_indicator_diff


def run_analysis(
    healthy_csv: str | Path,
    damaged_csv: str | Path,
    cfg: ShmConfig,
    output_dir: str | Path | None = None,
    save: bool = False,
    overlay: bool = False,
) -> dict[str, object]:
    """Run the full SHM analysis pipeline from CSV files to plots.

    Steps
    -----
    1. Load healthy and damaged CSV files (wide or long format).
    2. Compute spectra and detect natural frequencies.
    3. Extract mode shapes using the configured method.
    4. Compute modal damage indicators (MD, MS, MC).
    5. Generate and optionally save all plots.

    Parameters
    ----------
    healthy_csv : str or Path
        Path to the healthy-state CSV file.
    damaged_csv : str or Path
        Path to the damaged-state CSV file.
    cfg : ShmConfig
        Analysis configuration dataclass.
    output_dir : str or Path or None, optional
        Directory for saved figures.  Required when ``save=True``.
        Created automatically if it does not exist.
    save : bool, optional
        If ``True``, save all figures to *output_dir* instead of displaying.
    overlay : bool, optional
        If ``True``, use overlay mode for time-history and spectrum plots.

    Returns
    -------
    dict
        Dictionary containing computed results with keys:

        - ``"coords"`` : NDArray — sensor coordinates
        - ``"peak_freqs_healthy"`` : NDArray — natural frequencies (healthy)
        - ``"peak_freqs_damaged"`` : NDArray — natural frequencies (damaged)
        - ``"mode_shapes_healthy"`` : NDArray — mode shape matrix (healthy)
        - ``"mode_shapes_damaged"`` : NDArray — mode shape matrix (damaged)
        - ``"damage_indicators"`` : dict — MD / MS / MC results

    Raises
    ------
    ValueError
        If ``save=True`` but ``output_dir`` is not provided.
    """
    if save and output_dir is None:
        raise ValueError("output_dir must be provided when save=True.")

    out = Path(output_dir) if output_dir else Path("output")
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if cfg.data_format == "wide":
        sensor_cols_h, coords, signals_h = load_wide(healthy_csv, cfg)
        sensor_cols_d, _,      signals_d = load_wide(damaged_csv, cfg)
    else:
        nodes_h, coords, signals_h = load_long(healthy_csv, cfg)
        nodes_d, _,      signals_d = load_long(damaged_csv, cfg)

    fs = cfg.fs

    # ------------------------------------------------------------------
    # 2. Spectra
    # ------------------------------------------------------------------
    if cfg.method == "fdd":
        freqs_h, sv_h, _ = compute_fdd(signals_h, fs, cfg)
        freqs_d, sv_d, _ = compute_fdd(signals_d, fs, cfg)
        amp_h = sv_h[:, 0]
        amp_d = sv_d[:, 0]
    else:
        freqs_h, amp_h = compute_spectrum(signals_h, fs, cfg)
        freqs_d, amp_d = compute_spectrum(signals_d, fs, cfg)

    peak_freqs_h = find_peak_freqs(
        freqs_h, amp_h,
        n_peaks=cfg.n_modes,
        freq_min=cfg.freq_min,
        freq_max=cfg.freq_max,
        prominence=cfg.peak_prominence,
        min_distance_hz=cfg.peak_min_distance_hz,
    )
    peak_freqs_d = find_peak_freqs(
        freqs_d, amp_d,
        n_peaks=cfg.n_modes,
        freq_min=cfg.freq_min,
        freq_max=cfg.freq_max,
        prominence=cfg.peak_prominence,
        min_distance_hz=cfg.peak_min_distance_hz,
    )

    # ------------------------------------------------------------------
    # 3. Mode shapes
    # ------------------------------------------------------------------
    peak_freqs_h, shapes_h = extract_mode_shapes(
        signals_h, fs, cfg, peak_freqs=peak_freqs_h
    )
    peak_freqs_d, shapes_d = extract_mode_shapes(
        signals_d, fs, cfg, peak_freqs=peak_freqs_d
    )

    # Align number of modes (take minimum available)
    n_modes = min(shapes_h.shape[1], shapes_d.shape[1])
    shapes_h = shapes_h[:, :n_modes]
    shapes_d = shapes_d[:, :n_modes]

    # ------------------------------------------------------------------
    # 4. Damage indicators
    # ------------------------------------------------------------------
    di = damage_indicators(shapes_h, shapes_d, coords)

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    _save = lambda name: (out / name) if save else None

    # Time history
    plot_time_history(
        time=np.arange(signals_h.shape[0]) / fs,
        signals=signals_d,
        labels=[f"Ch{i+1}" for i in range(signals_d.shape[1])],
        healthy_signals=signals_h,
        overlay=overlay,
        title="Acceleration — Healthy vs Damaged",
        save_path=_save("01_time_history.png"),
    )

    # Spectrum comparison
    ylabel = "Singular Value" if cfg.method == "fdd" else (
        "PSD" if cfg.method == "welch" else "Amplitude"
    )
    plot_spectrum_comparison(
        freqs_h, amp_h, freqs_d, amp_d,
        peak_freqs_healthy=peak_freqs_h,
        peak_freqs_damaged=peak_freqs_d,
        method=cfg.method,
        overlay=overlay,
        ylabel=ylabel,
        title="Spectrum — Healthy vs Damaged",
        save_path=_save("02_spectrum.png"),
    )

    # Mode shapes
    if n_modes > 0:
        plot_mode_shapes(
            coords=coords,
            healthy_shapes=shapes_h,
            damaged_shapes=shapes_d,
            title="Mode Shape Comparison",
            save_path=_save("03_mode_shapes.png"),
        )

        # Damage indicator diff plots — one per mode
        for m in range(n_modes):
            plot_damage_indicator_diff(
                coords=coords,
                di_result=di,
                mode_index=m,
                title=f"Damage Indicators — Mode {m+1}",
                save_path=_save(f"04_damage_indicators_mode{m+1}.png"),
            )

    return {
        "coords": coords,
        "peak_freqs_healthy": peak_freqs_h,
        "peak_freqs_damaged": peak_freqs_d,
        "mode_shapes_healthy": shapes_h,
        "mode_shapes_damaged": shapes_d,
        "damage_indicators": di,
    }