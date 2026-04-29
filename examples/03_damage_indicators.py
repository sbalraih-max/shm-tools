"""
examples/03_damage_indicators.py
==================================
Damage detection — modal curvature, slope, and displacement indicators.

Workflow
--------
1. Load healthy and damaged CSVs
2. Compute spectra and extract mode shapes at peak frequencies
3. Compute MD / MS / MC damage indicators
4. Plot overlay (healthy vs damaged shapes) and DI chart

  PLOT_TYPE = "bar"  → one figure:  |Δ| bar chart per sensor position
  PLOT_TYPE = "line" → two figures: healthy/damaged overlay with shaded area
                                  + |Δ| line chart per sensor position

Usage
-----
    python examples/03_damage_indicators.py
    python examples/03_damage_indicators.py --method welch
    python examples/03_damage_indicators.py --method fdd --save
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Public API ────────────────────────────────────────────────────────────────
from shm_tools import ShmConfig, load_wide, load_long
from shm_tools.loaders.long import resolve_nodes_by_x
from shm_tools.processing.spectra import extract_mode_shapes
from shm_tools.indicators.damage import damage_indicators
from shm_tools.plotting.modal import (
    plot_mode_shapes,
    plot_damage_indicators,
    plot_damage_indicator_diff,
)
from shm_tools.plotting.utils import save_or_show

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    # --- Input files ---------------------------------------------------------
    "HEALTHY_CSV"   : DATA_DIR / "healthy_wide.csv",
    "DAMAGED_CSV"   : DATA_DIR / "damaged_wide.csv",

    # --- Data format ---------------------------------------------------------
    "DATA_FORMAT"   : "wide",         # "wide" | "long"

    # --- Sampling & time window ----------------------------------------------
    "FS"            : 200.0,          # sampling frequency [Hz]
    "TIME_START"    : None,
    "TIME_END"      : None,

    # --- Wide-format sensor selection ----------------------------------------
    "SENSOR_PREFIX" : "BS",
    "SENSOR_PLOT"   : None,           # list[int] e.g. [1, 2, 3] — None = all

    # --- Wide-format spatial coordinates [m] ---------------------------------
    # Required only for wide format — one value per sensor channel.
    # For long format the X-coordinates are read directly from the CSV.
    "SENSOR_COORDS" : [0.2, 0.4, 0.6, 0.8, 1.0],

    # --- Long-format node selection (pick ONE, set the other to None) --------
    "NODE_PLOT_BY_X": [0.2, 0.4, 0.6, 0.8, 1.0],        # select by X [m]  ← active
    "NODE_PLOT"     : None,                             # select by node ID e.g. [1, 2, 3]
    "X_TOLERANCE"   : 0.05,
    "DIRECTIONS"    : ["AT2"],

    # --- Spectral method & peak detection ------------------------------------
    "METHOD"        : "fdd",          # "fft" | "welch" | "fdd"
    "FREQ_MIN"      : 0.5,
    "FREQ_MAX"      : 50.0,
    "N_MODES"       : 3,
    "PEAK_PROMINENCE"     : 0.1,
    "PEAK_MIN_DISTANCE_HZ": 1.0,

    # --- Welch parameters (used when METHOD = "welch" or "fdd") --------------
    "WELCH_NPERSEG" : 1024,
    "WELCH_OVERLAP" : 0.5,

    # --- Damage indicators ---------------------------------------------------
    "INDICATORS"    : ["MD", "MC"],         # subset or all three ["MD", "MS", "MC"]
    "MODE_INDEX"    : 1,                    # which mode to highlight in DI plot
    "PLOT_TYPE"     : "line",               # "bar" | "line"

    # --- Output --------------------------------------------------------------
    "OUTPUT_DIR"    : OUTPUT_DIR,
    "SAVE_FIGURES"  : False,
    "FIGURE_DPI"    : 150,
}
# =============================================================================

# Indicator display names and colours used by the line |Δ| figure
_IND_LABELS = {
    "MD": "Modal Displacement |ΔMD|",
    "MS": "Modal Slope |ΔMS|",
    "MC": "Modal Curvature |ΔMC|",
}
_IND_COLORS = {
    "MD": "steelblue",
    "MS": "darkorange",
    "MC": "mediumseagreen",
}


def _resolve_cfg(cfg: ShmConfig, csv_path: Path) -> ShmConfig:
    """Inject node_plot into cfg from X-coords or direct labels."""
    if CONFIG["NODE_PLOT_BY_X"]:
        resolved = resolve_nodes_by_x(csv_path, cfg)
        if resolved:
            return replace(cfg, node_plot=list(resolved.keys()))
    elif CONFIG["NODE_PLOT"]:
        return replace(cfg, node_plot=list(CONFIG["NODE_PLOT"]))
    return cfg  # None → load all nodes


def _load(
    csv_path: Path,
    cfg: ShmConfig,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (signals_2d, labels, coords) for the requested format."""
    if cfg.data_format == "wide":
        sensor_cols, _, signals_dict = load_wide(csv_path, cfg)
        arr    = np.column_stack([signals_dict[k]["accel"] for k in sensor_cols])
        labels = list(sensor_cols)
        coords = np.array(CONFIG["SENSOR_COORDS"], dtype=float)
        return arr, labels, coords
    else:
        nodes, coords_sorted, signals_dict = load_long(csv_path, cfg)
        direction = (cfg.directions or ["AT2"])[0]
        arr    = np.column_stack([signals_dict[n][direction]["accel"] for n in nodes])
        labels = [f"Node {n}" for n in nodes]
        return arr, labels, coords_sorted  # coords come directly from the CSV


def _plot_di_line_diff(
    coords: np.ndarray,
    di_result: dict,
    mode_idx: int,
    indicators: list[str],
    title: str,
    save_path: Path | None,
    dpi: int,
) -> None:
    """Line chart of |Δ| per sensor position — one panel per indicator."""
    n_ind = len(indicators)
    fig, axes = plt.subplots(1, n_ind, figsize=(5 * n_ind, 4), sharey=False)
    if n_ind == 1:
        axes = [axes]

    for ax, ind in zip(axes, indicators):
        diff  = di_result[ind]["diff"][:, mode_idx]
        color = _IND_COLORS.get(ind, "grey")

        ax.plot(coords, diff, color=color, linewidth=1.8,
                marker="o", markersize=5)
        ax.fill_between(coords, 0, diff, alpha=0.15, color=color)

        ax.set_title(_IND_LABELS.get(ind, ind), fontsize=10)
        ax.set_xlabel("Position (m)", fontsize=9)
        ax.set_ylabel("|Δ|", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_or_show(fig, save_path=save_path, dpi=dpi)


def run(method: str, save: bool) -> None:
    cfg = ShmConfig(
        data_format         = CONFIG["DATA_FORMAT"],
        fs                  = CONFIG["FS"],
        start               = CONFIG["TIME_START"],
        end                 = CONFIG["TIME_END"],
        sensor_prefix       = CONFIG["SENSOR_PREFIX"],
        sensor_plot         = CONFIG["SENSOR_PLOT"],
        directions          = CONFIG["DIRECTIONS"],
        node_plot_by_x      = CONFIG["NODE_PLOT_BY_X"],
        x_tolerance         = CONFIG["X_TOLERANCE"],
        method              = method,
        freq_min            = CONFIG["FREQ_MIN"],
        freq_max            = CONFIG["FREQ_MAX"],
        n_modes             = CONFIG["N_MODES"],
        peak_prominence     = CONFIG["PEAK_PROMINENCE"],
        peak_min_distance_hz= CONFIG["PEAK_MIN_DISTANCE_HZ"],
        welch_nperseg       = CONFIG["WELCH_NPERSEG"],
        welch_overlap       = CONFIG["WELCH_OVERLAP"],
    )

    # ── Resolve node selection for long format ────────────────────────────────
    if cfg.data_format == "long":
        cfg = _resolve_cfg(cfg, CONFIG["HEALTHY_CSV"])

    # ── Load ─────────────────────────────────────────────────────────────────
    sig_h, labels, coords_h = _load(CONFIG["HEALTHY_CSV"], cfg)
    sig_d, _,      coords_d = _load(CONFIG["DAMAGED_CSV"],  cfg)

    n_ch   = min(sig_h.shape[1], sig_d.shape[1])
    sig_h  = sig_h[:, :n_ch]
    sig_d  = sig_d[:, :n_ch]
    labels = labels[:n_ch]
    coords = coords_h[:n_ch]  # use healthy coords as reference

    print(f"  Method   : {method.upper()}")
    print(f"  Channels : {labels}")
    print(f"  Coords   : {coords.tolist()} m")
    print(f"  Samples  : {sig_h.shape[0]}")

    # ── Mode shapes ───────────────────────────────────────────────────────────
    peak_freqs_h, mode_shapes_h = extract_mode_shapes(sig_h, cfg.fs, cfg)
    peak_freqs_d, mode_shapes_d = extract_mode_shapes(sig_d, cfg.fs, cfg)

    n_modes       = min(mode_shapes_h.shape[1], mode_shapes_d.shape[1], cfg.n_modes)
    mode_shapes_h = mode_shapes_h[:, :n_modes]
    mode_shapes_d = mode_shapes_d[:, :n_modes]

    print(f"\n  Peak frequencies (healthy) : {[f'{f:.3f} Hz' for f in peak_freqs_h]}")
    print(f"  Peak frequencies (damaged) : {[f'{f:.3f} Hz' for f in peak_freqs_d]}")

    # ── Damage indicators ─────────────────────────────────────────────────────
    di_result = damage_indicators(
        healthy_shapes = mode_shapes_h,
        damaged_shapes = mode_shapes_d,
        coords         = coords,
        normalise      = True,
    )

    mode_labels = [
        f"Mode {i+1}: {f:.2f} Hz"
        for i, f in enumerate(peak_freqs_h[:n_modes])
    ]

    # ── Output paths ──────────────────────────────────────────────────────────
    out_dir  = Path(CONFIG["OUTPUT_DIR"])
    mode_idx = min(CONFIG["MODE_INDEX"], n_modes - 1)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    sp_shapes  = out_dir / "03_mode_shapes.png"             if save else None
    sp_bar     = out_dir / "03_damage_indicators.png"        if save else None
    sp_overlay = out_dir / "03_damage_indicators_overlay.png" if save else None
    sp_line    = out_dir / "03_damage_indicators_diff_line.png" if save else None

    # ── Plot 1: Mode shape comparison (always shown) ──────────────────────────
    plot_mode_shapes(
        coords         = coords,
        healthy_shapes = mode_shapes_h,
        damaged_shapes = mode_shapes_d,
        mode_labels    = mode_labels,
        title          = "Mode Shapes — Healthy vs Damaged",
        save_path      = sp_shapes,
        dpi            = CONFIG["FIGURE_DPI"],
    )

    # ── Plot 2: DI chart (bar or line) ────────────────────────────────────────
    plot_type = CONFIG["PLOT_TYPE"].lower()

    if plot_type == "bar":
        # Single figure — |Δ| bar chart
        plot_damage_indicator_diff(
            coords     = coords,
            di_result  = di_result,
            mode_index = mode_idx,
            indicators = CONFIG["INDICATORS"],
            title      = f"Damage Indicators — {mode_labels[mode_idx]}",
            save_path  = sp_bar,
            dpi        = CONFIG["FIGURE_DPI"],
        )

    elif plot_type == "line":
        # Figure A — healthy vs damaged overlay with shaded area
        plot_damage_indicators(
            coords     = coords,
            di_result  = di_result,
            mode_index = mode_idx,
            indicators = CONFIG["INDICATORS"],
            title      = f"Damage Indicators — {mode_labels[mode_idx]}",
            save_path  = sp_overlay,
            dpi        = CONFIG["FIGURE_DPI"],
        )
        # Figure B — |Δ| line chart per sensor position
        _plot_di_line_diff(
            coords     = coords,
            di_result  = di_result,
            mode_idx   = mode_idx,
            indicators = CONFIG["INDICATORS"],
            title      = f"Damage Indicator |Δ| — {mode_labels[mode_idx]}",
            save_path  = sp_line,
            dpi        = CONFIG["FIGURE_DPI"],
        )

    else:
        raise ValueError(f"PLOT_TYPE must be 'bar' or 'line', got '{plot_type}'")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n  Damage indicator summary (mode index {mode_idx}):")
    for ind in CONFIG["INDICATORS"]:
        diff_col = di_result[ind]["diff"][:, mode_idx]
        peak_pos = coords[np.argmax(diff_col)]
        print(f"    {ind}  max|Δ| = {diff_col.max():.4f}  @ x = {peak_pos:.3f} m")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute and plot damage indicators.")
    parser.add_argument("--method", default=CONFIG["METHOD"],
                        choices=["fft", "welch", "fdd"],
                        help="Spectral method for mode-shape extraction (default: %(default)s)")
    parser.add_argument("--save", action="store_true",
                        default=CONFIG["SAVE_FIGURES"],
                        help="Save figures to OUTPUT_DIR instead of displaying")
    args = parser.parse_args()

    try:
        run(method=args.method, save=args.save)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())