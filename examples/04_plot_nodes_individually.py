"""
examples/04_plot_nodes_individually.py
=======================================
Plot each selected sensor/node in its own figure — healthy vs damaged.

One figure per channel, saved as:
  - wide format : output/sensor_<NAME>_time_history.png
  - long format : output/node_<ID>_<DIR>_time_history.png

Usage
-----
    python examples/04_plot_nodes_individually.py
    python examples/04_plot_nodes_individually.py --format wide
    python examples/04_plot_nodes_individually.py --save
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from shm_tools import ShmConfig, load_wide, load_long
from shm_tools.loaders.long import resolve_nodes_by_x

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
    "FS"            : 200.0,
    "TIME_START"    : None,
    "TIME_END"      : None,

    # --- Wide-format sensor selection ----------------------------------------
    "SENSOR_PREFIX" : "BS",
    "SENSOR_PLOT"   : None,           # list[int] e.g. [1, 2, 3] — None = all

    # --- Long-format node selection (pick ONE, set the other to None) --------
    "NODE_PLOT_BY_X": [0.2, 0.4, 0.6, 0.8, 1.0],     # select by X [m]  ← active
    "NODE_PLOT"     : None,                          # list[int] e.g. [79, 53, 25, 2] select by node ID — or None = all
    "X_TOLERANCE"   : 0.05,
    "DIRECTIONS"    : ["AT2"],

    # --- Plot appearance -----------------------------------------------------
    "FIGSIZE"       : (12, 4),
    "DPI"           : 150,
    "COLOR_HEALTHY" : "steelblue",
    "COLOR_DAMAGED" : "tomato",
    "YLABEL"        : "Acceleration (m/s²)",

    # --- Output --------------------------------------------------------------
    "OUTPUT_DIR"    : OUTPUT_DIR,
    "SAVE_FIGURES"  : False,
}
# =============================================================================


def _resolve_cfg(cfg: ShmConfig, csv_path: Path) -> ShmConfig:
    """Inject node_plot into cfg from X-coords or direct labels."""
    if CONFIG["NODE_PLOT_BY_X"]:
        resolved = resolve_nodes_by_x(csv_path, cfg)
        if resolved:
            return replace(cfg, node_plot=list(resolved.keys()))
    elif CONFIG["NODE_PLOT"]:
        return replace(cfg, node_plot=list(CONFIG["NODE_PLOT"]))
    return cfg  # None → load all nodes


def _plot_single(
    time: np.ndarray,
    accel_h: np.ndarray,
    accel_d: np.ndarray,
    title: str,
    save_path: Path | None,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])

    ax.plot(time, accel_h,
            color=CONFIG["COLOR_HEALTHY"], linewidth=0.8,
            label="Healthy", alpha=0.9)
    ax.plot(time, accel_d,
            color=CONFIG["COLOR_DAMAGED"], linewidth=0.8,
            label="Damaged", alpha=0.9)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel(CONFIG["YLABEL"], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved → {save_path.resolve()}")
        plt.close(fig)
    else:
        plt.show()


def run_wide(cfg: ShmConfig, save: bool) -> None:
    sensor_cols_h, _, signals_h = load_wide(CONFIG["HEALTHY_CSV"], cfg)
    sensor_cols_d, _, signals_d = load_wide(CONFIG["DAMAGED_CSV"], cfg)

    out_dir = Path(CONFIG["OUTPUT_DIR"])
    print(f"\n  Plotting {len(sensor_cols_h)} sensor(s) individually ...\n")

    for col in sensor_cols_h:
        if col not in signals_d:
            print(f"  SKIP {col} — not found in damaged file.")
            continue

        accel_h = signals_h[col]["accel"]
        accel_d = signals_d[col]["accel"]
        time    = signals_h[col]["time"]

        n = min(len(accel_h), len(accel_d))
        accel_h, accel_d, time = accel_h[:n], accel_d[:n], time[:n]

        print(f"  Sensor {col:>6s}  samples={n}  ({n/cfg.fs:.2f} s)")

        save_path = out_dir / f"sensor_{col}_time_history.png" if save else None
        _plot_single(
            time     = time,
            accel_h  = accel_h,
            accel_d  = accel_d,
            title    = f"Sensor {col} — Healthy vs Damaged",
            save_path= save_path,
            dpi      = CONFIG["DPI"],
        )


def run_long(cfg: ShmConfig, save: bool) -> None:
    cfg = _resolve_cfg(cfg, CONFIG["HEALTHY_CSV"])

    nodes_h, coords_h, signals_h = load_long(CONFIG["HEALTHY_CSV"], cfg)
    nodes_d, _,        signals_d = load_long(CONFIG["DAMAGED_CSV"], cfg)

    direction = (cfg.directions or ["AT2"])[0]
    out_dir   = Path(CONFIG["OUTPUT_DIR"])

    print(f"\n  Plotting {len(nodes_h)} node(s) individually ...\n")

    for node_id, x_coord in zip(nodes_h, coords_h):
        if node_id not in signals_h or node_id not in signals_d:
            print(f"  SKIP Node {node_id} — not present in both files.")
            continue

        accel_h = signals_h[node_id][direction]["accel"]
        accel_d = signals_d[node_id][direction]["accel"]
        time    = signals_h[node_id][direction]["time"]

        n = min(len(accel_h), len(accel_d))
        accel_h, accel_d, time = accel_h[:n], accel_d[:n], time[:n]

        print(f"  Node {node_id:>4d}  X={x_coord:.4f} m  "
              f"samples={n}  ({n/cfg.fs:.2f} s)")

        save_path = (out_dir / f"node_{node_id}_{direction}_time_history.png"
                     if save else None)
        _plot_single(
            time     = time,
            accel_h  = accel_h,
            accel_d  = accel_d,
            title    = f"Node {node_id}  (X = {x_coord:.4f} m)  —  {direction}",
            save_path= save_path,
            dpi      = CONFIG["DPI"],
        )


def run(data_format: str, save: bool) -> None:
    cfg = ShmConfig(
        data_format    = data_format,
        fs             = CONFIG["FS"],
        start          = CONFIG["TIME_START"],
        end            = CONFIG["TIME_END"],
        sensor_prefix  = CONFIG["SENSOR_PREFIX"],
        sensor_plot    = CONFIG["SENSOR_PLOT"],
        directions     = CONFIG["DIRECTIONS"],
        node_plot_by_x = CONFIG["NODE_PLOT_BY_X"],
        x_tolerance    = CONFIG["X_TOLERANCE"],
    )

    if data_format == "wide":
        run_wide(cfg, save)
    else:
        run_long(cfg, save)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot each sensor/node time history in a separate figure.")
    parser.add_argument("--format", default=CONFIG["DATA_FORMAT"],
                        choices=["wide", "long"], dest="data_format")
    parser.add_argument("--save", action="store_true",
                        default=CONFIG["SAVE_FIGURES"],
                        help="Save figures to OUTPUT_DIR instead of displaying")
    args = parser.parse_args()

    try:
        run(data_format=args.data_format, save=args.save)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())