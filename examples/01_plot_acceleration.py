"""
examples/01_plot_acceleration.py
=================================
Time-history plotting — healthy vs damaged acceleration signals.

Usage
-----
    python examples/01_plot_acceleration.py
    python examples/01_plot_acceleration.py --format long
    python examples/01_plot_acceleration.py --overlay
    python examples/01_plot_acceleration.py --overlay --save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── Public API ────────────────────────────────────────────────────────────────
from shm_tools import ShmConfig, load_wide, load_long
from shm_tools.loaders.long import resolve_nodes_by_x
from shm_tools.plotting.time_history import plot_time_history

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

# =============================================================================
# CONFIG — edit these values to match your data
# =============================================================================
CONFIG = {
    # --- Input files ---------------------------------------------------------
    "HEALTHY_CSV"  : DATA_DIR / "healthy_wide.csv",
    "DAMAGED_CSV"  : DATA_DIR / "damaged_wide.csv",

    # --- Data format ---------------------------------------------------------
    "DATA_FORMAT"  : "wide",          # "wide" | "long"

    # --- Sampling & time window ----------------------------------------------
    "FS"           : 200.0,           # sampling frequency [Hz]
    "TIME_START"   : None,            # crop start [s] – None = from beginning
    "TIME_END"     : None,            # crop end   [s] – None = to end

    # --- Wide-format sensor selection ----------------------------------------
    "SENSOR_PREFIX": "BS",
    "SENSOR_PLOT"  : None,

    # --- Long-format node selection ------------------------------------------
    "DIRECTIONS"   : ["AT2"],
    "NODE_PLOT_BY_X": [0.2, 0.4, 0.6, 0.8, 1.0],        # select nodes by X [m]
    "NODE_PLOT"     : None,                             # list[int]  select by node ID — or None
    "X_TOLERANCE"  : 0.05,

    # --- Output --------------------------------------------------------------
    "OUTPUT_DIR"   : OUTPUT_DIR,
    "SAVE_FIGURES" : False,
    "OVERLAY"      : False,
    "FIGURE_DPI"   : 150,
}
# =============================================================================


def run(data_format: str, overlay: bool, save: bool) -> None:
    cfg = ShmConfig(
        data_format     = data_format,
        fs              = CONFIG["FS"],
        start           = CONFIG["TIME_START"],
        end             = CONFIG["TIME_END"],
        sensor_prefix   = CONFIG["SENSOR_PREFIX"],
        sensor_plot     = CONFIG["SENSOR_PLOT"],
        directions      = CONFIG["DIRECTIONS"],
        node_plot_by_x  = CONFIG["NODE_PLOT_BY_X"],
        x_tolerance     = CONFIG["X_TOLERANCE"],
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    if data_format == "wide":
        sensor_cols_h, _, signals_h = load_wide(CONFIG["HEALTHY_CSV"], cfg)
        sensor_cols_d, _, signals_d = load_wide(CONFIG["DAMAGED_CSV"], cfg)
        labels  = list(sensor_cols_h)
        sig_h   = np.column_stack([signals_h[k]["accel"] for k in sensor_cols_h])
        sig_d   = np.column_stack([signals_d[k]["accel"] for k in sensor_cols_d])
        time    = signals_h[sensor_cols_h[0]]["time"]

    else:
        # ── Resolve node IDs from target X-coordinates BEFORE loading ─────────
        from dataclasses import replace

        if CONFIG["NODE_PLOT_BY_X"]:
            # X-coordinate → node ID lookup
            resolved = resolve_nodes_by_x(CONFIG["HEALTHY_CSV"], cfg)
            if resolved:
                cfg = replace(cfg, node_plot=list(resolved.keys()))
        elif CONFIG["NODE_PLOT"]:
            # Direct node-label selection
            cfg = replace(cfg, node_plot=list(CONFIG["NODE_PLOT"]))
        # else: node_plot stays None → load all nodes

        nodes_h, _, signals_h = load_long(CONFIG["HEALTHY_CSV"], cfg)
        nodes_d, _, signals_d = load_long(CONFIG["DAMAGED_CSV"], cfg)

        direction = (cfg.directions or ["AT2"])[0]
        sig_h  = np.column_stack([signals_h[n][direction]["accel"] for n in nodes_h])
        sig_d  = np.column_stack([signals_d[n][direction]["accel"] for n in nodes_d])
        time   = signals_h[nodes_h[0]][direction]["time"]
        labels = [f"Node {n}" for n in nodes_h]

    n_ch   = min(sig_h.shape[1], sig_d.shape[1])
    sig_h  = sig_h[:, :n_ch]
    sig_d  = sig_d[:, :n_ch]
    labels = labels[:n_ch]

    print(f"  Format   : {data_format}")
    print(f"  Channels : {labels}")
    print(f"  Samples  : {len(time)}  ({len(time)/cfg.fs:.2f} s @ {cfg.fs} Hz)")

    # ── Save path ─────────────────────────────────────────────────────────────
    save_path = None
    if save:
        out = Path(CONFIG["OUTPUT_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        save_path = out / "01_time_history.png"

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_time_history(
        time            = time,
        signals         = sig_d,
        labels          = labels,
        healthy_signals = sig_h,
        overlay         = overlay,
        title           = "Acceleration — Healthy vs Damaged",
        save_path       = save_path,
        dpi             = CONFIG["FIGURE_DPI"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot acceleration time histories.")
    parser.add_argument("--format", default=CONFIG["DATA_FORMAT"],
                        choices=["wide", "long"], dest="data_format")
    parser.add_argument("--overlay", action="store_true", default=CONFIG["OVERLAY"])
    parser.add_argument("--save",    action="store_true", default=CONFIG["SAVE_FIGURES"])
    args = parser.parse_args()

    try:
        run(data_format=args.data_format, overlay=args.overlay, save=args.save)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())