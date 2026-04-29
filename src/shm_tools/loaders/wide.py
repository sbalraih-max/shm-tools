"""
shm_tools.loaders.wide
======================
Wide-format CSV loader.

In **wide format** every row is a time step and every accelerometer
channel occupies its own column (e.g. ``BS1``, ``BS2``, …).  This is the
layout produced by most real-time sensor data loggers.

Public API
----------
load_wide(filepath, cfg) -> tuple[list[str], NDArray, WideSignals]
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shm_tools.config.models import ShmConfig
from shm_tools.loaders.common import match_col, parse_time_wide
from shm_tools.types import ChannelData, WideSignals


def load_wide(
    filepath: str | Path,
    cfg: ShmConfig,
) -> tuple[list[str], NDArray[np.float64], WideSignals]:
    """Load a wide-format sensor CSV file.

    Reads only the time column and the requested sensor columns into
    memory (``usecols`` optimisation).  Converts timestamps to elapsed
    seconds, applies the optional time window, and returns per-channel
    NumPy arrays.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    cfg : ShmConfig
        Pipeline configuration.  Relevant fields:

        - ``sensor_prefix`` — column-name prefix for sensor channels
        - ``sensor_plot``   — list of channel numbers to include; ``None`` = all
        - ``start`` / ``end`` — time-window bounds in seconds
        - ``timecol_candidates`` — ordered list of time-column name variants

    Returns
    -------
    sensor_cols : list[str]
        Ordered list of loaded sensor column names (e.g. ``["BS1","BS2"]``).
    coords : NDArray[np.float64]
        Sensor positions.  If ``cfg.sensor_coords`` is not set, this is
        simply ``[0, 1, 2, …]`` (sensor index).
    signals : WideSignals
        ``{ channel_name: {"time": ndarray, "accel": ndarray} }``

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If no time column or no sensor columns matching the prefix are found.
    """
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {fp.resolve()}")

    print(f"  Loading wide: {fp.resolve()}")

    # ── Peek at headers (zero memory cost) ───────────────────────────────────
    header_df = pd.read_csv(fp, nrows=0)
    all_cols: list[str] = [c.strip() for c in header_df.columns]
    rename_map: dict[str, str] = {c: c.strip() for c in header_df.columns}

    # ── Resolve time column ──────────────────────────────────────────────────
    time_col = match_col(all_cols, cfg.timecol_candidates)
    if time_col is None:
        raise ValueError(
            f"No time column found in {fp.name}. "
            f"Detected columns: {all_cols}. "
            f"Add the column name to ShmConfig.timecol_candidates."
        )

    # ── Resolve sensor columns ───────────────────────────────────────────────
    prefix = cfg.sensor_prefix
    sensor_cols: list[str] = [c for c in all_cols if c.startswith(prefix)]
    if cfg.sensor_plot:
        wanted = {f"{prefix}{n}" for n in cfg.sensor_plot}
        sensor_cols = [c for c in sensor_cols if c in wanted]
    if not sensor_cols:
        raise ValueError(
            f"No sensor columns with prefix '{prefix}' found in {fp.name}. "
            f"Detected columns: {all_cols}"
        )

    # ── Read only needed columns ─────────────────────────────────────────────
    needed_raw = [
        c for c in header_df.columns
        if c.strip() in ({time_col} | set(sensor_cols))
    ]
    df = pd.read_csv(fp, usecols=needed_raw, dtype=str)
    df.rename(columns=rename_map, inplace=True)

    # ── Parse time → elapsed seconds ─────────────────────────────────────────
    t, order = parse_time_wide(df[time_col])

    # ── Apply time window ────────────────────────────────────────────────────
    mask = np.ones(len(t), dtype=bool)
    if cfg.start is not None:
        mask &= t >= float(cfg.start)
    if cfg.end is not None:
        mask &= t <= float(cfg.end)
    t = t[mask]

    # ── Build signals dict ───────────────────────────────────────────────────
    signals: WideSignals = {}
    for col in sensor_cols:
        a = (
            pd.to_numeric(df[col], errors="coerce")
            .to_numpy(dtype=np.float64, copy=True)[order][mask]
        )
        entry: ChannelData = {"time": t.copy(), "accel": a}
        signals[col] = entry

    # ── Sensor coordinates ───────────────────────────────────────────────────
    user_coords: list[float] | None = getattr(cfg, "sensor_coords", None)
    if user_coords is not None and len(user_coords) == len(sensor_cols):
        coords = np.array(user_coords, dtype=np.float64)
    else:
        if user_coords is not None:
            print(
                f"  WARNING: sensor_coords length mismatch "
                f"({len(user_coords)} vs {len(sensor_cols)}). Using indices."
            )
        coords = np.arange(len(sensor_cols), dtype=np.float64)

    del df
    gc.collect()

    print(f"    Channels : {sensor_cols}")
    print(f"    Samples  : {len(t)}")
    return sensor_cols, coords, signals
