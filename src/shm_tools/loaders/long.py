"""
shm_tools.loaders.long
======================
Long-format (Abaqus) CSV loader.

In **long format** every row represents one node at one time step.
Columns typically include a time stamp, a node label, an X-coordinate,
and one or more acceleration components (``AT1``, ``AT2``, ``AT3``).

Public API
----------
resolve_nodes_by_x(filepath, cfg) -> dict[int, float] | None
load_long(filepath, cfg)          -> tuple[list[int], NDArray, LongSignals]
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shm_tools.config.models import ShmConfig
from shm_tools.loaders.common import match_col
from shm_tools.types import ChannelData, LongSignals


def resolve_nodes_by_x(
    filepath: str | Path,
    cfg: ShmConfig,
) -> dict[int, float] | None:
    """Scan the CSV and return node labels closest to target X-coordinates.

    Reads only the node-label and X-coordinate columns (header scan +
    two-column read) to keep memory usage minimal.  Called *before*
    :func:`load_long` so the resolved labels can be injected into
    ``cfg.node_plot``.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    cfg : ShmConfig
        Must contain ``node_plot_by_x`` (list of target X values in metres)
        and ``x_tolerance`` (warning threshold in metres).

    Returns
    -------
    dict[int, float] or None
        ``{ node_id: actual_x }`` for each target X, or ``None`` if the
        feature is disabled (``node_plot_by_x`` is ``None`` or empty).
    """
    if not cfg.node_plot_by_x:
        return None

    fp = Path(filepath)
    if not fp.exists():
        return None

    header_df = pd.read_csv(fp, nrows=0)
    all_cols = [c.strip() for c in header_df.columns]
    rename_map = {c: c.strip() for c in header_df.columns}

    node_col = match_col(all_cols, cfg.nodecol_candidates)
    x_col    = match_col(all_cols, cfg.xcol_candidates)
    if node_col is None or x_col is None:
        print(
            f"  WARNING: Cannot resolve nodes by X — "
            f"{'node col' if node_col is None else 'X col'} missing."
        )
        return None

    needed = [c for c in header_df.columns if c.strip() in {node_col, x_col}]
    df = pd.read_csv(fp, usecols=needed, dtype=str)
    df.rename(columns=rename_map, inplace=True)
    df[node_col] = pd.to_numeric(df[node_col], errors="coerce")
    df[x_col]    = pd.to_numeric(df[x_col],    errors="coerce")
    df.dropna(subset=[node_col, x_col], inplace=True)

    # One representative X per node (first occurrence)
    node_x_map: pd.Series = df.groupby(node_col)[x_col].first()

    tol = cfg.x_tolerance
    selected: dict[int, float] = {}
    for x_target in cfg.node_plot_by_x:
        nearest_node = int((node_x_map - x_target).abs().idxmin())
        nearest_x    = float(node_x_map[nearest_node])
        dist         = abs(nearest_x - x_target)
        if dist > tol:
            print(
                f"  WARNING: X={x_target} — nearest node is {nearest_node} "
                f"at X={nearest_x:.4f}  (distance={dist:.4f} > tol={tol})"
            )
        else:
            print(f"  X={x_target} → Node {nearest_node}  X={nearest_x:.4f}")
        selected[nearest_node] = nearest_x

    del df
    gc.collect()
    return selected


def load_long(
    filepath: str | Path,
    cfg: ShmConfig,
) -> tuple[list[int], NDArray[np.float64], LongSignals]:
    """Load a long-format Abaqus CSV file.

    Groups rows by node label, sorts each group by time, applies the
    optional time window, and returns per-node, per-direction NumPy arrays.
    Nodes are returned sorted by their X-coordinate so spatial derivative
    operations (slope, curvature) are correctly ordered.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    cfg : ShmConfig
        Pipeline configuration.  Relevant fields:

        - ``directions``         — list of directions to load; ``None`` = all found
        - ``node_plot``          — list of node labels to include; ``None`` = all
        - ``node_plot_by_x``     — select nodes by X-coordinate (resolved before call)
        - ``start`` / ``end``    — time-window bounds in seconds
        - ``timecol_candidates`` / ``nodecol_candidates`` / ``xcol_candidates``
        - ``accelcol_candidates``

    Returns
    -------
    nodes_sorted : list[int]
        Node labels sorted by X-coordinate (ascending).
    coords_sorted : NDArray[np.float64]
        X-coordinates corresponding to *nodes_sorted*.
    signals : LongSignals
        ``{ node_id: { direction: {"time": ndarray, "accel": ndarray} } }``

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns (time, node) or acceleration columns are not found.
    """
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {fp.resolve()}")

    print(f"  Loading long: {fp.resolve()}")

    # ── Peek at headers ───────────────────────────────────────────────────────
    header_df = pd.read_csv(fp, nrows=0)
    all_cols  = [c.strip() for c in header_df.columns]
    rename_map = {c: c.strip() for c in header_df.columns}

    # ── Resolve required columns ──────────────────────────────────────────────
    time_col = match_col(all_cols, cfg.timecol_candidates)
    node_col = match_col(all_cols, cfg.nodecol_candidates)
    x_col    = match_col(all_cols, cfg.xcol_candidates)

    if time_col is None:
        raise ValueError(
            f"No time column found in {fp.name}. Detected: {all_cols}. "
            f"Add the name to ShmConfig.timecol_candidates."
        )
    if node_col is None:
        raise ValueError(
            f"No node column found in {fp.name}. Detected: {all_cols}. "
            f"Add the name to ShmConfig.nodecol_candidates."
        )

    # ── Resolve acceleration columns ──────────────────────────────────────────
    directions = cfg.directions or list(cfg.accelcol_candidates.keys())
    accel_map: dict[str, str] = {}
    for d in directions:
        m = match_col(all_cols, cfg.accelcol_candidates.get(d, []))
        if m:
            accel_map[d] = m
    if not accel_map:
        raise ValueError(
            f"No acceleration columns found in {fp.name}. Detected: {all_cols}"
        )

    # ── Read only needed columns ──────────────────────────────────────────────
    keep = {time_col, node_col} | set(accel_map.values())
    if x_col:
        keep.add(x_col)
    needed_raw = [c for c in header_df.columns if c.strip() in keep]
    df = pd.read_csv(fp, usecols=needed_raw, dtype=str)
    df.rename(columns=rename_map, inplace=True)

    # ── Type conversions ──────────────────────────────────────────────────────
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[node_col] = pd.to_numeric(df[node_col], errors="coerce").astype("Int64")
    df.dropna(subset=[time_col, node_col], inplace=True)
    for col in accel_map.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if x_col:
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")

    # ── Filter by node selection ──────────────────────────────────────────────
    if cfg.node_plot:
        df = df[df[node_col].isin(cfg.node_plot)]

    # ── Apply time window ─────────────────────────────────────────────────────
    if cfg.start is not None:
        df = df[df[time_col] >= float(cfg.start)]
    if cfg.end is not None:
        df = df[df[time_col] <= float(cfg.end)]

    # ── Build per-node signals dict ───────────────────────────────────────────
    signals: LongSignals  = {}
    node_x:  dict[int, float] = {}

    for node_id, grp in df.groupby(node_col, sort=True):
        nid = int(node_id)
        grp = grp.sort_values(time_col)
        t0  = float(grp[time_col].iloc[0])
        t   = grp[time_col].to_numpy(dtype=np.float64, copy=True) - t0

        entry: dict[str, ChannelData] = {}
        for d, col in accel_map.items():
            entry[d] = {
                "time":  t.copy(),
                "accel": grp[col].to_numpy(dtype=np.float64, copy=True),
            }
        signals[nid] = entry

        if x_col and grp[x_col].notna().any():
            node_x[nid] = float(grp[x_col].iloc[0])
        else:
            node_x[nid] = float(nid)

    # ── Sort nodes by X-coordinate ────────────────────────────────────────────
    nodes_sorted  = sorted(signals.keys(), key=lambda n: node_x[n])
    coords_sorted = np.array([node_x[n] for n in nodes_sorted], dtype=np.float64)

    del df
    gc.collect()

    print(f"    Nodes (X-sorted) : {nodes_sorted}")
    print(f"    X-coords         : {coords_sorted.tolist()}")
    print(f"    Directions       : {list(accel_map.keys())}")
    return nodes_sorted, coords_sorted, signals
