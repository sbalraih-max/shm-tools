"""
shm_tools.loaders.common
========================
Shared utilities used by both wide-format and long-format loaders:

  - ``match_col``        — case-insensitive first-match column lookup
  - ``parse_time_wide``  — convert various timestamp formats to elapsed seconds
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def match_col(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first column name whose stripped, lower-case form matches
    any entry in *candidates*.

    The comparison is case-insensitive and strip-whitespace safe on both
    sides, so ``"Step Time "`` matches candidate ``"step time"``.

    Parameters
    ----------
    columns : list[str]
        Column names as read from the CSV header.
    candidates : list[str]
        Ordered list of candidate names to try (first match wins).

    Returns
    -------
    str or None
        The matched column name (original casing from *columns*), or
        ``None`` if no candidate matches.

    Examples
    --------
    >>> match_col(["Step Time", "Node Label", "AT2"], ["step time", "time"])
    'Step Time'
    """
    col_map: dict[str, str] = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def parse_time_wide(series: pd.Series) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Convert a raw timestamp column to zero-origin elapsed seconds.

    Handles three formats in priority order:

    1. **``HH:MM:SS:SUBSEC``** — 4-part colon-separated string where the
       sub-second field is a variable-length nanosecond counter
       (e.g. ``"0:0:1:000500000"``).
    2. **``HH:MM:SS.fff``** — standard colon-separated with decimal
       sub-seconds (e.g. ``"0:0:1.500"``).  Handled via
       :func:`pandas.to_timedelta`.
    3. **Numeric** — already seconds (float or int); cast directly.

    The returned array is always sorted ascending (stable sort) and
    zero-origin (first sample = 0 s).

    Parameters
    ----------
    series : pd.Series
        Raw time column from the CSV (dtype ``str`` or numeric).

    Returns
    -------
    t : NDArray[np.float64]
        Elapsed seconds, sorted ascending, zero-origin.
    order : NDArray[np.intp]
        Sort indices so the caller can reorder accompanying data rows with
        ``df.iloc[order]``.

    Raises
    ------
    ValueError
        If the series cannot be parsed by any of the three strategies.
    """
    s = series.astype(str).str.strip()

    # ── Strategy 1: HH:MM:SS:SUBSEC (4 colon-separated parts) ───────────────
    if s.str.count(":").max() >= 3:
        parts = s.str.split(":", expand=True)
        hh = parts[0].astype(np.int64).to_numpy()
        mm = parts[1].astype(np.int64).to_numpy()
        ss = parts[2].astype(np.int64).to_numpy()
        sub_clean = parts[3].str.strip().apply(
            lambda x: "".join(filter(str.isdigit, x))
        )
        sub_padded = sub_clean.str[-9:].str.rjust(9, "0")
        sub_ns = sub_padded.astype(np.int64).to_numpy()
        t = hh * 3600 + mm * 60 + ss + sub_ns / 1e9
        t = t - t[0]
        order = np.argsort(t, kind="stable")
        return t[order], order

    # ── Strategy 2: timedelta-parseable (HH:MM:SS.fff, ISO 8601, etc.) ──────
    try:
        td = pd.to_timedelta(s)
        if td.notna().all():
            t = td.dt.total_seconds().to_numpy(dtype=np.float64)
            t = t - t[0]
            order = np.argsort(t, kind="stable")
            return t[order], order
    except Exception:
        pass

    # ── Strategy 3: already numeric seconds ─────────────────────────────────
    t = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
    t = t - t[0]
    order = np.argsort(t, kind="stable")
    return t[order], order
