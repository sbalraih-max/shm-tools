"""Shared type aliases used across shm_tools submodules."""
from __future__ import annotations
from typing import TypedDict
import numpy as np
from numpy.typing import NDArray


class ChannelData(TypedDict):
    """Time-series for one channel or node-direction."""
    time:  NDArray[np.float64]
    accel: NDArray[np.float64]


WideSignals = dict[str, ChannelData]
LongSignals = dict[int, dict[str, ChannelData]]


class DamageIndicatorResult(TypedDict):
    displacement: dict[str, NDArray[np.float64]]
    slope:        dict[str, NDArray[np.float64]]
    curvature:    dict[str, NDArray[np.float64]]
