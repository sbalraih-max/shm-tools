"""Gradient-based modal damage indicators.

Computes three spatial damage indicators from mode-shape vectors:
- Modal Displacement (MD)
- Modal Slope (MS)  — first spatial derivative via np.gradient
- Modal Curvature (MC) — second spatial derivative via np.gradient

np.gradient is used throughout because it is second-order accurate at
boundaries (uses one-sided differences), which is important for beam-like
structures where the first and last sensor positions are boundary nodes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def spatial_diff(
    mode_shape: NDArray[np.floating],
    coords: NDArray[np.floating],
    order: int = 1,
) -> NDArray[np.floating]:
    """Compute a spatial derivative of a mode-shape vector.

    Uses ``np.gradient`` with non-uniform spacing support, giving
    second-order accuracy at both interior points and boundaries.

    Parameters
    ----------
    mode_shape : NDArray[np.floating]
        1-D mode-shape vector, shape ``(n_sensors,)``.
    coords : NDArray[np.floating]
        1-D array of sensor spatial coordinates (e.g. X positions in metres),
        shape ``(n_sensors,)``. Must be strictly monotonically increasing.
    order : int, optional
        Derivative order.  ``1`` returns slope, ``2`` returns curvature.
        By default 1.

    Returns
    -------
    NDArray[np.floating]
        Spatial derivative of the same shape as *mode_shape*.

    Raises
    ------
    ValueError
        If *order* is not 1 or 2, or if array lengths do not match.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}.")
    if mode_shape.shape != coords.shape:
        raise ValueError(
            f"mode_shape and coords must have the same shape, "
            f"got {mode_shape.shape} vs {coords.shape}."
        )

    result = np.gradient(mode_shape, coords)
    if order == 2:
        result = np.gradient(result, coords)
    return result


def damage_indicators(
    healthy_shapes: NDArray[np.floating],
    damaged_shapes: NDArray[np.floating],
    coords: NDArray[np.floating],
    normalise: bool = True,
) -> dict[str, dict[str, NDArray[np.floating]]]:
    """Compute modal damage indicators for healthy and damaged mode shapes.

    Calculates Modal Displacement (MD), Modal Slope (MS), and Modal
    Curvature (MC) for each mode in both the healthy and damaged states,
    then computes the absolute difference as the damage indicator (DI).

    Parameters
    ----------
    healthy_shapes : NDArray[np.floating]
        Mode-shape matrix for the healthy structure,
        shape ``(n_sensors, n_modes)``.
    damaged_shapes : NDArray[np.floating]
        Mode-shape matrix for the damaged structure,
        shape ``(n_sensors, n_modes)``.  Must match *healthy_shapes* in shape.
    coords : NDArray[np.floating]
        1-D array of sensor spatial coordinates in metres,
        shape ``(n_sensors,)``.
    normalise : bool, optional
        If ``True`` (default), normalise each mode shape to unit maximum
        absolute value before computing derivatives.  This makes DI values
        comparable across modes with different amplitude scales.

    Returns
    -------
    dict[str, dict[str, NDArray[np.floating]]]
        Nested dictionary with the following structure::

            {
                "MD": {
                    "healthy":  ndarray (n_sensors, n_modes),
                    "damaged":  ndarray (n_sensors, n_modes),
                    "diff":     ndarray (n_sensors, n_modes),
                },
                "MS": { ... },   # Modal Slope
                "MC": { ... },   # Modal Curvature
            }

        ``"diff"`` is the absolute difference ``|damaged - healthy|``,
        which is the scalar damage indicator plotted against sensor position.

    Raises
    ------
    ValueError
        If *healthy_shapes* and *damaged_shapes* have different shapes,
        or if ``coords`` length does not match ``n_sensors``.

    Notes
    -----
    ``np.gradient`` is used for all derivatives because it applies
    second-order accurate one-sided differences at the boundaries,
    avoiding the accuracy loss that simple finite-difference stencils
    suffer at the first and last sensor positions.

    For a beam discretised at positions :math:`x_1, x_2, \\ldots, x_N`,
    the curvature at interior point :math:`i` is approximated as:

    .. math::

        \\phi''(x_i) \\approx
        \\frac{\\phi(x_{i+1}) - 2\\phi(x_i) + \\phi(x_{i-1})}
             {h^2}

    with analogous one-sided stencils at :math:`x_1` and :math:`x_N`.
    """
    if healthy_shapes.shape != damaged_shapes.shape:
        raise ValueError(
            f"healthy_shapes and damaged_shapes must have the same shape, "
            f"got {healthy_shapes.shape} vs {damaged_shapes.shape}."
        )
    n_sensors, n_modes = healthy_shapes.shape
    if coords.shape[0] != n_sensors:
        raise ValueError(
            f"coords length {coords.shape[0]} does not match "
            f"n_sensors {n_sensors}."
        )

    if normalise:
        healthy_shapes = _normalise_columns(healthy_shapes)
        damaged_shapes = _normalise_columns(damaged_shapes)

    # Initialise output arrays
    results: dict[str, dict[str, NDArray[np.floating]]] = {
        "MD": {
            "healthy": np.zeros((n_sensors, n_modes)),
            "damaged": np.zeros((n_sensors, n_modes)),
            "diff":    np.zeros((n_sensors, n_modes)),
        },
        "MS": {
            "healthy": np.zeros((n_sensors, n_modes)),
            "damaged": np.zeros((n_sensors, n_modes)),
            "diff":    np.zeros((n_sensors, n_modes)),
        },
        "MC": {
            "healthy": np.zeros((n_sensors, n_modes)),
            "damaged": np.zeros((n_sensors, n_modes)),
            "diff":    np.zeros((n_sensors, n_modes)),
        },
    }

    for m in range(n_modes):
        h_shape = healthy_shapes[:, m]
        d_shape = damaged_shapes[:, m]

        # --- Modal Displacement ----------------------------------------------
        results["MD"]["healthy"][:, m] = h_shape
        results["MD"]["damaged"][:, m] = d_shape
        results["MD"]["diff"][:, m] = np.abs(d_shape - h_shape)

        # --- Modal Slope (1st spatial derivative) ----------------------------
        h_slope = spatial_diff(h_shape, coords, order=1)
        d_slope = spatial_diff(d_shape, coords, order=1)
        results["MS"]["healthy"][:, m] = h_slope
        results["MS"]["damaged"][:, m] = d_slope
        results["MS"]["diff"][:, m] = np.abs(d_slope - h_slope)

        # --- Modal Curvature (2nd spatial derivative) ------------------------
        h_curv = spatial_diff(h_shape, coords, order=2)
        d_curv = spatial_diff(d_shape, coords, order=2)
        results["MC"]["healthy"][:, m] = h_curv
        results["MC"]["damaged"][:, m] = d_curv
        results["MC"]["diff"][:, m] = np.abs(d_curv - h_curv)

    return results


def _normalise_columns(
    matrix: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Normalise each column to unit maximum absolute value.

    Parameters
    ----------
    matrix : NDArray[np.floating]
        Shape ``(n_rows, n_cols)``.

    Returns
    -------
    NDArray[np.floating]
        Normalised matrix of the same shape.
    """
    max_abs = np.abs(matrix).max(axis=0, keepdims=True)
    max_abs[max_abs == 0] = 1.0
    return matrix / max_abs