"""Tests for modal damage indicators.

Covers:
- spatial_diff order=1 (slope) and order=2 (curvature) with known inputs
- damage_indicators output structure and keys
- Correct zero diff for identical inputs
- Non-zero diff for locally perturbed mode shape
- Damage localisation: largest MC diff at the damage location
- Normalisation behaviour
- Input validation errors
"""

from __future__ import annotations

import numpy as np
import pytest

from shm_tools.indicators.damage import damage_indicators, spatial_diff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _beam_coords(n: int = 10, length: float = 1.0) -> np.ndarray:
    """Uniformly spaced sensor positions along a beam."""
    return np.linspace(0.0, length, n)


def _first_mode(coords: np.ndarray) -> np.ndarray:
    """Theoretical first bending mode: sin(π x / L)."""
    L = coords[-1]
    return np.sin(np.pi * coords / L)


def _damage_at(
    mode: np.ndarray,
    index: int,
    factor: float = 0.6,
) -> np.ndarray:
    """Return a copy of mode with a local amplitude reduction at index."""
    damaged = mode.copy()
    damaged[index] *= factor
    return damaged


# ---------------------------------------------------------------------------
# spatial_diff tests
# ---------------------------------------------------------------------------

class TestSpatialDiff:

    def test_slope_of_linear_is_constant(self):
        """Slope of f(x) = 2x must be 2 everywhere."""
        coords = _beam_coords(n=20)
        f      = 2.0 * coords
        slope  = spatial_diff(f, coords, order=1)
        np.testing.assert_allclose(slope, 2.0, atol=1e-10)

    def test_curvature_of_quadratic_is_constant(self):
        """
        Curvature of f(x) = x² must be 2 everywhere at interior points.
        np.gradient uses one-sided stencils at boundaries so indices 0, 1,
        n-2, n-1 have reduced accuracy — we test only the interior.
        """
        coords = _beam_coords(n=50)
        f = coords ** 2
        curvature = spatial_diff(f, coords, order=2)
        # Interior points only (exclude first 2 and last 2 boundary points)
        interior = curvature[2:-2]
        np.testing.assert_allclose(interior, 2.0, atol=1e-6)

    def test_slope_of_constant_is_zero(self):
        coords = _beam_coords(n=15)
        f      = np.ones_like(coords) * 5.0
        slope  = spatial_diff(f, coords, order=1)
        np.testing.assert_allclose(slope, 0.0, atol=1e-12)

    def test_curvature_of_linear_is_zero(self):
        coords    = _beam_coords(n=15)
        f         = 3.0 * coords + 1.0
        curvature = spatial_diff(f, coords, order=2)
        np.testing.assert_allclose(curvature, 0.0, atol=1e-10)

    def test_output_same_shape_as_input(self):
        coords = _beam_coords(n=12)
        f      = np.random.default_rng(0).random(12)
        assert spatial_diff(f, coords, order=1).shape == f.shape
        assert spatial_diff(f, coords, order=2).shape == f.shape

    def test_invalid_order_raises(self):
        coords = _beam_coords()
        f      = np.ones(10)
        with pytest.raises(ValueError, match="order must be 1 or 2"):
            spatial_diff(f, coords, order=3)

    def test_shape_mismatch_raises(self):
        coords = _beam_coords(n=10)
        f      = np.ones(8)
        with pytest.raises(ValueError, match="same shape"):
            spatial_diff(f, coords, order=1)

    def test_non_uniform_spacing_slope(self):
        """spatial_diff handles non-uniform sensor spacing correctly."""
        coords = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        f      = 2.0 * coords
        slope  = spatial_diff(f, coords, order=1)
        np.testing.assert_allclose(slope, 2.0, atol=1e-10)


# ---------------------------------------------------------------------------
# damage_indicators output structure
# ---------------------------------------------------------------------------

class TestDamageIndicatorsStructure:

    def _make_inputs(self, n: int = 8, n_modes: int = 2):
        coords  = _beam_coords(n=n)
        healthy = np.column_stack([_first_mode(coords)] * n_modes)
        damaged = healthy.copy()
        damaged[n // 2, 0] *= 0.7
        return coords, healthy, damaged

    def test_output_has_three_keys(self):
        coords, healthy, damaged = self._make_inputs()
        result = damage_indicators(healthy, damaged, coords)
        assert set(result.keys()) == {"MD", "MS", "MC"}

    def test_each_key_has_healthy_damaged_diff(self):
        coords, healthy, damaged = self._make_inputs()
        result = damage_indicators(healthy, damaged, coords)
        for key in ("MD", "MS", "MC"):
            assert set(result[key].keys()) == {"healthy", "damaged", "diff"}

    def test_array_shapes_correct(self):
        n, n_modes = 8, 3
        coords  = _beam_coords(n=n)
        healthy = np.column_stack([_first_mode(coords)] * n_modes)
        damaged = healthy.copy()
        result  = damage_indicators(healthy, damaged, coords)
        for key in ("MD", "MS", "MC"):
            for sub in ("healthy", "damaged", "diff"):
                assert result[key][sub].shape == (n, n_modes), (
                    f"Shape mismatch for result['{key}']['{sub}']"
                )

    def test_diff_non_negative(self):
        """diff = |damaged - healthy|, must always be >= 0."""
        coords, healthy, damaged = self._make_inputs()
        result = damage_indicators(healthy, damaged, coords)
        for key in ("MD", "MS", "MC"):
            assert np.all(result[key]["diff"] >= 0), (
                f"Negative diff values found in indicator '{key}'"
            )


# ---------------------------------------------------------------------------
# Known input / output tests
# ---------------------------------------------------------------------------

class TestDamageIndicatorsKnownIO:

    def test_identical_inputs_give_zero_diff_without_normalise(self):
        """
        When healthy == damaged and normalise=False the diff must be
        exactly zero for all three indicators.
        """
        coords  = _beam_coords(n=10)
        mode    = _first_mode(coords).reshape(-1, 1)
        result  = damage_indicators(mode, mode.copy(), coords,
                                    normalise=False)
        for key in ("MD", "MS", "MC"):
            np.testing.assert_array_equal(
                result[key]["diff"],
                np.zeros((10, 1)),
                err_msg=f"Expected zero diff for identical inputs in '{key}'",
            )

    def test_local_damage_gives_nonzero_diff(self):
        """A local perturbation at sensor 4 must produce non-zero diff."""
        coords  = _beam_coords(n=10)
        healthy = _first_mode(coords).reshape(-1, 1)
        damaged = _damage_at(healthy[:, 0], index=4, factor=0.5).reshape(-1, 1)
        result  = damage_indicators(healthy, damaged, coords, normalise=False)
        for key in ("MD", "MS", "MC"):
            assert np.any(result[key]["diff"] > 0), (
                f"Expected non-zero diff in '{key}' after local damage"
            )

    def test_modal_displacement_diff_matches_manual(self):
        """
        MD diff = |damaged - healthy| after normalisation.
        With normalise=False it must exactly match a manual calculation.
        """
        coords  = _beam_coords(n=6)
        healthy = np.array([0.0, 0.5, 1.0, 0.8, 0.4, 0.0]).reshape(-1, 1)
        damaged = np.array([0.0, 0.5, 0.8, 0.8, 0.4, 0.0]).reshape(-1, 1)
        result  = damage_indicators(healthy, damaged, coords, normalise=False)
        expected_diff = np.abs(damaged - healthy)
        np.testing.assert_allclose(
            result["MD"]["diff"],
            expected_diff,
            atol=1e-12,
        )

    def test_curvature_diff_peaks_at_damage_location(self):
        """
        The largest MC diff should be at or adjacent to the damaged sensor.
        Modal curvature is the most sensitive damage localisation indicator.
        """
        n           = 20
        damage_idx  = 10   # mid-span
        coords      = _beam_coords(n=n)
        healthy     = _first_mode(coords).reshape(-1, 1)
        damaged_vec = _damage_at(healthy[:, 0], index=damage_idx, factor=0.3)
        damaged     = damaged_vec.reshape(-1, 1)

        result      = damage_indicators(healthy, damaged, coords,
                                        normalise=False)
        mc_diff     = result["MC"]["diff"][:, 0]
        peak_idx    = int(np.argmax(mc_diff))

        # Allow ±2 sensors tolerance for gradient boundary effects
        assert abs(peak_idx - damage_idx) <= 2, (
            f"MC diff peak at sensor {peak_idx}, expected near {damage_idx}"
        )

    def test_slope_diff_uses_gradient(self):
        """
        MS healthy must equal np.gradient(healthy_mode, coords).
        Confirms that spatial_diff wraps np.gradient correctly.
        """
        coords  = _beam_coords(n=10)
        healthy = _first_mode(coords).reshape(-1, 1)
        damaged = healthy.copy()
        result  = damage_indicators(healthy, damaged, coords, normalise=False)
        expected_slope = np.gradient(healthy[:, 0], coords).reshape(-1, 1)
        np.testing.assert_allclose(
            result["MS"]["healthy"],
            expected_slope,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Normalisation behaviour
# ---------------------------------------------------------------------------

class TestNormalisation:

    def test_normalised_shapes_have_unit_max(self):
        """After normalise=True each mode column must have max |value| = 1."""
        coords  = _beam_coords(n=10)
        healthy = np.column_stack([
            _first_mode(coords),
            0.5 * _first_mode(coords),
        ])
        damaged = healthy * 0.9
        result  = damage_indicators(healthy, damaged, coords, normalise=True)
        md_h = result["MD"]["healthy"]
        for m in range(md_h.shape[1]):
            max_abs = np.abs(md_h[:, m]).max()
            assert abs(max_abs - 1.0) < 1e-6, (
                f"Mode {m} max abs = {max_abs}, expected 1.0 after normalisation"
            )

    def test_normalise_false_preserves_raw_values(self):
        """With normalise=False the MD healthy values equal the raw input."""
        coords  = _beam_coords(n=8)
        healthy = (_first_mode(coords) * 3.7).reshape(-1, 1)
        damaged = healthy.copy()
        result  = damage_indicators(healthy, damaged, coords, normalise=False)
        np.testing.assert_allclose(
            result["MD"]["healthy"],
            healthy,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_shape_mismatch_raises(self):
        coords  = _beam_coords(n=8)
        healthy = np.ones((8, 2))
        damaged = np.ones((8, 3))   # wrong n_modes
        with pytest.raises(ValueError, match="same shape"):
            damage_indicators(healthy, damaged, coords)

    def test_coords_length_mismatch_raises(self):
        coords  = _beam_coords(n=6)
        healthy = np.ones((8, 2))   # 8 sensors but coords has 6
        damaged = np.ones((8, 2))
        with pytest.raises(ValueError, match="coords length"):
            damage_indicators(healthy, damaged, coords)