"""Tests for the FDD core module.

Covers:
- Output shapes from compute_fdd
- Batch SVD produces correct singular value ordering (descending)
- Dynamic phase alignment: real part dominates over imaginary part
- CPSD Hermitian symmetry (upper-triangle + conjugate mirror)
- Known two-channel sine input produces a detectable peak at the right frequency
"""

from __future__ import annotations

import numpy as np
import pytest

from shm_tools.config.models import ShmConfig
from shm_tools.fdd.core import compute_fdd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(nperseg: int = 256, overlap: float = 0.5) -> ShmConfig:
    return ShmConfig(
        method="fdd",
        welch_nperseg=nperseg,
        welch_overlap=overlap,
    )


def _sine_signals(
    freq_hz: float = 5.0,
    fs: float = 200.0,
    duration: float = 4.0,
    n_channels: int = 4,
    phase_offsets: list[float] | None = None,
) -> np.ndarray:
    """Generate multi-channel sine signals at a single frequency."""
    n = int(fs * duration)
    t = np.arange(n) / fs
    if phase_offsets is None:
        phase_offsets = [0.0] * n_channels
    signals = np.column_stack([
        np.sin(2 * np.pi * freq_hz * t + phi)
        for phi in phase_offsets
    ])
    return signals.astype(np.float64)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestFddOutputShapes:

    def test_freqs_shape_is_1d(self):
        signals = _sine_signals(n_channels=3)
        cfg = _make_cfg(nperseg=128)
        freqs, sv, modes = compute_fdd(signals, fs=200.0, cfg=cfg)
        assert freqs.ndim == 1

    def test_singular_values_shape(self):
        n_ch = 4
        signals = _sine_signals(n_channels=n_ch)
        cfg = _make_cfg(nperseg=128)
        freqs, sv, modes = compute_fdd(signals, fs=200.0, cfg=cfg)
        assert sv.ndim == 2
        assert sv.shape[1] == n_ch
        assert sv.shape[0] == len(freqs)

    def test_mode_shapes_shape(self):
        n_ch = 4
        signals = _sine_signals(n_channels=n_ch)
        cfg = _make_cfg(nperseg=128)
        freqs, sv, modes = compute_fdd(signals, fs=200.0, cfg=cfg)
        assert modes.ndim == 2
        assert modes.shape[0] == n_ch
        assert modes.shape[1] == len(freqs)

    def test_minimum_two_channels(self):
        signals = _sine_signals(n_channels=2)
        cfg = _make_cfg(nperseg=128)
        freqs, sv, modes = compute_fdd(signals, fs=200.0, cfg=cfg)
        assert modes.shape[0] == 2

    def test_many_channels(self):
        signals = _sine_signals(n_channels=8)
        cfg = _make_cfg(nperseg=256)
        freqs, sv, modes = compute_fdd(signals, fs=200.0, cfg=cfg)
        assert modes.shape[0] == 8


# ---------------------------------------------------------------------------
# Singular value ordering
# ---------------------------------------------------------------------------

class TestSingularValueOrdering:

    def test_sv_descending_at_every_freq(self):
        """Singular values must be non-increasing along the channel axis."""
        signals = _sine_signals(n_channels=4)
        cfg = _make_cfg(nperseg=128)
        _, sv, _ = compute_fdd(signals, fs=200.0, cfg=cfg)
        for k in range(sv.shape[1] - 1):
            assert np.all(sv[:, k] >= sv[:, k + 1] - 1e-10), (
                f"Singular values not descending at column {k} vs {k+1}"
            )

    def test_sv_non_negative(self):
        signals = _sine_signals(n_channels=3)
        cfg = _make_cfg(nperseg=128)
        _, sv, _ = compute_fdd(signals, fs=200.0, cfg=cfg)
        assert np.all(sv >= -1e-10)


# ---------------------------------------------------------------------------
# Phase alignment
# ---------------------------------------------------------------------------

class TestPhaseAlignment:

    def test_mode_shapes_are_real(self):
        signals = _sine_signals(n_channels=4)
        cfg = _make_cfg(nperseg=128)
        _, _, modes = compute_fdd(signals, fs=200.0, cfg=cfg)
        assert np.isrealobj(modes), "mode_shapes must be a real-valued array"

    def test_dynamic_ref_uses_max_abs_channel(self):
        """
        The dominant channel at peak frequency should have the largest
        absolute value in the mode-shape vector — confirming dynamic
        reference (argmax|u1|) was applied correctly.
        """
        freq_hz = 10.0
        fs = 200.0
        amplitudes = [0.1, 0.2, 2.0, 0.3]   # channel 2 dominates
        n = int(fs * 4.0)
        t = np.arange(n) / fs
        signals = np.column_stack([
            a * np.sin(2 * np.pi * freq_hz * t) for a in amplitudes
        ])
        cfg = _make_cfg(nperseg=256)
        freqs, sv, modes = compute_fdd(signals, fs=fs, cfg=cfg)

        peak_bin     = int(np.argmin(np.abs(freqs - freq_hz)))
        mode_at_peak = modes[:, peak_bin]
        dominant_ch  = int(np.argmax(np.abs(mode_at_peak)))

        assert dominant_ch == 2, (
            f"Expected channel 2 to dominate, got channel {dominant_ch}. "
            f"Mode at peak: {mode_at_peak}"
        )

    def test_mode_shapes_finite(self):
        signals = _sine_signals(
            freq_hz=8.0, fs=200.0, n_channels=4,
            phase_offsets=[0.0, 0.5, 1.0, 1.5],
        )
        cfg = _make_cfg(nperseg=256)
        _, _, modes = compute_fdd(signals, fs=200.0, cfg=cfg)
        assert np.all(np.isfinite(modes)), "Mode shapes contain NaN or Inf"


# ---------------------------------------------------------------------------
# CPSD Hermitian symmetry
# ---------------------------------------------------------------------------

class TestCpsdSymmetry:

    def test_sv_real_valued(self):
        """
        Singular values of a Hermitian PSD matrix are real and non-negative.
        If conjugate mirroring is correct, S_all is real.
        """
        signals = _sine_signals(n_channels=3)
        cfg = _make_cfg(nperseg=128)
        _, sv, _ = compute_fdd(signals, fs=200.0, cfg=cfg)
        assert np.isrealobj(sv) or np.allclose(np.imag(sv), 0, atol=1e-10)

    def test_fdd_peak_at_correct_frequency(self):
        """
        For a pure sine input the dominant singular value must peak at
        (or very close to) the input frequency.
        """
        freq_hz = 12.0
        fs      = 200.0
        signals = _sine_signals(
            freq_hz=freq_hz, fs=fs,
            duration=8.0, n_channels=3,
        )
        cfg = _make_cfg(nperseg=512, overlap=0.5)
        freqs, sv, _ = compute_fdd(signals, fs=fs, cfg=cfg)

        sv1         = sv[:, 0]
        peak_bin    = int(np.argmax(sv1))
        peak_freq   = float(freqs[peak_bin])
        freq_res    = float(freqs[1] - freqs[0])  # frequency resolution

        assert abs(peak_freq - freq_hz) <= 2 * freq_res, (
            f"FDD peak at {peak_freq:.3f} Hz, expected ~{freq_hz} Hz "
            f"(resolution {freq_res:.3f} Hz)"
        )