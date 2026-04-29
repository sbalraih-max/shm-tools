"""Tests for prominence-based peak detection.

Covers:
- Synthetic spectrum with known peaks detected correctly
- Frequency band filtering (freq_min / freq_max)
- Prominence threshold edge cases
- Return type and sorting
- Empty-spectrum and no-peak edge cases
"""

from __future__ import annotations

import numpy as np
import pytest

from shm_tools.processing.peaks import find_peak_freqs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spectrum_with_peaks(
    fs: float = 200.0,
    n_fft: int = 2048,
    peak_freqs_hz: list[float] | None = None,
    peak_amplitudes: list[float] | None = None,
    noise_level: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic spectrum with Gaussian bumps at known frequencies."""
    if peak_freqs_hz is None:
        peak_freqs_hz = [5.0, 15.0, 30.0]
    if peak_amplitudes is None:
        peak_amplitudes = [1.0, 0.6, 0.4]

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    amplitudes = np.full_like(freqs, noise_level)

    sigma = 0.3  # Hz — narrow Gaussian bump
    for fc, amp in zip(peak_freqs_hz, peak_amplitudes):
        amplitudes += amp * np.exp(-0.5 * ((freqs - fc) / sigma) ** 2)

    return freqs, amplitudes


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------

class TestFindPeakFreqsBasic:

    def test_detects_all_three_peaks(self):
        target_freqs = [5.0, 15.0, 30.0]
        freqs, amp = _spectrum_with_peaks(peak_freqs_hz=target_freqs)
        detected = find_peak_freqs(freqs, amp, n_peaks=3, prominence=0.05)
        assert len(detected) == 3

    def test_detected_freqs_close_to_targets(self):
        target_freqs = [5.0, 15.0, 30.0]
        freqs, amp = _spectrum_with_peaks(peak_freqs_hz=target_freqs)
        detected = find_peak_freqs(freqs, amp, n_peaks=3, prominence=0.05)
        for tf, df in zip(sorted(target_freqs), sorted(detected)):
            assert abs(tf - df) < 1.0, (
                f"Expected peak near {tf} Hz, got {df:.3f} Hz"
            )

    def test_returns_sorted_ascending(self):
        freqs, amp = _spectrum_with_peaks(peak_freqs_hz=[20.0, 5.0, 12.0])
        detected = find_peak_freqs(freqs, amp, n_peaks=3, prominence=0.05)
        assert list(detected) == sorted(detected), (
            "Returned peak frequencies must be sorted ascending"
        )

    def test_return_type_is_ndarray(self):
        freqs, amp = _spectrum_with_peaks()
        detected = find_peak_freqs(freqs, amp, n_peaks=3)
        assert isinstance(detected, np.ndarray)

    def test_n_peaks_limits_output(self):
        freqs, amp = _spectrum_with_peaks(peak_freqs_hz=[5.0, 15.0, 30.0])
        detected = find_peak_freqs(freqs, amp, n_peaks=2, prominence=0.05)
        assert len(detected) <= 2

    def test_top_peaks_by_prominence(self):
        """
        When n_peaks=1, the tallest (most prominent) peak must be returned,
        not just the first one in frequency order.
        """
        # Peak at 30 Hz is tallest
        freqs, amp = _spectrum_with_peaks(
            peak_freqs_hz=[5.0, 15.0, 30.0],
            peak_amplitudes=[0.3, 0.5, 1.0],
        )
        detected = find_peak_freqs(freqs, amp, n_peaks=1, prominence=0.05)
        assert len(detected) == 1
        assert abs(detected[0] - 30.0) < 1.0, (
            f"Expected most prominent peak near 30 Hz, got {detected[0]:.3f} Hz"
        )


# ---------------------------------------------------------------------------
# Frequency band filtering
# ---------------------------------------------------------------------------

class TestFrequencyBandFiltering:

    def test_freq_min_excludes_low_peaks(self):
        freqs, amp = _spectrum_with_peaks(peak_freqs_hz=[5.0, 20.0])
        detected = find_peak_freqs(
            freqs, amp, n_peaks=2,
            freq_min=10.0, prominence=0.05,
        )
        assert all(f >= 10.0 for f in detected), (
            f"Found peak below freq_min=10 Hz: {detected}"
        )

    def test_freq_max_excludes_high_peaks(self):
        freqs, amp = _spectrum_with_peaks(peak_freqs_hz=[5.0, 30.0])
        detected = find_peak_freqs(
            freqs, amp, n_peaks=2,
            freq_max=25.0, prominence=0.05,
        )
        assert all(f <= 25.0 for f in detected), (
            f"Found peak above freq_max=25 Hz: {detected}"
        )

    def test_band_isolates_single_peak(self):
        freqs, amp = _spectrum_with_peaks(peak_freqs_hz=[5.0, 15.0, 30.0])
        detected = find_peak_freqs(
            freqs, amp, n_peaks=3,
            freq_min=12.0, freq_max=18.0,
            prominence=0.05,
        )
        assert len(detected) == 1
        assert abs(detected[0] - 15.0) < 1.0

    def test_empty_band_returns_empty(self):
        """A band containing no signal returns an empty array."""
        freqs, amp = _spectrum_with_peaks(peak_freqs_hz=[5.0, 15.0])
        detected = find_peak_freqs(
            freqs, amp, n_peaks=3,
            freq_min=40.0, freq_max=50.0,
            prominence=0.05,
        )
        assert len(detected) == 0


# ---------------------------------------------------------------------------
# Prominence threshold edge cases
# ---------------------------------------------------------------------------

class TestProminenceThreshold:

    def test_high_prominence_suppresses_small_peaks(self):
        """With very high prominence only the tallest peak survives."""
        freqs, amp = _spectrum_with_peaks(
            peak_freqs_hz=[5.0, 15.0, 30.0],
            peak_amplitudes=[1.0, 0.05, 0.04],
        )
        detected = find_peak_freqs(
            freqs, amp, n_peaks=3,
            prominence=0.3,
        )
        assert len(detected) == 1
        assert abs(detected[0] - 5.0) < 1.0, (
            f"Expected only the 5 Hz peak, got {detected}"
        )

    def test_zero_prominence_finds_all_local_maxima(self):
        """With prominence=0 all three bumps should be found."""
        freqs, amp = _spectrum_with_peaks(
            peak_freqs_hz=[5.0, 15.0, 30.0],
            peak_amplitudes=[1.0, 0.6, 0.4],
            noise_level=0.0,
        )
        detected = find_peak_freqs(
            freqs, amp, n_peaks=5,
            prominence=0.0,
            min_distance_hz=2.0,
        )
        assert len(detected) >= 3

    def test_prominence_above_max_returns_empty(self):
        """Prominence > 1.0 means threshold > amplitude range → no peaks."""
        freqs, amp = _spectrum_with_peaks()
        detected = find_peak_freqs(freqs, amp, n_peaks=3, prominence=2.0)
        assert len(detected) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_flat_spectrum_returns_empty(self):
        """A perfectly flat spectrum has no peaks."""
        freqs = np.linspace(0, 100, 1024)
        amp   = np.ones(1024)
        detected = find_peak_freqs(freqs, amp, n_peaks=3, prominence=0.01)
        assert len(detected) == 0

    def test_single_spike_spectrum(self):
        """A spectrum with one spike returns exactly one peak."""
        freqs = np.fft.rfftfreq(2048, d=1.0 / 200.0)
        amp   = np.zeros_like(freqs)
        target = 10.0
        idx = int(np.argmin(np.abs(freqs - target)))
        amp[idx] = 1.0
        detected = find_peak_freqs(freqs, amp, n_peaks=3, prominence=0.05)
        assert len(detected) == 1
        assert abs(detected[0] - target) < 0.5

    def test_min_distance_prevents_duplicate_detection(self):
        """Two very close peaks separated by less than min_distance_hz
        should collapse to one detected peak."""
        freqs, amp = _spectrum_with_peaks(
            peak_freqs_hz=[10.0, 10.3],   # 0.3 Hz apart
            peak_amplitudes=[1.0, 0.9],
        )
        detected = find_peak_freqs(
            freqs, amp, n_peaks=5,
            prominence=0.05,
            min_distance_hz=2.0,          # enforce 2 Hz minimum separation
        )
        assert len(detected) == 1