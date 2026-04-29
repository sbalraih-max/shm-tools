"""Prominence-based spectral peak detection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


def find_peak_freqs(
    freqs: NDArray[np.floating],
    amplitudes: NDArray[np.floating],
    n_peaks: int = 3,
    freq_min: float = 0.0,
    freq_max: float | None = None,
    prominence: float = 0.1,
    min_distance_hz: float = 0.5,
) -> NDArray[np.floating]:
    """Detect the dominant peak frequencies in a spectrum using prominence.

    Parameters
    ----------
    freqs : NDArray[np.floating]
        Frequency axis in Hz, shape ``(N,)``.
    amplitudes : NDArray[np.floating]
        Spectral amplitude (or singular value) array, shape ``(N,)``.
        Must be non-negative.
    n_peaks : int, optional
        Maximum number of peaks to return, by default 3.
    freq_min : float, optional
        Lower bound of the frequency band of interest in Hz, by default 0.0.
    freq_max : float or None, optional
        Upper bound of the frequency band of interest in Hz.
        ``None`` means use the Nyquist limit (last element of *freqs*).
    prominence : float, optional
        Minimum prominence threshold as a fraction of the global amplitude
        range, by default 0.1 (i.e. 10 %).
    min_distance_hz : float, optional
        Minimum separation between returned peaks in Hz, by default 0.5 Hz.

    Returns
    -------
    NDArray[np.floating]
        Sorted array of peak frequencies in Hz, length <= *n_peaks*.

    Notes
    -----
    Prominence is defined relative to the amplitude range inside the band,
    so the threshold automatically adapts to different signal levels without
    requiring manual tuning per dataset.

    The frequency resolution of the returned values is limited by the
    spacing of *freqs*; no interpolation is applied.
    """
    if freq_max is None:
        freq_max = float(freqs[-1])

    # --- restrict to band of interest ----------------------------------------
    band_mask = (freqs >= freq_min) & (freqs <= freq_max)
    band_freqs = freqs[band_mask]
    band_amp = amplitudes[band_mask]

    if band_amp.size == 0:
        return np.array([], dtype=float)

    # --- absolute prominence threshold ----------------------------------------
    amp_range = float(band_amp.max() - band_amp.min())
    abs_prominence = prominence * amp_range if amp_range > 0 else 0.0

    # --- minimum distance in samples ------------------------------------------
    df = float(band_freqs[1] - band_freqs[0]) if band_freqs.size > 1 else 1.0
    min_distance_samples = max(1, int(np.round(min_distance_hz / df)))

    # --- peak detection -------------------------------------------------------
    peaks_idx, properties = find_peaks(
        band_amp,
        prominence=abs_prominence,
        distance=min_distance_samples,
    )

    if peaks_idx.size == 0:
        return np.array([], dtype=float)

    # --- sort by prominence (descending) and take top-n ----------------------
    prom_values = properties["prominences"]
    sorted_order = np.argsort(prom_values)[::-1]
    top_idx = peaks_idx[sorted_order[:n_peaks]]

    # --- return sorted by frequency -------------------------------------------
    peak_freqs = np.sort(band_freqs[top_idx])
    return peak_freqs