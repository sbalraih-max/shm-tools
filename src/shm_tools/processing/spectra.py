"""FFT and Welch PSD spectral estimation and mode-shape extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

from shm_tools.config.models import ShmConfig
from shm_tools.processing.peaks import find_peak_freqs


def compute_spectrum(
    signals: NDArray[np.floating],
    fs: float,
    cfg: ShmConfig,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute a frequency-domain spectrum from a multi-channel signal array.

    Parameters
    ----------
    signals : NDArray[np.floating]
        2-D array of shape ``(n_samples, n_channels)``.
    fs : float
        Sampling frequency in Hz.
    cfg : ShmConfig
        Configuration object.  ``cfg.method`` must be ``"fft"`` or
        ``"welch"``; use :func:`shm_tools.fdd.core.compute_fdd` for FDD.

    Returns
    -------
    freqs : NDArray[np.floating]
        One-sided frequency axis in Hz, shape ``(M,)``.
    amplitudes : NDArray[np.floating]
        Mean amplitude spectrum (FFT) or mean PSD (Welch) averaged across
        channels, shape ``(M,)``.

    Raises
    ------
    ValueError
        If ``cfg.method`` is not ``"fft"`` or ``"welch"``.
    """
    method = cfg.method.lower()

    if method == "fft":
        return _compute_fft(signals, fs)
    elif method == "welch":
        return _compute_welch(signals, fs, cfg)
    else:
        raise ValueError(
            f"compute_spectrum does not support method='{method}'. "
            "Use compute_fdd() for FDD."
        )


def _compute_fft(
    signals: NDArray[np.floating],
    fs: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute averaged one-sided FFT amplitude spectrum.

    Parameters
    ----------
    signals : NDArray[np.floating]
        Shape ``(n_samples, n_channels)``.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    freqs : NDArray[np.floating]
        One-sided frequency axis.
    amplitudes : NDArray[np.floating]
        Mean amplitude across channels (normalised by ``n_samples``).
    """
    n_samples = signals.shape[0]
    fft_vals = np.fft.rfft(signals, axis=0)
    amplitudes = np.abs(fft_vals) / n_samples
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mean_amp = amplitudes.mean(axis=1)
    return freqs, mean_amp


def _compute_welch(
    signals: NDArray[np.floating],
    fs: float,
    cfg: ShmConfig,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute averaged Welch PSD spectrum.

    Parameters
    ----------
    signals : NDArray[np.floating]
        Shape ``(n_samples, n_channels)``.
    fs : float
        Sampling frequency in Hz.
    cfg : ShmConfig
        Must supply ``welch_nperseg`` and ``welch_overlap``.

    Returns
    -------
    freqs : NDArray[np.floating]
        One-sided frequency axis.
    amplitudes : NDArray[np.floating]
        Mean PSD across channels.
    """
    nperseg = cfg.welch_nperseg
    noverlap = int(nperseg * cfg.welch_overlap)

    psds: list[NDArray[np.floating]] = []
    for ch in range(signals.shape[1]):
        f, pxx = welch(
            signals[:, ch],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
        )
        psds.append(pxx)

    freqs = f  # same for all channels
    mean_psd = np.mean(np.stack(psds, axis=0), axis=0)
    return freqs, mean_psd


def extract_mode_shapes(
    signals: NDArray[np.floating],
    fs: float,
    cfg: ShmConfig,
    peak_freqs: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Extract mode-shape vectors at dominant natural frequencies.

    Dispatches to FFT, Welch, or FDD depending on ``cfg.method``.

    Parameters
    ----------
    signals : NDArray[np.floating]
        Shape ``(n_samples, n_channels)``.
    fs : float
        Sampling frequency in Hz.
    cfg : ShmConfig
        Analysis configuration.
    peak_freqs : NDArray[np.floating] or None, optional
        Pre-computed peak frequencies in Hz.  If ``None``, peaks are
        detected automatically from the spectrum.

    Returns
    -------
    peak_freqs : NDArray[np.floating]
        Natural frequencies in Hz, shape ``(n_modes,)``.
    mode_shapes : NDArray[np.floating]
        Real-valued mode-shape matrix, shape ``(n_channels, n_modes)``.
        Each column is one mode shape, normalised to unit max absolute value.

    Notes
    -----
    For ``"fdd"`` the singular vectors at each peak are returned directly
    after phase alignment.  For ``"fft"`` and ``"welch"`` the mode shape
    at each peak is taken as the real part of the channel-wise FFT at the
    nearest frequency bin.
    """
    method = cfg.method.lower()

    if method == "fdd":
        from shm_tools.fdd.core import compute_fdd

        fdd_freqs, singular_values, mode_shapes = compute_fdd(signals, fs, cfg)
        if peak_freqs is None:
            peak_freqs = find_peak_freqs(
                fdd_freqs,
                singular_values[:, 0],
                n_peaks=cfg.n_modes,
                freq_min=cfg.freq_min,
                freq_max=cfg.freq_max,
                prominence=cfg.peak_prominence,
                min_distance_hz=cfg.peak_min_distance_hz,
            )
        # extract singular vectors at peak frequencies
        mode_shape_list: list[NDArray[np.floating]] = []
        for pf in peak_freqs:
            idx = int(np.argmin(np.abs(fdd_freqs - pf)))
            mode_shape_list.append(mode_shapes[:, idx])
        mode_matrix = np.column_stack(mode_shape_list) if mode_shape_list else np.empty((signals.shape[1], 0))
        return peak_freqs, _normalise_modes(mode_matrix)

    else:
        freqs, amplitudes = compute_spectrum(signals, fs, cfg)
        if peak_freqs is None:
            peak_freqs = find_peak_freqs(
                freqs,
                amplitudes,
                n_peaks=cfg.n_modes,
                freq_min=cfg.freq_min,
                freq_max=cfg.freq_max,
                prominence=cfg.peak_prominence,
                min_distance_hz=cfg.peak_min_distance_hz,
            )
        fft_vals = np.fft.rfft(signals, axis=0)
        fft_freqs = np.fft.rfftfreq(signals.shape[0], d=1.0 / fs)
        mode_shape_list = []
        for pf in peak_freqs:
            idx = int(np.argmin(np.abs(fft_freqs - pf)))
            vec = np.real(fft_vals[idx, :])
            mode_shape_list.append(vec)
        mode_matrix = np.column_stack(mode_shape_list) if mode_shape_list else np.empty((signals.shape[1], 0))
        return peak_freqs, _normalise_modes(mode_matrix)


def _normalise_modes(
    mode_matrix: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Normalise each mode-shape column to unit maximum absolute value."""
    if mode_matrix.size == 0:
        return mode_matrix
    max_abs = np.abs(mode_matrix).max(axis=0, keepdims=True)
    max_abs[max_abs == 0] = 1.0
    return mode_matrix / max_abs