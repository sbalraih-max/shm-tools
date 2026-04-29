"""Frequency Domain Decomposition (FDD) core implementation.

Preserves the following exactly from shm_damage_v1.7.py:
- Upper-triangle CPSD computation with conjugate mirroring
- Stacked (batch) SVD over the full frequency axis
- Phase alignment: per-frequency dynamic reference channel
  (the channel with the largest absolute component in u1 at each bin)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import csd

from shm_tools.config.models import ShmConfig


def compute_fdd(
    signals: NDArray[np.floating],
    fs: float,
    cfg: ShmConfig,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Run Frequency Domain Decomposition on a multi-channel signal.

    The CPSD matrix at each frequency line is assembled from upper-triangle
    cross-spectral densities computed via Welch's method, then mirrored to
    fill the lower triangle via the Hermitian conjugate.  A stacked SVD is
    performed over all frequency lines in a single ``np.linalg.svd`` call
    for efficiency.

    Phase alignment uses a **dynamic per-frequency reference channel**:
    at each frequency bin the channel with the largest absolute component
    in the dominant singular vector is chosen as the phase reference.
    This matches the original ``shm_damage_v1.7.py`` implementation exactly
    and guarantees numerical stability even when sensors sit near modal nodes.

    Parameters
    ----------
    signals : NDArray[np.floating]
        2-D array of shape ``(n_samples, n_channels)``.
    fs : float
        Sampling frequency in Hz.
    cfg : ShmConfig
        Configuration object. Uses ``welch_nperseg`` and ``welch_overlap``.

    Returns
    -------
    freqs : NDArray[np.floating]
        One-sided frequency axis in Hz, shape ``(M,)``.
    singular_values : NDArray[np.floating]
        Singular values at each frequency line, shape ``(M, n_channels)``.
        Column 0 is the dominant singular value used for peak picking.
    mode_shapes : NDArray[np.floating]
        Phase-aligned, real-valued dominant singular vectors at each
        frequency line, shape ``(n_channels, M)``.

    Notes
    -----
    Phase alignment formula (from shm_damage_v1.7.py):

        refidx   = argmax(|u1|, axis=channels)   # per frequency bin
        refvals  = u1[freq_idx, refidx]           # complex reference value
        u1_aligned = u1 * exp(-j * angle(refvals))

    The real part of ``u1_aligned`` is returned as the mode shape.
    """
    n_samples, n_channels = signals.shape
    nperseg  = cfg.welch_nperseg
    noverlap = int(nperseg * cfg.welch_overlap)

    # ------------------------------------------------------------------
    # Step 1: build CPSD tensor — upper triangle + conjugate mirror
    # ------------------------------------------------------------------
    # Use first auto-spectrum to determine frequency axis length
    f_ref, g00 = csd(
        signals[:, 0], signals[:, 0],
        fs=fs, nperseg=nperseg, noverlap=noverlap,
        window="hann", scaling="density", detrend="constant",
    )
    n_freqs = len(f_ref)

    # Pre-allocate: shape (n_channels, n_channels, n_freqs) → transpose later
    G = np.zeros((n_channels, n_channels, n_freqs), dtype=np.complex64)
    G[0, 0, :] = g00  # reuse already-computed auto-spectrum

    for i in range(n_channels):
        for j in range(i, n_channels):
            if i == 0 and j == 0:
                continue  # already filled above
            _, gij = csd(
                signals[:, i], signals[:, j],
                fs=fs, nperseg=nperseg, noverlap=noverlap,
                window="hann", scaling="density", detrend="constant",
            )
            G[i, j, :] = gij          # upper triangle
            if i != j:
                G[j, i, :] = np.conj(gij)  # conjugate mirror → Hermitian

    # Reshape to (n_freqs, n_channels, n_channels) for np.linalg.svd
    G_stack = np.transpose(G, (2, 0, 1))
    del G

    # ------------------------------------------------------------------
    # Step 2: batch SVD over all frequency lines at once
    # ------------------------------------------------------------------
    # np.linalg.svd with (..., M, N) input — processes all freq bins together
    U_all, S_all, _ = np.linalg.svd(G_stack, full_matrices=False,
                                     compute_uv=True)
    del G_stack
    # U_all : (n_freqs, n_channels, n_channels)
    # S_all : (n_freqs, n_channels)

    # ------------------------------------------------------------------
    # Step 3: dynamic per-frequency phase alignment (matches original)
    # ------------------------------------------------------------------
    # u1: dominant left singular vector at each frequency
    u1 = U_all[:, :, 0]          # shape (n_freqs, n_channels)

    # For each frequency bin pick the channel with the largest |u1| component
    refidx  = np.argmax(np.abs(u1), axis=1)               # (n_freqs,)
    refvals = u1[np.arange(n_freqs), refidx]              # (n_freqs,) complex
    phase   = np.exp(-1j * np.angle(refvals))             # (n_freqs,) rotation

    u1_aligned = u1 * phase[:, np.newaxis]                # (n_freqs, n_channels)
    del U_all

    # Real part → mode shapes; transpose to (n_channels, n_freqs)
    mode_shapes = np.real(u1_aligned).astype(np.float32).T

    return f_ref, S_all, mode_shapes