"""Signal processing subpackage: spectral estimation and peak detection."""

from shm_tools.processing.peaks import find_peak_freqs
from shm_tools.processing.spectra import compute_spectrum, extract_mode_shapes

__all__ = [
    "compute_spectrum",
    "extract_mode_shapes",
    "find_peak_freqs",
]