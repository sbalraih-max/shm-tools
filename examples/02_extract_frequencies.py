"""
examples/02_extract_frequencies.py
====================================
Spectral analysis — extract natural frequencies and compare spectra.

Supports FFT, Welch PSD, and FDD methods via CONFIG["METHOD"].

Usage
-----
    python examples/02_extract_frequencies.py
    python examples/02_extract_frequencies.py --method welch
    python examples/02_extract_frequencies.py --method fdd
    python examples/02_extract_frequencies.py --method welch --save
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# ── Public API ────────────────────────────────────────────────────────────────
from shm_tools import ShmConfig, load_wide, load_long
from shm_tools.loaders.long import resolve_nodes_by_x
from shm_tools.processing.spectra import compute_spectrum, extract_mode_shapes
from shm_tools.fdd.core import compute_fdd
from shm_tools.processing.peaks import find_peak_freqs
from shm_tools.plotting.spectra import plot_spectrum, plot_spectrum_comparison

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

# =============================================================================
# CONFIG — edit these values to match your data
# =============================================================================
CONFIG = {
    # --- Input files ---------------------------------------------------------
    "HEALTHY_CSV"  : DATA_DIR / "healthy_wide.csv",
    "DAMAGED_CSV"  : DATA_DIR / "damaged_wide.csv",

    # --- Data format ---------------------------------------------------------
    "DATA_FORMAT"  : "wide",          # "wide" | "long"

    # --- Sampling & time window ----------------------------------------------
    "FS"           : 200.0,           # sampling frequency [Hz]
    "TIME_START"   : None,
    "TIME_END"     : None,

    # --- Wide-format sensor selection ----------------------------------------
    "SENSOR_PREFIX": "BS",
    "SENSOR_PLOT"  : None,            # list[int] e.g. [1, 2, 3] — None = all

    # --- Long-format node selection (pick ONE, set the other to None) --------
    "NODE_PLOT_BY_X": [0.2, 0.4, 0.6, 0.8, 1.0],  # select by X [m]  ← active
    "NODE_PLOT"     : None,                        # select by node ID e.g. [1, 2, 3]
    "X_TOLERANCE"   : 0.05,
    "DIRECTIONS"    : ["AT2"],

    # --- Spectral analysis ---------------------------------------------------
    "METHOD"       : "fft",           # "fft" | "welch" | "fdd"
    "FREQ_MIN"     : 0.5,             # [Hz] lower bound for peak search
    "FREQ_MAX"     : 50.0,            # [Hz] upper bound for peak search
    "N_MODES"      : 3,               # number of peaks to detect
    "PEAK_PROMINENCE"     : 0.1,      # fraction of amplitude range
    "PEAK_MIN_DISTANCE_HZ": 1.0,      # minimum spacing between peaks [Hz]

    # --- Welch parameters (used when METHOD = "welch" or "fdd") --------------
    "WELCH_NPERSEG": 1024,
    "WELCH_OVERLAP": 0.5,

    # --- Output --------------------------------------------------------------
    "OUTPUT_DIR"   : OUTPUT_DIR,
    "SAVE_FIGURES" : False,
    "OVERLAY"      : True,
    "FIGURE_DPI"   : 150,
}
# =============================================================================


def _resolve_cfg(cfg: ShmConfig, csv_path: Path) -> ShmConfig:
    """Inject node_plot into cfg from X-coords or direct labels."""
    if CONFIG["NODE_PLOT_BY_X"]:
        resolved = resolve_nodes_by_x(csv_path, cfg)
        if resolved:
            return replace(cfg, node_plot=list(resolved.keys()))
    elif CONFIG["NODE_PLOT"]:
        return replace(cfg, node_plot=list(CONFIG["NODE_PLOT"]))
    return cfg  # None → load all nodes


def _load(csv_path: Path, cfg: ShmConfig) -> np.ndarray:
    """Load CSV and return a 2-D signals array (n_samples, n_channels)."""
    if cfg.data_format == "wide":
        _, _, signals_dict = load_wide(csv_path, cfg)
        keys = list(signals_dict.keys())
        return np.column_stack([signals_dict[k]["accel"] for k in keys])
    else:
        nodes, _, signals_dict = load_long(csv_path, cfg)
        direction = (cfg.directions or ["AT2"])[0]
        return np.column_stack([
            signals_dict[n][direction]["accel"] for n in nodes
        ])


def _compute_spectrum_for_method(
    signals: np.ndarray,
    cfg: ShmConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, amplitudes) regardless of method."""
    if cfg.method == "fdd":
        freqs, singular_values, _ = compute_fdd(signals, cfg.fs, cfg)
        return freqs, singular_values[:, 0]   # dominant singular value
    return compute_spectrum(signals, cfg.fs, cfg)


def run(method: str, overlay: bool, save: bool) -> None:
    cfg = ShmConfig(
        data_format         = CONFIG["DATA_FORMAT"],
        fs                  = CONFIG["FS"],
        start               = CONFIG["TIME_START"],
        end                 = CONFIG["TIME_END"],
        sensor_prefix       = CONFIG["SENSOR_PREFIX"],
        sensor_plot         = CONFIG["SENSOR_PLOT"],
        directions          = CONFIG["DIRECTIONS"],
        node_plot_by_x      = CONFIG["NODE_PLOT_BY_X"],
        x_tolerance         = CONFIG["X_TOLERANCE"],
        method              = method,
        freq_min            = CONFIG["FREQ_MIN"],
        freq_max            = CONFIG["FREQ_MAX"],
        n_modes             = CONFIG["N_MODES"],
        peak_prominence     = CONFIG["PEAK_PROMINENCE"],
        peak_min_distance_hz= CONFIG["PEAK_MIN_DISTANCE_HZ"],
        welch_nperseg       = CONFIG["WELCH_NPERSEG"],
        welch_overlap       = CONFIG["WELCH_OVERLAP"],
    )

    # ── Resolve node selection for long format ────────────────────────────────
    if cfg.data_format == "long":
        cfg = _resolve_cfg(cfg, CONFIG["HEALTHY_CSV"])

    # ── Load ─────────────────────────────────────────────────────────────────
    sig_h = _load(CONFIG["HEALTHY_CSV"], cfg)
    sig_d = _load(CONFIG["DAMAGED_CSV"], cfg)

    print(f"  Method   : {method.upper()}")
    print(f"  Samples  : {sig_h.shape[0]}  Channels: {sig_h.shape[1]}")

    # ── Spectra ───────────────────────────────────────────────────────────────
    freqs_h, amp_h = _compute_spectrum_for_method(sig_h, cfg)
    freqs_d, amp_d = _compute_spectrum_for_method(sig_d, cfg)

    # ── Peak detection ────────────────────────────────────────────────────────
    peaks_h = find_peak_freqs(
        freqs_h, amp_h,
        n_peaks         = cfg.n_modes,
        freq_min        = cfg.freq_min,
        freq_max        = cfg.freq_max,
        prominence      = cfg.peak_prominence,
        min_distance_hz = cfg.peak_min_distance_hz,
    )
    peaks_d = find_peak_freqs(
        freqs_d, amp_d,
        n_peaks         = cfg.n_modes,
        freq_min        = cfg.freq_min,
        freq_max        = cfg.freq_max,
        prominence      = cfg.peak_prominence,
        min_distance_hz = cfg.peak_min_distance_hz,
    )

    print("\n  Natural frequencies detected:")
    print(f"    Healthy : {[f'{f:.3f} Hz' for f in peaks_h]}")
    print(f"    Damaged : {[f'{f:.3f} Hz' for f in peaks_d]}")
    if len(peaks_h) == len(peaks_d) and len(peaks_h) > 0:
        shifts = peaks_d - peaks_h
        print(f"    Δf      : {[f'{s:+.3f} Hz' for s in shifts]}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    ylabel = "Singular Value" if method == "fdd" else (
             "PSD" if method == "welch" else "Amplitude")

    out_dir = Path(CONFIG["OUTPUT_DIR"])
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    plot_spectrum_comparison(
        freqs_healthy      = freqs_h,
        amplitudes_healthy = amp_h,
        freqs_damaged      = freqs_d,
        amplitudes_damaged = amp_d,
        peak_freqs_healthy = peaks_h,
        peak_freqs_damaged = peaks_d,
        method  = method,
        overlay = overlay,
        title   = "Spectrum — Healthy vs Damaged",
        ylabel  = ylabel,
        save_path = out_dir / "02_spectrum_comparison.png" if save else None,
        dpi       = CONFIG["FIGURE_DPI"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Spectral frequency extraction.")
    parser.add_argument("--method", default=CONFIG["METHOD"],
                        choices=["fft", "welch", "fdd"],
                        help="Spectral method (default: %(default)s)")
    parser.add_argument("--overlay", action="store_true",
                        default=CONFIG["OVERLAY"],
                        help="Overlay healthy and damaged spectra")
    parser.add_argument("--save", action="store_true",
                        default=CONFIG["SAVE_FIGURES"],
                        help="Save figure to OUTPUT_DIR instead of displaying")
    args = parser.parse_args()

    try:
        run(method=args.method, overlay=args.overlay, save=args.save)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())