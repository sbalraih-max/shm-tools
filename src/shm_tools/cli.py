"""Command-line interface for shm_tools.

Entry point registered in pyproject.toml as ``shm-extract``.

Usage
-----
    shm-extract --healthy data/healthy.csv --damaged data/damaged.csv
    shm-extract --healthy data/healthy.csv --damaged data/damaged.csv --save
    shm-extract --healthy data/healthy.csv --damaged data/damaged.csv \\
                --method fdd --format wide --overlay --save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shm_tools.config.models import ShmConfig
from shm_tools.api import run_analysis


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for ``shm-extract``."""
    parser = argparse.ArgumentParser(
        prog="shm-extract",
        description=(
            "shm_tools — Structural Health Monitoring extraction CLI.\n"
            "Loads two CSV files (healthy and damaged), runs spectral\n"
            "analysis, extracts mode shapes, computes damage indicators,\n"
            "and produces comparison plots."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- required inputs --------------------------------------------------
    parser.add_argument(
        "--healthy",
        required=True,
        metavar="PATH",
        help="Path to the healthy-state CSV file.",
    )
    parser.add_argument(
        "--damaged",
        required=True,
        metavar="PATH",
        help="Path to the damaged/current-state CSV file.",
    )

    # --- method & format --------------------------------------------------
    parser.add_argument(
        "--method",
        default="fdd",
        choices=["fft", "welch", "fdd"],
        help="Signal processing method (default: fdd).",
    )
    parser.add_argument(
        "--format",
        default="wide",
        choices=["wide", "long"],
        dest="data_format",
        help="CSV data format: 'wide' (sensor columns) or 'long' (Abaqus). (default: wide)",
    )

    # --- sampling ---------------------------------------------------------
    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        metavar="HZ",
        help="Sampling frequency in Hz. Overrides ShmConfig default.",
    )

    # --- frequency band ---------------------------------------------------
    parser.add_argument(
        "--freq-min",
        type=float,
        default=None,
        metavar="HZ",
        help="Lower bound of frequency band of interest in Hz.",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=None,
        metavar="HZ",
        help="Upper bound of frequency band of interest in Hz.",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=None,
        metavar="N",
        help="Number of modes to extract (default: from ShmConfig).",
    )

    # --- output flags -----------------------------------------------------
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to --output-dir instead of displaying them.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Use overlay mode for time-history and spectrum plots.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Directory for saved figures (default: ./output).",
    )

    # --- sensor prefix (wide format) --------------------------------------
    parser.add_argument(
        "--sensor-prefix",
        default=None,
        metavar="PREFIX",
        help="Column prefix for sensor channels in wide format "
             "(e.g. 'ACC', 'CH'). Overrides ShmConfig default.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — parses arguments and runs the analysis pipeline.

    Parameters
    ----------
    argv : list[str] or None, optional
        Argument list (defaults to ``sys.argv[1:]`` when ``None``).

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- validate file paths ----------------------------------------------
    healthy_path = Path(args.healthy)
    damaged_path = Path(args.damaged)

    if not healthy_path.exists():
        print(
            f"[shm-extract] ERROR: healthy CSV not found: {healthy_path}",
            file=sys.stderr,
        )
        return 1
    if not damaged_path.exists():
        print(
            f"[shm-extract] ERROR: damaged CSV not found: {damaged_path}",
            file=sys.stderr,
        )
        return 1

    # --- build config -----------------------------------------------------
    cfg = ShmConfig(
        method=args.method,
        data_format=args.data_format,
    )

    # Apply CLI overrides to config fields where provided
    if args.fs is not None:
        cfg.fs = args.fs
    if args.freq_min is not None:
        cfg.freq_min = args.freq_min
    if args.freq_max is not None:
        cfg.freq_max = args.freq_max
    if args.n_modes is not None:
        cfg.n_modes = args.n_modes
    if args.sensor_prefix is not None:
        cfg.sensor_prefix = args.sensor_prefix

    # --- run --------------------------------------------------------------
    print(f"[shm-extract] Method   : {cfg.method.upper()}")
    print(f"[shm-extract] Format   : {cfg.data_format}")
    print(f"[shm-extract] Healthy  : {healthy_path}")
    print(f"[shm-extract] Damaged  : {damaged_path}")
    print(f"[shm-extract] Save     : {args.save}")
    print(f"[shm-extract] Overlay  : {args.overlay}")
    if args.save:
        print(f"[shm-extract] Output   : {Path(args.output_dir).resolve()}")

    try:
        results = run_analysis(
            healthy_csv=healthy_path,
            damaged_csv=damaged_path,
            cfg=cfg,
            output_dir=args.output_dir if args.save else None,
            save=args.save,
            overlay=args.overlay,
        )
    except Exception as exc:
        print(f"[shm-extract] ERROR: {exc}", file=sys.stderr)
        return 1

    # --- summary ----------------------------------------------------------
    print("\n[shm-extract] ── Results ──────────────────────────────────")
    pf_h = results["peak_freqs_healthy"]
    pf_d = results["peak_freqs_damaged"]
    print(f"  Natural frequencies (healthy) : "
          f"{[f'{f:.3f} Hz' for f in pf_h]}")
    print(f"  Natural frequencies (damaged) : "
          f"{[f'{f:.3f} Hz' for f in pf_d]}")
    n_modes = results["mode_shapes_healthy"].shape[1]
    print(f"  Modes extracted               : {n_modes}")
    di = results["damage_indicators"]
    for ind in ["MD", "MS", "MC"]:
        max_diff = di[ind]["diff"].max()
        max_pos  = results["coords"][di[ind]["diff"][:, 0].argmax()]
        print(f"  Max |Δ{ind}| (mode 1)          : "
              f"{max_diff:.6f}  at x = {max_pos:.4f} m")
    print("[shm-extract] ─────────────────────────────────────────────")
    if args.save:
        print(f"[shm-extract] Figures saved to: {Path(args.output_dir).resolve()}")
    print("[shm-extract] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())