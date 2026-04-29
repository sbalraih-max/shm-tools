# shm-tools

A Python library for **Structural Health Monitoring (SHM) signal processing**.
`shm-tools` loads acceleration time-histories from wide-format or long-format (Abaqus) CSV files,
extracts natural frequencies via FFT / Welch PSD / FDD, computes mode-shape-based damage indicators,
and produces publication-ready comparison plots — all from a single clean configuration block.

---

## Features

- **Two CSV formats** — wide-format (one column per sensor) and long-format (Abaqus node output)
- **Three spectral methods** — FFT, Welch PSD, Frequency Domain Decomposition (FDD)
- **Damage indicators** — Modal Displacement (MD), Modal Slope (MS), Modal Curvature (MC)
- **Overlay plots** — healthy vs. damaged time histories, spectra, and mode shapes
- **CLI entry point** — `shm-extract` for quick command-line use
- **Fully typed** — `ShmConfig` dataclass with explicit field names and defaults

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/sbalraih-max/shm-tools.git
cd shm-tools
pip install -e ".[dev]"
```

> **Python 3.10 or newer is required.**

To install without development tools:

```bash
pip install -e .
```

---

## Quick Start

### 1. Generate sample data

```bash
python data/generate_sample_data.py
```

This creates `healthy_wide.csv`, `damaged_wide.csv`, `healthy_long.csv`, and `damaged_long.csv`
inside the `data/` folder.

### 2. Run an example script

```bash
# Plot acceleration time histories
python examples/01_plot_acceleration.py

# Extract and compare natural frequencies
python examples/02_extract_frequencies.py --method fdd

# Compute and plot damage indicators
python examples/03_damage_indicators.py --method fdd --save
```

### 3. Use the library directly

```python
from shm_tools import ShmConfig, load_wide
from shm_tools.processing.spectra import extract_mode_shapes
from shm_tools.indicators.damage import damage_indicators

cfg = ShmConfig(
    dataformat="wide",
    fs=200.0,
    method="fdd",
    freq_min=0.5,
    freq_max=50.0,
    n_modes=3,
)

sensor_cols, coords, signals_dict = load_wide("data/healthy_wide.csv", cfg)
```

### 4. Use the CLI

```bash
shm-extract --method fdd --format long --overlay --save
```

| Flag | Description |
|---|---|
| `--method` | Spectral method: `fft`, `welch`, or `fdd` (default: from config) |
| `--format` | Data format: `wide` or `long` (default: from config) |
| `--overlay` | Overlay healthy and damaged plots |
| `--save` | Save figures to `output/` instead of displaying |

---

## Project Structure

```
shm-tools/
├── src/
│   └── shm_tools/          # Library source
│       ├── __init__.py     # Public API exports
│       ├── api.py          # run_analysis() top-level entry
│       ├── cli.py          # shm-extract CLI
│       ├── types.py        # Type aliases
│       ├── config/         # ShmConfig dataclass
│       ├── loaders/        # CSV loaders (wide + long formats)
│       ├── processing/     # Spectra, FDD, peak detection
│       ├── fdd/            # Frequency Domain Decomposition core
│       ├── indicators/     # MD / MS / MC damage indicators
│       └── plotting/       # Time-history, spectra, modal plots
├── examples/               # Runnable example scripts (01-04)
├── tests/                  # pytest unit tests
├── data/                   # Sample data + generator script
├── output/                 # Saved figures (git-ignored)
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.26 | Array operations, FFT |
| `scipy` | ≥ 1.12 | Welch PSD, SVD (FDD), peak finding |
| `pandas` | ≥ 2.1 | CSV loading |
| `matplotlib` | ≥ 3.8 | All plotting |

Install all dependencies automatically with `pip install -e .`.

---

## Running Tests

```bash
pytest
# or with coverage:
pytest --cov=shm_tools --cov-report=term-missing
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
