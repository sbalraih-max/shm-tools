#!/usr/bin/env python3
"""Generate synthetic wide-format and long-format CSVs for testing."""
from pathlib import Path
import numpy as np
import pandas as pd

FS    = 200.0
T     = 10.0
N     = int(FS * T)
MODES = [5.0, 12.5, 24.0]
NODES = [81, 91, 101, 111]
X_POS = [0.24, 0.50, 0.78, 1.02]
OUT   = Path(__file__).parent


def synthetic_signal(damage: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t   = np.arange(N) / FS
    sig = np.zeros(N)
    for i, f in enumerate(MODES):
        amp = 1.0 if not damage else 1.0 - 0.15 * (i + 1)
        sig += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, np.pi))
    sig += rng.normal(0, 0.05, N)
    return sig


def write_wide(label: str, damage: bool) -> None:
    t: dict[str, np.ndarray] = {"time": np.arange(N) / FS}
    for i in range(1, 5):
        t[f"BS{i}"] = synthetic_signal(damage, seed=i * (10 if damage else 1))
    pd.DataFrame(t).to_csv(OUT / f"{label}_wide.csv", index=False)
    print(f"  wrote {label}_wide.csv")


def write_long(label: str, damage: bool) -> None:
    t   = np.arange(N) / FS
    dfs = []
    for node, x in zip(NODES, X_POS):
        sig = synthetic_signal(damage, seed=node * (2 if damage else 1))
        dfs.append(pd.DataFrame({
            "Step Time": t, "Node Label": node, "X": x, "AT2": sig,
        }))
    pd.concat(dfs, ignore_index=True).to_csv(OUT / f"{label}_long.csv", index=False)
    print(f"  wrote {label}_long.csv")


if __name__ == "__main__":
    write_wide("healthy", damage=False)
    write_wide("damaged", damage=True)
    write_long("healthy", damage=False)
    write_long("damaged", damage=True)
    print("Done.")
