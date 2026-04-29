# data/

Place your CSV files here before running the examples or CLI.

## Wide format  (real sensor logger)

One row per time step; one column per sensor channel.

| time    | BS1    | BS2    | BS3    |
|---------|--------|--------|--------|
| 0:0:1:0 | 0.012  | -0.005 | 0.031  |

- Time column: any name from `timecol_candidates` (e.g. `time`, `step time`)
- Sensor columns: share a common prefix (default `BS`)

## Long format  (Abaqus output)

One row per node per time step.

| Step Time | Node Label | X    | AT2    |
|-----------|------------|------|--------|
| 0.001     | 81         | 0.24 | 0.012  |
| 0.001     | 91         | 0.50 | -0.005 |

- Supported directions: `AT1`, `AT2`, `AT3`
- Node selection: by label (`node_plot`) or by X-coord (`node_plot_by_x`)

## Generating synthetic data

```bash
python data/generate_sample_data.py
```

Writes `healthy_wide.csv`, `damaged_wide.csv`, `healthy_long.csv`,
and `damaged_long.csv` into this folder.
