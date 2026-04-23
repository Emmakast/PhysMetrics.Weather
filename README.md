# PhysBench

A benchmark toolkit for evaluating the **physical consistency** of AI weather
forecast models.  It computes eight physics-based diagnostic metrics against an
ERA5 or IFS HRES reference dataset and generates publication-quality plots and
summary tables.

---

## What does it evaluate?

| Metric | What it measures |
|---|---|
| **Dry Air Mass drift** | Is global dry-air mass conserved? |
| **Water Mass drift** | Is global water-vapour mass conserved? |
| **Total Energy drift** | Is global total energy conserved? |
| **Effective Resolution** | At what spatial scale does the model lose skill? |
| **Spectral Divergence** | How much does the kinetic-energy spectrum diverge from ERA5? |
| **Spectral Residual** | Excess/deficit small-scale energy vs ERA5 |
| **Hydrostatic Balance** | How well does the model satisfy hydrostatic balance? |
| **Geostrophic Balance** | How well does the model satisfy geostrophic balance? |

Metrics are computed at multiple forecast lead times (default: 12 h, 5 days,
10 days) and written to a long-format CSV for downstream analysis.

---

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for fast, reproducible
dependency management.

```bash
# 1. Install uv (one-time, if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/Emmakast/Physical-consistency-benchmark.git
cd Physical-consistency-benchmark

# 3. Create a virtual environment and install all dependencies
uv sync

# 4. Activate the environment
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

After installation the following commands are available on your `PATH`:

| Command | Purpose |
|---|---|
| `wb-eval` | Run physics evaluation and produce a CSV of metrics |
| `wb-spectrum` | Compute kinetic-energy / moisture spectra |
| `wb-summarize` | Aggregate per-date CSV into a summary table |
| `wb-plot-ts` | Plot timeseries of conservation metrics |
| `wb-plot-table` | Render colour-coded comparison tables |

### Alternative: plain pip

```bash
pip install -e .
```

---

## Inputs: what do you need to provide?

### Option A — use WeatherBench 2 data (no download required)

The tool is pre-configured to stream data directly from the public
[WeatherBench 2](https://weatherbench2.readthedocs.io/) Google Cloud Storage
buckets.  **No files need to be downloaded** — xarray reads only the slices it
needs over the network.

Running the command below evaluates Aurora predictions against ERA5 for the
year 2022 using the WB2 default paths:

```bash
wb-eval --year 2022 --model aurora --output results/aurora_2022.csv
```

Other models that are already in WB2:

```bash
# Pangu-Weather
wb-eval --model pangu \
  --prediction-zarr gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr \
  --year 2022 --output results/pangu_2022.csv

# GraphCast
wb-eval --model graphcast \
  --prediction-zarr gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr \
  --year 2020 --output results/graphcast_2020.csv
```

### Option B — bring your own prediction file

Point `--prediction-zarr` at **any Zarr store** — local or remote:

```bash
# Local Zarr directory
wb-eval --model my_model \
  --prediction-zarr /data/my_model_forecasts.zarr \
  --output results/my_model_2022.csv

# S3
wb-eval --model my_model \
  --prediction-zarr s3://my-bucket/forecasts.zarr \
  --output results/my_model_2022.csv

# Custom GCS bucket
wb-eval --model my_model \
  --prediction-zarr gs://my-bucket/forecasts.zarr \
  --output results/my_model_2022.csv
```

The Zarr store must contain forecast data with at least the following
variables (or common aliases):

| Physical quantity | Accepted variable names |
|---|---|
| Surface pressure | `surface_pressure`, `sp`, `ps` |
| Temperature (pressure levels) | `temperature`, `t` |
| U wind (pressure levels) | `u_component_of_wind`, `u` |
| V wind (pressure levels) | `v_component_of_wind`, `v` |
| Specific humidity | `specific_humidity`, `q` |
| Geopotential (pressure levels) | `geopotential`, `z` |

The ERA5 reference is always read from the public WB2 bucket by default;
you can override it with `--era5-zarr`.

---

## Running the full pipeline

### 1. Evaluate a model

```bash
wb-eval \
  --model aurora \
  --year 2022 \
  --workers 8 \
  --output results/physics_evaluation_aurora_2022.csv
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--year` | 2022 | Year to evaluate |
| `--dates` | — | Evaluate specific dates, e.g. `2022-01-01 2022-02-01` |
| `--month` | — | Evaluate a full month, e.g. `2022-06` |
| `--model` | `aurora` | Model name (used in the output filename) |
| `--prediction-zarr` | Aurora WB2 path | Zarr store for the model predictions |
| `--era5-zarr` | ERA5 WB2 path | Zarr store for the ERA5 reference |
| `--reference` | `era5` | Use `era5` or `ifs` (IFS HRES t=0) as the reference |
| `--lead-times` | `12h,5d,10d` | Comma-separated lead times to evaluate |
| `--workers` | 8 | Number of parallel workers |
| `--output` | auto | Output CSV path |
| `--quiet` | off | Suppress progress output |

### 2. Compute spectra

```bash
wb-spectrum ke \
  --model aurora \
  --year 2022 \
  --prediction-zarr gs://weatherbench2/datasets/aurora/2022-1440x721.zarr \
  --output results/ke_spectrum_aurora_2022.csv
```

Spectrum types: `ke` (500 hPa kinetic energy), `ke_850hpa`, `q` (moisture).

### 3. Summarize results

```bash
wb-summarize \
  --input results/physics_evaluation_aurora_2022.csv \
  --output results/physics_summary_aurora_2022.csv
```

### 4. Plot timeseries

```bash
# Single model
wb-plot-ts single results/physics_evaluation_aurora_2022.csv

# Multi-model overlay (auto-discovers CSVs in results/)
wb-plot-ts combined --exclude aurora_2022
```

### 5. Plot summary table

```bash
wb-plot-table  # auto-discovers summary CSVs and produces one PNG per lead time
```

---

## Output format

`wb-eval` writes a **long-format CSV**:

```
date,lead_time_hours,metric_name,model_value,era5_value,n_levels,sp_method
2022-01-01,12,geostrophic_rmse,2.45,2.10,13,direct_sp
2022-01-01,12,hydrostatic_rmse,1.23,0.98,13,direct_sp
2022-01-01,120,dry_mass_drift_pct_per_day,-0.002,,,
...
```

---

## Requirements

- Python ≥ 3.10
- Internet access for WeatherBench 2 data (GCS)  
  *(or a locally available Zarr store if using Option B)*

All Python dependencies are pinned in `uv.lock` and installed automatically
by `uv sync`.
