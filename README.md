# PhysMetrics.Weather

An open-source, unified framework for evaluating the **physical consistency** of Machine Learning Weather Prediction (MLWP) models.  It computes nine physics-based diagnostic metrics against an
ERA5 or IFS HRES reference dataset and generates plots and
summary tables.

---

## What does it evaluate?

| Metric | What it measures |
|---|---|
| **Dry Air Mass drift** | Is global dry-air mass conserved? |
| **Water Mass drift** | Is global water-vapour mass conserved? |
| **Total Energy drift** | Is global total energy conserved? |
| **Effective Resolution** | At what spatial scale does the model lose skill? |
| **Spectral Divergence** | How much does the kinetic-energy spectrum diverge from reference data? |
| **Spectral Residual** | Excess/deficit small-scale energy vs reference data |
| **Hydrostatic Balance** | How well does the model satisfy hydrostatic balance? |
| **Geostrophic Balance** | How well does the model satisfy geostrophic balance? |
| **Lapse Rate** | How much does the lapse rate distribution diverge from the reference data? |

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
git clone https://github.com/Emmakast/PhysMetrics.Weather.git
cd PhysMetrics.Weather

# 3. Create a virtual environment and install all dependencies
uv sync

# 4. Activate the environment
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```
**NOTE:** this uv installation does not work for an anonymized GitHub. To still allow for easy verification of the code, there is a Notebook folder containing plots.

After installation the following commands are available on your `PATH`:

| Command | Purpose |
|---|---|
| `physmetrics-run` | Run physics evaluation and produce a CSV of metrics |
| `physmetrics-plot` | Generate all plots from evaluation CSVs |

You can also invoke them without activating the environment:

```bash
uv run physmetrics-run --help
uv run physmetrics-plot --help
```

### Alternative: plain pip

```bash
pip install -e .
```

---

## Inputs: what do you need to provide?

### Option A — use WeatherBench 2 data (no download required)

The framework is pre-configured to stream data directly from the public
[WeatherBench 2](https://weatherbench2.readthedocs.io/) Google Cloud Storage
buckets.  **No files need to be downloaded** — xarray reads only the slices it
needs over the network.

Running the command below evaluates Pangu-Weather predictions against ERA5 for the
year 2022 using the WB2 default paths:

```bash
uv run physmetrics-run --year 2022 --model pangu \
  --prediction-zarr gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr \
  --output results/pangu_2022.csv
```

Other models that are already in WB2:

```bash
# GraphCast
uv run physmetrics-run --model graphcast \
  --prediction-zarr gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr \
  --year 2020 --output results/graphcast_2020.csv
```

### Option B — bring your own prediction file

Point `--prediction-zarr` at **any Zarr store** — local or remote:

```bash
# Local Zarr directory
uv run physmetrics-run --model my_model \
  --prediction-zarr /data/my_model_forecasts.zarr \
  --output results/my_model_2022.csv

# S3
uv run physmetrics-run --model my_model \
  --prediction-zarr s3://my-bucket/forecasts.zarr \
  --output results/my_model_2022.csv

# Custom GCS bucket
uv run physmetrics-run --model my_model \
  --prediction-zarr gs://my-bucket/forecasts.zarr \
  --output results/my_model_2022.csv
```

You can also supply your own reference dataset instead of ERA5 (e.g., the IFS HRES analysis):

```bash
uv run physmetrics-run --model my_model \
  --prediction-zarr /data/my_model_forecasts.zarr \
  --reference ifs \
  --ref-zarr gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr \
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
you can override it with `--ref-zarr`.

---

## Running the full pipeline

### 1. Evaluate a model

```bash
uv run physmetrics-run \
  --model pangu \
  --prediction-zarr gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr \
  --year 2022 \
  --workers 8 \
  --output results/physics_evaluation_pangu_2022.csv
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--year` | 2022 | Year to evaluate |
| `--dates` | — | Evaluate specific dates, e.g. `2022-01-01 2022-02-01` |
| `--month` | — | Evaluate a full month, e.g. `2022-06` |
| `--model` | `model` | Model name (used in the output filename) |
| `--prediction-zarr` | WB2 path | Zarr store for the model predictions |
| `--ref-zarr` | ERA5 WB2 path | Zarr store for the reference dataset |
| `--reference` | `era5` | Use `era5` or `ifs` (IFS HRES t=0) as the reference |
| `--lead-times` | `12h,5d,10d` | Comma-separated lead times to evaluate |
| `--workers` | 16 | Number of parallel workers |
| `--output` | auto | Output CSV path |
| `--extended-spectra` | off | Also compute the q spectrum and the KE spectrum at 850 hPa (in addition to the default 500 hPa KE spectrum) |
| `--quiet` | off | Suppress progress output |

### 2. Plot results

```bash
uv run physmetrics-plot \
  --results-dir results/ \
  --outdir plots/
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--results-dir` | `../results` | Directory containing evaluation CSVs |
| `--outdir` | `../plots` | Directory to write plot images |

---

## SLURM example

```bash
#!/bin/bash
#SBATCH --job-name=eval_hres
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

MODEL="hres"
YEAR="${YEAR:-2020}"
WORKERS="${WORKERS:-16}"
PREDICTION_ZARR="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
REF_ZARR="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
OUTPUT_CSV="results/physics_evaluation_${MODEL}_${YEAR}.csv"

uv run physmetrics-run \
  --model "$MODEL" \
  --prediction-zarr "$PREDICTION_ZARR" \
  --reference ifs \
  --ref-zarr "$REF_ZARR" \
  --year "$YEAR" \
  --lead-times "12h,5d,10d" \
  --workers "$WORKERS" \
  --output "$OUTPUT_CSV"
```

---

## Output format

`physmetrics-run` writes a **long-format CSV**:

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

---

## License

This project is licensed under the [MIT License](LICENSE).

