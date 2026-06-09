#!/usr/bin/env python3
"""
Visualisation script for NeurIPS paper metrics.
Plots:
1. Spectra (12h, 120h, 240h)
2. Summary tables (RMSE, Conservation, Spectral) including Wasserstein Distance.
3. Timeseries (Dry Air, Water, Energy, Hydrostatic, Geostrophic).
4. Lapse Rate Distributions (3 regions, 3 lead times).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance

# ── Config ───────────────────────────────────────────────────────────────────

# Explicitly defining the global order for tables and legends
MODELS = ["hres", "pangu", "graphcast", "neuralgcm", "fuxi", "aurora"]
NICE = {
    "hres": "HRES", "pangu": "Pangu", "graphcast": "GraphCast",
    "neuralgcm": "NeuralGCM", "fuxi": "FuXi", "aurora": "Aurora",
}
MODEL_STYLES = {
    "aurora":    {"color": "#0072B2", "marker": "o"},  # Blue
    "pangu":     {"color": "#D55E00", "marker": "s"},  # Vermilion
    "fuxi":      {"color": "#009E73", "marker": "^"},  # Bluish Green
    "graphcast": {"color": "#000000", "marker": "D"},  # Black (High Contrast CB-safe)
    "neuralgcm": {"color": "#E69F00", "marker": "v"},  # Orange
    "hres":      {"color": "#56B4E9", "marker": "P"},  # Sky Blue
}

EARTH_RADIUS_KM = 6371.0


def infer_reference_label(results_dir: Path) -> str:
    """Infer reference dataset label from results directory naming."""
    name = results_dir.name.lower()
    if "ifs" in name:
        return "IFS"
    return "ERA5"


def pretty_region_name(region: str) -> str:
    """Map region keys to publication-friendly names."""
    mapping = {
        "tropics": "Tropics",
        "nh_mid": "Nor. HS",
        "sh_mid": "Sou. HS",
    }
    return mapping.get(region, region.replace("_", " ").title())

def get_model_baselines(summaries: dict[str, pd.DataFrame], metric: str) -> dict[str, float]:
    """Extract reference baselines for a specific metric for each model."""
    bases = {}
    for m, df in summaries.items():
        metric_col = next((c for c in ["metric_name", "metric", "variable", "name"] if c in df.columns), None)
        if metric_col:
            sub = df[df[metric_col] == metric]
            if not sub.empty:
                ref_col = next((c for c in ["ref_value", "mean_ref"] if c in sub.columns), None)
                if ref_col:
                    vals = pd.to_numeric(sub[ref_col], errors="coerce")
                    if vals.notna().any():
                        bases[m] = float(vals.mean())
    return bases

# ── 1. Timeseries plotting ───────────────────────────────────────────────────

def plot_timeseries(results_dir: Path, outdir: Path):
    """Plot timeseries across models."""
    csv_paths = list(results_dir.glob("time_series_*.csv"))
    if not csv_paths:
        print("No time_series_*.csv found.")
        return

    summaries = load_summaries(results_dir)

    frames = []
    for path in csv_paths:
        model = path.stem.replace("time_series_", "").split("_")[0]
        if model not in MODELS: continue
        df = pd.read_csv(path)
        df["model"] = model
        frames.append(df)
    
    if not frames: return
    df_all = pd.concat(frames, ignore_index=True)
    
    metrics = {
        "dry_mass_Eg": "Dry Air Mass (Eg)",
        "water_mass_kg": "Water Mass (kg)",
        "total_energy_J": "Total Energy (J)",
        "hydrostatic_rmse": "Hydrostatic RMSE Δ",
        "geostrophic_rmse": "Geostrophic RMSE Δ"
    }

    outdir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    for col, title in metrics.items():
        if col not in df_all.columns: continue

        bases = get_model_baselines(summaries, col) if "rmse" in col else {}
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Define a default ylabel just in case
        ylabel = title
        
        for model in MODELS:
            mdf = df_all[df_all["model"] == model].copy()
            if mdf.empty:
                continue
            style = MODEL_STYLES.get(model, {"color": "grey", "marker": "."})
            
            # For conservation metrics, compute relative change per date first,
            # then aggregate mean/std across dates at each forecast hour.
            if col in ["dry_mass_Eg", "water_mass_kg", "total_energy_J"]:
                rel_df = mdf[["date", "forecast_hour", col]].dropna().copy()
                if rel_df.empty:
                    continue

                # Baseline at the earliest lead for each initialization date.
                base = (
                    rel_df.sort_values("forecast_hour")
                    .groupby("date", as_index=False)
                    .first()[["date", col]]
                    .rename(columns={col: "base_val"})
                )
                rel_df = rel_df.merge(base, on="date", how="left")
                rel_df = rel_df[rel_df["base_val"].abs() > 0]
                if rel_df.empty: continue
            
                rel_df["rel_pct"] = (rel_df[col] - rel_df["base_val"]) / rel_df["base_val"] * 100.0
                agg = rel_df.groupby("forecast_hour")["rel_pct"].agg(["mean", "std"]).reset_index()
                
                style = MODEL_STYLES.get(model, {"color": "grey", "marker": "."})
                ax.plot(agg["forecast_hour"], agg["mean"], label=NICE.get(model, model), color=style["color"], marker=style["marker"], markersize=3)
                ax.fill_between(agg["forecast_hour"], agg["mean"] - agg["std"].fillna(0), agg["mean"] + agg["std"].fillna(0), color=style["color"], alpha=0.18, linewidth=0)
                
            else:
                # Raw means for Hydrostatic and Geostrophic RMSE
                mdf_clean = mdf.copy()
                if "rmse" in col:
                    base_val = bases.get(model, 0.0)
                    mdf_clean[col] = pd.to_numeric(mdf_clean[col], errors="coerce") - base_val

                agg = mdf_clean.groupby("forecast_hour")[col].agg(["mean", "std"]).reset_index()
                if agg.empty:
                    continue
                x = agg["forecast_hour"].values
                y = agg["mean"].values
                y_sigma = agg["std"].fillna(0.0).values
                
                # Abbreviate the y-axis label for RMSE 
                if col == "hydrostatic_rmse":
                    ylabel = "Δ RMSE (m²/s²)"
                elif col == "geostrophic_rmse":
                    ylabel = "Δ RMSE (m/s)"
                else:
                    ylabel = title
                
                style = MODEL_STYLES.get(model, {"color": "grey", "marker": "."})
                ax.plot(x, y, label=NICE.get(model, model), color=style["color"], marker=style["marker"], markersize=3)
                ax.fill_between(x, y - y_sigma, y + y_sigma, color=style["color"], alpha=0.18, linewidth=0)
                
            ax.set_title(title, fontsize=35)
            ax.set_xlabel("Forecast Hour", fontsize=30)
            ax.set_ylabel(ylabel, fontsize=35)
            
            # Specific legend placement: remove from hydrostatic, dry mass, water mass
            if col == "geostrophic_rmse":
                ax.legend(fontsize=24, loc="upper left")
            elif col == "total_energy_J":
                ax.legend(fontsize=24, bbox_to_anchor=(1.05, 1), loc="upper left")
                
            ax.tick_params(axis='both', which='major', labelsize=30)
            
            # Use bbox_inches="tight" to prevent the external legend from being clipped
            fig.savefig(outdir / f"ts_{col}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved timeseries plot for {col}")

# ── 2. Spectra plotting ──────────────────────────────────────────────────────

def plot_spectra(results_dir: Path, outdir: Path, leads=[12, 120, 240], reference_label: str | None = None):
    """Plot spectra for target lead times."""
    if reference_label is None:
        reference_label = infer_reference_label(results_dir)

    csv_paths = list(results_dir.glob("spectra_*.csv"))
    if not csv_paths:
        print("No spectra_*.csv found.")
        return

    frames = []
    for path in csv_paths:
        model = path.stem.replace("spectra_", "").split("_")[0]
        if model not in MODELS: continue
        df = pd.read_csv(path)
        df["model"] = model
        frames.append(df)
        
    if not frames: return
    df_all = pd.concat(frames, ignore_index=True)
    outdir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    for lt in leads:
        sub = df_all[(df_all["lead_hours"] == lt) & (df_all["variable"] == "KE") & (df_all["wavenumber"] > 0)]
        if sub.empty: continue
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot reference from mean across dates
        ref_agg = sub.groupby("wavenumber")["power_ref"].mean().reset_index()
        if not ref_agg.empty:
            wl = 2.0 * np.pi * EARTH_RADIUS_KM / ref_agg["wavenumber"].values
            ax.loglog(wl, ref_agg["power_ref"].values, color="black", linewidth=2, label=reference_label, zorder=5)

        for model in MODELS:
            msub = sub[sub["model"] == model]
            if msub.empty: continue
            msub_agg = msub.groupby("wavenumber")["power_pred"].mean().reset_index()
            style = MODEL_STYLES.get(model, {"color": "grey"})
            wl = 2.0 * np.pi * EARTH_RADIUS_KM / msub_agg["wavenumber"].values
            ax.loglog(wl, msub_agg["power_pred"].values, color=style["color"], linewidth=1.5, label=NICE.get(model, model))
            
        ax.set_title(f"KE Spectrum - {lt}h", fontsize=45)
        ax.set_xlabel("Wavelength (km)", fontsize=35)
        ax.set_ylabel("Kinetic Energy", fontsize=35)
        ax.set_xlim(40000, 100) # Prevents matplotlib log-locator hang on inverted axis
        
        # Only show legend for 240h plot
        if lt == 240:
            ax.legend(fontsize=24, bbox_to_anchor=(1.05, 1), loc="upper left")
            
        ax.tick_params(axis='both', which='major', labelsize=30)
        
        fig.savefig(outdir / f"spectra_ke_{lt}h.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved spectra plot for {lt}h")

# ── 3. Lapse Rate Distributions ──────────────────────────────────────────────

def plot_lapse_rates(results_dir: Path, outdir: Path, leads=[12, 120, 240], reference_label: str | None = None):
    """Plot lapse rate distributions as line curves for easier model comparison."""
    if reference_label is None:
        reference_label = infer_reference_label(results_dir)

    csv_paths = list(results_dir.glob("lapse_rate_dist_*.csv"))
    if not csv_paths:
        print("No lapse_rate_dist_*.csv found.")
        return

    frames = []
    for path in csv_paths:
        model = path.stem.replace("lapse_rate_dist_", "").split("_")[0]
        if model not in MODELS: continue
        df = pd.read_csv(path)
        df["model"] = model
        frames.append(df)
        
    if not frames: return
    df_all = pd.concat(frames, ignore_index=True)
    outdir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    regions = df_all["region"].unique()
    
    for region in regions:
        fig, axes = plt.subplots(1, len(leads), figsize=(18, 5), sharey=True)
        if len(leads) == 1: axes = [axes]
        legend_handles = None
        legend_labels = None
        
        for ax, lt in zip(axes, leads):
            sub = df_all[(df_all["lead_hours"] == lt) & (df_all["region"] == region)]
            if sub.empty: continue
            y_max = 0.0
            
            # Determine bin width for center conversion (supports uneven bins via median).
            b_unique = np.sort(sub["bin_edge_lower"].unique())
            width = float(np.median(np.diff(b_unique))) if len(b_unique) > 1 else 0.5

            # Plot reference as a thick black line.
            ref_agg = (
                sub.groupby("bin_edge_lower", as_index=False)["freq_ref"]
                .mean()
                .sort_values("bin_edge_lower")
            )
            if not ref_agg.empty:
                x_ref = ref_agg["bin_edge_lower"].to_numpy() + 0.5 * width
                y_ref = ref_agg["freq_ref"].to_numpy()
                if y_ref.size:
                    y_max = max(y_max, float(np.nanmax(y_ref)))
                ax.plot(x_ref, y_ref, color="black", linewidth=2.4, label=reference_label, zorder=10)

            # Plot model distributions as colored lines.
            for i, model in enumerate(MODELS):
                msub = sub[sub["model"] == model]
                if msub.empty: continue
                style = MODEL_STYLES.get(model, {"color": "grey"})
                m_agg = (
                    msub.groupby("bin_edge_lower", as_index=False)["freq_pred"]
                    .mean()
                    .sort_values("bin_edge_lower")
                )
                x = m_agg["bin_edge_lower"].to_numpy() + 0.5 * width
                y = m_agg["freq_pred"].to_numpy()
                if y.size:
                    y_max = max(y_max, float(np.nanmax(y)))
                ax.plot(
                    x,
                    y,
                    color=style["color"],
                    linewidth=1.7,
                    alpha=0.95,
                    label=NICE.get(model, model),
                    zorder=3 + i,
                )
            
            # Restrict x-axis to the non-zero region
            non_zero = sub[(sub["freq_ref"].fillna(0) > 1e-5) | (sub["freq_pred"].fillna(0) > 1e-5)]
            if not non_zero.empty:
                lower_bound = non_zero["bin_edge_lower"].min()
                upper_bound = non_zero["bin_edge_lower"].max()
                # give a tiny pad
                pad = (upper_bound - lower_bound) * 0.05
                ax.set_xlim(lower_bound - pad, upper_bound + pad)

            if y_max > 0:
                ax.set_ylim(0, y_max * 1.20)
            else:
                ax.set_ylim(bottom=0)
            ax.set_title(f"{pretty_region_name(region)} - {lt}h", fontsize=45)
            ax.set_xlabel("Lapse Rate (K/km)", fontsize=35)
            if ax == axes[0]: ax.set_ylabel("Density", fontsize=35)

            ax.tick_params(axis='both', which='major', labelsize=30)

            # Cache legend entries from the first populated panel.
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            
        if legend_handles:
            axes[-1].legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=28)
        fig.tight_layout()
        fig.savefig(outdir / f"lapse_rate_{region}.png", dpi=200)
        plt.close(fig)
        print(f"Saved lapse rate plot for {region}")


def plot_lapse_rates_240h_compact(results_dir: Path, outdir: Path, reference_label: str | None = None):
    """Create a compact 240h lapse-rate figure with three regional panels."""
    if reference_label is None:
        reference_label = infer_reference_label(results_dir)

    csv_paths = list(results_dir.glob("lapse_rate_dist_*.csv"))
    if not csv_paths:
        return

    frames = []
    for path in csv_paths:
        model = path.stem.replace("lapse_rate_dist_", "").split("_")[0]
        if model not in MODELS:
            continue
        df = pd.read_csv(path)
        df["model"] = model
        frames.append(df)

    if not frames:
        return

    df_all = pd.concat(frames, ignore_index=True)
    sub_240 = df_all[df_all["lead_hours"] == 240]
    if sub_240.empty:
        return

    # Fixed regional order for paper figure.
    regions = [r for r in ["tropics", "sh_mid", "nh_mid"] if r in set(sub_240["region"].unique())]
    if not regions:
        regions = sorted(sub_240["region"].unique())

    fig, axes = plt.subplots(1, len(regions), figsize=(10, 5), sharey=True)
    if len(regions) == 1:
        axes = [axes]

    legend_handles = None
    legend_labels = None
    global_ymax = 0.0

    for panel_idx, (ax, region) in enumerate(zip(axes, regions)):
        sub = sub_240[sub_240["region"] == region]
        if sub.empty:
            continue

        b_unique = np.sort(sub["bin_edge_lower"].unique())
        width = float(np.median(np.diff(b_unique))) if len(b_unique) > 1 else 0.5

        ref_agg = (
            sub.groupby("bin_edge_lower", as_index=False)["freq_ref"]
            .mean()
            .sort_values("bin_edge_lower")
        )
        if not ref_agg.empty:
            x_ref = ref_agg["bin_edge_lower"].to_numpy() + 0.5 * width
            y_ref = ref_agg["freq_ref"].to_numpy()
            if y_ref.size:
                global_ymax = max(global_ymax, float(np.nanmax(y_ref)))
            ax.plot(x_ref, y_ref, color="black", linewidth=2.2, label=reference_label, zorder=10)

        for i, model in enumerate(MODELS):
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue
            style = MODEL_STYLES.get(model, {"color": "grey"})
            m_agg = (
                msub.groupby("bin_edge_lower", as_index=False)["freq_pred"]
                .mean()
                .sort_values("bin_edge_lower")
            )
            x = m_agg["bin_edge_lower"].to_numpy() + 0.5 * width
            y = m_agg["freq_pred"].to_numpy()
            if y.size:
                global_ymax = max(global_ymax, float(np.nanmax(y)))
            ax.plot(x, y, color=style["color"], linewidth=1.4, alpha=0.95,
                    label=NICE.get(model, model), zorder=3 + i)

        ax.set_xlim(1, 10)
        ax.set_title(pretty_region_name(region), fontsize=28)
        
        # Only set xlabel for the middle lapse rate panel to save space
        if region == regions[1] if len(regions) > 1 else regions[0]:
            ax.set_xlabel("Lapse Rate", fontsize=24)
            
        ax.tick_params(axis='both', which='major', labelsize=18)
        
    for ax in axes[2:]:
        if global_ymax > 0: ax.set_ylim(0, global_ymax * 1.15)
        else: ax.set_ylim(bottom=0)
    axes[2].set_ylabel("Density", fontsize=24)

    handles, labels = _get_shared_legend_handles(axes)
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=len(labels), fontsize=22)

    fig.tight_layout()
    fig.savefig(outdir / "lapse_rate_240h_compact.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved compact 240h lapse rate plot")

# ── 4. Summary Table & Wasserstein ──────────────────────────────────────────

def calculate_wasserstein(results_dir: Path) -> dict:
    """Calculate Wasserstein distances for lapse rates to include in balance table."""
    csv_paths = list(results_dir.glob("lapse_rate_dist_*.csv"))
    w_dists = {}
    for path in csv_paths:
        model = path.stem.replace("lapse_rate_dist_", "").split("_")[0]
        df = pd.read_csv(path)
        # Approximate computation from binned data
        model_dists = {}
        for lt in [12, 120, 240]:
            sub = df[df["lead_hours"] == lt]
            if sub.empty: continue
            # Average across regions for a global mean Wasserstein distance
            lt_dists = []
            for region in sub["region"].unique():
                r_sub = sub[sub["region"] == region]
                u = r_sub["freq_pred"].values
                v = r_sub["freq_ref"].values
                lt_dists.append(wasserstein_distance(u, v))
            model_dists[lt] = np.mean(lt_dists)
        w_dists[model] = model_dists
    return w_dists

# ── Summary Tables (Three Separate Tables) ──────────────────────────────────

def load_summaries(results_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for m in MODELS + ["era5"]:
        # Check standard names
        paths = [
            results_dir / f"physics_evaluation_{m}_2020.csv",
            results_dir / f"physics_summary_{m}_2020.csv",
            results_dir / f"physics_summary_{m}_2020_vs_era5.csv",
            results_dir / f"physics_summary_{m}_s3_2022.csv",
            results_dir / f"physics_summary_{m}_2022.csv"
        ]
        for p in paths:
            if p.exists():
                out[m] = pd.read_csv(p)
                break
    return out

def get_value(df: pd.DataFrame, metric: str, is_ref: bool = False) -> float:
    # Dynamically find the appropriate columns
    metric_col = next((c for c in ["metric_name", "metric", "variable", "name"] if c in df.columns), None)
    if not metric_col: return np.nan
    
    sub = df[df[metric_col] == metric]
    if sub.empty: return np.nan
    
    val_col = next((c for c in (["ref_value", "mean_ref"] if is_ref else ["model_value", "mean_value", "mean_model", "value", "mean", "score"]) if c in sub.columns), None)
    if not val_col: return np.nan
    
    val = sub[val_col].astype(float).mean()
    return float(val) if pd.notna(val) else np.nan

def fmt(val: float, metric: str) -> str:
    if np.isnan(val): return "—"
    if "drift" in metric: return f"{val:+.4f}"
    if "rmse" in metric: return f"{val:.2f}"
    if "resolution" in metric: return f"{val:.1f}"
    if "wasserstein" in metric: return f"{val:.4f}"
    return f"{val:.4f}"


def _read_hardcoded_rmse(path: Path) -> dict[str, float]:
    """Read hardcoded RMSE values from a simple CSV-like text file."""
    out: dict[str, float] = {}
    if not path.exists():
        return out

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        metric, value = parts
        try:
            out[metric] = float(value)
        except ValueError:
            continue
    return out


def _metric_mean(df: pd.DataFrame, metric: str, prefer_ref: bool = True) -> float:
    """Return global mean for metric from either ref_value or model_value."""
    metric_col = next((c for c in ["metric_name", "metric", "variable", "name"] if c in df.columns), None)
    if metric_col is None:
        return np.nan

    sub = df[df[metric_col] == metric]
    if sub.empty:
        return np.nan

    pref_cols = ["ref_value", "mean_ref"] if prefer_ref else ["model_value", "mean_value", "mean_model", "value", "mean", "score"]
    fallback_cols = ["model_value", "mean_value", "mean_model", "value", "mean", "score"] if prefer_ref else ["ref_value", "mean_ref"]

    for col in pref_cols + fallback_cols:
        if col in sub.columns:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if not vals.empty:
                return float(vals.mean())
    return np.nan


def get_or_compute_hardcoded_rmse(results_dir: Path, summaries: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Load or compute hardcoded RMSE constants used for delta tables."""
    target_file = results_dir / ("ifs_0_7_rmse.txt" if "ifs" in results_dir.name else "era5_0_7_rmse.txt")

    # Backward-compatible names that may already exist.
    candidate_files = [
        target_file,
        results_dir / "era5_0_7_rmse.txt",
        results_dir / "ifs_0_7_rmse.txt",
        results_dir / "hres_0_7_rmse.txt",
    ]
    for f in candidate_files:
        vals = _read_hardcoded_rmse(f)
        if "hydrostatic_rmse" in vals and "geostrophic_rmse" in vals:
            print(f"Using hardcoded RMSE from {f}")
            return vals

    # Compute constants first, then persist for reproducibility.
    source = summaries.get("era5") or summaries.get("hres")
    if source is None:
        print("WARNING: Could not compute hardcoded RMSE (missing era5/hres summary).")
        return {}

    out = {
        "hydrostatic_rmse": _metric_mean(source, "hydrostatic_rmse", prefer_ref=True),
        "geostrophic_rmse": _metric_mean(source, "geostrophic_rmse", prefer_ref=True),
    }

    # If still missing, retry from model values as fallback.
    if np.isnan(out["hydrostatic_rmse"]):
        out["hydrostatic_rmse"] = _metric_mean(source, "hydrostatic_rmse", prefer_ref=False)
    if np.isnan(out["geostrophic_rmse"]):
        out["geostrophic_rmse"] = _metric_mean(source, "geostrophic_rmse", prefer_ref=False)

    if np.isnan(out["hydrostatic_rmse"]) or np.isnan(out["geostrophic_rmse"]):
        print("WARNING: Incomplete hardcoded RMSE computation; using available values only.")
        return {k: v for k, v in out.items() if not np.isnan(v)}

    target_file.write_text(
        "hydrostatic_rmse," + f"{out['hydrostatic_rmse']:.12f}" + "\n"
        "geostrophic_rmse," + f"{out['geostrophic_rmse']:.12f}" + "\n"
    )
    print(f"Computed and saved hardcoded RMSE to {target_file}")
    return out

def plot_summary_tables(results_dir: Path, outdir: Path, leads=[12, 120, 240]):
    """Render three grouped transposed tables, injecting Wasserstein to Balance."""
    outdir.mkdir(exist_ok=True)
    summaries = load_summaries(results_dir)
    w_dists = calculate_wasserstein(results_dir)
    hardcoded_rmse = get_or_compute_hardcoded_rmse(results_dir, summaries)
    
    models_to_plot = [m for m in MODELS if m != "aurora"]
    
    groups = {
        "Conservation_Variability": [
            ("dry_mass_drift_pct_per_day", "Dry Mass Drift →0 [%/day]"),
            ("water_mass_drift_pct_per_day", "Water Mass Drift →0 [%/day]"),
            ("total_energy_drift_pct_per_day", "Total Energy Drift →0 [%/day]"),
        ],
        "Spectral": [
            ("effective_resolution_km", "Eff. Resolution ↓111.5 [km]"),
            ("spectral_residual", "Spec. Residual ↓0"),
            ("spectral_divergence", "Spec. Divergence ↓0"),
        ],
        "Balance": [
            ("geostrophic_rmse", "Geostrophic RMSE Δ →0 [m/s]"),
            ("hydrostatic_rmse", "Hydrostatic RMSE Δ →0 [m²/s²]"),
            ("lapse_rate_wasserstein", "Mean Lapse Rate W-Dist ↓0"),
        ]
    }
    
    header_color = np.array([0.9, 0.9, 0.9])
    red = np.array([1.0, 0.75, 0.75])
    white = np.array([1.0, 1.0, 1.0])

    for group_name, metrics_list in groups.items():
        current_models = [m for m in models_to_plot if m != "fuxi"] if group_name == "Conservation_Variability" else models_to_plot
        model_labels = [NICE.get(m, m) for m in current_models]

        max_abs = {}
        for metric, _ in metrics_list:
            vals = []
            for m in current_models:
                if metric == "lapse_rate_wasserstein":
                    if m in w_dists: vals.extend(w_dists[m].values())
                elif m in summaries:
                    df = summaries[m]
                    for lead in leads:
                        lead_col = next((c for c in ["lead_hours", "lead_time_hours", "lead_time", "forecast_hour"] if c in df.columns), None)
                        df_lt = df[df[lead_col] == lead] if lead_col else df
                        if df_lt.empty and lead_col:
                            avail = sorted(df[lead_col].dropna().unique())
                            if avail: df_lt = df[df[lead_col] == min(avail, key=lambda x: abs(x - lead))]
                        if not df_lt.empty:
                            val = get_value(df_lt, metric)
                            if metric in ["hydrostatic_rmse", "geostrophic_rmse"] and not np.isnan(val):
                                ref_val = get_value(df_lt, metric, is_ref=True)
                                if not np.isnan(ref_val):
                                    val -= ref_val
                            if not np.isnan(val): vals.append(val)
            max_abs[metric] = max([abs(v) for v in vals]) if vals else 1.0
            if max_abs[metric] == 0: max_abs[metric] = 1.0

        cell_texts = [["Metric", "Lead Time"] + model_labels]
        cell_colors = [[header_color] * len(cell_texts[0])]
            
        for metric, m_label in metrics_list:
            for l_idx, lead in enumerate(leads):
                text_label = m_label if l_idx == len(leads)//2 else ""
                row_t = [text_label, f"{lead}h"]
                row_c = [white.copy(), white.copy()]

                for m in current_models:
                    val = np.nan
                    suffix = ""
                    if metric == "lapse_rate_wasserstein":
                        if m in w_dists and lead in w_dists[m]:
                            val = w_dists[m][lead]
                    elif m in summaries:
                        df = summaries[m]
                        lead_col = next((c for c in ["lead_hours", "lead_time_hours", "lead_time", "forecast_hour"] if c in df.columns), None)
                        if lead_col:
                            df_lt = df[df[lead_col] == lead]
                            if df_lt.empty:
                                avail = sorted(df[lead_col].dropna().unique())
                                if avail:
                                    nearest = min(avail, key=lambda x: abs(x - lead))
                                    df_lt = df[df[lead_col] == nearest]
                                    val = get_value(df_lt, metric)
                                    suffix = " *"
                            else:
                                val = get_value(df_lt, metric)
                                
                    if metric in ["hydrostatic_rmse", "geostrophic_rmse"] and not np.isnan(val):
                        ref_val = get_value(df_lt, metric, is_ref=True)
                        if not np.isnan(ref_val):
                            val -= ref_val
                            
                    if np.isnan(val):
                        row_t.append("—")
                        row_c.append(white)
                    else:
                        row_t.append(fmt(val, metric) + suffix)
                        if metric == "effective_resolution_km" and m in ["hres", "neuralgcm"]:
                            row_c.append(white)
                        else:
                            intensity = min(abs(val) / max_abs[metric], 1.0) * 0.8
                            row_c.append(white * (1 - intensity) + red * intensity)
                    
                cell_texts.append(row_t)
                cell_colors.append(row_c)

        n_cols = len(cell_texts[0])
        n_rows = len(cell_texts)
        # Match vertical multiplier to font size so rows aren't overlapping
        fig, ax = plt.subplots(figsize=(max(1.8 * n_cols, 12), max(0.3 * n_rows, 3.0)))
        ax.axis("off")
        
        # Make the metric name column and the data columns slightly wider
        colWidths = [0.45 if group_name in ["Balance", "Conservation_Variability"] else 0.35, 0.12] + [0.15] * len(current_models)
        table = ax.table(
            cellText=cell_texts,
            cellColours=[[tuple(c) for c in row] for row in cell_colors],
            colWidths=colWidths, loc="center", cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(14)  # Increased font size 
        # Set vertical scaling to provide enough padding for the larger font
        table.scale(1.0, 1.6)

        for j in range(n_cols):
            table[0, j].set_text_props(fontweight="bold")
            table[0, j].set_facecolor(tuple(header_color))

        row_idx = 1
        for _ in metrics_list:
            for r in range(row_idx, row_idx + len(leads)):
                if r == row_idx: table[r, 0].visible_edges = 'LRT'
                elif r == row_idx + len(leads) - 1: table[r, 0].visible_edges = 'LRB'
                else: table[r, 0].visible_edges = 'LR'
                table[r, 0].set_text_props(fontweight="bold")
            row_idx += len(leads)

        fig.savefig(outdir / f"neurips_table_{group_name.lower()}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved summary table for {group_name}")


# ── 5. Combined Multi-Panel Plots ────────────────────────────────────────────

def _get_shared_legend_handles(axes):
    handles_dict = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in handles_dict:
                handles_dict[label] = handle
    return list(handles_dict.values()), list(handles_dict.keys())

def plot_combined_conservation(results_dir: Path, outdir: Path):
    """1x3 Combined panel for Dry air, Water mass, Total energy."""
    frames = []
    for path in results_dir.glob("time_series_*.csv"):
        m = path.stem.replace("time_series_", "").split("_")[0]
        if m in MODELS:
            df = pd.read_csv(path)
            df["model"] = m
            frames.append(df)
    if not frames: return
    df_all = pd.concat(frames, ignore_index=True)

    metrics = [("dry_mass_Eg", "Dry Air Mass (Eg)"),
               ("water_mass_kg", "Water Mass (kg)"),
               ("total_energy_J", "Total Energy (J)")]
               
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    for ax, (col, title) in zip(axes, metrics):
        for model in MODELS:
            mdf = df_all[df_all["model"] == model]
            if mdf.empty or col not in mdf.columns: continue
            
            rel_df = mdf[["date", "forecast_hour", col]].dropna().copy()
            if rel_df.empty: continue
            
            base = rel_df.sort_values("forecast_hour").groupby("date", as_index=False).first()[["date", col]].rename(columns={col: "base_val"})
            rel_df = rel_df.merge(base, on="date", how="left")
            rel_df = rel_df[rel_df["base_val"].abs() > 0]
            if rel_df.empty: continue
            
            rel_df["rel_pct"] = (rel_df[col] - rel_df["base_val"]) / rel_df["base_val"] * 100.0
            agg = rel_df.groupby("forecast_hour")["rel_pct"].agg(["mean", "std"]).reset_index()
            
            style = MODEL_STYLES.get(model, {"color": "grey", "marker": "."})
            ax.plot(agg["forecast_hour"], agg["mean"], label=NICE.get(model, model), color=style["color"], marker=style["marker"], markersize=3)
            ax.fill_between(agg["forecast_hour"], agg["mean"] - agg["std"].fillna(0), agg["mean"] + agg["std"].fillna(0), color=style["color"], alpha=0.18, linewidth=0)
            
        ax.set_title(title, fontsize=28)
        ax.set_xlabel("Forecast Hour", fontsize=24)
        if ax == axes[0]: ax.set_ylabel("Relative Change (%)", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)

    handles, labels = _get_shared_legend_handles(axes)
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=len(labels), fontsize=22)
        
    fig.tight_layout()
    fig.savefig(outdir / "combined_conservation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved combined conservation plot")

def plot_combined_spectra(results_dir: Path, outdir: Path, reference_label: str):
    """1x3 Combined panel for Spectra at 12h, 120h, 240h."""
    frames = []
    for path in results_dir.glob("spectra_*.csv"):
        m = path.stem.replace("spectra_", "").split("_")[0]
        if m in MODELS:
            df = pd.read_csv(path)
            df["model"] = m
            frames.append(df)
    if not frames: return
    df_all = pd.concat(frames, ignore_index=True)

    leads = [12, 120, 240]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    for ax, lt in zip(axes, leads):
        sub = df_all[(df_all["lead_hours"] == lt) & (df_all["variable"] == "KE") & (df_all["wavenumber"] > 0)]
        if sub.empty: continue
        
        ref_agg = sub.groupby("wavenumber")["power_ref"].mean().reset_index()
        if not ref_agg.empty:
            wl = 2.0 * np.pi * EARTH_RADIUS_KM / ref_agg["wavenumber"].values
            ax.loglog(wl, ref_agg["power_ref"].values, color="black", linewidth=2, label=reference_label, zorder=5)

        for model in MODELS:
            msub = sub[sub["model"] == model]
            if msub.empty: continue
            msub_agg = msub.groupby("wavenumber")["power_pred"].mean().reset_index()
            style = MODEL_STYLES.get(model, {"color": "grey"})
            wl = 2.0 * np.pi * EARTH_RADIUS_KM / msub_agg["wavenumber"].values
            ax.loglog(wl, msub_agg["power_pred"].values, color=style["color"], linewidth=1.5, label=NICE.get(model, model))
            
        ax.set_title(f"{lt}h", fontsize=28)
        ax.set_xlabel("Wavelength (km)", fontsize=24)
        if ax == axes[0]: ax.set_ylabel("Kinetic Energy", fontsize=24)
        ax.set_xlim(40000, 100)
        ax.tick_params(axis='both', which='major', labelsize=20)

    handles, labels = _get_shared_legend_handles(axes)
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=len(labels), fontsize=22)
        
    fig.tight_layout()
    fig.savefig(outdir / "combined_spectra.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved combined spectra plot")

def plot_combined_balance(results_dir: Path, outdir: Path, reference_label: str):
    """Combined panel for Balance (Geostrophic RMSE, Hydrostatic RMSE, 240h Lapse Rates)."""
    ts_frames = []
    for path in results_dir.glob("time_series_*.csv"):
        m = path.stem.replace("time_series_", "").split("_")[0]
        if m in MODELS:
            df = pd.read_csv(path)
            df["model"] = m
            ts_frames.append(df)
    
    lr_frames = []
    for path in results_dir.glob("lapse_rate_dist_*.csv"):
        m = path.stem.replace("lapse_rate_dist_", "").split("_")[0]
        if m in MODELS:
            df = pd.read_csv(path)
            df["model"] = m
            lr_frames.append(df)

    if not ts_frames or not lr_frames: return
    df_ts = pd.concat(ts_frames, ignore_index=True)
    df_lr = pd.concat(lr_frames, ignore_index=True)
    sub_240 = df_lr[df_lr["lead_hours"] == 240]
    
    summaries = load_summaries(results_dir)

    # Create a 1x5 grid where RMSE plots take 2/3 (3+3 parts) and Lapse Rates take 1/3 (1+1+1 parts)
    fig = plt.figure(figsize=(24, 4))
    gs = fig.add_gridspec(1, 5, width_ratios=[3, 3, 1, 1, 1])
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    
    # Plot RMSEs
    for ax, col in zip(axes[:2], ["geostrophic_rmse", "hydrostatic_rmse"]):
        bases = get_model_baselines(summaries, col)
        
        for model in MODELS:
            mdf = df_ts[df_ts["model"] == model]
            if mdf.empty or col not in mdf.columns: continue
            
            mdf_clean = mdf.copy()
            base_val = bases.get(model, 0.0)
            mdf_clean[col] = pd.to_numeric(mdf_clean[col], errors="coerce") - base_val
            
            agg = mdf_clean.groupby("forecast_hour")[col].agg(["mean", "std"]).reset_index()
            if agg.empty: continue
            
            style = MODEL_STYLES.get(model, {"color": "grey", "marker": "."})
            ax.plot(agg["forecast_hour"], agg["mean"], label=NICE.get(model, model), color=style["color"], marker=style["marker"], markersize=3)
            ax.fill_between(agg["forecast_hour"], agg["mean"] - agg["std"].fillna(0), agg["mean"] + agg["std"].fillna(0), color=style["color"], alpha=0.18, linewidth=0)
            
        title_base = "Geostrophic RMSE Δ" if col.startswith("geo") else "Hydrostatic RMSE Δ"
        ylabel = "Δ RMSE (m/s)" if col.startswith("geo") else "Δ RMSE (m²/s²)"
        ax.set_title(title_base, fontsize=28)
        ax.set_xlabel("Forecast Hour", fontsize=24)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)

    # Plot Lapse Rates
    regions = [r for r in ["tropics", "sh_mid", "nh_mid"] if r in set(sub_240["region"].unique())]
    if not regions: regions = sorted(sub_240["region"].unique())
    global_ymax = 0.0
    
    for ax, region in zip(axes[2:], regions):
        sub = sub_240[sub_240["region"] == region]
        if sub.empty: continue

        b_unique = np.sort(sub["bin_edge_lower"].unique())
        width = float(np.median(np.diff(b_unique))) if len(b_unique) > 1 else 0.5

        ref_agg = sub.groupby("bin_edge_lower", as_index=False)["freq_ref"].mean().sort_values("bin_edge_lower")
        if not ref_agg.empty:
            x_ref = ref_agg["bin_edge_lower"].to_numpy() + 0.5 * width
            y_ref = ref_agg["freq_ref"].to_numpy()
            if y_ref.size: global_ymax = max(global_ymax, float(np.nanmax(y_ref)))
            ax.plot(x_ref, y_ref, color="black", linewidth=2.2, label=reference_label, zorder=10)

        for i, model in enumerate(MODELS):
            msub = sub[sub["model"] == model]
            if msub.empty: continue
            style = MODEL_STYLES.get(model, {"color": "grey"})
            m_agg = msub.groupby("bin_edge_lower", as_index=False)["freq_pred"].mean().sort_values("bin_edge_lower")
            x = m_agg["bin_edge_lower"].to_numpy() + 0.5 * width
            y = m_agg["freq_pred"].to_numpy()
            if y.size: global_ymax = max(global_ymax, float(np.nanmax(y)))
            ax.plot(x, y, color=style["color"], linewidth=1.4, alpha=0.95, label=NICE.get(model, model), zorder=3 + i)

        ax.set_xlim(1, 10)
        ax.set_title(pretty_region_name(region), fontsize=28)
        
        # Only set xlabel for the middle lapse rate panel to save space
        if region == regions[1] if len(regions) > 1 else regions[0]:
            ax.set_xlabel("Lapse Rate", fontsize=24)
            
        ax.tick_params(axis='both', which='major', labelsize=18)
        
    for ax in axes[2:]:
        if global_ymax > 0: ax.set_ylim(0, global_ymax * 1.15)
        else: ax.set_ylim(bottom=0)
    axes[2].set_ylabel("Density", fontsize=24)

    handles, labels = _get_shared_legend_handles(axes)
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=len(labels), fontsize=26)

    fig.tight_layout()
    fig.savefig(outdir / "combined_balance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved combined balance plot")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent.parent / "plots"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    reference_label = infer_reference_label(results_dir)
    
    print("Plotting Timeseries...")
    plot_timeseries(results_dir, outdir)
    
    print("Plotting Spectra...")
    plot_spectra(results_dir, outdir, reference_label=reference_label)
    
    print("Plotting Lapse Rates...")
    plot_lapse_rates(results_dir, outdir, reference_label=reference_label)

    print("Plotting Compact 240h Lapse Rate...")
    plot_lapse_rates_240h_compact(results_dir, outdir, reference_label=reference_label)
    
    print("Plotting Summary Tables...")
    plot_summary_tables(results_dir, outdir)
    
    print("Plotting Combined Panels...")
    plot_combined_conservation(results_dir, outdir)
    plot_combined_spectra(results_dir, outdir, reference_label)
    plot_combined_balance(results_dir, outdir, reference_label)
    
    print(f"Done! Plots saved to {outdir}")

if __name__ == "__main__":
    main()
