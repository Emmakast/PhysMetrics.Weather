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

MODELS = ["hres", "pangu", "graphcast", "fuxi", "neuralgcm", "aurora"]
NICE = {
    "hres": "HRES", "fuxi": "FuXi", "graphcast": "GraphCast",
    "neuralgcm": "NeuralGCM", "pangu": "Pangu", "aurora": "Aurora",
}
MODEL_STYLES = {
    "aurora":    {"color": "#1f77b4", "marker": "o"},
    "pangu":     {"color": "#ff7f0e", "marker": "s"},
    "fuxi":      {"color": "#2ca02c", "marker": "^"},
    "graphcast": {"color": "#d62728", "marker": "D"},
    "neuralgcm": {"color": "#9467bd", "marker": "v"},
    "hres":      {"color": "#8c564b", "marker": "P"},
}

EARTH_RADIUS_KM = 6371.0

# ── 1. Timeseries plotting ───────────────────────────────────────────────────

def plot_timeseries(results_dir: Path, outdir: Path):
    """Plot timeseries across models."""
    csv_paths = list(results_dir.glob("time_series_*.csv"))
    if not csv_paths:
        print("No time_series_*.csv found.")
        return

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
        "hydrostatic_rmse": "Hydrostatic RMSE",
        "geostrophic_rmse": "Geostrophic RMSE"
    }

    outdir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    for col, title in metrics.items():
        if col not in df_all.columns: continue
        
        # Pre-aggregate
        agg = df_all.groupby(["model", "forecast_hour"])[col].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        for model in MODELS:
            mdf = agg[agg["model"] == model]
            if mdf.empty: continue
            style = MODEL_STYLES.get(model, {"color": "grey", "marker": "."})
            
            # Using relative change for conservation metrics
            if col in ["dry_mass_Eg", "water_mass_kg", "total_energy_J"]:
                first_val = mdf[col].iloc[0] if len(mdf) > 0 else 1
                y = (mdf[col] - first_val) / first_val * 100
                ylabel = "Relative Change (%)"
            else:
                y = mdf[col]
                ylabel = title
                
            ax.plot(mdf["forecast_hour"], y, label=NICE.get(model, model), 
                    color=style["color"], marker=style["marker"], markersize=5)
        
        ax.set_title(f"{title} Timeseries", fontsize=16)
        ax.set_xlabel("Forecast Hour", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(outdir / f"ts_{col}.png", dpi=200)
        plt.close(fig)
        print(f"Saved timeseries plot for {col}")

# ── 2. Spectra plotting ──────────────────────────────────────────────────────

def plot_spectra(results_dir: Path, outdir: Path, leads=[12, 120, 240]):
    """Plot spectra for target lead times."""
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
            ax.loglog(wl, ref_agg["power_ref"].values, color="black", linewidth=2, label="ERA5 / IFS", zorder=5)

        for model in MODELS:
            msub = sub[sub["model"] == model]
            if msub.empty: continue
            msub_agg = msub.groupby("wavenumber")["power_pred"].mean().reset_index()
            style = MODEL_STYLES.get(model, {"color": "grey"})
            wl = 2.0 * np.pi * EARTH_RADIUS_KM / msub_agg["wavenumber"].values
            ax.loglog(wl, msub_agg["power_pred"].values, color=style["color"], linewidth=1.5, label=NICE.get(model, model))
            
        ax.set_title(f"KE Spectrum - {lt}h", fontsize=16)
        ax.set_xlabel("Wavelength (km)", fontsize=14)
        ax.set_ylabel("Kinetic Energy", fontsize=14)
        ax.set_xlim(40000, 100) # Prevents matplotlib log-locator hang on inverted axis
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"spectra_ke_{lt}h.png", dpi=200)
        plt.close(fig)
        print(f"Saved spectra plot for {lt}h")

# ── 3. Lapse Rate Distributions ──────────────────────────────────────────────

def plot_lapse_rates(results_dir: Path, outdir: Path, leads=[12, 120, 240]):
    """Plot lapse rate distributions for different regions and lead times."""
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
        
        for ax, lt in zip(axes, leads):
            sub = df_all[(df_all["lead_hours"] == lt) & (df_all["region"] == region)]
            if sub.empty: continue
            
            # Determine bin width
            b_unique = sorted(sub["bin_edge_lower"].unique())
            width = np.diff(b_unique)[0] if len(b_unique) > 1 else 0.5

            # Plot models in the back
            for i, model in enumerate(MODELS):
                msub = sub[sub["model"] == model]
                if msub.empty: continue
                style = MODEL_STYLES.get(model, {"color": "grey"})
                ax.bar(msub["bin_edge_lower"], msub["freq_pred"], width=width, align="edge",
                       color=style["color"], alpha=0.6, edgecolor=style["color"],
                       linewidth=1.0, label=NICE.get(model, model), zorder=2+i)
            
            # Plot Reference in the front
            ref_agg = sub.groupby("bin_edge_lower")["freq_ref"].mean().reset_index()
            if not ref_agg.empty:
                ax.bar(ref_agg["bin_edge_lower"], ref_agg["freq_ref"], width=width, align="edge",
                       color="black", alpha=0.3, edgecolor="black", linewidth=2.0,
                       label="ERA5", zorder=10)
            
            # Restrict x-axis to the non-zero region
            non_zero = sub[(sub["freq_ref"].fillna(0) > 1e-5) | (sub["freq_pred"].fillna(0) > 1e-5)]
            if not non_zero.empty:
                lower_bound = non_zero["bin_edge_lower"].min()
                upper_bound = non_zero["bin_edge_lower"].max()
                # give a tiny pad
                pad = (upper_bound - lower_bound) * 0.05
                ax.set_xlim(lower_bound - pad, upper_bound + pad)

            ax.set_ylim(bottom=0)
            ax.set_title(f"{region.capitalize()} - {lt}h", fontsize=14)
            ax.set_xlabel("Lapse Rate (K/km)", fontsize=12)
            if ax == axes[0]: ax.set_ylabel("Density", fontsize=12)
            
        axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(outdir / f"lapse_rate_{region}.png", dpi=200)
        plt.close(fig)
        print(f"Saved lapse rate plot for {region}")

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

def plot_summary_tables(results_dir: Path, outdir: Path, leads=[12, 120, 240]):
    """Render three grouped transposed tables, injecting Wasserstein to Balance."""
    outdir.mkdir(exist_ok=True)
    summaries = load_summaries(results_dir)
    w_dists = calculate_wasserstein(results_dir)
    
    models_to_plot = [m for m in MODELS if m != "aurora"]
    
    groups = {
        "Conservation_Variability": [
            ("dry_mass_drift_pct_per_day", "Dry Mass Drift →0 [%/day]"),
            ("water_mass_drift_pct_per_day", "Water Mass Drift →0 [%/day]"),
            ("total_energy_drift_pct_per_day", "Total Energy Drift →0 [%/day]"),
        ],
        "Spectral": [
            ("effective_resolution_km", "Eff. Resolution ↓111.5 [km]"),
            ("spectral_residual", "Spec. Residual ↓0 [-]"),
            ("spectral_divergence", "Spec. Divergence ↓0 [-]"),
        ],
        "Balance": [
            ("hydrostatic_rmse", "Hydrostatic RMSE Δ →0 [Pa]"),
            ("geostrophic_rmse", "Geostrophic RMSE Δ →0 [Pa]"),
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
                                if m == "neuralgcm":  # Force override because 0.7 ref in CSV is corrupted/NaN
                                    ref_val = 254.281359 if metric == "hydrostatic_rmse" else 4.719532
                                if not np.isnan(ref_val): val -= ref_val
                                elif "era5" in summaries:
                                    df_lt_era5 = summaries["era5"][summaries["era5"][lead_col] == lead] if lead_col and lead_col in summaries["era5"].columns else summaries["era5"]
                                    if not df_lt_era5.empty: val -= np.nan_to_num(get_value(df_lt_era5, metric))
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
                        ref_val = get_value(df_lt, metric, is_ref=True) if m in summaries and not df_lt.empty else np.nan
                        if m == "neuralgcm":  # Force override because 0.7 ref in CSV is corrupted/NaN
                            ref_val = 254.281359 if metric == "hydrostatic_rmse" else 4.719532
                        if not np.isnan(ref_val): val -= ref_val
                        elif "era5" in summaries:
                            df_era5 = summaries["era5"]
                            lead_col_era5 = next((c for c in ["lead_hours", "lead_time_hours", "lead_time", "forecast_hour"] if c in df_era5.columns), None)
                            df_lt_era5 = df_era5[df_era5[lead_col_era5] == lead] if lead_col_era5 else df_era5
                            if not df_lt_era5.empty: val -= np.nan_to_num(get_value(df_lt_era5, metric))
                            
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
        fig, ax = plt.subplots(figsize=(max(1.5 * n_cols, 10), max(0.4 * n_rows, 4)))
        ax.axis("off")
        
        colWidths = [0.35 if group_name in ["Balance", "Conservation_Variability"] else 0.25, 0.1] + [0.12] * len(current_models)
        table = ax.table(
            cellText=cell_texts,
            cellColours=[[tuple(c) for c in row] for row in cell_colors],
            colWidths=colWidths, loc="center", cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent.parent / "plots"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("Plotting Timeseries...")
    plot_timeseries(results_dir, outdir)
    
    print("Plotting Spectra...")
    plot_spectra(results_dir, outdir)
    
    print("Plotting Lapse Rates...")
    plot_lapse_rates(results_dir, outdir)
    
    print("Plotting Summary Tables...")
    plot_summary_tables(results_dir, outdir)
    
    print(f"Done! Plots saved to {outdir}")

if __name__ == "__main__":
    main()
