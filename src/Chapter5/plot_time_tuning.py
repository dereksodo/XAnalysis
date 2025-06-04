import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from src.DataProcessing.utils import Paths
from src.DataProcessing.evaluate import compute_metrics
from src.DataProcessing.utils import get_scores

# --------------------------------------------------------------------- #
# Helper: load *_best_summary.csv files and annotate with model/period
def _load_best(files: list[str]) -> pd.DataFrame:
    """
    Load *_best_summary.csv files explicitly listed in *files*.
    Adds columns: `model` (derived from file name), `period` (if found in name).
    """
    all_dfs = []
    for f in files:
        if not os.path.exists(f):
            print(f"[warn] File not found: {f}")
            continue
        name = os.path.basename(f)
        model = name.split("_best_summary")[0]   # e.g. XGB, RF, ARIMA
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[warn] cannot read {f}: {e}")
            continue
        need = {"country", "target", "guiding_score"}
        if not need.issubset(df.columns):
            print(f"[warn] {f} missing required columns {need}")
            continue
        df = df[list(need)].copy()
        df["model"] = model
        m = re.search(r"(19\d{2}_20\d{2}|20\d{2}_20\d{2})", name)
        df["period"] = m.group(1) if m else ""
        all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError("No valid *_best_summary.csv files loaded!")
    return pd.concat(all_dfs, ignore_index=True)

# --------------------------------------------------------------------- #
# Helper: build Naive summary DataFrame from prediction files
def _build_naive_summary(pred_dir: str) -> pd.DataFrame:
    """
    Build a DataFrame identical in shape to *_best_summary.csv* for Naive baseline.

    Each file in *pred_dir* is expected to be <countrycode>_<featurecode>.csv
    with at least columns 'y_true' and 'y_pred'. If additional columns exist,
    they are ignored.

    Returns an empty DataFrame if directory missing or no valid files found.
    """
    rows = []
    if not os.path.isdir(pred_dir):
        print(f"[warn] Naive directory not found: {pred_dir}")
        return pd.DataFrame(columns=["country", "target", "guiding_score", "model", "period"])

    for f in glob.glob(os.path.join(pred_dir, "*.csv")):
        name = os.path.basename(f).split(".")[0]            # USA_SP.DYN.LE00.IN
        if "_" not in name:
            continue
        country, target = name.split("_", 1)                # split only first underscore
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[warn] cannot read {f}: {e}")
            continue
        if not {"y_true", "y_pred"}.issubset(df.columns):
            print(f"[warn] {f} missing y_true / y_pred")
            continue

        y_true = df["y_true"].to_numpy()
        y_pred = df["y_pred"].to_numpy()
        m = compute_metrics(y_true, y_pred)
        g = get_scores.guiding_score(m)
        rows.append({"country": country,
                     "target": target,
                     "guiding_score": g,
                     "model": "Naive",
                     "period": ""})
    if not rows:
        print("[warn] No Naive prediction files processed.")
        return pd.DataFrame(columns=["country", "target", "guiding_score", "model", "period"])
    return pd.DataFrame(rows)

TUNING_DIR = os.path.join(Paths.FIGURE_DIR, "Chapter5tuning")  # default
# --------------------------------------------------------------------- #
def _pick_baseline(df: pd.DataFrame) -> str:
    """Return 'Naive' if present else 'ARIMA' (or first available)."""
    for cand in ["Naive", "ARIMA"]:
        if cand in df["model"].unique():
            return cand
    return df["model"].unique()[0]

# --------------------------------------------------------------------- #
def fig1_grouped_bar(df_all: pd.DataFrame, outdir: str):
    """Average guiding per indicator – grouped bar Naive vs Tuned."""
    baseline = _pick_baseline(df_all)
    keep = df_all[df_all["model"].isin([baseline, "XGB", "RF"])]
    bar_df = (
        keep.groupby(["target", "model"])["guiding_score"]
        .mean().unstack("model").reindex(columns=[baseline, "XGB", "RF"])
    )
    bar_df.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Average Guiding Score")
    plt.title("Indicator‑level Guiding Score: Naive vs Tuned")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig1_grouped_bar.png"), dpi=300)
    plt.close()

# --------------------------------------------------------------------- #
def fig2_delta_box(df_all: pd.DataFrame, outdir: str):
    """Box‑plot of delta (Tuned – Baseline)."""
    baseline = _pick_baseline(df_all)
    n_df = df_all[df_all["model"] == baseline][["country", "target", "guiding_score"]]
    n_df = n_df.rename(columns={"guiding_score": "base"})
    merge = df_all[df_all["model"].isin(["XGB", "RF"])].merge(
        n_df, on=["country", "target"], how="inner"
    )
    merge["delta"] = merge["guiding_score"] - merge["base"]
    fig, ax = plt.subplots(figsize=(6, 6))
    groups = [merge.loc[merge["model"] == m, "delta"].dropna() for m in ["XGB", "RF"]]
    ax.boxplot(groups, tick_labels=["XGB", "RF"], vert=True, showfliers=True)
    ax.axhline(0, linestyle="--")
    ax.set_ylabel(f"ΔGuiding (Tuned − {baseline})")
    ax.set_title("Distribution of Guiding Improvement after Tuning")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig2_delta_box.png"), dpi=300)
    plt.close()

# --------------------------------------------------------------------- #
def fig3_scatter(df_all: pd.DataFrame, outdir: str):
    """Scatter: Baseline vs Tuned (use XGB)."""
    baseline = _pick_baseline(df_all)
    base_df = df_all[df_all["model"] == baseline][["country", "target", "guiding_score"]]
    base_df = base_df.rename(columns={"guiding_score": "base"})
    xgb = df_all[df_all["model"] == "XGB"][["country", "target", "guiding_score"]]
    xgb = xgb.rename(columns={"guiding_score": "xgb"})
    m = xgb.merge(base_df, on=["country", "target"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(m["base"], m["xgb"], alpha=0.6, s=20)
    lims = [min(m[["base", "xgb"]].min()) - 0.5,
            max(m[["base", "xgb"]].max()) + 0.5]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlabel(f"{baseline} Guiding")
    ax.set_ylabel("XGB Tuned Guiding")
    ax.set_title(f"{baseline} vs Tuned Guiding (all country‑indicator pairs)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig3_scatter_naive.png"), dpi=300)
    plt.close()

# --------------------------------------------------------------------- #
def fig4_heatmap(df_all: pd.DataFrame, outdir: str):
    """Heat‑map of ΔGuiding (Tuned − Baseline) by country × indicator."""
    baseline = _pick_baseline(df_all)
    naive = df_all[df_all["model"] == baseline][["country", "target", "guiding_score"]]
    naive = naive.rename(columns={"guiding_score": "base"})
    best_tuned = (
        df_all[df_all["model"].isin(["XGB", "RF"])]
        .sort_values("guiding_score", ascending=False)
        .drop_duplicates(subset=["country", "target"])
    )
    best_tuned = best_tuned[["country", "target", "guiding_score"]]
    best_tuned = best_tuned.rename(columns={"guiding_score": "tuned"})
    diff = best_tuned.merge(naive, on=["country", "target"])
    diff["delta"] = diff["tuned"] - diff["base"]
    pivot = diff.pivot(index="country", columns="target", values="delta")
    fig, ax = plt.subplots(figsize=(12, 8))

    # handle all‑NaN or constant matrices gracefully
    data_vals = pivot.values
    finite_mask = np.isfinite(data_vals)
    if not finite_mask.any():
        print("ΔGuiding heatmap skipped: all entries are NaN.")
        plt.close(fig)
        return

    vmin, vmax = np.nanmin(data_vals), np.nanmax(data_vals)
    if vmin == vmax:
        norm = None  # constant matrix: fall back to default norm
    else:
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

    im = ax.imshow(pivot, aspect="auto", cmap="coolwarm", norm=norm)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"ΔGuiding (Tuned − {baseline}) Heat‑map")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("ΔGuiding")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig4_heatmap_delta.png"), dpi=300)
    plt.close()

# --------------------------------------------------------------------- #
def fig5_time_window(df_all: pd.DataFrame, outdir: str):
    """Line plot: time‑window average Guiding (Baseline vs Tuned)."""
    avail_periods = df_all["period"].unique()
    valid = [p for p in avail_periods if p]
    if len(valid) < 2:
        # if period info not available, skip
        return
    baseline = _pick_baseline(df_all)
    dfp = df_all[df_all["period"].isin(valid) & df_all["model"].isin([baseline, "XGB"])]
    line_df = (
        dfp.groupby(["period", "model"])["guiding_score"]
        .mean().unstack("model").reindex(columns=[baseline, "XGB"])
    )
    line_df = line_df.sort_index()  # chronological
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(line_df.index, line_df[baseline], marker="o", label=baseline)
    ax.plot(line_df.index, line_df["XGB"], marker="o", label="XGB Tuned")
    ax.set_ylabel("Average Guiding Score")
    ax.set_title("Average Guiding Score by Time Window")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig5_time_window.png"), dpi=300)
    plt.close()

# --------------------------------------------------------------------- #
def main():
    # 1. Load best_summary files for ARIMA, RF, XGB (require 'guiding_score')
    summary_dir = os.path.join(Paths.FIGURE_DIR, "Chapter5tuning")
    files = [
        os.path.join(summary_dir, "ARIMA_best_summary.csv"),
        os.path.join(summary_dir, "RF_best_summary.csv"),
        os.path.join(summary_dir, "XGB_best_summary.csv"),
    ]
    model_dfs = []
    for file in files:
        file_exists = os.path.exists(file)
        print(f"[info] Checking for file: {file} ... Exists: {file_exists}")
        if not file_exists:
            print(f"[warn] File not found: {file}")
            continue
        try:
            df = pd.read_csv(file)
            print(f"[info] Successfully read {file}: {len(df)} rows loaded.")
        except Exception as e:
            print(f"[warn] cannot read {file}: {e}")
            continue
        if not {"country", "target", "guiding_score"}.issubset(df.columns):
            print(f"[warn] {file} missing required columns")
            continue
        model = os.path.basename(file).split("_best_summary")[0]
        df = df[["country", "target", "guiding_score"]].copy()
        df["model"] = model
        model_dfs.append(df)
    if not model_dfs:
        raise RuntimeError("No valid *_best_summary.csv files loaded!")
    best_df = pd.concat(model_dfs, ignore_index=True)

    # 2. Load all Naive predictions, compute guiding_score using compute_metrics and guiding_score
    from src.DataProcessing.utils import get_scores
    naive_pred_dir = os.path.join(Paths.FIGURE_DIR, "Chapter5", "predictions", "Naive")
    naive_rows = []
    naive_files = glob.glob(os.path.join(naive_pred_dir, "*.csv"))
    print(f"[info] Found {len(naive_files)} Naive prediction CSV files in {naive_pred_dir}")
    for f in naive_files:
        name = os.path.basename(f).split(".")[0]
        if "_" not in name:
            continue
        country, target = name.split("_", 1)
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[warn] cannot read {f}: {e}")
            continue
        if not {"y_true", "y_pred"}.issubset(df.columns):
            print(f"[warn] {f} missing y_true / y_pred")
            continue
        y_true = df["y_true"].to_numpy()
        y_pred = df["y_pred"].to_numpy()
        # Use compute_metrics and guiding_score from guiding_metrics
        m = compute_metrics(y_true, y_pred)
        g = get_scores.guiding_score(m)
        naive_rows.append({
            "country": country,
            "target": target,
            "guiding_score": g,
            "model": "Naive"
        })
    print(f"[info] Processed {len(naive_rows)} valid Naive prediction files.")
    naive_df = pd.DataFrame(naive_rows)

    # 3. Combine all into a single DataFrame
    df_all = pd.concat([best_df, naive_df], ignore_index=True)
    print(f"[info] Concatenated all models: df_all shape = {df_all.shape}")

    os.makedirs(TUNING_DIR, exist_ok=True)

    # Helper for plotting: get unique indicators for consistent order
    indicators = sorted(df_all["target"].unique())
    models = ["Naive", "XGB", "ARIMA"]

    # a) Bar plot comparing guiding scores across 10 indicators for Naive, XGB, ARIMA
    print("[info] Starting grouped bar plot (Naive, XGB, ARIMA) ...")
    if not df_all.empty:
        bar_df = (
            df_all[df_all["model"].isin(models)]
            .groupby(["target", "model"])["guiding_score"]
            .mean().unstack("model").reindex(columns=models)
        )
        bar_df = bar_df.reindex(indicators)
        bar_df.plot(kind="bar", figsize=(12, 6))
        plt.ylabel("Average Guiding Score")
        plt.title("Indicator-level Guiding Score: Naive vs XGB vs ARIMA")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(TUNING_DIR, "plot_time_tuning_bar.png"), dpi=300)
        plt.close()
        print("[info] Grouped bar plot saved.")
    else:
        print("[warn] df_all is empty, skipping grouped bar plot.")

    # b) Boxplot of delta guiding score (XGB - Naive and ARIMA - Naive)
    print("[info] Starting boxplot of delta guiding score (XGB - Naive, ARIMA - Naive) ...")
    def calc_delta(df, model_name):
        m = df_all[df_all["model"] == model_name][["country", "target", "guiding_score"]]
        n = df_all[df_all["model"] == "Naive"][["country", "target", "guiding_score"]]
        merged = m.merge(n, on=["country", "target"], suffixes=("_model", "_naive"))
        merged["delta"] = merged["guiding_score_model"] - merged["guiding_score_naive"]
        return merged["delta"]
    deltas = {m: calc_delta(df_all, m) for m in ["XGB", "ARIMA"]}
    if deltas["XGB"].shape[0] > 0 or deltas["ARIMA"].shape[0] > 0:
        plt.figure(figsize=(6, 6))
        plt.boxplot([deltas["XGB"].dropna(), deltas["ARIMA"].dropna()], labels=["XGB", "ARIMA"])
        plt.axhline(0, linestyle="--")
        plt.ylabel("ΔGuiding (Model − Naive)")
        plt.title("Distribution of Guiding Improvement (Tuned vs Naive)")
        plt.tight_layout()
        plt.savefig(os.path.join(TUNING_DIR, "plot_time_tuning_deltabox.png"), dpi=300)
        plt.close()
        print("[info] Boxplot saved.")
    else:
        print("[warn] Not enough data for delta boxplot.")

    # c) Scatter plot of (Naive vs XGB) and (Naive vs ARIMA) guiding scores
    print("[info] Starting scatter plot (Naive vs XGB/ARIMA) ...")
    xgb = df_all[df_all["model"] == "XGB"][["country", "target", "guiding_score"]]
    arima = df_all[df_all["model"] == "ARIMA"][["country", "target", "guiding_score"]]
    naive = df_all[df_all["model"] == "Naive"][["country", "target", "guiding_score"]]
    merged_xgb = xgb.merge(naive, on=["country", "target"], suffixes=("_model", "_naive"))
    merged_arima = arima.merge(naive, on=["country", "target"], suffixes=("_model", "_naive"))
    if not merged_xgb.empty or not merged_arima.empty:
        plt.figure(figsize=(6, 6))
        if not merged_xgb.empty:
            plt.scatter(merged_xgb["guiding_score_naive"], merged_xgb["guiding_score_model"],
                        alpha=0.6, s=20, label="XGB")
        if not merged_arima.empty:
            plt.scatter(merged_arima["guiding_score_naive"], merged_arima["guiding_score_model"],
                        alpha=0.6, s=20, label="ARIMA")
        lims = [
            min(
                merged_xgb["guiding_score_naive"].min() if not merged_xgb.empty else np.inf,
                merged_xgb["guiding_score_model"].min() if not merged_xgb.empty else np.inf,
                merged_arima["guiding_score_naive"].min() if not merged_arima.empty else np.inf,
                merged_arima["guiding_score_model"].min() if not merged_arima.empty else np.inf,
            ) - 0.5,
            max(
                merged_xgb["guiding_score_naive"].max() if not merged_xgb.empty else -np.inf,
                merged_xgb["guiding_score_model"].max() if not merged_xgb.empty else -np.inf,
                merged_arima["guiding_score_naive"].max() if not merged_arima.empty else -np.inf,
                merged_arima["guiding_score_model"].max() if not merged_arima.empty else -np.inf,
            ) + 0.5,
        ]
        plt.plot(lims, lims, linestyle="--", color="gray")
        plt.xlabel("Naive Guiding Score")
        plt.ylabel("Model Guiding Score")
        plt.title("Naive vs Tuned Guiding Score (XGB, ARIMA)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(TUNING_DIR, "plot_time_tuning_scatter.png"), dpi=300)
        plt.close()
        print("[info] Scatter plot saved.")
    else:
        print("[warn] Not enough data for scatter plot.")

    print("[info] Starting heatmap of ΔGuiding (max(XGB,ARIMA) - Naive) by (country, target) ...")
    # For each (country, target): max(XGB, ARIMA) - Naive
    pivot_rows = []
    for (country, target), group in df_all.groupby(["country", "target"]):
        naive_val = group.loc[group["model"] == "Naive", "guiding_score"]
        xgb_val = group.loc[group["model"] == "XGB", "guiding_score"]
        arima_val = group.loc[group["model"] == "ARIMA", "guiding_score"]
        if naive_val.empty:
            continue
        best_tuned = max(
            [v for v in [xgb_val.values[0] if not xgb_val.empty else None,
                         arima_val.values[0] if not arima_val.empty else None]
             if v is not None],
            default=None,
        )
        if best_tuned is None:
            continue
        delta = best_tuned - naive_val.values[0]
        pivot_rows.append({
            "country": country,
            "target": target,
            "delta": delta
        })

    print(f"[info] Number of valid (country, target) pairs for heatmap: {len(pivot_rows)}")
    if pivot_rows:
        print("[info] Sample entry for heatmap:", pivot_rows[0])

    heat_df = pd.DataFrame(pivot_rows)
    heat_pivot = heat_df.pivot(index="country", columns="target", values="delta")
    # Plot heatmap
    if not heat_pivot.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        data_vals = heat_pivot.values
        finite_mask = np.isfinite(data_vals)
        if not finite_mask.any():
            print("[warn] ΔGuiding heatmap skipped: all entries are NaN.")
            plt.close(fig)
        else:
            vmin, vmax = np.nanmin(data_vals), np.nanmax(data_vals)
            if vmin == vmax:
                norm = None
            else:
                norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
            im = ax.imshow(heat_pivot, aspect="auto", cmap="coolwarm", norm=norm)
            ax.set_xticks(range(len(heat_pivot.columns)))
            ax.set_xticklabels(heat_pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(heat_pivot.index)))
            ax.set_yticklabels(heat_pivot.index)
            ax.set_title("ΔGuiding (Best of XGB/ARIMA − Naive) Heatmap")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("ΔGuiding")
            plt.tight_layout()
            plt.savefig(os.path.join(TUNING_DIR, "plot_time_tuning_heatmap.png"), dpi=300)
            plt.close()
            print("[info] Heatmap saved.")
    else:
        print("[warn] No data for heatmap, skipping.")

    print("Plots saved to", TUNING_DIR)


if __name__ == "__main__":
    main()