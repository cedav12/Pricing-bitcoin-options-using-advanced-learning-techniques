"""
report_ann_results.py
---------------------
Lightweight reporting and visualization helper for ANN prediction outputs.

Usage:
    python3 src/analysis/report_ann_results.py \\
        --ann output/ann/<run>/predict_test_<ts>/predictions.csv \\
        [--bs data/processed/predictions_bs.csv] \\
        [--bs-col model_price] \\
        [--out output/reporting/]

Guaranteed columns in ANN predictions.csv
------------------------------------------
These are always present after ann_predict runs (enforced in ANNPredictPipeline):
  - timestamp
  - strike
  - underlying_price
  - option_type
  - expiry
  - module_id
  - actual_price
  - predicted_price

If module_columns were used during training (e.g. ["mon_bin", "ttm_bin"]),
those bin columns are also guaranteed to be present (enforced via pipeline).

Black-Scholes prediction column
---------------------------------
Default: "model_price"
Override with --bs-col if your BS output uses a different column name.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = y_pred - y_true
    mae  = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    bias = float(np.mean(residuals))
    return {"MAE": mae, "RMSE": rmse, "Bias": bias, "count": len(y_true)}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _identity_lim(series_list):
    """Compute a common [lo, hi] range across multiple series."""
    combined = np.concatenate([s.dropna().values for s in series_list])
    lo, hi = combined.min(), combined.max()
    pad = (hi - lo) * 0.02
    return lo - pad, hi + pad


def _scatter(ax, x, y, lo, hi, title, xlabel="Actual Price", ylabel="Predicted Price"):
    ax.scatter(x, y, alpha=0.3, s=3, linewidths=0)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)


# ─── Main ─────────────────────────────────────────────────────────────────────

def report_ann_results(ann_preds_path: str,
                       bs_preds_path: str | None = None,
                       bs_col: str = "model_price",
                       out_dir: str | None = None):

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(ann_preds_path), "reporting")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[REPORT] Loading ANN predictions: {ann_preds_path}")
    ann_df = pd.read_csv(ann_preds_path)

    # ── Validate required columns ────────────────────────────────────────────
    required = ["actual_price", "predicted_price", "module_id"]
    missing  = [c for c in required if c not in ann_df.columns]
    if missing:
        raise ValueError(f"ANN predictions CSV is missing required columns: {missing}")

    y_true = ann_df["actual_price"].values
    y_pred = ann_df["predicted_price"].values

    lo, hi = _identity_lim([ann_df["actual_price"], ann_df["predicted_price"]])

    # ── 1. Aggregated scatter ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    _scatter(ax, y_true, y_pred, lo, hi, "Aggregated: Actual vs Predicted (ANN)")
    m = compute_metrics(y_true, y_pred)
    ax.text(0.04, 0.96,
            f"MAE={m['MAE']:.5f}\nRMSE={m['RMSE']:.5f}\nBias={m['Bias']:.5f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aggregated_scatter.png"), dpi=150)
    plt.close(fig)
    print("[REPORT] aggregated_scatter.png saved.")

    # ── 2. Residual / error diagnostic ───────────────────────────────────────
    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Residual histogram
    axes[0].hist(residuals, bins=60, edgecolor="none", alpha=0.8)
    axes[0].axvline(0, color="r", lw=1.2, linestyle="--")
    axes[0].set_xlabel("Residual (predicted − actual)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Residual Distribution")
    axes[0].grid(True, alpha=0.25)

    # (b) Absolute error vs actual price
    axes[1].scatter(y_true, abs_errors, alpha=0.3, s=3, linewidths=0)
    axes[1].set_xlabel("Actual Price")
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title("|Error| vs Actual Price")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "residual_diagnostics.png"), dpi=150)
    plt.close(fig)
    print("[REPORT] residual_diagnostics.png saved.")

    # ── 3. Per-module scatter + metrics table ─────────────────────────────────
    mod_dir = os.path.join(out_dir, "module_scatters")
    os.makedirs(mod_dir, exist_ok=True)

    mod_metrics = []
    for mod_id, mdf in ann_df.groupby("module_id", sort=True):
        mae, rmse, bias = (compute_metrics(mdf["actual_price"].values, mdf["predicted_price"].values)[k]
                           for k in ("MAE", "RMSE", "Bias"))
        mod_metrics.append({"module_id": mod_id, "MAE": mae, "RMSE": rmse, "Bias": bias, "count": len(mdf)})

        fig, ax = plt.subplots(figsize=(5, 5))
        _scatter(ax, mdf["actual_price"], mdf["predicted_price"], lo, hi,
                 f"Module: {mod_id} (n={len(mdf)})")
        ax.text(0.04, 0.96, f"MAE={mae:.5f}\nBias={bias:.5f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        fig.tight_layout()
        safe = str(mod_id).replace("/", "_").replace("\\", "_")
        fig.savefig(os.path.join(mod_dir, f"scatter_{safe}.png"), dpi=120)
        plt.close(fig)

    metrics_df = pd.DataFrame(mod_metrics)
    metrics_df.to_csv(os.path.join(out_dir, "module_metrics.csv"), index=False)
    print(f"[REPORT] Per-module scatters saved to {mod_dir}/")

    # ── 4. Module heatmaps (mon_bin × ttm_bin) ───────────────────────────────
    bin_rows = "mon_bin"
    bin_cols = "ttm_bin"
    if bin_rows in ann_df.columns and bin_cols in ann_df.columns:
        for metric in ("MAE", "RMSE", "Bias"):
            idx = 0 if metric == "MAE" else 1 if metric == "RMSE" else 2
            pivot = ann_df.groupby([bin_rows, bin_cols], observed=True).apply(
                lambda g: compute_metrics(g["actual_price"].values, g["predicted_price"].values)[metric],
                include_groups=False
            ).unstack(bin_cols)

            pivot.to_csv(os.path.join(out_dir, f"heatmap_table_{metric}.csv"))

            fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1]), max(4, pivot.shape[0])))
            cmap = "coolwarm" if metric == "Bias" else "viridis"
            im = ax.imshow(pivot.values.astype(float), cmap=cmap, aspect="auto")
            plt.colorbar(im, ax=ax, label=metric)
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_xlabel(bin_cols)
            ax.set_ylabel(bin_rows)
            ax.set_title(f"{metric} by Module Bins")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"heatmap_{metric}.png"), dpi=150)
            plt.close(fig)
        print("[REPORT] Module heatmaps saved.")
    else:
        print(f"[REPORT] Skipping heatmaps: '{bin_rows}' and/or '{bin_cols}' not in predictions CSV.")

    # ── 5. ANN vs Black-Scholes comparison ───────────────────────────────────
    if bs_preds_path and os.path.exists(bs_preds_path):
        print(f"[REPORT] Loading BS predictions: {bs_preds_path}  (column='{bs_col}')")
        bs_df = pd.read_csv(bs_preds_path, low_memory=False)

        if bs_col not in bs_df.columns:
            available = [c for c in bs_df.columns if "price" in c.lower() or "pred" in c.lower()]
            print(f"[REPORT_WARNING] Column '{bs_col}' not found in BS CSV.")
            print(f"  Candidates: {available}")
            print("  Skipping BS comparison. Use --bs-col to specify the correct column.")
        else:
            join_keys = ["timestamp", "strike", "expiry", "option_type"]
            valid_keys = [k for k in join_keys if k in ann_df.columns and k in bs_df.columns]
            missing_keys = [k for k in join_keys if k not in valid_keys]
            if missing_keys:
                print(f"[REPORT_WARNING] Join keys missing from one dataset: {missing_keys}. Skipping BS comparison.")
            else:
                merged = pd.merge(
                    ann_df, bs_df[valid_keys + [bs_col]],
                    on=valid_keys, how="inner"
                )
                print(f"[REPORT] Merged {len(merged)} rows (ANN had {len(ann_df)}, BS had {len(bs_df)}).")

                if len(merged) == 0:
                    print("[REPORT_WARNING] Merge resulted in 0 rows. Check key alignment.")
                else:
                    ann_m = compute_metrics(merged["actual_price"].values, merged["predicted_price"].values)
                    bs_m  = compute_metrics(merged["actual_price"].values, merged[bs_col].values)

                    comp_df = pd.DataFrame([
                        {"Model": "ANN",           **ann_m},
                        {"Model": "Black-Scholes",  **bs_m},
                    ])
                    print("\n[REPORT] Global comparison:")
                    print(comp_df.to_string(index=False))
                    comp_df.to_csv(os.path.join(out_dir, "ann_vs_bs_comparison.csv"), index=False)

                    # Per-module comparison table
                    if "module_id" in merged.columns:
                        rows = []
                        for mod_id, mdf in merged.groupby("module_id", sort=True):
                            a = compute_metrics(mdf["actual_price"].values, mdf["predicted_price"].values)
                            b = compute_metrics(mdf["actual_price"].values, mdf[bs_col].values)
                            rows.append({
                                "module_id": mod_id, "count": len(mdf),
                                "ANN_MAE": a["MAE"], "BS_MAE": b["MAE"],
                                "ANN_RMSE": a["RMSE"], "BS_RMSE": b["RMSE"],
                                "ANN_Bias": a["Bias"], "BS_Bias": b["Bias"],
                            })
                        pd.DataFrame(rows).to_csv(
                            os.path.join(out_dir, "ann_vs_bs_module_comparison.csv"), index=False)

                    # Side-by-side scatter — axis limits from merged data
                    m_lo, m_hi = _identity_lim([
                        merged["actual_price"], merged["predicted_price"], merged[bs_col]
                    ])
                    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
                    _scatter(axes[0], merged["actual_price"], merged["predicted_price"],
                             m_lo, m_hi, f"ANN  (MAE={ann_m['MAE']:.5f})")
                    _scatter(axes[1], merged["actual_price"], merged[bs_col],
                             m_lo, m_hi, f"Black-Scholes  (MAE={bs_m['MAE']:.5f})",
                             ylabel="BS Predicted Price")
                    fig.suptitle("ANN vs Black-Scholes: Actual vs Predicted", fontsize=11)
                    fig.tight_layout()
                    fig.savefig(os.path.join(out_dir, "ann_vs_bs_scatter.png"), dpi=150)
                    plt.close(fig)
                    print("[REPORT] ann_vs_bs_scatter.png saved.")
    elif bs_preds_path:
        print(f"[REPORT_WARNING] BS predictions file not found: {bs_preds_path}")

    print(f"\n[REPORT] Done. All artifacts in: {out_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ANN reporting plots and (optionally) ANN vs BS comparison."
    )
    parser.add_argument("--ann",    required=True,        help="Path to ANN predictions.csv")
    parser.add_argument("--bs",     default=None,         help="Path to Black-Scholes predictions CSV (optional)")
    parser.add_argument("--bs-col", default="model_price",
                        help="Column name for BS predicted price (default: 'model_price')")
    parser.add_argument("--out",    default=None,         help="Output directory (default: <ann_dir>/reporting/)")
    args = parser.parse_args()

    report_ann_results(
        ann_preds_path=args.ann,
        bs_preds_path=args.bs,
        bs_col=args.bs_col,
        out_dir=args.out,
    )
