import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    bias = np.mean(y_pred - y_true)
    return mae, rmse, bias

def report_ann_results(ann_preds_path, bs_preds_path=None, out_dir=None):
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(ann_preds_path), "reporting")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading ANN predictions from {ann_preds_path}")
    ann_df = pd.read_csv(ann_preds_path)
    
    # 1. Aggregated Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(ann_df['actual_price'], ann_df['predicted_price'], alpha=0.3, s=2)
    
    min_val = min(ann_df['actual_price'].min(), ann_df['predicted_price'].min())
    max_val = max(ann_df['actual_price'].max(), ann_df['predicted_price'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Aggregated Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "aggregated_scatter.png"))
    plt.close()
    
    # 2. Per-module scatter plots
    if 'module_id' in ann_df.columns:
        mod_dir = os.path.join(out_dir, "module_scatters")
        os.makedirs(mod_dir, exist_ok=True)
        
        modules = ann_df['module_id'].unique()
        mod_metrics = []
        
        for mod in modules:
            mdf = ann_df[ann_df['module_id'] == mod]
            plt.figure(figsize=(6, 6))
            plt.scatter(mdf['actual_price'], mdf['predicted_price'], alpha=0.3, s=2)
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.title(f'Module: {mod}')
            plt.xlabel('Actual Price')
            plt.ylabel('Predicted Price')
            plt.grid(True, alpha=0.3)
            # Safe filename
            safe_mod = str(mod).replace("/", "_").replace("\\", "_")
            plt.savefig(os.path.join(mod_dir, f"scatter_{safe_mod}.png"))
            plt.close()
            
            # compute metrics
            mae, rmse, bias = compute_metrics(mdf['actual_price'], mdf['predicted_price'])
            mod_metrics.append({'module_id': mod, 'MAE': mae, 'RMSE': rmse, 'Bias': bias, 'count': len(mdf)})
            
        metrics_df = pd.DataFrame(mod_metrics)
        metrics_df.to_csv(os.path.join(out_dir, "module_metrics.csv"), index=False)
        
        # 3. Module-level heatmap tables (using pivot on categorical module axes)
        if 'mon_bin' in ann_df.columns and 'ttm_bin' in ann_df.columns:
            for metric in ['MAE', 'RMSE', 'Bias']:
                idx_dim = 'mon_bin'
                col_dim = 'ttm_bin'
                grouped = ann_df.groupby([idx_dim, col_dim], observed=False).apply(
                    lambda g: compute_metrics(g['actual_price'], g['predicted_price'])[
                        0 if metric == 'MAE' else 1 if metric == 'RMSE' else 2
                    ], include_groups=False
                ).unstack(col_dim)
                
                grouped.to_csv(os.path.join(out_dir, f"heatmap_table_{metric}.csv"))
                
                plt.figure(figsize=(8, 6))
                # Using imshow to plot simple heatmap
                im = plt.imshow(grouped.values, cmap='viridis' if metric != 'Bias' else 'coolwarm', aspect='auto')
                plt.colorbar(im, label=metric)
                
                plt.yticks(ticks=range(len(grouped.index)), labels=grouped.index)
                plt.xticks(ticks=range(len(grouped.columns)), labels=grouped.columns, rotation=45)
                plt.ylabel(idx_dim)
                plt.xlabel(col_dim)
                plt.title(f'{metric} Heatmap by Module Bins')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"heatmap_{metric}.png"))
                plt.close()

    # 4. ANN vs BS Comparison
    if bs_preds_path and os.path.exists(bs_preds_path):
        print(f"Loading BS predictions from {bs_preds_path}")
        bs_df = pd.read_csv(bs_preds_path, low_memory=False)
        
        join_keys = ['timestamp', 'strike', 'expiry', 'option_type']
        valid_keys = [k for k in join_keys if k in ann_df.columns and k in bs_df.columns]
        
        if len(valid_keys) == 4:
            print(f"Merging ANN and BS on keys: {valid_keys}")
            bs_pred_col = 'model_price' if 'model_price' in bs_df.columns else 'prediction'
            
            merged = pd.merge(ann_df, bs_df[valid_keys + [bs_pred_col]], on=valid_keys, how='inner')
            print(f"Merged dataframe has {len(merged)} rows (from {len(ann_df)} ANN rows).")
            
            if len(merged) > 0:
                ann_mae, ann_rmse, ann_bias = compute_metrics(merged['actual_price'], merged['predicted_price'])
                bs_mae, bs_rmse, bs_bias = compute_metrics(merged['actual_price'], merged[bs_pred_col])
                
                comp = {
                    "Model": ["ANN", "Black-Scholes"],
                    "MAE": [ann_mae, bs_mae],
                    "RMSE": [ann_rmse, bs_rmse],
                    "Bias": [ann_bias, bs_bias],
                    "Count": [len(merged), len(merged)]
                }
                
                comp_df = pd.DataFrame(comp)
                print("\nGlobal Comparison:")
                print(comp_df)
                comp_df.to_csv(os.path.join(out_dir, "ann_vs_bs_comparison.csv"), index=False)
                
                # Module-level comparison table
                if 'module_id' in merged.columns:
                    mod_comps = []
                    for mod in merged['module_id'].unique():
                        mmeta = merged[merged['module_id'] == mod]
                        a_m, a_r, a_b = compute_metrics(mmeta['actual_price'], mmeta['predicted_price'])
                        b_m, b_r, b_b = compute_metrics(mmeta['actual_price'], mmeta[bs_pred_col])
                        mod_comps.append({
                            "module_id": mod, "Count": len(mmeta),
                            "ANN_MAE": a_m, "BS_MAE": b_m,
                            "ANN_RMSE": a_r, "BS_RMSE": b_r,
                            "ANN_Bias": a_b, "BS_Bias": b_b
                        })
                    pd.DataFrame(mod_comps).to_csv(os.path.join(out_dir, "ann_vs_bs_module_comparison.csv"), index=False)
                
                # Comparative Scatter Plot
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.scatter(merged['actual_price'], merged['predicted_price'], alpha=0.3, s=2)
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                plt.title('ANN: Actual vs Predicted')
                plt.xlabel('Actual Price')
                plt.ylabel('ANN Predicted Price')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.scatter(merged['actual_price'], merged[bs_pred_col], alpha=0.3, s=2)
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                plt.title('Black-Scholes: Actual vs Predicted')
                plt.xlabel('Actual Price')
                plt.ylabel('BS Predicted Price')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "ann_vs_bs_scatter.png"))
                plt.close()
        else:
            print("Required join keys ['timestamp', 'strike', 'expiry', 'option_type'] missing in one of datasets.")
            
    print(f"Reporting artifacts safely completed in: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Report ANN Results")
    parser.add_argument("--ann", required=True, type=str, help="Path to ANN predictions.csv")
    parser.add_argument("--bs", type=str, default=None, help="Path to Black-Scholes predictions.csv (optional)")
    parser.add_argument("--out", type=str, default=None, help="Output directory specifying plot drops")
    args = parser.parse_args()
    
    report_ann_results(args.ann, args.bs, args.out)
