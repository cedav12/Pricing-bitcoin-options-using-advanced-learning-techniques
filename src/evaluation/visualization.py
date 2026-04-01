"""
Visualization Layer for Model Evaluation
Generates clean, publication-ready diagnostic plots for option pricing models.
Applies safe plotting filters to remove tick-level noise from visuals while
preserving the underlying statistical evaluation intact elsewhere.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def generate_diagnostic_plots(
    df: pd.DataFrame, 
    model_name: str, 
    out_dir: str, 
    error_type: str = "relative",
    min_price: float = 0.001
):
    """
    Generates the core 6-panel or split-panel diagnostic plots for a specific model.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Plotting Filter: Clean up visual noise (the actual metrics may use full data)
    # Visualizations of relative/log errors explode near 0, making plots unreadable.
    df_plot = df.copy()
    if error_type in ["relative", "log"]:
        df_plot = df_plot[df_plot["market_price"] > min_price]
        
    if df_plot.empty:
        print(f"[Visualization] No valid data to plot for {model_name} after noise filters.")
        return

    # Ensure error exists in the df_plot specifically
    err_col = f"error_{error_type}" if error_type in ["abs", "rel", "log"] else "error_rel"
    # Fallback if names differ slightly
    if err_col not in df_plot.columns:
        if error_type == "relative":
            df_plot[err_col] = (df_plot["model_price"] - df_plot["market_price"]) / df_plot["market_price"]
        elif error_type == "log":
            df_plot[err_col] = np.log(df_plot["model_price"] / df_plot["market_price"])
        else:
            df_plot[err_col] = df_plot["model_price"] - df_plot["market_price"]

    # Sample for performance if dataset is massive (> 50k rows for scatter plots)
    df_scatter = df_plot.sample(n=min(len(df_plot), 50000), random_state=42) if len(df_plot) > 50000 else df_plot
    
    # Setup Figure
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Diagnostics: {model_name} ({error_type.capitalize()} Error)", fontsize=18, y=0.98)
    
    # Plot 1: Market vs Model (Log-Log)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(df_scatter["market_price"], df_scatter["model_price"], alpha=0.1, s=2)
    mmax = max(df_scatter["market_price"].max(), df_scatter["model_price"].max())
    mmin = min(df_scatter["market_price"].min(), df_scatter["model_price"].min())
    if mmin <= 0: mmin = 1e-4
    ax1.plot([mmin, mmax], [mmin, mmax], "r--")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Market Price (BTC)")
    ax1.set_ylabel("Model Price (BTC)")
    ax1.set_title("Market vs Model (Log-Log)")
    
    # Plot 2: Error Distribution
    ax2 = fig.add_subplot(2, 3, 2)
    errors = df_plot[err_col].dropna()
    # Clip extreme outliers globally for the histogram to keep it readable
    q_low, q_high = errors.quantile(0.01), errors.quantile(0.99)
    hist_data = errors[(errors >= q_low) & (errors <= q_high)]
    ax2.hist(hist_data, bins=50, color='skyblue', edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--')
    ax2.set_title(f"{error_type.capitalize()} Error Distribution (1st-99th %ile)")
    
    # Plot 3: Error vs Moneyness
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(df_scatter["log_moneyness"], df_scatter[err_col], alpha=0.1, s=2)
    ax3.axhline(0, color='r', linestyle='--')
    ax3.set_xlabel("Log Moneyness")
    ax3.set_ylabel(f"{error_type.capitalize()} Error")
    ax3.set_title("Error vs Moneyness")
    
    # Plot 4: Error vs Maturity
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(df_scatter["time_to_maturity"], df_scatter[err_col], alpha=0.1, s=2)
    ax4.axhline(0, color='r', linestyle='--')
    ax4.set_xlabel("Time to Maturity (Years)")
    ax4.set_ylabel(f"{error_type.capitalize()} Error")
    ax4.set_title("Error vs Maturity")

    # Plot 5: Heatmap (Moneyness x Maturity)
    ax5 = fig.add_subplot(2, 3, 5)
    if "seg_maturity" in df_plot.columns and "seg_moneyness" in df_plot.columns:
        heat_df = df_plot.copy()
        heat_df["abs_err"] = np.abs(heat_df[err_col])
        heatmap_data = heat_df.groupby(["seg_maturity", "seg_moneyness"], observed=False)["abs_err"].mean()
        heatmap_data = heatmap_data.unstack(level="seg_moneyness")
        sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax5)
        ax5.set_title(f"Mean Abs {error_type.capitalize()} Error")
        ax5.set_ylabel("Maturity")
        ax5.set_xlabel("Moneyness")
    else:
        ax5.text(0.5, 0.5, "Segments Not Computed", ha='center')

    # Plot 6: Error vs Price Bucket (Boxplot or Scatter)
    ax6 = fig.add_subplot(2, 3, 6)
    if "seg_price" in df_plot.columns:
        sns.boxplot(x="seg_price", y=err_col, data=df_plot, showfliers=False, ax=ax6, color="lightgreen")
        ax6.axhline(0, color='r', linestyle='--')
        ax6.set_title("Error by Price Bracket")
        ax6.set_xlabel("Market Price (BTC)")
        ax6.tick_params(axis='x', rotation=45)
    else:
        ax6.text(0.5, 0.5, "Price Segment Not Computed", ha='center')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    file_path = os.path.join(out_dir, f"{error_type}_diagnostics.png")
    plt.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved {file_path}")
