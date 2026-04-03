"""
Model Evaluation Orchestrator
Coordinates the model-agnostic evaluation framework, running preprocessing,
segmentation, metrics computation, and visualization against any provided predictions file.
"""
import os
import pandas as pd

from src.evaluation.preprocessing import preprocess_dataset
from src.evaluation.segmentation import apply_segments
from src.evaluation.metrics import compute_price_metrics, compute_diagnostic_metrics, add_error_columns
from src.evaluation.visualization import generate_diagnostic_plots

class ModelEvaluator:
    def __init__(self, predictions_path: str):
        self.predictions_path = predictions_path
        self.output_dir = "output/evaluation"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(
        self,
        option_filter: str = "call",
        error_type: str = "relative",
        eval_mode: str = "stable",
        min_price: float = 0.001,
        min_time_value: float = -1.0,
        segments: list = None,
        sample_size: int = None
    ):
        """
        Executes the full evaluation pipeline.
        """
        if segments is None:
            segments = ["moneyness", "maturity", "price"]

        print(f"[Evaluator] Loading '{self.predictions_path}'...")
        try:
            df = pd.read_csv(self.predictions_path, nrows=sample_size, low_memory=False)
        except Exception as e:
            print(f"[Evaluator] ERROR: Could not load data. {e}")
            return

        print(f"  Loaded {len(df):,} rows.")

        # 1. Preprocessing (Data preparation only, no systematic structural bias)
        df = preprocess_dataset(df, option_filter=option_filter)
        print(f"  After structural preprocessing: {len(df):,} rows.")

        if df.empty:
            print("[Evaluator] No valid rows after preprocessing.")
            return

        # 2. Segmentation
        df = apply_segments(df, segments)

        # Add basic un-filtered error columns for visualization and broad tracking
        df = add_error_columns(df)

        models = df["model_name"].unique()
        model_rankings = []

        # 3. Execution per model
        for model in models:
            mdf = df[df["model_name"] == model]
            print(f"\n{'='*60}\n  Evaluating Model: {model} (N={len(mdf):,})\n{'='*60}")

            # Global Metrics
            btc_metrics = compute_price_metrics(
                mdf,
                y_true_col="market_price",
                y_pred_col="model_price"
            )

            usd_metrics = compute_price_metrics(
                mdf,
                y_true_col="market_price_usd",
                y_pred_col="model_price_usd"
            )

            diag_metrics = compute_diagnostic_metrics(
                mdf,
                eval_mode=eval_mode,
                min_price=min_price,
                min_time_value=min_time_value
            )

            print(f"  Diagnostic Mode: {eval_mode.upper()}")
            print(f"  BTC Count : {int(btc_metrics['count']):,} rows")
            print(f"  BTC MAE   : {btc_metrics['MAE']:11.6f}")
            print(f"  BTC RMSE  : {btc_metrics['RMSE']:11.6f}")
            print(f"  BTC Bias  : {btc_metrics['Bias']:11.6f}")
            print(f"  BTC R²    : {btc_metrics['R2']:11.4f}")

            print(f"  USD Count : {int(usd_metrics['count']):,} rows")
            print(f"  USD MAE   : {usd_metrics['MAE']:11.6f}")
            print(f"  USD RMSE  : {usd_metrics['RMSE']:11.6f}")
            print(f"  USD Bias  : {usd_metrics['Bias']:11.6f}")
            print(f"  USD R²    : {usd_metrics['R2']:11.4f}")

            print(f"  Stable Count: {int(diag_metrics['count']):,} rows")
            print(f"  MARE      : {diag_metrics['MARE']:11.6f}")
            print(f"  MALE      : {diag_metrics['MALE']:11.6f}")
            print(f"  MANE      : {diag_metrics['MANE']:11.6f}")

            # Record Ranking Data
            rank_data = {
                "model_name": model,
                "count_btc": btc_metrics["count"],
                "MAE_BTC": btc_metrics["MAE"],
                "RMSE_BTC": btc_metrics["RMSE"],
                "Bias_BTC": btc_metrics["Bias"],
                "R2_BTC": btc_metrics["R2"],
                "count_usd": usd_metrics["count"],
                "MAE_USD": usd_metrics["MAE"],
                "RMSE_USD": usd_metrics["RMSE"],
                "Bias_USD": usd_metrics["Bias"],
                "R2_USD": usd_metrics["R2"],
                "count_diag": diag_metrics["count"],
                "MARE": diag_metrics["MARE"],
                "MALE": diag_metrics["MALE"],
                "MANE": diag_metrics["MANE"]
            }
            model_rankings.append(rank_data)

            # Segmented Metrics
            out_dir = os.path.join(self.output_dir, model)
            os.makedirs(out_dir, exist_ok=True)

            for seg in segments:
                seg_col = f"seg_{seg}"
                if seg_col in mdf.columns:
                    grouped = mdf.groupby(seg_col, observed=False).apply(
                        lambda g: compute_price_metrics(
                            g,
                            y_true_col="market_price",
                            y_pred_col="model_price"
                        )
                    ).unstack()

                    # Print and Save
                    print(f"\n--- By {seg.capitalize()} ---")
                    print(grouped[["count", "MAE", "RMSE", "Bias", "R2"]])
                    csv_path = os.path.join(out_dir, f"evaluation_{seg}.csv")
                    grouped.to_csv(csv_path)

            # Extra tabular output: by Option Type (useful if filter was 'both')
            if "option_type" in mdf.columns:
                by_type = mdf.groupby("option_type", observed=False).apply(
                    lambda g: compute_price_metrics(
                        g,
                        y_true_col="market_price",
                        y_pred_col="model_price"
                    )
                ).unstack()

                print(f"\n--- By Option Type ---\n{by_type[['count', 'MAE', 'RMSE', 'Bias', 'R2']]}")
                by_type.to_csv(os.path.join(out_dir, "evaluation_option_type.csv"))

            # 4. Visualization
            print(f"\n[Evaluator] Generating absolute-error plots for {model}...")
            generate_diagnostic_plots(
                mdf, model_name=model, out_dir=out_dir,
                error_type="abs", min_price=min_price
            )

        # 5. Model Ranking
        if model_rankings:
            rank_df = pd.DataFrame(model_rankings)
            rank_path = os.path.join(self.output_dir, "model_ranking.csv")
            rank_df.to_csv(rank_path, index=False)
            print(f"\n[Evaluator] Saved model comparison ranking to '{rank_path}'")