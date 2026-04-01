"""
main.py – Bitcoin Option Pricing Project entry-point
=====================================================
Usage examples
--------------
  python main.py --mode build_dataset
  python main.py --mode bs_pricing --volatility rolling_std_24h
  python main.py --mode evaluate_model --input data/processed/predictions_bs.csv --option-filter call
"""

import argparse
from src.dataset_builder import DatasetBuilder
from src.btc_descriptives import BTCDescriptiveAnalyzer
from src.pipelines.bs_pricing import BlackScholesPipeline
from src.evaluation.model_evaluation import ModelEvaluator

SUPPORTED_VOL_COLUMNS = [
    "rolling_std_24h", "rolling_std_7d",
    "realized_volatility", "parkinson_volatility",
    "garman_klass_volatility", "garch_volatility"
]

def main():
    parser = argparse.ArgumentParser(
        description="Bitcoin Option Pricing Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default="config", metavar="PATH",
        help="Optional Path to JSON config file for mode-specific parameters.",
    )

    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["build_dataset", "bs_pricing", "evaluate_model", "btc_descriptives"],
        help=(
            "Execution mode:\n"
            "  build_dataset    – run the raw-data processing pipeline\n"
            "  bs_pricing       – compute model predictions and save to CSV\n"
            "  evaluate_model   – run model-agnostic evaluation and plot generation\n"
            "  btc_descriptives – run BTC price analysis\n"
        ),
    )

    # Arguments for bs_pricing mode
    parser.add_argument(
        "--dataset", type=str, default="data/processed/options_dataset.csv", metavar="PATH",
        help="Path to the processed options CSV (default: data/processed/options_dataset.csv).",
    )
    parser.add_argument(
        "--volatility", type=str, default="rolling_std_24h", metavar="VOL_COLUMN",
        help=f"Volatility estimator column. Supported: {', '.join(SUPPORTED_VOL_COLUMNS)}.",
    )

    # Arguments for evaluate_model mode
    parser.add_argument(
        "--input", type=str, default="data/processed/predictions_bs.csv", metavar="PATH",
        help="Path to predictions CSV for evaluate_model.",
    )
    parser.add_argument(
        "--option-filter", type=str, default="call", choices=["call", "put", "both"],
        help="Filter predictions by option type before evaluation (default: call).",
    )
    parser.add_argument(
        "--error-type", type=str, default="relative", choices=["absolute", "relative", "log"],
        help="Target error metrics for visualization. Absolute, Relative, or Log (default: relative).",
    )
    parser.add_argument(
        "--evaluation-mode", type=str, default="stable", choices=["full", "stable"],
        help="Full (no threshold filtering) or Stable (drops extreme noisy data from scale metrics). Default: stable.",
    )
    parser.add_argument(
        "--min-price", type=float, default=0.001,
        help="Minimum market price filter for stable metrics and plots (default: 0.001).",
    )
    parser.add_argument(
        "--min-time-value", type=float, default=0.001,
        help="Minimum time value filter for stable metrics (default: 0.001).",
    )
    parser.add_argument(
        "--segments", nargs="+", default=["moneyness", "maturity", "price"],
        help="List of segments to compute. Supported: moneyness, maturity, price, liquidity, volatility.",
    )

    # General / debugging limits
    parser.add_argument(
        "--sample-size", type=int, default=None, metavar="N",
        help="Max number of rows to load.",
    )

    args = parser.parse_args()

    # ── build_dataset ────────────────────────────────────────────────────────
    if args.mode == "build_dataset":
        print("Starting dataset preparation pipeline…")
        builder = DatasetBuilder()
        builder.build_dataset()
        print("Dataset preparation complete.")

    # ── bs_pricing ───────────────────────────────────────────────────────────
    elif args.mode == "bs_pricing":
        pipeline = BlackScholesPipeline(dataset_path=args.dataset)
        pipeline.run(
            vol_column=args.volatility,
            sample_size=args.sample_size,
        )

    # ── evaluate_model ───────────────────────────────────────────────────────
    elif args.mode == "evaluate_model":
        evaluator = ModelEvaluator(predictions_path=args.input)
        evaluator.run(
            option_filter=args.option_filter,
            error_type=args.error_type,
            eval_mode=args.evaluation_mode,
            min_price=args.min_price,
            min_time_value=args.min_time_value,
            segments=args.segments,
            sample_size=args.sample_size
        )

    # ── btc_descriptives ─────────────────────────────────────────────────────
    elif args.mode == "btc_descriptives":
        analysis = BTCDescriptiveAnalyzer(args.config)
        analysis.run()


if __name__ == "__main__":
    main()
