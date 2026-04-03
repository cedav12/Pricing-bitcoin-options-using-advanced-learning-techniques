"""
main.py – Bitcoin Option Pricing Project entry-point
=====================================================
Usage examples
--------------
  python main.py --mode build_dataset
  python main.py --mode bs_pricing --volatility rolling_std_24h
  python main.py --mode evaluate_model --input data/processed/predictions_bs.csv --option-filter call
"""

import os
import json
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

def load_mode_config(config_arg, mode):
    if not config_arg:
        return {}
    if os.path.isdir(config_arg):
        path = os.path.join(config_arg, f"{mode}.json")
    else:
        path = config_arg

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}

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
        "--dataset", type=str, default=None, metavar="PATH",
        help="Path to the processed options CSV (default: data/processed/options_dataset_filtered.csv).",
    )
    parser.add_argument(
        "--volatility", type=str, default=None, metavar="VOL_COLUMN",
        help=f"Volatility estimator column. Supported: {', '.join(SUPPORTED_VOL_COLUMNS)}.",
    )

    # Arguments for evaluate_model mode
    parser.add_argument(
        "--input", type=str, default=None, metavar="PATH",
        help="Path to predictions CSV for evaluate_model.",
    )
    parser.add_argument(
        "--option-filter", type=str, default=None, choices=["call", "put", "both"],
        help="Filter predictions by option type before evaluation (default: call).",
    )
    parser.add_argument(
        "--error-type", type=str, default=None, choices=["absolute", "relative", "log"],
        help="Target error metrics for visualization. Absolute, Relative, or Log (default: relative).",
    )
    parser.add_argument(
        "--evaluation-mode", type=str, default=None, choices=["full", "stable"],
        help="Full (no threshold filtering) or Stable (drops extreme noisy data from scale metrics). Default: stable.",
    )
    parser.add_argument(
        "--min-price", type=float, default=None,
        help="Minimum market price filter for stable metrics and plots (default: 0.001).",
    )
    parser.add_argument(
        "--min-time-value", type=float, default=None,
        help="Minimum time value filter for stable metrics (default: 0.001).",
    )
    parser.add_argument(
        "--segments", nargs="+", default=None,
        help="List of segments to compute. Supported: moneyness, maturity, price, liquidity, volatility.",
    )

    # General / debugging limits
    parser.add_argument(
        "--sample-size", type=int, default=None, metavar="N",
        help="Max number of rows to load.",
    )

    args = parser.parse_args()

    # Helper to resolve prioritization: CLI > Config > Default
    def get_val(cli_val, config_dict, key, default):
        if cli_val is not None:
            return cli_val
        return config_dict.get(key, default)

    mode_config = load_mode_config(args.config, args.mode) if args.mode != "btc_descriptives" else {}

    # ── build_dataset ────────────────────────────────────────────────────────
    if args.mode == "build_dataset":
        print("Starting dataset preparation pipeline…")
        builder = DatasetBuilder()
        builder.build_dataset()
        print("Dataset preparation complete.")

    # ── bs_pricing ───────────────────────────────────────────────────────────
    elif args.mode == "bs_pricing":
        dataset_val = get_val(args.dataset, mode_config, "dataset", "data/processed/options_dataset_filtered.csv")
        vol_val = get_val(args.volatility, mode_config, "volatility", "rolling_std_7d")
        sample_size_val = get_val(args.sample_size, mode_config, "sample_size", None)

        pipeline = BlackScholesPipeline(dataset_path=dataset_val)
        pipeline.run(
            vol_column=vol_val,
            sample_size=sample_size_val,
        )

    # ── evaluate_model ───────────────────────────────────────────────────────
    elif args.mode == "evaluate_model":
        input_val = get_val(args.input, mode_config, "input", "data/processed/predictions_bs.csv")
        opt_filter_val = get_val(args.option_filter, mode_config, "option_filter", "call")
        error_val = get_val(args.error_type, mode_config, "error_type", "relative")
        eval_mode_val = get_val(args.evaluation_mode, mode_config, "evaluation_mode", "stable")
        min_price_val = get_val(args.min_price, mode_config, "min_price", 0.001)
        min_time_val = get_val(args.min_time_value, mode_config, "min_time_value", 0.001)
        segments_val = get_val(args.segments, mode_config, "segments", ["moneyness", "maturity", "price"])
        sample_size_val = get_val(args.sample_size, mode_config, "sample_size", None)

        evaluator = ModelEvaluator(predictions_path=input_val)
        evaluator.run(
            option_filter=opt_filter_val,
            error_type=error_val,
            eval_mode=eval_mode_val,
            min_price=min_price_val,
            min_time_value=min_time_val,
            segments=segments_val,
            sample_size=sample_size_val
        )

    # ── btc_descriptives ─────────────────────────────────────────────────────
    elif args.mode == "btc_descriptives":
        analysis = BTCDescriptiveAnalyzer(args.config)
        analysis.run()


if __name__ == "__main__":
    main()

