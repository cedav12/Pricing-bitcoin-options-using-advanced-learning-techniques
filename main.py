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
from src.pipelines.ann_pricing import ANNDatasetPipeline
from src.pipelines.bs_pricing import BlackScholesPipeline
from src.evaluation.model_evaluation import ModelEvaluator
from src.analysis.dataset_descriptives import run_descriptives_pipeline
from src.dataset_filter import DatasetFilterPipeline
from src.pipelines.ann_train import ANNTrainPipeline
from src.pipelines.ann_predict import ANNPredictPipeline

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
        choices=["build_dataset", "bs_pricing", "evaluate_model", "btc_descriptives", "dataset_descriptives", "filter_dataset", "ann_dataset", "ann_train", "ann_predict"],
        help=(
            "Execution mode:\n"
            "  build_dataset        – run the raw-data processing pipeline\n"
            "  bs_pricing           – compute model predictions and save to CSV\n"
            "  evaluate_model       – run model-agnostic evaluation and plot generation\n"
            "  btc_descriptives     – run BTC price analysis\n"
            "  dataset_descriptives – run options dataset metrics and diagnostics\n"
            "  filter_dataset       – flexible config-based dataset filtration\n"
            "  ann_dataset          – verify PyTorch dataset preparation\n"
            "  ann_train            – run modular ANN training pipeline\n"
            "  ann_predict          – run modular ANN inference pipeline\n"
        ),
    )


    # General / debugging limits
    parser.add_argument(
        "--sample-size", type=int, default=None, metavar="N",
        help="Max number of rows to load.",
    )

    args = parser.parse_args()

    # Helper to resolve prioritization: Config > Default
    def get_val(config_dict, key, default):
        return config_dict.get(key, default)

    mode_config = load_mode_config(args.config, args.mode) if args.mode != "btc_descriptives" else {}
    if args.sample_size is not None:
        mode_config["sample_size"] = args.sample_size

    # ── build_dataset ────────────────────────────────────────────────────────
    if args.mode == "build_dataset":
        print("Starting dataset preparation pipeline…")
        builder = DatasetBuilder()
        builder.build_dataset()
        print("Dataset preparation complete.")

    # ── bs_pricing ───────────────────────────────────────────────────────────
    elif args.mode == "bs_pricing":
        dataset_val = get_val(mode_config, "dataset", "data/processed/options_dataset_filtered.csv")
        vol_val = get_val(mode_config, "volatility", "rolling_std_7d")
        sample_size_val = get_val(mode_config, "sample_size", None)

        pipeline = BlackScholesPipeline(dataset_path=dataset_val)
        pipeline.run(
            vol_column=vol_val,
            sample_size=sample_size_val,
        )

    # ── evaluate_model ───────────────────────────────────────────────────────
    elif args.mode == "evaluate_model":
        input_val = get_val(mode_config, "input", "data/processed/predictions_bs.csv")
        input_val = get_val(mode_config, "input", "data/processed/predictions_bs.csv")
        opt_filter_val = get_val(mode_config, "option_filter", "call")
        error_val = get_val(mode_config, "error_type", "relative")
        eval_mode_val = get_val(mode_config, "evaluation_mode", "stable")
        min_price_val = get_val(mode_config, "min_price", 0.001)
        min_time_val = get_val(mode_config, "min_time_value", 0.001)
        segments_val = get_val(mode_config, "segments", ["moneyness", "maturity", "price"])
        sample_size_val = get_val(mode_config, "sample_size", None)

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

    # ── dataset_descriptives ─────────────────────────────────────────────────
    elif args.mode == "dataset_descriptives":
        input_val = get_val(mode_config, "dataset", "data/processed/options_dataset_filtered.csv")
        output_dir_val = get_val(mode_config, "output_dir", "output/dataset_descriptives")
        call_only_val = get_val(mode_config, "call_only", False)
        trade_count_val = get_val(mode_config, "filter_trade_count_positive", False)
        stale_val = get_val(mode_config, "run_stale_check", False)
        
        bucket_meth_val = get_val(mode_config, "bucket_method", "quantile")
        mon_bins_val = get_val(mode_config, "moneyness_bins", 3)
        ttm_bins_val = get_val(mode_config, "ttm_bins", 3)
        
        mon_edges_val = get_val(mode_config, "moneyness_edges", None)
        ttm_edges_val = get_val(mode_config, "ttm_edges", None)
        price_edges_val = get_val(mode_config, "price_edges", None)
        tc_edges_val = get_val(mode_config, "trade_count_edges", None)
        vol_edges_val = get_val(mode_config, "volume_edges", None)

        run_descriptives_pipeline(
            input_path=input_val,
            output_dir=output_dir_val,
            filter_trade_count_positive=trade_count_val,
            call_only=call_only_val,
            bucket_method=bucket_meth_val,
            moneyness_bins=mon_bins_val,
            moneyness_edges=mon_edges_val,
            ttm_bins=ttm_bins_val,
            ttm_edges=ttm_edges_val,
            price_edges=price_edges_val,
            trade_count_edges=tc_edges_val,
            volume_edges=vol_edges_val,
            run_stale_check=stale_val
        )

    # ── filter_dataset ───────────────────────────────────────────────────────
    elif args.mode == "filter_dataset":
        print("Starting dataset filtration pipeline...")
        pipeline = DatasetFilterPipeline(mode_config)
        pipeline.run()

    # ── ann_dataset ──────────────────────────────────────────────────────────
    elif args.mode == "ann_dataset":
        pipeline = ANNDatasetPipeline(mode_config)
        pipeline.run()

    # ── ann_train ────────────────────────────────────────────────────────────
    elif args.mode == "ann_train":
        pipeline = ANNTrainPipeline(mode_config)
        pipeline.run()

    # ── ann_predict ──────────────────────────────────────────────────────────
    elif args.mode == "ann_predict":
        pipeline = ANNPredictPipeline(mode_config)
        pipeline.run()


if __name__ == "__main__":
    main()

