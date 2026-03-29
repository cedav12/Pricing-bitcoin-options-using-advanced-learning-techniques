"""
main.py – Bitcoin Option Pricing Project entry-point
=====================================================
Usage examples
--------------
  python main.py --mode build_dataset
  python main.py --mode black_scholes_benchmark
  python main.py --mode black_scholes_benchmark --volatility realized_volatility
  python main.py --mode black_scholes_benchmark --volatility garch_volatility --no-save-csv
  python main.py --mode black_scholes_benchmark --skip-smile
"""

import argparse
from src.dataset_builder import DatasetBuilder
from src.black_scholes import BlackScholesBenchmark, SUPPORTED_VOL_COLUMNS
from src.btc_descriptives import BTCDescriptiveAnalyzer



def main():
    parser = argparse.ArgumentParser(
        description="Bitcoin Option Pricing Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config",
        metavar="PATH",
        help="Optional Path to JSON config file for mode-specific parameters.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["build_dataset", "black_scholes_benchmark", "bs_pricing", "bs_analysis", "btc_descriptives"],
        help=(
            "Execution mode:\n"
            "  build_dataset           – run the raw-data processing pipeline\n"
            "  bs_pricing              – Stage 1: Compute BS prices and implied vols\n"
            "  bs_analysis             – Stage 2: Compute stats and diagnostic plots\n"
            "  black_scholes_benchmark – (legacy) run both stages together\n"
            "  btc_descriptives        – run BTC price analysis\n"
        ),
    )

    parser.add_argument(
        "--volatility",
        type=str,
        default="rolling_std_24h",
        metavar="VOL_COLUMN",
        help=(
            f"Volatility estimator column to use for the BS benchmark. "
            f"Supported: {', '.join(SUPPORTED_VOL_COLUMNS)}. "
            f"Default: rolling_std_24h"
        ),
    )

    parser.add_argument(
        "--no-save-csv",
        action="store_true",
        default=False,
        help="Skip saving output/bs_benchmark_results.csv (faster for large datasets).",
    )

    parser.add_argument(
        "--skip-smile",
        action="store_true",
        default=False,
        help=(
            "Skip the volatility smile / implied-vol plot. "
            "Useful on very large datasets where IV inversion is slow."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/options_dataset.csv",
        metavar="PATH",
        help="Path to the processed options CSV (default: data/processed/options_dataset.csv).",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=5_000_000,
        metavar="N",
        help="Max number of rows to load for benchmarking to prevent OOM (default: 5,000,000).",
    )

    args = parser.parse_args()

    # ── build_dataset ────────────────────────────────────────────────────────
    if args.mode == "build_dataset":
        print("Starting dataset preparation pipeline…")
        builder = DatasetBuilder()
        builder.build_dataset()
        print("Dataset preparation complete.")

    # ── bs_pricing (Stage 1) ────────────────────────────────────────────────
    elif args.mode == "bs_pricing":
        benchmark = BlackScholesBenchmark(dataset_path=args.dataset)
        benchmark.run_pricing_mode(
            vol_column=args.volatility,
            sample_size=args.sample_size,
        )

    # ── bs_analysis (Stage 2) ───────────────────────────────────────────────
    elif args.mode == "bs_analysis":
        benchmark = BlackScholesBenchmark(dataset_path=args.dataset)
        benchmark.run_analysis_mode(
            sample_size=args.sample_size,
        )

    # ── black_scholes_benchmark (Legacy/Combined) ───────────────────────────
    elif args.mode == "black_scholes_benchmark":
        benchmark = BlackScholesBenchmark(dataset_path=args.dataset)
        # Reorganized to run both stages sequentially
        benchmark.run_pricing_mode(
            vol_column=args.volatility,
            sample_size=args.sample_size,
        )
        benchmark.run_analysis_mode(
            sample_size=args.sample_size,
        )

    elif args.mode == "btc_descriptives":
        analysis = BTCDescriptiveAnalyzer(args.config)
        analysis.run()


if __name__ == "__main__":
    main()
