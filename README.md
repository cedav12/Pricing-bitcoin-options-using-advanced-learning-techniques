# Bitcoin Option Pricing & Benchmarking

Advanced pipeline for processing Deribit Bitcoin (BTC) options data and benchmarking the Black-Scholes model against market prices using various volatility estimators.

## Features

- **Data Processing**: Streams large-scale BTC options and perpetual futures data, performs hourly aggregation, and merges with macro financial data (Risk-free rate, VIX).
- **Feature Engineering**:
  - **Option Metrics**: Time to maturity, log-moneyness.
  - **Volatility Estimators**: Rolling historical-standard deviation (24h, 7d), Realized Volatility (5m returns), Parkinson, Garman-Klass, and GARCH(1,1).
- **Black-Scholes Benchmark**:
  - **Two-Stage Architecture**: Scalable split between heavy pricing (Stage 1) and instant diagnostics (Stage 2).
  - Vectorized pricer optimized with `scipy.special.ndtr`.
  - Per-row implied volatility (IV) inversion with **No-Arbitrage Bounds** checks.
  - Segmentation of pricing errors (MAE, RMSE, MAPE, Bias, R²) by moneyness and maturity.
  - Aggregated volatility smile analytics.
  - Headless diagnostic plotting (matplotlib).
- **Git Integration**: Clean project structure with `.gitignore` for large data files and environment caches.

## Installation

```bash
git clone git@github.com:cedav12/Pricing-bitcoin-options-using-advanced-learning-techniques.git
cd bitcoin_option_pricing
# Recommended: Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Build the Dataset

Processes raw CSVs from `data/raw/`, aligns timestamps, and produces a consolidated dataset.

```bash
python3 main.py --mode build_dataset
```

### 2. Black-Scholes Benchmarking

The benchmarking pipeline is split into two stages for scalability:

**Stage 1: Pricing & IV Inversion**
Computes theoretical prices and implied volatility for the entire dataset. Results are saved to `data/processed/options_with_bs.csv`.
```bash
python3 main.py --mode bs_pricing --volatility rolling_std_24h
```

**Stage 2: Analysis & Visualization**
Generates statistics (MAE, RMSE, R², Bias, etc.) and diagnostic plots from the precomputed dataset.
```bash
python3 main.py --mode bs_analysis
```

*Note: Use `--sample-size N` to limit rows for faster testing.*

## Project Structure

- `src/`: Core implementation logic.
  - `black_scholes.py`: Optimized vectorized pricer and benchmarking engine.
  - `btc_feature_engineering.py`: Volatility and return feature generation.
  - `dataset_builder.py`: Data alignment and merging pipeline.
- `tests/`: Unittest suite.
  - `test_core_logic.py`: Mathematics and basic feature checks.
  - `test_final_dataset.py`: Large-scale data integrity and no-arbitrage checks.
- `output/`: Generated stats, plots, and benchmark results.

## Testing

Run the test suite to ensure mathematical and data integrity:

```bash
python3 -m unittest discover tests
```
