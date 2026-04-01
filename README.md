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

### 2. Model Evaluation (Two-Stage Pipeline)

The benchmarking pipeline is split into a pure pricing phase and a model-agnostic evaluate phase:

**Stage 1: Pricing**
Computes predictions for the dataset. Results are saved to `data/processed/predictions_bs.csv`.
```bash
python3 main.py --mode bs_pricing --volatility rolling_std_24h
```

**Stage 2: Model-Agnostic Evaluation**
Generates segmented statistics (MAE, RMSE, MAPE, Bias, R²) and diagnostic plots/heatmaps.
```bash
# Evaluate CALL options (Default)
python3 main.py --mode evaluate_model --input data/processed/predictions_bs.csv --option-filter call

# Evaluate PUT options
python3 main.py --mode evaluate_model --input data/processed/predictions_bs.csv --option-filter put

# Evaluate BOTH options
python3 main.py --mode evaluate_model --input data/processed/predictions_bs.csv --option-filter both
```

*Note: Use `--sample-size N` to limit rows and test quickly.*

## Project Structure

- `src/`: Core implementation logic.
  - `models/black_scholes.py`: Pure, high-performance Black-Scholes math.
  - `pipelines/bs_pricing.py`: Data pipeline generating model predictions.
  - `evaluation/model_evaluation.py`: Model-agnostic statistical evaluator and plotter.
  - `btc_feature_engineering.py`: Volatility and return feature generation.
  - `dataset_builder.py`: Data alignment and merging pipeline.
- `tests/`: Unittest suite.
  - `test_core_logic.py`: Mathematics and basic feature checks.
  - `test_final_dataset.py`: Large-scale data integrity and no-arbitrage checks.
- `output/`: Generated stats, plots, and benchmark evaluation results (CSV + PNG).

## Testing

Run the test suite to ensure mathematical and data integrity:

```bash
python3 -m unittest discover tests
```
