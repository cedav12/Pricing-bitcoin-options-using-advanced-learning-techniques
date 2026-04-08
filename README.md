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

The project relies heavily on a central CLI and JSON configurations. The `--mode` argument drives the entry point, while `--config` points to optional JSON overrides for parameters.

### 1. Build the Dataset
Processes raw CSVs from `data/raw/`, aligns timestamps, and produces a consolidated dataset.
```bash
python3 main.py --mode build_dataset
```

### 2. Dataset Descriptives & Analysis
Audit and analyze the generated options dataset for liquidity, trade characteristics, and model readiness:
```bash
python3 main.py --mode dataset_descriptives --config config/dataset_descriptives.json
```

### 3. Filter Dataset
Applies configurable criteria (e.g., call-only, minimum trade counts, negative time value checks) keeping rigorous summary drops:
```bash
python3 main.py --mode filter_dataset --config config/filter_dataset.json
```

### 4. Model Evaluation (Two-Stage Pipeline)
The benchmarking pipeline is split into a pure pricing phase and a model-agnostic evaluate phase:

**Stage 1: Pricing**
Computes Black-Scholes predictions using a specified volatility column. 
```bash
python3 main.py --mode bs_pricing --config config/bs_pricing.json
```

**Stage 2: Model-Agnostic Evaluation**
Generates segmented statistics (MAE, RMSE, MAPE, Bias, R²) and diagnostic heatmaps over prediction sets.
```bash
python3 main.py --mode evaluate_model --config config/evaluate_model.json
```

*Note: Use `--sample-size N` to limit rows and test quickly for any mode.*

## Project Structure

- `src/`: Core implementation logic.
  - `dataset_builder.py`: Data alignment and merging pipeline.
  - `dataset_filter.py`: Quality filtration engine dropping arbitrary rows.
  - `btc_feature_engineering.py`: Volatility and return feature generation.
  - `btc_descriptives.py`: Macro-oriented timeseries diagnostics.
  - `analysis/dataset_descriptives.py`: Metric reporting matrices and trade distribution layers.
  - `models/black_scholes.py`: Pure, high-performance Black-Scholes math.
  - `pipelines/bs_pricing.py`: Data pipeline generating model predictions.
  - `evaluation/model_evaluation.py`: Model-agnostic statistical evaluator and plotter.
- `config/`: Configuration JSONs matching explicitly to the `--mode` arguments.
- `tests/`: Unittest suite covering math, integrity, and limits.
- `output/`: Generated stats, CSV tables, plots, and evaluation results.

## Testing

Run the test suite to ensure mathematical and data integrity:

```bash
python3 -m unittest discover tests
```
