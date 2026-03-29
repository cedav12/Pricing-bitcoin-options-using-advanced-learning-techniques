import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BTCDescriptiveAnalyzer:
    """
    Config-driven descriptive analysis for BTC price / return / volatility data.

    Expected JSON config example
    ----------------------------
    {
      "data": "data/processed/btc_volatility.csv",
      "output_dir": "output/btc_descriptives",
      "periods_per_year": 8760,
      "timestamp_unit": "ms",
      "save_tables": true,
      "save_plots": true
    }

    Notes
    -----
    Assumptions:
    - realized_volatility, parkinson_volatility, garman_klass_volatility
      are NOT annualised in input and will be annualised for comparison plots.
    - realized_variance, positive_semivariance, negative_semivariance
      are variance-type measures and will be annualised using periods_per_year.
    - rolling_std_24h, rolling_std_7d, garch_volatility are assumed to
      already be annualised and are not rescaled.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "data": "data/processed/btc_volatility.csv",
        "output_dir": "output/btc_descriptives",
        "periods_per_year": 24 * 365,
        "timestamp_unit": "ms",
        "save_tables": True,
        "save_plots": True,
    }

    PRICE_COLUMNS = [
        "btc_price_close",
    ]

    RETURN_COLUMNS = [
        "btc_return",
    ]

    VARIANCE_COLUMNS = [
        "realized_variance",
        "positive_semivariance",
        "negative_semivariance",
    ]

    NON_ANNUALIZED_VOL_COLUMNS = [
        "realized_volatility",
        "parkinson_volatility",
        "garman_klass_volatility",
    ]

    ALREADY_ANNUALIZED_VOL_COLUMNS = [
        "rolling_std_24h",
        "rolling_std_7d",
        "garch_volatility",
    ]

    VOL_COLUMNS = NON_ANNUALIZED_VOL_COLUMNS + ALREADY_ANNUALIZED_VOL_COLUMNS

    def __init__(self, config_path: Union[str, Path]) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

        self.data_path = Path(self.config["data"])
        self.output_dir = Path(self.config["output_dir"])
        self.periods_per_year = int(self.config["periods_per_year"])
        self.timestamp_unit = str(self.config["timestamp_unit"])
        self.save_tables = bool(self.config["save_tables"])
        self.save_plots = bool(self.config["save_plots"])

        if self.periods_per_year <= 0:
            raise ValueError("'periods_per_year' must be a positive integer.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df: Optional[pd.DataFrame] = None
        self.stats: Optional[pd.DataFrame] = None
        self.plot_paths: Dict[str, str] = {}
        self.table_paths: Dict[str, str] = {}

    @classmethod
    def _load_config(cls, config_path: Path) -> Dict[str, Any]:
        """
        Load config from JSON file or from config directory.

        Behavior:
        - if config_path is a JSON file -> use it directly
        - if config_path is a directory -> use config_path / 'btc_descriptives.json'
        - otherwise -> interpret as directory-like path and append btc_descriptives.json
        """
        if config_path.is_dir():
            config_file = config_path / "btc_descriptives.json"
        elif config_path.suffix.lower() == ".json":
            config_file = config_path
        else:
            config_file = config_path / "btc_descriptives.json"

        if not config_file.exists():
            raise FileNotFoundError(
                f"BTC descriptives config file not found: {config_file}"
            )

        with config_file.open("r", encoding="utf-8") as f:
            loaded = json.load(f)

        if not isinstance(loaded, dict):
            raise ValueError("BTC descriptives config must be a JSON object.")

        config = dict(cls.DEFAULT_CONFIG)
        config.update(loaded)

        if not config.get("data"):
            raise ValueError("Config must contain non-empty 'data' path.")

        return config

    @property
    def annualized_vol_columns(self) -> list[str]:
        return [
            "realized_volatility_annualized",
            "parkinson_volatility_annualized",
            "garman_klass_volatility_annualized",
            "rolling_std_24h",
            "rolling_std_7d",
            "garch_volatility",
        ]

    @property
    def annualized_variance_columns(self) -> list[str]:
        return [
            "realized_variance_annualized",
            "positive_semivariance_annualized",
            "negative_semivariance_annualized",
        ]

    def load_data(self) -> pd.DataFrame:
        """
        Load and prepare BTC descriptive dataset.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {self.data_path}")

        df = pd.read_csv(self.data_path)

        if "timestamp" not in df.columns:
            raise ValueError("Input dataset must contain 'timestamp' column.")

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit=self.timestamp_unit,
            errors="coerce",
        )
        df = df.sort_values("timestamp").reset_index(drop=True)

        numeric_cols = [
            c for c in (
                self.PRICE_COLUMNS
                + self.RETURN_COLUMNS
                + self.VARIANCE_COLUMNS
                + self.VOL_COLUMNS
            )
            if c in df.columns
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        vol_factor = np.sqrt(self.periods_per_year)
        var_factor = self.periods_per_year

        # Annualise non-annualised volatility estimators
        annualize_vol_map = {
            "realized_volatility": "realized_volatility_annualized",
            "parkinson_volatility": "parkinson_volatility_annualized",
            "garman_klass_volatility": "garman_klass_volatility_annualized",
        }
        for src, dst in annualize_vol_map.items():
            if src in df.columns:
                df[dst] = df[src] * vol_factor

        # Annualise variance / semivariance estimators
        annualize_var_map = {
            "realized_variance": "realized_variance_annualized",
            "positive_semivariance": "positive_semivariance_annualized",
            "negative_semivariance": "negative_semivariance_annualized",
        }
        for src, dst in annualize_var_map.items():
            if src in df.columns:
                df[dst] = df[src] * var_factor

        self.df = df
        return df

    def compute_descriptive_stats(self) -> pd.DataFrame:
        """
        Compute descriptive statistics for all available core columns.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        cols = [
            c for c in (
                self.PRICE_COLUMNS
                + self.RETURN_COLUMNS
                + self.VARIANCE_COLUMNS
                + self.VOL_COLUMNS
                + self.annualized_vol_columns
                + self.annualized_variance_columns
            )
            if c in self.df.columns
        ]

        stats = self.df[cols].describe().T
        stats["missing_count"] = self.df[cols].isna().sum()
        stats["missing_pct"] = self.df[cols].isna().mean() * 100.0
        stats["skew"] = self.df[cols].skew(numeric_only=True)
        stats["kurtosis"] = self.df[cols].kurtosis(numeric_only=True)

        self.stats = stats

        if self.save_tables:
            out_path = self.output_dir / "descriptive_stats.csv"
            stats.to_csv(out_path)
            self.table_paths["descriptive_stats"] = str(out_path)

        return stats

    def _save_current_figure(self, name: str) -> None:
        if not self.save_plots:
            plt.close()
            return

        out_path = self.output_dir / f"{name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        self.plot_paths[name] = str(out_path)
        plt.close()

    def plot_price_and_returns(self) -> None:
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        available = [c for c in ["btc_price_close", "btc_return"] if c in self.df.columns]
        if not available:
            return

        fig, axes = plt.subplots(len(available), 1, figsize=(12, 7), sharex=True)
        if len(available) == 1:
            axes = [axes]

        for ax, col in zip(axes, available):
            ax.plot(self.df["timestamp"], self.df[col])
            ax.set_title(col)
            ax.grid(True, alpha=0.3)

        self._save_current_figure("price_and_returns")

    def plot_volatility_estimators_raw(self) -> None:
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        cols = [c for c in self.VOL_COLUMNS if c in self.df.columns]
        if not cols:
            return

        plt.figure(figsize=(12, 6))
        for col in cols:
            plt.plot(self.df["timestamp"], self.df[col], label=col)

        plt.title("Raw volatility estimators")
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._save_current_figure("volatility_estimators_raw")

    def plot_volatility_estimators_annualized(self) -> None:
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        cols = [c for c in self.annualized_vol_columns if c in self.df.columns]
        if not cols:
            return

        plt.figure(figsize=(12, 6))
        for col in cols:
            plt.plot(self.df["timestamp"], self.df[col], label=col)

        plt.title("Annualised volatility comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._save_current_figure("volatility_estimators_annualized")

    def plot_annualized_variance_estimators(self) -> None:
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        cols = [c for c in self.annualized_variance_columns if c in self.df.columns]
        if not cols:
            return

        plt.figure(figsize=(12, 6))
        for col in cols:
            plt.plot(self.df["timestamp"], self.df[col], label=col)

        plt.title("Annualised variance / semivariance comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._save_current_figure("annualized_variance_estimators")

    def plot_return_distribution(self) -> None:
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        if "btc_return" not in self.df.columns:
            return

        plt.figure(figsize=(10, 5))
        series = self.df["btc_return"].dropna()
        plt.hist(series, bins=100)
        plt.title("BTC return distribution")
        plt.grid(True, alpha=0.3)
        self._save_current_figure("return_distribution")

    def plot_semivariance_balance(self) -> None:
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        cols = [c for c in ["positive_semivariance", "negative_semivariance"] if c in self.df.columns]
        if len(cols) < 2:
            return

        plt.figure(figsize=(12, 6))
        for col in cols:
            plt.plot(self.df["timestamp"], self.df[col], label=col)

        plt.title("Positive vs negative semivariance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._save_current_figure("semivariance_balance")

    def plot_correlation_heatmap(self) -> None:
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        cols = [
            c for c in (
                self.RETURN_COLUMNS
                + self.VARIANCE_COLUMNS
                + self.VOL_COLUMNS
                + self.annualized_vol_columns
                + self.annualized_variance_columns
            )
            if c in self.df.columns
        ]
        if len(cols) < 2:
            return

        corr = self.df[cols].corr()

        plt.figure(figsize=(12, 9))
        plt.imshow(corr, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
        plt.yticks(range(len(cols)), cols)
        plt.title("Correlation heatmap")
        self._save_current_figure("correlation_heatmap")

        if self.save_tables:
            out_path = self.output_dir / "correlation_matrix.csv"
            corr.to_csv(out_path)
            self.table_paths["correlation_matrix"] = str(out_path)

    def plot_volatility_boxplot(self) -> None:
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        cols = [c for c in self.annualized_vol_columns if c in self.df.columns]
        if not cols:
            return

        plt.figure(figsize=(12, 6))
        clean = [self.df[c].dropna().values for c in cols]
        plt.boxplot(clean, tick_labels=cols)
        plt.xticks(rotation=30, ha="right")
        plt.title("Annualised volatility estimator boxplot")
        plt.grid(True, alpha=0.3)
        self._save_current_figure("volatility_boxplot")

    def plot_annualized_variance_boxplot(self) -> None:
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        cols = [c for c in self.annualized_variance_columns if c in self.df.columns]
        if not cols:
            return

        plt.figure(figsize=(10, 6))
        clean = [self.df[c].dropna().values for c in cols]
        plt.boxplot(clean, tick_labels=cols)
        plt.xticks(rotation=30, ha="right")
        plt.title("Annualised variance / semivariance boxplot")
        plt.grid(True, alpha=0.3)
        self._save_current_figure("annualized_variance_boxplot")

    def generate_all_plots(self) -> None:
        self.plot_price_and_returns()
        self.plot_volatility_estimators_raw()
        self.plot_volatility_estimators_annualized()
        self.plot_annualized_variance_estimators()
        self.plot_return_distribution()
        self.plot_semivariance_balance()
        self.plot_correlation_heatmap()
        self.plot_volatility_boxplot()
        self.plot_annualized_variance_boxplot()

    def run(self) -> Dict[str, Any]:
        """
        Full pipeline entry-point used by main.py.

        Returns
        -------
        dict
            {
                "stats": pd.DataFrame,
                "plot_paths": dict,
                "table_paths": dict,
                "output_dir": str
            }
        """
        print(f"Loading BTC descriptive config from: {self.config_path}")
        print(f"Reading dataset: {self.data_path}")

        self.load_data()
        self.compute_descriptive_stats()
        self.generate_all_plots()

        print("BTC descriptive analysis complete.")
        print(f"Output directory: {self.output_dir}")

        if self.plot_paths:
            print("\nGenerated plots:")
            for name, path in self.plot_paths.items():
                print(f"  - {name}: {path}")

        if self.table_paths:
            print("\nGenerated tables:")
            for name, path in self.table_paths.items():
                print(f"  - {name}: {path}")

        return {
            "stats": self.stats,
            "plot_paths": self.plot_paths,
            "table_paths": self.table_paths,
            "output_dir": str(self.output_dir),
        }