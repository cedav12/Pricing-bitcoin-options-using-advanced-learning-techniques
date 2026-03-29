"""
Black-Scholes Benchmarking Module
==================================
Provides:
  - black_scholes_price()        : vectorised BS call/put pricer
  - compute_implied_volatility() : per-row IV inversion via Brent's method
  - BlackScholesBenchmark        : full benchmark pipeline (load, price,
                                   metrics, plots, CLI entry-point)
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # headless – safe for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_VOL_COLUMNS = [
    "rolling_std_24h",
    "rolling_std_7d",
    "realized_volatility",
    "garch_volatility",
    "parkinson_volatility",
    "garman_klass_volatility",
]

_MONEYNESS_BINS  = [-np.inf, -0.20, -0.05, 0.05, 0.20, np.inf]
_MONEYNESS_LABELS = ["Deep OTM", "OTM", "ATM", "ITM", "Deep ITM"]

_MATURITY_BINS  = [0, 7/365, 28/365, 91/365, 365/365, np.inf]
_MATURITY_LABELS = ["< 1 week", "1–4 weeks", "1–3 months", "3–12 months", "> 1 year"]


# ---------------------------------------------------------------------------
# Core pricing function
# ---------------------------------------------------------------------------

def black_scholes_price(
    S: "array-like",
    K: "array-like",
    T: "array-like",
    r: "float | array-like",
    sigma: "float | array-like",
    option_type: "array-like",
) -> np.ndarray:
    """Vectorised Black-Scholes pricer for European call/put options.

    Parameters
    ----------
    S : underlying price
    K : strike price
    T : time to maturity in years
    r : risk-free rate (scalar or array)
    sigma : annualised volatility (scalar or per-row Series/array)
    option_type : 'call' or 'put' (case-insensitive), Series or array

    Returns
    -------
    np.ndarray of theoretical prices
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Numerical safety guards
    T_safe   = np.where(T <= 0.0,   1e-8,  T)
    S_safe   = np.where(S <= 0.0,   1e-8,  S)
    K_safe   = np.where(K <= 0.0,   1e-8,  K)
    sig_safe = np.where(sigma <= 0.0, 1e-8, sigma)

    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S_safe / K_safe) + (r + 0.5 * sig_safe ** 2) * T_safe) / (sig_safe * sqrt_T)
    d2 = d1 - sig_safe * sqrt_T

    call_price = S_safe * ndtr(d1) - K_safe * np.exp(-r * T_safe) * ndtr(d2)
    put_price  = K_safe * np.exp(-r * T_safe) * ndtr(-d2) - S_safe * ndtr(-d1)

    # Resolve option type
    if isinstance(option_type, pd.Series):
        is_call = option_type.str.lower().values == "call"
    else:
        opt_arr = np.asarray(option_type)
        is_call = np.char.lower(opt_arr.astype(str)) == "call"

    return np.where(is_call, call_price, put_price)


# ---------------------------------------------------------------------------
# Implied-volatility inversion
# ---------------------------------------------------------------------------

def _bs_scalar(sigma: float, S: float, K: float, T: float,
               r: float, is_call: bool) -> float:
    """Scalar BS price for a single option row (used inside root-finder)."""
    T_s = max(T, 1e-8)
    sig_s = max(sigma, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sig_s ** 2) * T_s) / (sig_s * np.sqrt(T_s))
    d2 = d1 - sig_s * np.sqrt(T_s)
    if is_call:
        return S * ndtr(d1) - K * np.exp(-r * T_s) * ndtr(d2)
    return K * np.exp(-r * T_s) * ndtr(-d2) - S * ndtr(-d1)


def _iv_one_row(market_price: float, S: float, K: float,
                T: float, r: float, is_call: bool) -> float:
    """Return implied vol for one row; NaN if Brent's method fails or bounds are violated."""
    if not np.isfinite(market_price) or market_price <= 0:
        return np.nan
    
    # 1. No-Arbitrage Bounds Check (USD space)
    df_k = K * np.exp(-r * max(T, 0))
    if is_call:
        lower = max(0, S - df_k)
        upper = S
    else:
        lower = max(0, df_k - S)
        upper = df_k
    
    # Allow a small tolerance for market microstructure / bid-ask
    tol = 1e-7 * S
    if market_price < lower - tol or market_price > upper + tol:
        return np.nan

    # 2. Inversion
    try:
        iv = brentq(
            lambda sig: _bs_scalar(sig, S, K, T, r, is_call) - market_price,
            1e-6, 20.0,
            xtol=1e-6, maxiter=200,
        )
        return iv
    except (ValueError, RuntimeError):
        return np.nan


_vec_iv = np.vectorize(_iv_one_row)


def compute_implied_volatility(
    market_price: "array-like",
    S: "array-like",
    K: "array-like",
    T: "array-like",
    r: "array-like",
    option_type: "array-like",
) -> np.ndarray:
    """Compute per-row implied volatility via Brent root-finding.

    Returns an array of implied vols; rows outside no-arbitrage bounds → NaN.
    """
    if isinstance(option_type, pd.Series):
        is_call = (option_type.str.lower().values == "call")
    else:
        is_call = (np.char.lower(np.asarray(option_type).astype(str)) == "call")

    return _vec_iv(
        np.asarray(market_price, dtype=float),
        np.asarray(S,            dtype=float),
        np.asarray(K,            dtype=float),
        np.asarray(T,            dtype=float),
        np.asarray(r,            dtype=float),
        is_call,
    )


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class BlackScholesBenchmark:
    """End-to-end Black-Scholes benchmark pipeline.

    Parameters
    ----------
    dataset_path : path to the processed options CSV
    output_dir   : root output directory (plots + results)
    """

    def __init__(
        self,
        dataset_path: str = "data/processed/options_dataset.csv",
        output_dir: str = "output",
    ):
        self.dataset_path = dataset_path
        self.output_dir   = output_dir
        self.plots_dir    = os.path.join(output_dir, "plots")
        self.df: Optional[pd.DataFrame] = None
        self._vol_column: Optional[str] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        vol_column: str = "rolling_std_24h",
        chunksize: int = 500_000,
        max_rows: Optional[int] = 5_000_000,
    ) -> pd.DataFrame:
        """Load the processed dataset in chunks, keeping only usable rows.

        If max_rows is set, the loader will stop after reaching the limit
        or sample from the file to stay within memory bounds.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at '{self.dataset_path}'. "
                "Run --mode build_dataset first."
            )

        required_cols = {
            "option_price", "underlying_price", "strike",
            "time_to_maturity", "option_type", vol_column,
        }

        chunks = []
        loaded_count = 0
        print(f"  Loading '{self.dataset_path}' in chunks of {chunksize:,} …")
        if max_rows:
            print(f"  (Capping dataset at {max_rows:,} rows for memory stability)")

        for chunk in pd.read_csv(self.dataset_path, chunksize=chunksize, low_memory=False):
            missing = required_cols - set(chunk.columns)
            if missing:
                raise ValueError(f"Dataset is missing required columns: {missing}")

            # Risk-free rate fallback
            if "risk_free_rate" not in chunk.columns:
                chunk["risk_free_rate"] = 0.0
            else:
                chunk["risk_free_rate"] = chunk["risk_free_rate"].fillna(0.0)

            # Drop unusable rows
            mask = (
                chunk["option_price"].notna()        & (chunk["option_price"]      > 0) &
                chunk["underlying_price"].notna()    & (chunk["underlying_price"]  > 0) &
                chunk["strike"].notna()              & (chunk["strike"]            > 0) &
                chunk["time_to_maturity"].notna()    & (chunk["time_to_maturity"]  > 0) &
                chunk[vol_column].notna()            & (chunk[vol_column]          > 0)
            )
            filtered_chunk = chunk.loc[mask]
            
            if max_rows and (loaded_count + len(filtered_chunk) > max_rows):
                remaining = max_rows - loaded_count
                if remaining > 0:
                    chunks.append(filtered_chunk.iloc[:remaining])
                break
            
            chunks.append(filtered_chunk)
            loaded_count += len(filtered_chunk)

        df = pd.concat(chunks, ignore_index=True)
        print(f"  Loaded {len(df):,} rows after filtering.")
        return df

    # ------------------------------------------------------------------
    # BS pricing
    # ------------------------------------------------------------------

    def compute_bs_prices(self, df: pd.DataFrame, vol_column: str) -> pd.DataFrame:
        """Add 'bs_price' and 'pricing_error' columns to *df* (in-place copy).
        
        Note: Black-Scholes formula returns prices in USD (same as S and K).
        Deribit options are quoted in BTC, so we convert the model output:
        Price_BTC = Price_USD / Underlying_Price
        """
        df = df.copy()
        bs_price_usd = black_scholes_price(
            S=df["underlying_price"],
            K=df["strike"],
            T=df["time_to_maturity"],
            r=df["risk_free_rate"],
            sigma=df[vol_column],
            option_type=df["option_type"],
        )
        df["bs_price"] = bs_price_usd / df["underlying_price"]
        df["pricing_error"] = df["bs_price"] - df["option_price"]
        return df

    # ------------------------------------------------------------------
    # Bucket helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_moneyness_bucket(log_moneyness: pd.Series) -> pd.Categorical:
        return pd.cut(
            log_moneyness,
            bins=_MONEYNESS_BINS,
            labels=_MONEYNESS_LABELS,
            right=True,
        )

    @staticmethod
    def _assign_maturity_bucket(ttm: pd.Series) -> pd.Categorical:
        return pd.cut(
            ttm,
            bins=_MATURITY_BINS,
            labels=_MATURITY_LABELS,
            right=True,
        )

    # ------------------------------------------------------------------
    # Error metrics
    # ------------------------------------------------------------------

    def compute_error_metrics(self, df: pd.DataFrame) -> dict:
        """Compute MAE, RMSE, MAPE, mean bias – overall and grouped.

        Returns a dict with keys:
          'overall'         -> dict of scalar metrics
          'by_moneyness'    -> grouped DataFrame
          'by_maturity'     -> grouped DataFrame
        """
        err   = df["pricing_error"]
        price = df["option_price"]

        mae  = err.abs().mean()
        rmse = np.sqrt((err ** 2).mean())
        # MAPE: guard against zero market prices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mape = (err.abs() / price.replace(0, np.nan)).mean() * 100
        bias = err.mean()

        # Moneyness bucket – compute log_moneyness on the fly if missing
        if "log_moneyness" in df.columns:
            lm = df["log_moneyness"]
        else:
            lm = np.log(df["underlying_price"] / df["strike"])

        df = df.copy()
        df["_mon_bucket"] = self._assign_moneyness_bucket(lm)
        df["_mat_bucket"] = self._assign_maturity_bucket(df["time_to_maturity"])

        def _group_stats(group_col: str) -> pd.DataFrame:
            g = df.groupby(group_col, observed=True)
            out = pd.DataFrame({
                "count": g["pricing_error"].count(),
                "MAE":   g["pricing_error"].apply(lambda x: x.abs().mean()),
                "RMSE":  g["pricing_error"].apply(lambda x: np.sqrt((x**2).mean())),
                "Bias":  g["pricing_error"].mean(),
            })
            return out

        by_moneyness = _group_stats("_mon_bucket")
        by_maturity  = _group_stats("_mat_bucket")

        return {
            "overall": {
                "MAE":  mae,
                "RMSE": rmse,
                "MAPE": mape,
                "Bias": bias,
            },
            "by_moneyness": by_moneyness,
            "by_maturity":  by_maturity,
        }

    # ------------------------------------------------------------------
    # Pretty-print summary
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(metrics: dict, vol_column: str) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  Black-Scholes Benchmark  |  vol = {vol_column}")
        print(sep)
        ov = metrics["overall"]
        print(f"  MAE  : {ov['MAE']:>12.4f}")
        print(f"  RMSE : {ov['RMSE']:>12.4f}")
        print(f"  MAPE : {ov['MAPE']:>11.2f}%")
        print(f"  Bias : {ov['Bias']:>12.4f}  (positive = model overprices)")
        print(f"\n--- By Moneyness Bucket ---")
        print(metrics["by_moneyness"].to_string())
        print(f"\n--- By Maturity Bucket ---")
        print(metrics["by_maturity"].to_string())
        print(sep + "\n")

    # ------------------------------------------------------------------
    # Volatility smile / smirk plot
    # ------------------------------------------------------------------

    def plot_volatility_smile(
        self,
        df: pd.DataFrame,
        n_timestamps: int = 6,
    ) -> str:
        """Plot implied volatility vs log-moneyness for selected timestamps.

        Saves to output/plots/volatility_smile_example.png and returns the path.
        """
        os.makedirs(self.plots_dir, exist_ok=True)

        # Work on a sample to keep IV calculation tractable
        sample_size = min(200_000, len(df))
        sample = df.sample(n=sample_size, random_state=42).copy()

        print("  Computing implied volatilities …")
        # market_price must be in USD for the pricer solver
        market_price_usd = sample["option_price"] * sample["underlying_price"]
        sample["implied_vol"] = compute_implied_volatility(
            market_price=market_price_usd,
            S=sample["underlying_price"],
            K=sample["strike"],
            T=sample["time_to_maturity"],
            r=sample["risk_free_rate"],
            option_type=sample["option_type"],
        )
        sample = sample.dropna(subset=["implied_vol"])
        sample = sample[sample["implied_vol"] > 0]

        if "log_moneyness" not in sample.columns:
            sample["log_moneyness"] = np.log(
                sample["underlying_price"] / sample["strike"]
            )

        # Select representative timestamps
        if "timestamp" in sample.columns:
            ts_values = (
                sample["timestamp"]
                .drop_duplicates()
                .sample(min(n_timestamps, sample["timestamp"].nunique()), random_state=7)
                .sort_values()
                .values
            )
        else:
            ts_values = None

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        palette = plt.cm.plasma(np.linspace(0.15, 0.9, n_timestamps))

        if ts_values is not None and len(ts_values) > 1:
            for i, ts in enumerate(ts_values):
                subset = sample[sample["timestamp"] == ts]
                if len(subset) < 3:
                    continue
                subset_sorted = subset.sort_values("log_moneyness")
                ax.scatter(
                    subset_sorted["log_moneyness"], subset_sorted["implied_vol"],
                    color=palette[i], s=15, alpha=0.7,
                    label=str(pd.to_datetime(ts, unit="ms", errors="coerce"))[:16],
                )
        else:
            # Fallback: plot all as a single scatter
            sample_sorted = sample.sort_values("log_moneyness")
            sc = ax.scatter(
                sample_sorted["log_moneyness"], sample_sorted["implied_vol"],
                c=sample_sorted["implied_vol"], cmap="plasma", s=8, alpha=0.5,
            )
            plt.colorbar(sc, ax=ax, label="Implied Vol")

        ax.set_xlabel("Log Moneyness  ln(S/K)", color="white", fontsize=12)
        ax.set_ylabel("Implied Volatility", color="white", fontsize=12)
        ax.set_title("Volatility Smile / Smirk", color="white", fontsize=14, fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444466")
        if ts_values is not None and len(ts_values) > 1:
            ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white", title="Timestamp", title_fontsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        plt.tight_layout()
        out_path = os.path.join(self.plots_dir, "volatility_smile_example.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------

    def plot_diagnostics(self, df: pd.DataFrame, vol_column: str) -> str:
        """4-panel diagnostic figure.

        Panels:
          1. market_price vs bs_price scatter
          2. pricing_error distribution
          3. pricing_error vs moneyness bucket (box)
          4. pricing_error vs maturity bucket (box)

        Saves to output/plots/diagnostics_<vol_column>.png and returns path.
        """
        os.makedirs(self.plots_dir, exist_ok=True)

        plot_df = df.copy()
        if "log_moneyness" not in plot_df.columns:
            plot_df["log_moneyness"] = np.log(
                plot_df["underlying_price"] / plot_df["strike"]
            )
        plot_df["_mon_bucket"] = self._assign_moneyness_bucket(plot_df["log_moneyness"])
        plot_df["_mat_bucket"] = self._assign_maturity_bucket(plot_df["time_to_maturity"])

        # Cap to 500k for plot speed
        if len(plot_df) > 500_000:
            plot_df = plot_df.sample(500_000, random_state=42)

        dark_bg   = "#1a1a2e"
        panel_bg  = "#16213e"
        accent    = "#e94560"
        accent2   = "#0f3460"
        text_col  = "white"
        grid_col  = "#2a2a4a"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(dark_bg)
        fig.suptitle(
            f"Black-Scholes Diagnostics  |  vol = {vol_column}",
            color=text_col, fontsize=15, fontweight="bold", y=1.01
        )

        for ax in axes.flat:
            ax.set_facecolor(panel_bg)
            ax.tick_params(colors=text_col)
            ax.spines[:].set_color("#444466")
            ax.xaxis.label.set_color(text_col)
            ax.yaxis.label.set_color(text_col)
            ax.title.set_color(text_col)
            ax.grid(color=grid_col, linewidth=0.5, alpha=0.7)

        # ── Panel 1: market vs BS price scatter ──────────────────────────
        ax1 = axes[0, 0]
        price_max = np.percentile(
            plot_df[["option_price", "bs_price"]].dropna().values.ravel(), 99
        )
        ax1.scatter(
            plot_df["option_price"], plot_df["bs_price"],
            s=3, alpha=0.25, color=accent, rasterized=True,
        )
        ax1.plot([0, price_max], [0, price_max], "w--", linewidth=1, alpha=0.6, label="Perfect fit")
        ax1.set_xlim(0, price_max)
        ax1.set_ylim(0, price_max)
        ax1.set_xlabel("Market Price")
        ax1.set_ylabel("BS Price")
        ax1.set_title("Market vs BS Price")
        ax1.legend(fontsize=8, facecolor=panel_bg, labelcolor=text_col)

        # ── Panel 2: pricing error distribution ──────────────────────────
        ax2 = axes[0, 1]
        err_vals = plot_df["pricing_error"].dropna()
        clip_lo, clip_hi = np.percentile(err_vals, [1, 99])
        err_vals_clipped = err_vals.clip(clip_lo, clip_hi)
        ax2.hist(err_vals_clipped, bins=100, color="#5c7aea", edgecolor="none", alpha=0.85)
        ax2.axvline(0, color="white", linewidth=1.5, linestyle="--", alpha=0.7)
        ax2.axvline(err_vals.mean(), color=accent, linewidth=1.5, linestyle="--",
                    label=f"Mean={err_vals.mean():.2f}")
        ax2.set_xlabel("Pricing Error (BS − Market)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Pricing Error Distribution")
        ax2.legend(fontsize=8, facecolor=panel_bg, labelcolor=text_col)

        # ── Panel 3: error vs moneyness bucket (box) ─────────────────────
        ax3 = axes[1, 0]
        groups_mon = [
            plot_df.loc[plot_df["_mon_bucket"] == lbl, "pricing_error"].dropna().values
            for lbl in _MONEYNESS_LABELS
        ]
        bp3 = ax3.boxplot(
            groups_mon,
            labels=_MONEYNESS_LABELS,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=1.5),
            flierprops=dict(marker=".", color=accent, alpha=0.3, markersize=2),
            whiskerprops=dict(color="#8888aa"),
            capprops=dict(color="#8888aa"),
        )
        colors3 = plt.cm.viridis(np.linspace(0.2, 0.85, len(_MONEYNESS_LABELS)))
        for patch, c in zip(bp3["boxes"], colors3):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)
        ax3.axhline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
        ax3.set_xlabel("Moneyness Bucket")
        ax3.set_ylabel("Pricing Error")
        ax3.set_title("Error vs Moneyness")
        ax3.tick_params(axis="x", rotation=20)

        # ── Panel 4: error vs maturity bucket (box) ───────────────────────
        ax4 = axes[1, 1]
        groups_mat = [
            plot_df.loc[plot_df["_mat_bucket"] == lbl, "pricing_error"].dropna().values
            for lbl in _MATURITY_LABELS
        ]
        bp4 = ax4.boxplot(
            groups_mat,
            labels=_MATURITY_LABELS,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=1.5),
            flierprops=dict(marker=".", color=accent, alpha=0.3, markersize=2),
            whiskerprops=dict(color="#8888aa"),
            capprops=dict(color="#8888aa"),
        )
        colors4 = plt.cm.magma(np.linspace(0.2, 0.85, len(_MATURITY_LABELS)))
        for patch, c in zip(bp4["boxes"], colors4):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)
        ax4.axhline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
        ax4.set_xlabel("Maturity Bucket")
        ax4.set_ylabel("Pricing Error")
        ax4.set_title("Error vs Maturity")
        ax4.tick_params(axis="x", rotation=20)

        plt.tight_layout()
        safe_vol = vol_column.replace("/", "_")
        out_path = os.path.join(self.plots_dir, f"diagnostics_{safe_vol}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------

    def save_results(self, df: pd.DataFrame, path: Optional[str] = None) -> str:
        """Save key columns to a CSV file."""
        os.makedirs(self.output_dir, exist_ok=True)
        if path is None:
            path = os.path.join(self.output_dir, "bs_benchmark_results.csv")
        keep = [c for c in [
            "timestamp", "option_type", "strike", "underlying_price",
            "time_to_maturity", "log_moneyness",
            "option_price", "bs_price", "pricing_error",
        ] if c in df.columns]
        df[keep].to_csv(path, index=False)
        print(f"  Results saved → {path}  ({len(df):,} rows)")
        return path

    # ------------------------------------------------------------------
    # Stage 1: Pricing
    # ------------------------------------------------------------------

    def run_pricing_mode(
        self,
        vol_column: str = "rolling_std_24h",
        chunksize: int = 100_000,
        sample_size: Optional[int] = None,
    ):
        """Stage 1: Compute BS prices and implied vols and save to options_with_bs.csv."""
        output_path = os.path.join(
            os.path.dirname(self.dataset_path), "options_with_bs.csv"
        )
        print(f"\n[BS Pricing] Starting Stage 1")
        print(f"[BS Pricing] Input  = {self.dataset_path}")
        print(f"[BS Pricing] Output = {output_path}")

        if os.path.exists(output_path):
            os.remove(output_path)

        loaded_total = 0
        first_chunk = True

        reader = pd.read_csv(self.dataset_path, chunksize=chunksize, low_memory=False)
        
        for chunk in reader:
            if sample_size and (loaded_total >= sample_size):
                break

            # 1. Clean and Prepare
            if "risk_free_rate" not in chunk.columns:
                chunk["risk_free_rate"] = 0.0
            chunk["risk_free_rate"] = chunk["risk_free_rate"].fillna(0.0)

            # Drop missing essential values
            mask = (
                chunk["option_price"].notna()        & (chunk["option_price"]      > 0) &
                chunk["underlying_price"].notna()    & (chunk["underlying_price"]  > 0) &
                chunk["strike"].notna()              & (chunk["strike"]            > 0) &
                chunk["time_to_maturity"].notna()    & (chunk["time_to_maturity"]  > 0) &
                chunk[vol_column].notna()            & (chunk[vol_column]          > 0)
            )
            chunk = chunk.loc[mask].copy()
            
            if chunk.empty:
                continue

            # 2. Compute BS Prices (BTC)
            bs_price_usd = black_scholes_price(
                S=chunk["underlying_price"],
                K=chunk["strike"],
                T=chunk["time_to_maturity"],
                r=chunk["risk_free_rate"],
                sigma=chunk[vol_column],
                option_type=chunk["option_type"],
            )
            chunk["bs_price"] = bs_price_usd / chunk["underlying_price"]
            chunk["pricing_error"] = chunk["bs_price"] - chunk["option_price"]

            # 3. Compute Intrinsic and Time Value (BTC)
            is_call = chunk["option_type"].str.lower() == "call"
            intrinsic_usd = np.where(
                is_call,
                np.maximum(chunk["underlying_price"] - chunk["strike"], 0),
                np.maximum(chunk["strike"] - chunk["underlying_price"], 0)
            )
            chunk["intrinsic_value"] = intrinsic_usd / chunk["underlying_price"]
            chunk["time_value"] = chunk["option_price"] - chunk["intrinsic_value"]

            # 4. Compute Implied Volatility
            market_price_usd = chunk["option_price"] * chunk["underlying_price"]
            chunk["implied_vol"] = compute_implied_volatility(
                market_price=market_price_usd,
                S=chunk["underlying_price"],
                K=chunk["strike"],
                T=chunk["time_to_maturity"],
                r=chunk["risk_free_rate"],
                option_type=chunk["option_type"],
            )

            # 5. Incremental Save
            chunk.to_csv(output_path, mode='a', index=False, header=first_chunk)
            
            loaded_total += len(chunk)
            first_chunk = False
            print(f"  Processed {loaded_total:,} rows …", end="\r")
            
            if sample_size and (loaded_total >= sample_size):
                break

        print(f"\n[BS Pricing] Completed. Saved {loaded_total:,} rows to '{output_path}'.")

    # ------------------------------------------------------------------
    # Stage 2: Analysis
    # ------------------------------------------------------------------

    def _compute_smile_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate mean/median/std IV by log-moneyness bins."""
        bins = [-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0]
        labels = [
            "[-1.0, -0.5]", "[-0.5, -0.25]", "[-0.25, -0.1]",
            "[-0.1, 0.1]", "[0.1, 0.25]", "[0.25, 0.5]", "[0.5, 1.0]"
        ]
        
        df = df.dropna(subset=["implied_vol", "log_moneyness"]).copy()
        df["mon_bin"] = pd.cut(df["log_moneyness"], bins=bins, labels=labels)
        
        stats = df.groupby(["option_type", "mon_bin"], observed=True)["implied_vol"].agg([
            ("count", "count"),
            ("mean", "mean"),
            ("median", "median"),
            ("std", "std")
        ]).reset_index()
        
        return stats

    def run_analysis_mode(
        self,
        sample_size: Optional[int] = 5_000_000,
    ):
        """Stage 2: Statistics and Visualization from precomputed dataset."""
        input_path = os.path.join(
            os.path.dirname(self.dataset_path), "options_with_bs.csv"
        )
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing '{input_path}'. Run --mode bs_pricing first.")

        print(f"\n[BS Analysis] Starting Stage 2")
        print(f"[BS Analysis] Loading {input_path} …")
        
        # Load a representative sample for analysis
        df = pd.read_csv(input_path, nrows=sample_size, low_memory=False)
        print(f"  Loaded {len(df):,} rows.")

        # 1. Global Stats
        actual = df["option_price"]
        pred = df["bs_price"]
        err = df["pricing_error"]
        
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err**2))
        mape = np.mean(np.abs(err / actual)) * 100
        bias = np.mean(err)
        
        # R2
        ss_res = np.sum(err**2)
        ss_tot = np.sum((actual - np.mean(actual))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        print("\n" + "="*60)
        print(f"  Overall Performance Metrics (N={len(df):,})")
        print("="*60)
        print(f"  MAE  :  {mae:10.6f}")
        print(f"  RMSE :  {rmse:10.6f}")
        print(f"  MAPE :  {mape:9.2f}%")
        print(f"  Bias :  {bias:10.6f}")
        print(f"  R²   :  {r2:10.4f}")

        # 2. Grouped Stats (Reuse existing logic)
        metrics = self.compute_error_metrics(df)
        print("\n--- By Moneyness Bucket ---")
        print(metrics["by_moneyness"][["count", "MAE", "RMSE", "Bias"]])
        print("\n--- By Maturity Bucket ---")
        print(metrics["by_maturity"][["count", "MAE", "RMSE", "Bias"]])

        # 3. Aggregated Smile
        smile_stats = self._compute_smile_stats(df)
        print("\n--- Aggregated Volatility Smile ---")
        print(smile_stats)

        # 4. Plots
        print("\n[BS Analysis] Generating plots …")
        diag_path = self.plot_diagnostics(df, "precomputed")
        
        # smile combined
        plt.figure(figsize=(10, 6))
        for otype in ["call", "put"]:
            subset = smile_stats[smile_stats["option_type"].str.lower() == otype]
            plt.plot(subset["mon_bin"], subset["mean"], marker="o", label=otype.title())
        
        plt.title("Aggregated Volatility Smile (Mean IV per Bin)")
        plt.xlabel("Log Moneyness Bin")
        plt.ylabel("Implied Volatility")
        plt.legend()
        plt.grid(True, alpha=0.3)
        smile_path = os.path.join(self.output_dir, "plots", "volatility_smile.png")
        os.makedirs(os.path.dirname(smile_path), exist_ok=True)
        plt.savefig(smile_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Diagnostics → {diag_path}")
        print(f"  Smile Plot  → {smile_path}")
        print(f"\n[BS Analysis] Complete.")

    def run(
        self,
        vol_column: str = "rolling_std_24h",
        save_csv: bool = True,
        skip_smile_plot: bool = False,
        sample_size: Optional[int] = 5_000_000,
    ) -> dict:
        """Orchestrate the full benchmark pipeline.

        Parameters
        ----------
        vol_column      : volatility estimator column to use
        save_csv        : whether to save output/bs_benchmark_results.csv
        skip_smile_plot : set True to skip the slow implied-vol inversion step
        sample_size     : max rows to load from the dataset

        Returns
        -------
        dict with 'metrics' and 'paths' (saved file paths)
        """
        if vol_column not in SUPPORTED_VOL_COLUMNS:
            print(
                f"[Warning] '{vol_column}' is not in the standard list "
                f"{SUPPORTED_VOL_COLUMNS}. Proceeding anyway."
            )

        print(f"\n[BS Benchmark] Starting  |  vol_column = '{vol_column}'")
        print(f"[BS Benchmark] Dataset   = '{self.dataset_path}'")

        # 1. Load -------------------------------------------------------
        print("\n[1/5] Loading dataset …")
        df = self.load_dataset(vol_column=vol_column, max_rows=sample_size)
        self.df = df
        self._vol_column = vol_column

        # 2. Compute BS prices -----------------------------------------
        print("\n[2/5] Computing Black-Scholes prices …")
        df = self.compute_bs_prices(df, vol_column)

        # 3. Error metrics ---------------------------------------------
        print("\n[3/5] Computing error metrics …")
        metrics = self.compute_error_metrics(df)
        self.print_summary(metrics, vol_column)

        saved_paths = []

        # 4. Diagnostic plots ------------------------------------------
        print("\n[4/5] Generating diagnostic plots …")
        diag_path = self.plot_diagnostics(df, vol_column)
        saved_paths.append(diag_path)

        # 5. Volatility smile (optional) --------------------------------
        if not skip_smile_plot:
            print("\n[5/5] Generating volatility smile plot …")
            smile_path = self.plot_volatility_smile(df)
            saved_paths.append(smile_path)
        else:
            print("\n[5/5] Skipping volatility smile plot (skip_smile_plot=True).")

        # Save CSV -------------------------------------------------------
        if save_csv:
            csv_path = self.save_results(df)
            saved_paths.append(csv_path)

        print("\n[BS Benchmark] Complete.")
        return {"metrics": metrics, "paths": saved_paths}
