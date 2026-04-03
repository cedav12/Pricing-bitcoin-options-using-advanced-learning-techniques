"""
Metrics computation for Model Evaluation
Generates MAE, RMSE, Bias, R2, MARE, MALE.
Filters data conditionally based on `error_type` and `eval_mode` to maintain statistical integrity.
"""
import numpy as np
import pandas as pd


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the base BTC-denominated and diagnostic error columns to the dataframe."""
    df = df.copy()
    y_true = df["market_price"].values
    y_pred = df["model_price"].values

    # Absolute Error
    df["error_abs"] = y_pred - y_true

    # Relative Error
    # Safely handle division by zero or near-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = (y_pred - y_true) / y_true
        df["error_rel"] = np.where(y_true > 1e-8, rel, np.nan)

    # Log Error
    with np.errstate(divide='ignore', invalid='ignore'):
        log_e = np.log(y_pred / y_true)
        df["error_log"] = np.where((y_true > 1e-8) & (y_pred > 1e-8), log_e, np.nan)

    # Normalized Error (scale by underlying price)
    with np.errstate(divide='ignore', invalid='ignore'):
        df["error_norm"] = np.where(
            df["underlying_price"] > 1e-8,
            (y_pred - y_true) / df["underlying_price"],
            np.nan
        )

    if "market_price_usd" in df.columns and "model_price_usd" in df.columns:
        df["error_abs_usd"] = df["model_price_usd"] - df["market_price_usd"]

    return df


def apply_diagnostic_filters(
        df: pd.DataFrame,
        min_price: float = 0.001,
        min_time_value: float = 0.001
) -> pd.DataFrame:
    """
    Applies filters for diagnostic metrics where tiny prices / tiny time value
    would make scale-free error measures unstable.
    """
    df_filtered = df.copy()

    if min_time_value > 0 and "time_value" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["time_value"] > min_time_value]

    df_filtered = df_filtered[df_filtered["market_price"] > min_price]
    return df_filtered


def compute_price_metrics(
        df: pd.DataFrame,
        y_true_col: str = "market_price",
        y_pred_col: str = "model_price"
) -> pd.Series:
    """
    Computes core absolute pricing metrics for arbitrary target/prediction columns.
    """
    df_eval = df[[y_true_col, y_pred_col]].dropna()

    if df_eval.empty:
        return pd.Series({
            "count": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
            "R2": np.nan
        })

    y_true = df_eval[y_true_col].values
    y_pred = df_eval[y_pred_col].values
    err = y_pred - y_true

    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    bias = np.mean(err)

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return pd.Series({
        "count": len(df_eval),
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "R2": r2
    })


def compute_diagnostic_metrics(
        df: pd.DataFrame,
        eval_mode: str = "stable",
        min_price: float = 0.001,
        min_time_value: float = -1.0
) -> pd.Series:
    """
    Computes scale-free diagnostic metrics (relative/log/normalized).
    """
    df_eval = df.copy()

    if eval_mode != "full":
        df_eval = apply_diagnostic_filters(df_eval, min_price, min_time_value)

    if df_eval.empty:
        return pd.Series({
            "count": 0,
            "MARE": np.nan,
            "MALE": np.nan,
            "MANE": np.nan
        })

    df_eval = add_error_columns(df_eval)

    err_rel = df_eval["error_rel"].dropna().values
    err_log = df_eval["error_log"].dropna().values
    err_norm = df_eval["error_norm"].dropna().values

    mare = np.mean(np.abs(err_rel)) if len(err_rel) > 0 else np.nan
    male = np.mean(np.abs(err_log)) if len(err_log) > 0 else np.nan
    mane = np.mean(np.abs(err_norm)) if len(err_norm) > 0 else np.nan

    return pd.Series({
        "count": len(df_eval),
        "MARE": mare,
        "MALE": male,
        "MANE": mane
    })