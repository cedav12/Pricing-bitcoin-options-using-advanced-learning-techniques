"""
Metrics computation for Model Evaluation
Generates MAE, RMSE, Bias, R2, MARE, MALE.
Filters data conditionally based on `error_type` and `eval_mode` to maintain statistical integrity.
"""
import numpy as np
import pandas as pd

def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the base error columns to the dataframe."""
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
        # Only valid strictly positive prices
        df["error_log"] = np.where((y_true > 1e-8) & (y_pred > 1e-8), log_e, np.nan)
        
    return df

def apply_evaluation_filters(
    df: pd.DataFrame, 
    error_type: str = "relative", 
    eval_mode: str = "stable", 
    min_price: float = 0.001,
    min_time_value: float = 0.001
) -> pd.DataFrame:
    """
    Applies filters ONLY where appropriate for the requested evaluation mode and target error type.
    """
    if eval_mode == "full":
        return df
        
    df_filtered = df.copy()
    
    # Time value filter is generally a stability filter across all metrics if requested
    if min_time_value > 0 and "time_value" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["time_value"] > min_time_value]
        
    # Relative error specifically explodes near 0, filtering is mandatory for 'stable'
    if error_type == "relative":
        df_filtered = df_filtered[df_filtered["market_price"] > min_price]
        
    return df_filtered

def compute_metrics(
    df: pd.DataFrame, 
    error_type: str = "relative", 
    eval_mode: str = "stable",
    min_price: float = 0.001,
    min_time_value: float = -1.0
) -> pd.Series:
    """
    Computes all core metrics on the provided dataframe.
    """
    # 1. Conditionally filter
    df_eval = apply_evaluation_filters(df, error_type, eval_mode, min_price, min_time_value)
    
    if df_eval.empty:
        return pd.Series({
            "count": 0, "MAE": np.nan, "RMSE": np.nan, 
            "Bias": np.nan, "R2": np.nan, "MARE": np.nan, "MALE": np.nan
        })

    # 2. Add error cols
    df_eval = add_error_columns(df_eval)
    
    # 3. Compute Metrics
    count = len(df_eval)
    y_true = df_eval["market_price"].values
    y_pred = df_eval["model_price"].values
    
    err_abs = df_eval["error_abs"].values
    err_rel = df_eval["error_rel"].dropna().values
    err_log = df_eval["error_log"].dropna().values
    
    mae = np.mean(np.abs(err_abs))
    rmse = np.sqrt(np.mean(err_abs ** 2))
    bias = np.mean(err_abs)
    
    ss_res = np.sum(err_abs ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    mare = np.mean(np.abs(err_rel)) if len(err_rel) > 0 else np.nan
    male = np.mean(np.abs(err_log)) if len(err_log) > 0 else np.nan
    
    return pd.Series({
        "count": count,
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "R2": r2,
        "MARE": mare,
        "MALE": male
    })
