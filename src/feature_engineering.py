import numpy as np
import pandas as pd

def compute_time_to_maturity(timestamps: pd.Series, expiries: pd.Series) -> pd.Series:
    """
    Computes time to maturity in years.
    Both timestamps and expiries should be datetime-like objects or Unix timestamps (ms or s).
    Assuming timestamps are in milliseconds because crypto data often is.
    If they are datetime objects, we can use total_seconds().
    """
    # Convert to datetime if not already
    is_unix_ms = False
    if len(timestamps) > 0:
        first_val = timestamps.iloc[0]
        # Check if it's a large number (typical of ms timestamps)
        if isinstance(first_val, (int, float, np.integer, np.floating)) and first_val > 1e11:
            is_unix_ms = True
            
    timestamps_dt = pd.to_datetime(timestamps, unit='ms' if is_unix_ms else None)
    expiries_dt = pd.to_datetime(expiries)
    
    # Calculate difference in seconds and convert to years (365 days)
    # Using 365.25 for leap years approximation, or simply 365.
    delta_seconds = (expiries_dt - timestamps_dt).dt.total_seconds()
    
    ttm_years = delta_seconds / (365.0 * 24 * 3600)
    
    # Ensure TTM is non-negative (can happen precisely at expiration or slightly after)
    return np.maximum(ttm_years, 0.0)

def compute_log_moneyness(underlying_price: pd.Series, strike: pd.Series) -> pd.Series:
    """
    Computes log moneyness: ln(S / K)
    """
    return np.log(underlying_price / strike)

def append_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends required features to the combined dataframe.
    Requires: 'timestamp', 'expiry', 'underlying_price', 'strike'
    """
    df = df.copy()
    
    if 'timestamp' in df.columns and 'expiry' in df.columns:
        df['time_to_maturity'] = compute_time_to_maturity(df['timestamp'], df['expiry'])
        
    if 'underlying_price' in df.columns and 'strike' in df.columns:
        df['log_moneyness'] = compute_log_moneyness(df['underlying_price'], df['strike'])
        
    return df
