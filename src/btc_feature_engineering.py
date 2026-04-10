import pandas as pd
import numpy as np
import warnings
from arch import arch_model

def compute_parkinson_vol(high: pd.Series, low: pd.Series) -> float:
    """
    Computes Parkinson volatility estimator.
    Formula: sqrt ( (1 / (4 * N * ln(2))) * sum(ln(High/Low)^2) )
    """
    N = len(high)
    if N == 0:
        return 0.0
    hl_log_sq = (np.log(high / low))**2
    return np.sqrt((1.0 / (4.0 * N * np.log(2.0))) * hl_log_sq.sum())

def compute_garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    """
    Computes Garman-Klass volatility estimator.
    Formula: sqrt ( (1/N) * sum( 0.5 * ln(High/Low)^2 - (2*ln(2)-1) * ln(Close/Open)^2 ) )
    """
    N = len(close)
    if N == 0:
        return 0.0
    hl_log_sq = (np.log(high / low))**2
    co_log_sq = (np.log(close / open_))**2
    gk_var = 0.5 * hl_log_sq - (2 * np.log(2) - 1) * co_log_sq
    return np.sqrt(max(0.0, gk_var.sum() / N))  # max ensures no negative values due to floating point inaccuracies

def preprocess_btc_data(btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Loads BTC_PERP 5-minute data (already loaded inside btc_df).
    2. Aggregates it into hourly intervals.
    3. Computes volatility-related statistics from the 5-minute returns inside each hour.
    
    Expected input:
    timestamp, open, high, low, close, volume, trade_count
    """
    if btc_df.empty:
        return pd.DataFrame()

    df = btc_df.copy()
    
    # Cleanly convert timestamp into datetime unconditionally handling ms / ISO strings
    if df['timestamp'].dtype == 'object' or df['timestamp'].dtype == 'string':
        df['datetime'] = pd.to_datetime(df['timestamp'])
        # If timezone aware, convert to timezone naive for merging
        if df['datetime'].dt.tz is not None:
            df['datetime'] = df['datetime'].dt.tz_localize(None)
    elif len(df) > 0 and df['timestamp'].iloc[0] > 1e11:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    df = df.sort_values('datetime')
    
    # Compute 5-minute log returns for realized volatility
    # r_t = ln(S_t / S_{t-1})
    df['log_return_5m'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_5m'] = df['log_return_5m'].fillna(0.0)
    
    # Set datetime as index for resampling
    df.set_index('datetime', inplace=True)
    
    # Define aggregation dict for generating true hourly OHLC bars
    agg_funcs = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    
    # Aggregate OHLC to hourly map
    hourly_ohlc = df.resample('1h').agg(agg_funcs)
    
    # Extract realized volatility components from the 5-minute bars
    def hourly_vol_features(group: pd.DataFrame) -> pd.Series:
        r = group['log_return_5m']
        
        realized_var = (r**2).sum()
        realized_vol = np.sqrt(realized_var)
        
        pos_r = r[r > 0]
        neg_r = r[r < 0]
        
        pos_semivar = (pos_r**2).sum()
        neg_semivar = (neg_r**2).sum()
        
        return pd.Series({
            'realized_variance': realized_var,
            'realized_volatility': realized_vol,
            'positive_semivariance': pos_semivar,
            'negative_semivariance': neg_semivar
        })
    
    hourly_realized = df.resample('1h').apply(hourly_vol_features)
    
    # Merge hourly OHLC and realized metrics
    hourly_df = hourly_ohlc.join(hourly_realized)
    
    # Drop rows with NaN (hours without data)
    hourly_df = hourly_df.dropna()
    
    # Compute true hourly variables based on the hourly OHLC structure
    hourly_df['btc_price_close'] = hourly_df['close']
    hourly_df['btc_return'] = np.log(hourly_df['close'] / hourly_df['close'].shift(1)).fillna(0.0)
    
    # Apply Parkinson and Garman-Klass using the hourly OHLC
    # The formulas expect Series so we can apply directly across vectors.
    N_hours = len(hourly_df)
    
    # Parkinson Volatility:
    hl_log_sq = (np.log(hourly_df['high'] / hourly_df['low']))**2
    # In earlier iteration, it averaged over the entire grouping length N but we want an hourly estimator over 1 bar right? No, the standard formula is an average estimator. Per user request: "Compute this from BTC hourly high and low prices" 
    # Usually you compute it per bar: (1/(4*ln2)) * ln(H_i / L_i)^2. If you want a rolling Parkinson you average it over N. 
    # Since it's to be aligned with hourly timestamps, we provide the instantaneous vol estimate per hour.
    hourly_df['parkinson_volatility'] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * hl_log_sq)
    
    # Garman-Klass Volatility per hour bar:
    co_log_sq = (np.log(hourly_df['close'] / hourly_df['open']))**2
    gk_var = 0.5 * hl_log_sq - (2 * np.log(2) - 1) * co_log_sq
    hourly_df['garman_klass_volatility'] = np.sqrt(np.maximum(0.0, gk_var))
    
    # Compute rolling historical standardized volatility
    # Annualized multiplier assuming hourly continuous data: sqrt(365.25 * 24) = ~93.6
    annualization_factor = np.sqrt(365.25 * 24)
    
    hourly_df['rolling_std_24h'] = hourly_df['btc_return'].rolling(window=24).std() * annualization_factor
    hourly_df['rolling_std_7d'] = hourly_df['btc_return'].rolling(window=168).std() * annualization_factor
    
    # Forward fill NaNs created by rolling windows.
    hourly_df['rolling_std_24h'] = hourly_df['rolling_std_24h'].bfill()
    hourly_df['rolling_std_7d'] = hourly_df['rolling_std_7d'].bfill()
    
    # Fit GARCH(1,1) and extract conditional volatility
    # Use btc_return (which is already log return)
    # arch dataset prefers scaled returns (x100) for optimization convergence, so we multiply by 100 before fitting
    # then divide by 100 after.
    returns_for_garch = hourly_df['btc_return'] * 100.0
    
    hourly_df['garch_volatility'] = np.nan
    
    if len(returns_for_garch) >= 168: # Only fit if we have enough data (e.g. at least 1 week)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                am = arch_model(returns_for_garch, p=1, q=1, vol='Garch', dist='Normal')
                res = am.fit(update_freq=0, disp='off')
                # res.conditional_volatility is the model's standard deviation given past info. 
                # We extract it and unscale
                hourly_df['garch_volatility'] = (res.conditional_volatility / 100.0) * annualization_factor
        except Exception as e:
            print(f"[Warning] GARCH model fit failed: {e}")
            hourly_df['garch_volatility'] = 0.0
    else:
        hourly_df['garch_volatility'] = 0.0

    # Reset index so timestamp/datetime is a column
    hourly_df = hourly_df.reset_index()
    
    # Convert datetime back to uniform ms timestamp format for proper merging
    # datetime is in nanoseconds, so divide by 10**6
    hourly_df['timestamp'] = (hourly_df['datetime'].astype('int64') // 10**6)
        
    expected_cols = [
        'timestamp', 'btc_price_close', 'btc_return', 
        'realized_variance', 'realized_volatility',
        'positive_semivariance', 'negative_semivariance',
        'parkinson_volatility', 'garman_klass_volatility',
        'rolling_std_24h', 'rolling_std_7d',
        'garch_volatility'
    ]
    
    return hourly_df[expected_cols]
