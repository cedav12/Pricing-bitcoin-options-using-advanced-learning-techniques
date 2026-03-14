import yfinance as yf
import pandas as pd

def download_and_prepare_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads risk-free rate (^IRX) and VIX index (^VIX), aligns and forward-fills 
    to hourly resolution.
    
    Args:
        start_date: Start date string (e.g., '2021-01-01')
        end_date: End date string (e.g., '2024-01-01')
        
    Returns:
        pd.DataFrame containing 'timestamp', 'risk_free_rate', 'vix_index'
    """
    print(f"Downloading macro data from {start_date} to {end_date}...")
    
    # 1. Download risk-free rate (^IRX) which is 13-week Treasury bill rate
    irx = yf.download("^IRX", start=start_date, end=end_date)
    # yfinance sometimes returns a MultiIndex DataFrame for single ticker. 
    # Use 'Close' safely.
    if isinstance(irx.columns, pd.MultiIndex):
        irx = irx['Close']
    else:
        irx = irx[['Close']]
    
    irx = irx.rename(columns={'Close': 'risk_free_rate', '^IRX': 'risk_free_rate'})
    
    # Convert percentage to decimal form
    irx['risk_free_rate'] = irx['risk_free_rate'] / 100.0

    # 2. Download VIX index (^VIX)
    vix = yf.download("^VIX", start=start_date, end=end_date)
    if isinstance(vix.columns, pd.MultiIndex):
        vix = vix['Close']
    else:
        vix = vix[['Close']]
        
    vix = vix.rename(columns={'Close': 'vix_index', '^VIX': 'vix_index'})

    # 3. Combine both into a single daily dataframe
    # Dropna to keep only exact market days where both are tracked
    macro_daily = irx.join(vix, how='outer')
    # Because macro_daily uses timezone-aware timestamps sometimes via yfinance, 
    # localize or drop timezone for consistent merging.
    if macro_daily.index.tz is not None:
        macro_daily.index = macro_daily.index.tz_localize(None)
    
    # Forward fill any daily gaps before resampling
    macro_daily = macro_daily.ffill()

    # 4. Resample to hourly and forward-fill
    # To resample to hourly, we create a full hourly date range
    hourly_range = pd.date_range(start=macro_daily.index.min(), end=macro_daily.index.max(), freq='h')
    
    # Reindex and forward fill over the hourly grid
    macro_hourly = macro_daily.reindex(hourly_range).ffill()
    
    # Forward-fill and backward-fill boundary edges (e.g. if starting on a weekend)
    macro_hourly = macro_hourly.ffill().bfill()
    
    # Reset index to make timestamp a column
    macro_hourly = macro_hourly.reset_index()
    macro_hourly = macro_hourly.rename(columns={'index': 'timestamp'})
    
    # Convert to numeric (milliseconds) to match existing pipeline
    macro_hourly['timestamp'] = macro_hourly['timestamp'].astype('int64') // 10**6
    
    return macro_hourly
