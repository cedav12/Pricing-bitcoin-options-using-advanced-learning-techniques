import os
import pandas as pd
from src.data_loader import DataLoader
from src.feature_engineering import compute_time_to_maturity, compute_log_moneyness
from src.btc_feature_engineering import preprocess_btc_data
from src.macro_data_loader import download_and_prepare_macro_data

class DatasetBuilder:
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.loader = DataLoader(raw_data_dir)
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def build_dataset(self) -> str:
        """
        Builds the complete dataset by aligning BTC and option data,
        computing initial features, and saving to CSV.
        Returns the path to the saved dataset.
        """
        print("Loading BTC perpetual data...")
        btc_df = self.loader.load_btc_data()
        
        if btc_df.empty:
            print("[Error] Missing BTC perpetual data.")
            return ""
            
        print("Processing BTC perpetual data (computing 1h volatility features)...")
        hourly_btc_features = preprocess_btc_data(btc_df)
        
        # Ensure timestamp is the same type
        hourly_btc_features['timestamp'] = pd.to_numeric(hourly_btc_features['timestamp'])
        hourly_btc_features = hourly_btc_features.sort_values('timestamp')
        
        print("Downloading and preparing macro data...")
        
        min_ts = pd.to_datetime(hourly_btc_features['timestamp'].min(), unit='ms')
        max_ts = pd.to_datetime(hourly_btc_features['timestamp'].max(), unit='ms')
        # Add a 5-day buffer to start and end dates to ensure we cover the entire period
        start_date = (min_ts - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        end_date = (max_ts + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        
        try:
            macro_hourly = download_and_prepare_macro_data(start_date, end_date)
            macro_hourly = macro_hourly.sort_values('timestamp')
        except Exception as e:
            print(f"[Warning] Failed to download macro data: {e}")
            macro_hourly = pd.DataFrame(columns=['timestamp', 'risk_free_rate', 'vix_index'])
        
        print("Streaming Options data and building dataset...")
        
        output_path = os.path.join(self.processed_data_dir, "options_dataset.csv")
        # Remove old output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
            
        # Helper function to parse timestamps safely
        def normalize_timestamp(series: pd.Series) -> pd.Series:
            if series.dtype == 'object' or series.dtype == 'string':
                # Attempt to parse strings like "2025-08-27T05:00:00Z"
                try:
                    # Convert to datetime, then to ms timestamp
                    dt = pd.to_datetime(series)
                    return dt.astype('int64') // 10**6
                except Exception:
                    pass
            # Fallback to numeric
            return pd.to_numeric(series)
        
        expected_cols = [
            'timestamp', 'option_price', 'underlying_price', 
            'option_type', 'strike', 'expiry', 
            'time_to_maturity', 'log_moneyness', 
            'volume', 'trade_count',
            'btc_return', 'realized_variance', 'realized_volatility',
            'positive_semivariance', 'negative_semivariance',
            'parkinson_volatility', 'garman_klass_volatility',
            'rolling_std_24h', 'rolling_std_7d', 'garch_volatility',
            'risk_free_rate', 'vix_index'
        ]
        
        is_first_chunk = True
        total_rows = 0
        
        for options_df in self.loader.stream_options_data():
            
            options_df['timestamp'] = normalize_timestamp(options_df['timestamp'])
            options_df = options_df.sort_values('timestamp')
            
            # Use merge_asof to find the latest BTC data at or before the option timestamp
            merged_df = pd.merge_asof(
                options_df, 
                hourly_btc_features, 
                on='timestamp', 
                direction='backward'
            )
            
            # Use merge_asof to merge macro data
            if not macro_hourly.empty:
                merged_df = pd.merge_asof(
                    merged_df,
                    macro_hourly,
                    on='timestamp',
                    direction='backward'
                )
            else:
                merged_df['risk_free_rate'] = pd.NA
                merged_df['vix_index'] = pd.NA
            
            # Rename btc_price_close to underlying_price for compatibility
            if 'btc_price_close' in merged_df.columns:
                merged_df = merged_df.rename(columns={'btc_price_close': 'underlying_price'})
            
            # Computing basic option features
            merged_df['time_to_maturity'] = compute_time_to_maturity(
                merged_df['timestamp'], 
                merged_df['expiry']
            )
            
            merged_df['log_moneyness'] = compute_log_moneyness(
                merged_df['underlying_price'], 
                merged_df['strike']
            )
            
            if 'close' in merged_df.columns:
                merged_df = merged_df.rename(columns={'close': 'option_price'})
                
            # ensure columns exist
            final_cols = [c for c in expected_cols if c in merged_df.columns]
            final_df = merged_df[final_cols]
            
            # Append to file
            final_df.to_csv(output_path, mode='a', header=is_first_chunk, index=False)
            is_first_chunk = False
            total_rows += len(final_df)
            
        print(f"Dataset generated with {total_rows} observations.")
        
        return output_path
