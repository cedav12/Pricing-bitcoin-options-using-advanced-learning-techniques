import pandas as pd
import glob
import os
from .utils import parse_option_instrument

class DataLoader:
    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = raw_data_dir

    def load_btc_data(self) -> pd.DataFrame:
        """
        Loads historical BTC perpetual futures data.
        Assumes it's a single CSV file ending in something like BTC_PERP.csv.
        """
        pattern = os.path.join(self.raw_data_dir, "*BTC-PERP*.csv")
        files = glob.glob(pattern)
        if not files:
            # For testing without real files, return empty df
            print("[Warning] No BTC-PERP data found.")
            return pd.DataFrame()
            
        file_path = files[0]
        df = pd.read_csv(file_path)
        # Assuming format: timestamp,open,high,low,close,volume,trade_count
        return df

    def stream_options_data(self):
        """
        Yields filtered options dataframes one by one.
        Assumes options files don't have BTC_PERP in the name but end in .csv
        Also extracts instrument details from the filename if not present in the CSV.
        """
        discovered_count = 0
        skipped_count = 0
        skipped_count_structure = {
            "skipped_count_empty": 0,
            "skipped_count_zero_volume": 0,
            "skipped_count_zero_close": 0
        }


        processed_count = 0
        
        for file_path in glob.glob(os.path.join(self.raw_data_dir, "*.csv")):
            if "BTC_PERP" in file_path or "BTC-PERPETUAL" in file_path:
                continue
                
            discovered_count += 1
            filename = os.path.basename(file_path)
            # Remove .csv to get instrument name, e.g. BTC-25MAR22-40000-C
            instrument_name = filename.replace(".csv", "")
            
            try:
                # Need to handle potentially empty files cleanly
                if os.path.getsize(file_path) == 0:
                    skipped_count += 1
                    skipped_count_structure["skipped_count_empty"] += 1
                    continue
                    
                df = pd.read_csv(file_path)
                
                if df.empty:
                    skipped_count += 1
                    skipped_count_structure["skipped_count_empty"] += 1
                    continue
                    
                # Filtering logic
                # Skip if volume is always zero
                if 'volume' in df.columns and (df['volume'] == 0).all():
                    skipped_count += 1
                    skipped_count_structure["skipped_count_zero_volume"] += 1
                    continue
                    
                # Skip if trade_count is always zero
                if 'trade_count' in df.columns and (df['trade_count'] == 0).all():
                    skipped_count += 1
                    continue
                    
                # Skip if option_price (close) is always zero
                if 'close' in df.columns and (df['close'] == 0).any():
                    skipped_count += 1
                    skipped_count_structure["skipped_count_zero_close"] += 1
                    continue
                
                # If instrument details aren't in columns, parse from filename
                if 'strike' not in df.columns:
                    parsed = parse_option_instrument(instrument_name)
                    if parsed:
                        df['instrument'] = instrument_name
                        df['strike'] = parsed['strike']
                        df['expiry'] = parsed['expiry_datetime']
                        df['option_type'] = parsed['option_type']
                
                processed_count += 1
                yield df
                
            except Exception as e:
                print(f"[Warning] Failed to read {file_path}: {e}")
                skipped_count += 1
            
        print(f"Data Loader Stats:")
        print(f" - Discovered elements: {discovered_count}")
        print(f" - Skipped elements: {skipped_count}")
        for k, v in skipped_count_structure.items():
            print(f"    - {k}: {v}")
        print(f" - Processed elements: {processed_count}")
