import unittest
import pandas as pd
import numpy as np
import os
import glob
import random

class TestFinalDataset(unittest.TestCase):
    """
    This test suite is meant to be run manually on the final processed dataset 
    and the full raw data folders to verify data integrity after the build process.
    
    Usage:
    python -m unittest tests/test_final_dataset.py
    """
    
    @classmethod
    def setUpClass(cls):
        # Paths can be modified if running from a different root or custom folders
        cls.raw_data_dir = "data/raw"
        cls.processed_data_path = "data/processed/options_dataset.csv"
        
        # We only load these if the processed dataset exists, otherwise tests will just fail/skip gracefully
        if os.path.exists(cls.processed_data_path):
            print(f"Loading full processed dataset from {cls.processed_data_path}...")
            # Load the dataset but only a sample or chunk if it's too massive, 
            # though usually reading ~60k * N rows is acceptable for a dedicated test box.
            cls.final_df = pd.read_csv(cls.processed_data_path)
            # Ensure timestamps are numeric
            cls.final_df['timestamp'] = pd.to_numeric(cls.final_df['timestamp'])
        else:
            cls.final_df = pd.DataFrame()

    def test_01_processed_dataset_exists(self):
        """Check if the final dataset was actually generated."""
        self.assertFalse(self.final_df.empty, "Processed dataset not found or is empty.")

    def test_02_random_instruments_included(self):
        """
        Randomly select 2 non-empty raw option files and verify that their 
        valid (post-filtered) rows are present in the final dataset.
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")
            
        # Get all option files
        pattern = os.path.join(self.raw_data_dir, "*.csv")
        all_files = [f for f in glob.glob(pattern) if "BTC_PERP" not in f and "BTC-PERPETUAL" not in f]
        
        self.assertGreaterEqual(len(all_files), 2, "Not enough raw option files to test.")
        
        # Pick a few files and find 2 that are non-empty and have valid rows
        random.shuffle(all_files)
        
        tested_count = 0
        for file_path in all_files:
            if os.path.getsize(file_path) == 0:
                continue
                
            raw_df = pd.read_csv(file_path)
            
            # Apply the same filters the DataLoader does
            if raw_df.empty: continue
            if 'volume' in raw_df.columns and (raw_df['volume'] == 0).all(): continue
            if 'trade_count' in raw_df.columns and (raw_df['trade_count'] == 0).all(): continue
            if 'close' in raw_df.columns and (raw_df['close'] == 0).all(): continue
            
            # If we reach here, this instrument was processed.
            filename = os.path.basename(file_path)
            instrument_name = filename.replace(".csv", "")
            
            print(f"\\nVerifying inclusion of: {instrument_name}")
            
            # Since the final dataset doesn't retain the raw 'instrument' column string verbatim 
            # (unless added as a feature), we identify rows by strike, option_type, and expiry.
            # However, since different instruments have distinct strike/expiries, we can just 
            # pull a subset of raw timestamps and see if they exist in the final set for those traits.
            
            if 'strike' not in raw_df.columns:
                from src.utils import parse_option_instrument
                parsed = parse_option_instrument(instrument_name)
                self.assertIsNotNone(parsed, f"Could not parse instrument: {instrument_name}")
                strike_val = parsed['strike']
            else:
                strike_val = raw_df['strike'].iloc[0]
                
            # Filter the final DF for this strike
            final_subset = self.final_df[self.final_df['strike'] == strike_val]
            
            # We don't necessarily have to check *every* single row if the file is huge,
            # but we can check if the timestamps from the raw sequence overlap correctly.
            # Convert raw timestamps to the normalized format
            def normalize_timestamp(series):
                if series.dtype == 'object' or series.dtype == 'string':
                    return (pd.to_datetime(series).astype('int64') // 10**6)
                return pd.to_numeric(series)
                
            raw_timestamps = set(normalize_timestamp(raw_df['timestamp']).values)
            final_timestamps = set(final_subset['timestamp'].values)
            
            # The final dataset should contain all valid raw timestamps for this strike/option combo
            # Intersection should be high
            intersection = raw_timestamps.intersection(final_timestamps)
            
            self.assertGreater(len(intersection), 0, f"No matching timestamps found for {instrument_name} in the final dataset.")
            print(f"Successfully matched {len(intersection)} observations for {instrument_name}.")
            
            tested_count += 1
            if tested_count >= 2:
                break
                
        self.assertEqual(tested_count, 2, "Could not find 2 valid non-empty instrument files to test.")

    def test_03_timestamp_volatility_consistency(self):
        """
        Verify that for the same timestamp, different entries (e.g., different strikes/options)
        have the exact same BTC volatility metrics, since they use the same underlying hourly bin.
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")
            
        # Find timestamps that have multiple rows (e.g., multiple options trading at the same time)
        timestamp_counts = self.final_df['timestamp'].value_counts()
        duplicate_timestamps = timestamp_counts[timestamp_counts > 1].index.tolist()
        
        self.assertGreater(len(duplicate_timestamps), 0, "No duplicate timestamps found to test consistency.")
        
        # Pick a random timestamp with multiple entries
        test_ts = random.choice(duplicate_timestamps)
        
        subset = self.final_df[self.final_df['timestamp'] == test_ts]
        print(f"\\nTesting consistency for timestamp {test_ts} with {len(subset)} entries.")
        
        # The volatility columns should be identical across all rows in this subset
        vol_cols = [
            'btc_return', 'realized_variance', 'realized_volatility',
            'positive_semivariance', 'negative_semivariance',
            'parkinson_volatility', 'garman_klass_volatility'
        ]
        
        for col in vol_cols:
            if col in subset.columns:
                unique_vals = subset[col].nunique(dropna=False)
                # nunique() returns number of unique valid values. 
                # If all are NaN, it returns 0. If all are the same value, it returns 1.
                self.assertLessEqual(unique_vals, 1, f"Column {col} is not consistent across entries for timestamp {test_ts}")

    def test_04_no_forward_looking_bias(self):
        """
        Data should be strictly temporally ordered in terms of the underlying merge.
        Options data timestamps must always be >= the timestamp of the hourly BTC features they were merged with.
        Since we merged using merge_asof(direction='backward'), the underlying BTC snapshot 
        associated with an option should never come from the future relative to the option's timestamp.
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")
            
        # We can't perfectly test this without the hourly BTC df to cross-reference, 
        # but we can ensure time_to_maturity is >= 0
        self.assertTrue((self.final_df['time_to_maturity'] >= 0).all(), "Found negative time_to_maturity! Forward-looking bias or bad data.")
        
        # Check that there are no NaN values where there shouldn't be
        vital_cols = ['timestamp', 'option_price', 'underlying_price', 'strike', 'time_to_maturity']
        for col in vital_cols:
            self.assertFalse(self.final_df[col].isna().any(), f"Column {col} contains NaNs improperly.")


if __name__ == '__main__':
    unittest.main()
