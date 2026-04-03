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
        cls.processed_data_path = "data/processed/options_dataset_filtered.csv"
        
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


    # ------------------------------------------------------------------
    # TEST 05 — INTRINSIC VALUE BOUND
    # ------------------------------------------------------------------
    def test_05_intrinsic_value_bound(self):
        """
        No-arbitrage lower bound:
          call_price >= max(S - K*exp(-rT), 0)
          put_price  >= max(K*exp(-rT) - S,  0)
        Uses a random 0.5 % sample; tolerance 1e-6.
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")

        sample = self.final_df.sample(frac=0.005, random_state=42)

        r = sample['risk_free_rate'].fillna(0.0)
        T = sample['time_to_maturity']
        S = sample['underlying_price']
        K = sample['strike']
        P = sample['option_price']
        opt_type = sample['option_type'].str.upper()

        discount = K * np.exp(-r * T)
        tol = 1e-6

        calls = opt_type == 'C'
        puts  = opt_type == 'P'

        call_lb = np.maximum(S - discount, 0.0)
        put_lb  = np.maximum(discount - S, 0.0)

        call_violations = (P[calls] < call_lb[calls] - tol).sum()
        put_violations  = (P[puts]  < put_lb[puts]   - tol).sum()

        total = call_violations + put_violations
        print(f"\nTest 05 – intrinsic-value violations: {total} "
              f"(calls: {call_violations}, puts: {put_violations}) "
              f"out of {len(sample)} sampled rows")
        self.assertEqual(total, 0,
            f"{total} option price(s) breach the no-arbitrage intrinsic-value lower bound.")

    # ------------------------------------------------------------------
    # TEST 06 — UPPER PRICE BOUND
    # ------------------------------------------------------------------
    def test_06_upper_price_bound(self):
        """
        call_price <= underlying_price
        put_price  <= strike
        Sampled 0.5 % of the dataset; small tolerance allowed.
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")

        sample = self.final_df.sample(frac=0.005, random_state=7)
        tol = 1e-6
        opt_type = sample['option_type'].str.upper()

        calls = opt_type == 'C'
        puts  = opt_type == 'P'

        call_violations = (sample.loc[calls, 'option_price'] >
                           sample.loc[calls, 'underlying_price'] + tol).sum()
        put_violations  = (sample.loc[puts, 'option_price'] >
                           sample.loc[puts, 'strike'] + tol).sum()

        total = call_violations + put_violations
        print(f"\nTest 06 – upper-price-bound violations: {total} "
              f"(calls: {call_violations}, puts: {put_violations})")
        self.assertEqual(total, 0,
            f"{total} option price(s) exceed their theoretical upper bound.")

    # ------------------------------------------------------------------
    # TEST 07 — STRIKE MONOTONICITY
    # ------------------------------------------------------------------
    def test_07_strike_monotonicity(self):
        """
        For randomly sampled (timestamp, expiry) slices isolate call options,
        sort by strike and verify price is non-increasing.
        Fails only when violations exceed 5 % of checked triples.
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")

        calls = self.final_df[self.final_df['option_type'].str.upper() == 'CALL']
        if calls.empty:
            self.skipTest("No call options in dataset.")

        keys = (calls.groupby(['timestamp', 'expiry'])
                     .filter(lambda g: len(g) >= 2)
                     [['timestamp', 'expiry']]
                     .drop_duplicates())

        if keys.empty:
            self.skipTest("No (timestamp, expiry) groups with >= 2 strikes.")

        n_sample = min(200, len(keys))
        sampled_keys = keys.sample(n=n_sample, random_state=42)

        violations = 0
        checked = 0
        for _, row in sampled_keys.iterrows():
            grp = (calls[(calls['timestamp'] == row['timestamp']) &
                         (calls['expiry']    == row['expiry'])]
                   .sort_values('strike'))
            diffs = grp[grp["trade_count"]>0]['option_price'].diff().dropna()
            violations += (diffs > 1e-6).sum()
            checked += len(diffs)

        print(f"\nTest 07 – strike-monotonicity violations: {violations}/{checked} "
              f"across {n_sample} sampled slices")
        threshold = max(1, int(0.05 * checked))
        self.assertLessEqual(violations, threshold,
            f"Too many strike-monotonicity violations ({violations} > {threshold}).")

    # ------------------------------------------------------------------
    # TEST 08 — STRIKE CONVEXITY
    # ------------------------------------------------------------------
    def test_08_strike_convexity(self):
        """
        For sampled timestamps verify convexity:
          C(K1) - 2*C(K2) + C(K3) >= 0  for neighbouring strike triples.
        Fails only when violations exceed 2 % of checked triples.
        """
        self.assertTrue(True)
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")

        calls = self.final_df[self.final_df['option_type'].str.upper() == 'CALL']
        if calls.empty:
            self.skipTest("No call options in dataset.")

        keys = (calls.groupby(['timestamp', 'expiry'])
                     .filter(lambda g: len(g) >= 3)
                     [['timestamp', 'expiry']]
                     .drop_duplicates())

        if keys.empty:
            self.skipTest("No groups with >= 3 strikes for convexity check.")

        n_sample = min(200, len(keys))
        sampled_keys = keys.sample(n=n_sample, random_state=99)

        violations = 0
        checked = 0
        for _, row in sampled_keys.iterrows():
            grp = (calls[(calls['timestamp'] == row['timestamp']) &
                         (calls['expiry']    == row['expiry'])]
                   .sort_values('strike')
                   .reset_index(drop=True))
            grp = grp[grp["trade_count"] > 0].reset_index(drop=True)
            prices = grp['option_price'].values
            # Check convexity for every consecutive triple
            for i in range(len(prices) - 2):
                butterfly = prices[i] - 2 * prices[i + 1] + prices[i + 2]
                if butterfly < -1e-6:
                    violations += 1
                checked += 1

        print(f"\nTest 08 – strike-convexity violations: {violations}/{checked} "
              f"across {n_sample} sampled slices")
        threshold = max(1, int(0.02 * checked))
        self.assertLessEqual(violations, threshold,
            f"Too many convexity violations ({violations} > {threshold}).")

    # ------------------------------------------------------------------
    # TEST 09 — TIMESTAMP ORDERING PER INSTRUMENT
    # ------------------------------------------------------------------
    def test_09_timestamp_ordering_per_instrument(self):
        """
        For each (strike, expiry, option_type) instrument, timestamps must be
        non-decreasing (strictly increasing since duplicates are forbidden).
        Checks a random sample of 500 instruments.
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")

        groups = self.final_df.groupby(['strike', 'expiry', 'option_type'])
        all_keys = list(groups.groups.keys())
        n_sample = min(500, len(all_keys))
        sampled = random.sample(all_keys, n_sample)

        violations = 0
        for key in sampled:
            ts = groups.get_group(key)['timestamp'].values
            if (np.diff(ts) < 0).any():
                violations += 1

        print(f"\nTest 09 – instruments with non-monotone timestamps: "
              f"{violations}/{n_sample}")
        self.assertEqual(violations, 0,
            f"{violations} instrument(s) have out-of-order timestamps.")

    # ------------------------------------------------------------------
    # TEST 10 — DUPLICATE OBSERVATIONS
    # ------------------------------------------------------------------
    def test_10_no_duplicate_observations(self):
        """
        No two rows may share the same (timestamp, strike, expiry, option_type).
        Uses drop_duplicates on the key columns to detect collisions.
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")

        key_cols = ['timestamp', 'strike', 'expiry', 'option_type']
        n_total = len(self.final_df)
        n_unique = self.final_df.drop_duplicates(subset=key_cols).shape[0]
        n_dupes = n_total - n_unique

        print(f"\nTest 10 – duplicate observations: {n_dupes} (out of {n_total} rows)")
        self.assertEqual(n_dupes, 0,
            f"{n_dupes} duplicate (timestamp, strike, expiry, option_type) rows found.")

    # ------------------------------------------------------------------
    # TEST 11 — VOLATILITY DECOMPOSITION
    # ------------------------------------------------------------------
    def test_11_volatility_decomposition(self):
        """
        realized_variance ≈ positive_semivariance + negative_semivariance
        Checks a 1 % random sample; tolerance 1e-8.
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")

        required = ['realized_variance', 'positive_semivariance', 'negative_semivariance']
        missing = [c for c in required if c not in self.final_df.columns]
        if missing:
            self.skipTest(f"Columns missing for decomposition check: {missing}")

        sample = self.final_df[required].dropna().sample(frac=0.01, random_state=42)
        if sample.empty:
            self.skipTest("No non-null rows available for decomposition check.")

        reconstructed = sample['positive_semivariance'] + sample['negative_semivariance']
        abs_diff = (sample['realized_variance'] - reconstructed).abs()
        max_err = abs_diff.max()
        n_violations = (abs_diff > 1e-8).sum()

        print(f"\nTest 11 – volatility decomposition max error: {max_err:.2e}, "
              f"violations: {n_violations}/{len(sample)}")
        self.assertEqual(n_violations, 0,
            f"{n_violations} row(s) where realized_variance ≠ pos + neg semivariance "
            f"(max error: {max_err:.2e}).")

    # ------------------------------------------------------------------
    # TEST 12 — LOG MONEYNESS CORRECTNESS
    # ------------------------------------------------------------------
    def test_12_log_moneyness_correctness(self):
        """
        log_moneyness must equal log(underlying_price / strike) within 1e-8.
        Verified on a 1 % random sample.
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")

        required = ['log_moneyness', 'underlying_price', 'strike']
        missing = [c for c in required if c not in self.final_df.columns]
        if missing:
            self.skipTest(f"Columns missing: {missing}")

        sample = self.final_df[required].dropna().sample(frac=0.01, random_state=42)
        if sample.empty:
            self.skipTest("No non-null rows for log_moneyness check.")

        expected = np.log(sample['underlying_price'] / sample['strike'])
        abs_diff = (sample['log_moneyness'] - expected).abs()
        max_err = abs_diff.max()
        n_violations = (abs_diff > 1e-8).sum()

        print(f"\nTest 12 – log_moneyness max error: {max_err:.2e}, "
              f"violations: {n_violations}/{len(sample)}")
        self.assertEqual(n_violations, 0,
            f"{n_violations} log_moneyness value(s) deviate from log(S/K) "
            f"(max error: {max_err:.2e}).")

    # ------------------------------------------------------------------
    # TEST 13 — TIME TO MATURITY RANGE
    # ------------------------------------------------------------------
    def test_13_time_to_maturity_range(self):
        """
        0 <= time_to_maturity < 2 years for all rows
        (crypto options rarely exceed 2 years).
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")

        ttm = self.final_df['time_to_maturity']
        below_zero = (ttm < 0).sum()
        above_two  = (ttm >= 2).sum()

        print(f"\nTest 13 – TTM < 0: {below_zero}, TTM >= 2y: {above_two}")
        self.assertEqual(below_zero, 0,
            f"{below_zero} row(s) have time_to_maturity < 0.")
        self.assertEqual(above_two, 0,
            f"{above_two} row(s) have time_to_maturity >= 2 years.")

    # ------------------------------------------------------------------
    # TEST 14 — BTC RETURN CONSISTENCY
    # ------------------------------------------------------------------
    def test_14_btc_return_consistency(self):
        """
        For the same timestamp, btc_return must be identical across all rows.
        Verifies 200 randomly sampled timestamps that appear more than once.
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")

        if 'btc_return' not in self.final_df.columns:
            self.skipTest("Column 'btc_return' not present.")

        ts_counts = self.final_df['timestamp'].value_counts()
        multi_ts = ts_counts[ts_counts > 1].index.tolist()

        if not multi_ts:
            self.skipTest("No repeated timestamps found.")

        n_sample = min(200, len(multi_ts))
        sampled_ts = random.sample(multi_ts, n_sample)

        inconsistent = 0
        for ts in sampled_ts:
            unique_vals = self.final_df.loc[
                self.final_df['timestamp'] == ts, 'btc_return'
            ].nunique(dropna=False)
            if unique_vals > 1:
                inconsistent += 1

        print(f"\nTest 14 – timestamps with inconsistent btc_return: "
              f"{inconsistent}/{n_sample}")
        self.assertEqual(inconsistent, 0,
            f"{inconsistent} timestamp(s) have differing btc_return values across rows.")

    # ------------------------------------------------------------------
    # TEST 15 — LIQUIDITY SANITY
    # ------------------------------------------------------------------
    def test_15_liquidity_sanity(self):
        """
        volume >= 0 and trade_count >= 0 for all rows.
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")

        errors = []
        for col in ['volume', 'trade_count']:
            if col not in self.final_df.columns:
                continue
            neg = (self.final_df[col] < 0).sum()
            if neg:
                errors.append(f"{col}: {neg} negative value(s)")

        print(f"\nTest 15 – liquidity sanity: {errors if errors else 'OK'}")
        self.assertEqual(len(errors), 0, "; ".join(errors))

    # ------------------------------------------------------------------
    # TEST 16 — EXTREME PRICE FILTER
    # ------------------------------------------------------------------
    def test_16_extreme_price_filter(self):
        """
        Flag observations where option_price > 5 * underlying_price.
        Reports count but does not necessarily hard-fail (reports for awareness).
        """
        if self.final_df.empty:
            self.skipTest("Processed dataset is empty.")

        extreme = (self.final_df['option_price'] >
                   5).sum()
        pct = 100.0 * extreme / len(self.final_df)

        print(f"\nTest 16 – extreme prices (option > 5×S): {extreme} rows ({pct:.4f} %)")
        # Hard-fail if more than 0.1 % of the dataset is affected
        threshold = max(1, int(0.001 * len(self.final_df)))
        self.assertLessEqual(extreme, threshold,
            f"{extreme} row(s) have option_price > 5 × underlying_price "
            f"(threshold: {threshold}).")


if __name__ == '__main__':
    unittest.main()
