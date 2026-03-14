import unittest
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import patch
from src.dataset_builder import DatasetBuilder

class TestDatasetBuilder(unittest.TestCase):
    def setUp(self):
        # Create temp dirs for testing
        self.test_raw_dir = "data/test_raw"
        self.test_processed_dir = "data/test_processed"
        os.makedirs(self.test_raw_dir, exist_ok=True)
        os.makedirs(self.test_processed_dir, exist_ok=True)
        
        # Create dummy BTC data
        btc_data = {
            'timestamp': [1609459200000, 1609462800000], # 2021-01-01 00:00, 2021-01-01 01:00
            'open': [29000, 29100],
            'high': [29200, 29300],
            'low': [28900, 29000],
            'close': [29100, 29200], # This is the underlying_price
            'volume': [1000, 1100],
            'trade_count': [500, 550]
        }
        pd.DataFrame(btc_data).to_csv(os.path.join(self.test_raw_dir, "BTC-PERPETUAL.csv"), index=False)
        
        # Create dummy Options data
        # Note: the timestamp here needs to align, using ISO strings to test the fix
        options_data = {
            'timestamp': ['2021-01-01T00:00:00Z', '2021-01-01T01:00:00Z'],
            'open': [0.1, 0.11],
            'high': [0.12, 0.13],
            'low': [0.09, 0.10],
            'close': [0.11, 0.12], # This is the option_price
            'volume': [10, 20],
            'trade_count': [5, 10]
        }
        pd.DataFrame(options_data).to_csv(os.path.join(self.test_raw_dir, "BTC-25MAR21-30000-C.csv"), index=False)

    def tearDown(self):
        shutil.rmtree(self.test_raw_dir)
        shutil.rmtree(self.test_processed_dir)

    @patch('src.dataset_builder.download_and_prepare_macro_data')
    def test_build_dataset(self, mock_macro):
        # Create dummy macro data to avoid actual yfinance downloads during tests
        macro_dummy = pd.DataFrame({
            'timestamp': [1609459200000, 1609462800000],
            'risk_free_rate': [0.05, 0.05],
            'vix_index': [20.0, 21.0]
        })
        mock_macro.return_value = macro_dummy
        
        builder = DatasetBuilder(raw_data_dir=self.test_raw_dir, processed_data_dir=self.test_processed_dir)
        output_path = builder.build_dataset()
        
        self.assertTrue(os.path.exists(output_path))
        
        # Load the generated dataset and check its contents
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 2)
        
        # Check if columns are correct
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
        for col in expected_cols:
            self.assertIn(col, df.columns)
            
        # Check alignment values
        # The first row should have underlying_price 29100, option_price 0.11
        self.assertEqual(df.iloc[0]['underlying_price'], 29100)
        self.assertEqual(df.iloc[0]['option_price'], 0.11)
        self.assertEqual(df.iloc[0]['strike'], 30000.0)
        self.assertEqual(df.iloc[0]['option_type'], 'call')
        self.assertEqual(df.iloc[0]['risk_free_rate'], 0.05)
        self.assertEqual(df.iloc[0]['vix_index'], 20.0)

if __name__ == '__main__':
    unittest.main()
