import unittest
import pandas as pd
import numpy as np
from src.feature_engineering import compute_time_to_maturity, compute_log_moneyness
from src.models.black_scholes import black_scholes_price
from src.utils import parse_option_instrument

class TestFeatureEngineering(unittest.TestCase):
    def test_compute_time_to_maturity(self):
        # 1 year apart in milliseconds
        timestamps = pd.Series([1609459200000]) # 2021-01-01
        expiries = pd.Series([pd.Timestamp('2022-01-01')])
        
        ttm = compute_time_to_maturity(timestamps, expiries)
        # Should be roughly 1.0 years
        self.assertAlmostEqual(ttm.iloc[0], 1.0, places=2)

        # Ensure non-negative
        ttm_past = compute_time_to_maturity(pd.Series([1640995200000]), expiries) # 2022-01-01 approx equal
        self.assertGreaterEqual(ttm_past.iloc[0], 0.0)

    def test_compute_log_moneyness(self):
        underlying = pd.Series([50000.0, 40000.0])
        strikes = pd.Series([50000.0, 50000.0])
        
        lm = compute_log_moneyness(underlying, strikes)
        self.assertAlmostEqual(lm.iloc[0], 0.0) # ln(1) = 0
        self.assertLess(lm.iloc[1], 0.0) # ln(0.8) < 0

class TestBlackScholes(unittest.TestCase):
    def test_black_scholes_price(self):
        S = pd.Series([100.0, 100.0])
        K = pd.Series([100.0, 100.0])
        T = pd.Series([1.0, 1.0])
        r = 0.05
        sigma = 0.2
        option_type = pd.Series(['call', 'put'])
        
        prices = black_scholes_price(S, K, T, r, sigma, option_type)
        
        # Approximate call price for ATM, S=100, K=100, T=1, r=0.05, sigma=0.2 is ~ 10.45
        # Approximate put price is ~ 5.57
        self.assertAlmostEqual(prices[0], 10.45, places=2)
        self.assertAlmostEqual(prices[1], 5.57, places=2)

class TestUtils(unittest.TestCase):
    def test_parse_instrument(self):
        instrument = "BTC-25MAR22-40000-C"
        parsed = parse_option_instrument(instrument)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['underlying'], 'BTC')
        self.assertEqual(parsed['strike'], 40000.0)
        self.assertEqual(parsed['option_type'], 'call')
        
        # Test Put
        instrument_put = "BTC-25MAR22-40000-P"
        parsed_put = parse_option_instrument(instrument_put)
        self.assertEqual(parsed_put['option_type'], 'put')
        
        # Test Invalid
        self.assertIsNone(parse_option_instrument("BTC-25MAR22"))

if __name__ == '__main__':
    unittest.main()
