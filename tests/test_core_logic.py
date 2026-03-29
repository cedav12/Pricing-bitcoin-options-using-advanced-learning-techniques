import unittest
import pandas as pd
import numpy as np
from src.feature_engineering import compute_time_to_maturity, compute_log_moneyness
from src.black_scholes import black_scholes_price
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


class TestImpliedVolatility(unittest.TestCase):
    """Round-trip: price with BS → recover sigma via IV inversion."""

    def test_iv_roundtrip_call(self):
        from src.black_scholes import compute_implied_volatility
        S, K, T, r, sigma_true = 50000.0, 50000.0, 0.25, 0.02, 0.80
        # Price a call
        price = black_scholes_price(
            pd.Series([S]), pd.Series([K]), pd.Series([T]), r, sigma_true,
            pd.Series(["call"])
        )
        # Invert to recover sigma
        iv = compute_implied_volatility(
            pd.Series(price), pd.Series([S]), pd.Series([K]),
            pd.Series([T]), pd.Series([r]), pd.Series(["call"])
        )
        self.assertAlmostEqual(float(iv[0]), sigma_true, places=4)

    def test_iv_roundtrip_put(self):
        from src.black_scholes import compute_implied_volatility
        S, K, T, r, sigma_true = 48000.0, 50000.0, 0.5, 0.03, 0.60
        price = black_scholes_price(
            pd.Series([S]), pd.Series([K]), pd.Series([T]), r, sigma_true,
            pd.Series(["put"])
        )
        iv = compute_implied_volatility(
            pd.Series(price), pd.Series([S]), pd.Series([K]),
            pd.Series([T]), pd.Series([r]), pd.Series(["put"])
        )
        self.assertAlmostEqual(float(iv[0]), sigma_true, places=4)

    def test_iv_nan_on_negative_price(self):
        from src.black_scholes import compute_implied_volatility
        iv = compute_implied_volatility(
            pd.Series([-1.0]), pd.Series([50000.0]), pd.Series([50000.0]),
            pd.Series([0.25]), pd.Series([0.02]), pd.Series(["call"])
        )
        self.assertTrue(np.isnan(iv[0]))


class TestErrorMetrics(unittest.TestCase):
    """Verify MAE / RMSE / MAPE / Bias on a tiny synthetic dataset."""

    def _make_df(self):
        from src.black_scholes import BlackScholesBenchmark
        data = {
            "option_price":      [100.0, 200.0, 150.0],
            "underlying_price":  [50000.0, 50000.0, 50000.0],
            "strike":            [50000.0, 48000.0, 52000.0],
            "time_to_maturity":  [0.25, 0.5, 0.1],
            "option_type":       ["call", "put", "call"],
            "rolling_std_24h":   [0.7, 0.8, 0.6],
            "risk_free_rate":    [0.02, 0.02, 0.02],
        }
        df = pd.DataFrame(data)
        df["bs_price"] = [110.0, 190.0, 160.0]
        df["pricing_error"] = df["bs_price"] - df["option_price"]
        df["log_moneyness"] = np.log(df["underlying_price"] / df["strike"])
        return df, BlackScholesBenchmark()

    def test_overall_metrics(self):
        df, bench = self._make_df()
        metrics = bench.compute_error_metrics(df)
        ov = metrics["overall"]
        # error vector: [10, -10, 10]
        self.assertAlmostEqual(ov["MAE"],  10.0, places=4)
        self.assertAlmostEqual(ov["RMSE"], 10.0, places=4)
        self.assertAlmostEqual(ov["Bias"], 10/3, places=4)

    def test_grouped_keys(self):
        df, bench = self._make_df()
        metrics = bench.compute_error_metrics(df)
        self.assertIn("by_moneyness", metrics)
        self.assertIn("by_maturity",  metrics)
        self.assertIn("MAE",  metrics["by_moneyness"].columns)
        self.assertIn("RMSE", metrics["by_maturity"].columns)


class TestMoneynessAndMaturityBuckets(unittest.TestCase):
    def test_moneyness_labels(self):
        from src.black_scholes import BlackScholesBenchmark
        bench = BlackScholesBenchmark()
        lm = pd.Series([-0.5, -0.10, 0.0, 0.10, 0.5])
        buckets = bench._assign_moneyness_bucket(lm)
        self.assertEqual(str(buckets.iloc[0]), "Deep OTM")
        self.assertEqual(str(buckets.iloc[2]), "ATM")
        self.assertEqual(str(buckets.iloc[4]), "Deep ITM")

    def test_maturity_labels(self):
        from src.black_scholes import BlackScholesBenchmark
        bench = BlackScholesBenchmark()
        ttm = pd.Series([2/365, 14/365, 60/365, 180/365, 400/365])
        buckets = bench._assign_maturity_bucket(ttm)
        self.assertEqual(str(buckets.iloc[0]), "< 1 week")
        self.assertEqual(str(buckets.iloc[2]), "1–3 months")
        self.assertEqual(str(buckets.iloc[4]), "> 1 year")


if __name__ == '__main__':
    unittest.main()
