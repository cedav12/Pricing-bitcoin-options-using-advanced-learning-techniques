import numpy as np
import pandas as pd
from scipy.stats import norm
import os

def black_scholes_price(S: pd.Series, K: pd.Series, T: pd.Series, r: float, sigma: float, option_type: pd.Series) -> pd.Series:
    """
    Calculate the Black-Scholes theoretical price for European options.
    S: Underlying Price
    K: Strike Price
    T: Time to Maturity (Years)
    r: Risk-free interest rate
    sigma: Volatility
    option_type: 'call' or 'put'
    """
    # Avoid division by zero when T = 0
    T_safe = np.where(T <= 0.0, 1e-8, T)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
    d2 = d1 - sigma * np.sqrt(T_safe)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    put_price = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # Select price based on option type
    is_call = option_type.str.lower() == 'call'
    return np.where(is_call, call_price, put_price)

class BlackScholesBenchmark:
    def __init__(self, dataset_path: str = "data/processed/options_dataset.csv"):
        self.dataset_path = dataset_path

    def run_benchmark(self, r: float = 0.0, sigma: float = 0.8):
        """
        Loads dataset, computes BS price with an assumed constant volatility,
        and computes error metrics against market option_price.
        A fixed vol of 80% (0.8) is used here as a placeholder. In future iterations,
        implied vol could be computed or a term structure could be used.
        """
        if not os.path.exists(self.dataset_path):
            print(f"[Error] Dataset not found at {self.dataset_path}. Run build_dataset first.")
            return

        print(f"Loading dataset from {self.dataset_path}...")
        df = pd.read_csv(self.dataset_path)
        
        print("Computing Black-Scholes theoretical prices...")
        # Ensure correct types
        df['theoretical_price'] = black_scholes_price(
            S=df['underlying_price'],
            K=df['strike'],
            T=df['time_to_maturity'],
            r=r,
            sigma=sigma,
            option_type=df['option_type']
        )
        
        print("Calculating error metrics...")
        errors = df['theoretical_price'] - df['option_price']
        
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        print("\\n--- Black-Scholes Benchmark Results ---")
        print(f"Assumed constant Volatility: {sigma:.2%}")
        print(f"Assumed constant Risk-Free Rate: {r:.2%}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print("---------------------------------------\\n")
        
        return {'mae': mae, 'rmse': rmse}
