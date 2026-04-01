"""
Preprocessing logic for the Model-Agnostic Evaluation Framework
"""
import numpy as np
import pandas as pd

def preprocess_dataset(df: pd.DataFrame, option_filter: str = "call") -> pd.DataFrame:
    """
    Applies strict data validity filtering and feature derivation.
    Crucially, it DOES NOT filter by price or time value here to avoid dataset bias.
    """
    df = df.copy()

    # 1. Structural Filtering
    df = df[df["time_to_maturity"] > 0]
    
    option_filter_lower = option_filter.lower()
    if option_filter_lower in ["call", "put"]:
        df = df[df["option_type"].str.lower() == option_filter_lower]

    # 2. Feature Derivation
    if "log_moneyness" not in df.columns:
        df["log_moneyness"] = np.log(df["underlying_price"] / df["strike"])
        
    if "time_value" not in df.columns:
        is_call = df["option_type"].str.lower() == "call"
        intrinsic_usd = np.where(
            is_call,
            np.maximum(df["underlying_price"] - df["strike"], 0),
            np.maximum(df["strike"] - df["underlying_price"], 0)
        )
        # Assuming market_price is in BTC because Deribit is inverse
        intrinsic_btc = intrinsic_usd / df["underlying_price"]
        df["time_value"] = df["market_price"] - intrinsic_btc

    return df
