"""
Segmentation engine for Model Evaluation.
Provides dynamic bucketing mechanisms for evaluating models across different regimes.
"""
import numpy as np
import pandas as pd

def segment_moneyness(df: pd.DataFrame) -> pd.Series:
    """Bins log_moneyness into discrete buckets."""
    if "log_moneyness" not in df.columns:
        return pd.Series(index=df.index, dtype="category")
    bins = [-np.inf, -0.10, -0.05, 0.05, 0.10, np.inf]
    labels = ["Deep OTM", "OTM", "ATM", "ITM", "Deep ITM"]
    return pd.cut(df["log_moneyness"], bins=bins, labels=labels)

def segment_maturity(df: pd.DataFrame) -> pd.Series:
    """Bins time_to_maturity into practical duration buckets."""
    if "time_to_maturity" not in df.columns:
        return pd.Series(index=df.index, dtype="category")
    bins = [-np.inf, 7/365, 30/365, 90/365, 1.0, np.inf]
    labels = ["< 1 week", "1-4 weeks", "1-3 months", "3-12 months", "> 1 year"]
    return pd.cut(df["time_to_maturity"], bins=bins, labels=labels)

def segment_price(df: pd.DataFrame) -> pd.Series:
    """Bins market_price into magnitude buckets to isolate tick-size noise."""
    if "market_price" not in df.columns:
        return pd.Series(index=df.index, dtype="category")
    # Using specific boundaries reflecting BTC price levels
    bins = [-np.inf, 0.001, 0.01, 0.1, 1.0, np.inf]
    labels = ["< 0.001", "0.001-0.01", "0.01-0.10", "0.10-1.0", "> 1.0"]
    return pd.cut(df["market_price"], bins=bins, labels=labels)

def segment_liquidity(df: pd.DataFrame) -> pd.Series:
    """Bins volume for liquidity regime analysis if available."""
    if "volume" not in df.columns:
        return pd.Series(index=df.index, dtype="category")
    bins = [-np.inf, 0, 10, 100, np.inf]
    labels = ["Zero", "Illiquid (0-10)", "Liquid (10-100)", "Highly Liquid (>100)"]
    return pd.cut(df["volume"], bins=bins, labels=labels)

def segment_volatility(df: pd.DataFrame) -> pd.Series:
    """Bins realized_volatility for volatility regime analysis if available."""
    if "realized_volatility" not in df.columns:
        return pd.Series(index=df.index, dtype="category")
    bins = [-np.inf, 0.30, 0.60, 1.0, np.inf]
    labels = ["Low (<30%)", "Normal (30-60%)", "High (60-100%)", "Extreme (>100%)"]
    return pd.cut(df["realized_volatility"], bins=bins, labels=labels)

SEGMENT_FUNCTIONS = {
    "moneyness": segment_moneyness,
    "maturity": segment_maturity,
    "price": segment_price,
    "liquidity": segment_liquidity,
    "volatility": segment_volatility
}

def apply_segments(df: pd.DataFrame, segments: list) -> pd.DataFrame:
    """Applies a list of requested segments to the dataframe."""
    df = df.copy()
    for seg_name in segments:
        seg_lower = seg_name.lower()
        if seg_lower in SEGMENT_FUNCTIONS:
            df[f"seg_{seg_lower}"] = SEGMENT_FUNCTIONS[seg_lower](df)
        else:
            print(f"[Segmentation] Warning: Unknown segment '{seg_name}'")
    return df
