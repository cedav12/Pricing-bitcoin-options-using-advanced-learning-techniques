import pandas as pd
import numpy as np
from src.models.ann.schema import validate_columns, check_numeric_columns

def prepare_ann_dataframe(config: dict) -> pd.DataFrame:
    """
    Load mapping logic parsing explicit targets out of filtered option sets.
    Validates numerical requirements and NaN dropping logic explicitly mapping against config inputs.
    """
    input_path = config.get("input_path")
    if not input_path:
        raise ValueError("config must specify 'input_path'")
        
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Loaded dataset is completely empty.")
        
    features = config.get("feature_columns", [])
    target = config.get("target_column")
    metadata = config.get("metadata_columns", [])
    
    if not features:
        raise ValueError("No 'feature_columns' specified.")
    if not target:
        raise ValueError("No 'target_column' specified.")
        
    validate_columns(df, features, " for features")
    validate_columns(df, [target], " for target")
    validate_columns(df, metadata, " for metadata")
    
    check_numeric_columns(df, features, " for features")
    check_numeric_columns(df, [target], " for target")
    
    # Keep only explicitly targeted columns dynamically tracking explicit structure limits 
    all_needed = list(set(features + [target] + metadata))
    df = df[all_needed].copy()
    
    # Replace inf with NaN for uniform dropping evaluations matching native missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if config.get("drop_na", True):
        # Strictly evaluate missing representations impacting models
        model_vars = features + [target]
        pre_len = len(df)
        df.dropna(subset=model_vars, inplace=True)
        post_len = len(df)
        if pre_len != post_len:
            print(f"Dropped {pre_len - post_len} rows containing NaN/inf in model vars.")
            
    if df.empty:
        raise ValueError("Dataset is empty after dropping missing values.")
        
    dtype_str = config.get("dtype", "float32")
    
    # Process only model representations saving uncast metadata fields globally
    for c in features + [target]:
        df[c] = df[c].astype(dtype_str)
        
    return df
