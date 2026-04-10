import pandas as pd
import numpy as np

def prepare_ann_dataframe(config: dict) -> pd.DataFrame:
    """
    Validates config and loads dataset with correct typings and required columns.
    Returns a clean DataFrame ready for the split manager.
    """
    input_path = config.get("input_path")
    if not input_path:
        raise ValueError("config must specify 'input_path'")
        
    features = config.get("feature_columns")
    if not features or not isinstance(features, list) or len(features) == 0:
        raise ValueError("config must specify a non-empty list for 'feature_columns'")
        
    target = config.get("target_column")
    if not target:
        raise ValueError("config must specify 'target_column'")
        
    ts_col = config.get("timestamp_column")
    if not ts_col:
        raise ValueError("config must specify 'timestamp_column'")
        
    module_cols = config.get("module_columns", [])
    metadata = config.get("metadata_columns", [])
    dtype_str = config.get("dtype", "float32")
    
    all_needed = list(set(features + [target] + metadata + module_cols + [ts_col]))
    
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path, columns=all_needed)
    else:
        df = pd.read_csv(input_path, usecols=all_needed)
        
    if df.empty:
        raise ValueError("Loaded dataset is completely empty.")
        
    # Explicit Timestamp check
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found in dataset.")
    if df[ts_col].isna().any():
        raise ValueError(f"Timestamp column '{ts_col}' contains missing values, which violates chronological splitting.")
        
    # Explicit Module check
    missing_modules = [m for m in module_cols if m not in df.columns]
    if missing_modules:
        raise ValueError(f"Required module columns not found in dataset: {missing_modules}")
        
    df[features + [target]] = df[features + [target]].replace([np.inf, -np.inf], np.nan)
    
    if config.get("drop_na", True):
        df.dropna(subset=features + [target], inplace=True)
            
    if df.empty:
        raise ValueError("Dataset is empty after dropping missing values.")
        
    for c in features + [target]:
        try:
            df[c] = df[c].astype(dtype_str)
        except ValueError as e:
            raise TypeError(f"Failed to cast target/feature column '{c}' to numeric dtype '{dtype_str}'. Inner error: {e}")
        except TypeError as e:
            raise TypeError(f"Failed to cast target/feature column '{c}' to numeric dtype '{dtype_str}'. Inner error: {e}")
            
    df.reset_index(drop=True, inplace=True)
    return df
