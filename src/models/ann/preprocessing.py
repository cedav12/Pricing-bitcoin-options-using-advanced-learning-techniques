import pandas as pd
import numpy as np
from src.models.ann.schema import validate_columns, check_numeric_columns


def prepare_ann_dataframe(config: dict) -> pd.DataFrame:
    """
    Load mapping logic parsing explicit targets out of filtered option sets.
    Optimizes loading sequences via `usecols`, scales gracefully supporting CSV/Parquet interfaces respectively.
    """
    input_path = config.get("input_path")
    if not input_path:
        raise ValueError("config must specify 'input_path'")

    features = config.get("feature_columns", [])
    target = config.get("target_column")
    metadata = config.get("metadata_columns", [])

    if not features:
        raise ValueError("No 'feature_columns' specified.")
    if not target:
        raise ValueError("No 'target_column' specified.")

    # We load strictly explicit configurations maximizing global system stability scaling data
    all_needed = list(set(features + [target] + metadata))

    print(f"Loading data from {input_path}...")
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path, columns=all_needed)
    else:
        df = pd.read_csv(input_path, usecols=all_needed)

    if df.empty:
        raise ValueError("Loaded dataset is completely empty.")

    check_numeric_columns(df, features, " for features")
    check_numeric_columns(df, [target], " for target")

    # Replace inf mathematically
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if config.get("drop_na", True):
        model_vars = features + [target]
        pre_len = len(df)
        df.dropna(subset=model_vars, inplace=True)
        post_len = len(df)
        if pre_len != post_len:
            print(f"Dropped {pre_len - post_len} rows containing NaN/inf in model vars.")

    if df.empty:
        raise ValueError("Dataset is empty after dropping missing values.")

    dtype_str = config.get("dtype", "float32")

    for c in features + [target]:
        df[c] = df[c].astype(dtype_str)

    # Reset index efficiently preparing clean arrays safely wrapping indices naturally tracking outputs dynamically
    df.reset_index(drop=True, inplace=True)
    return df
