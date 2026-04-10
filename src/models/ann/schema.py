"""
Data validation schemas and helpers for ANN dataset preparation.
"""

def validate_columns(columns_list, required_columns, context=""):
    """
    Validates that a list of columns exists.
    """
    missing = [c for c in required_columns if c not in columns_list]
    if missing:
        raise ValueError(f"Missing required columns{context}: {missing}")

def check_numeric_columns(df, numeric_columns, context=""):
    """
    Validates that the specified columns are numeric.
    """
    import pandas as pd
    non_numeric = [c for c in numeric_columns if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Columns must be numeric{context}: {non_numeric}")
