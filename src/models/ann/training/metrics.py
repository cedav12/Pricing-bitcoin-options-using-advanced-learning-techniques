import numpy as np
from typing import Dict

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes basic regression evaluation metrics.
    - y_true: actual prices
    - y_pred: predicted prices
    """
    # Ensure they are flat arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Error
    error = y_pred - y_true
    
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))
    
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias
    }
