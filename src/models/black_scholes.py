"""
Pure Black-Scholes Model
========================
High-performance vectorized implementation of the Black-Scholes European call/put pricing model.
Contains NO data loading, plotting, IV inversion, or evaluation logic.
"""

import numpy as np
import pandas as pd
from scipy.special import ndtr

def black_scholes_price(
    S: "array-like",
    K: "array-like",
    T: "array-like",
    r: "float | array-like",
    sigma: "float | array-like",
    option_type: "array-like",
) -> np.ndarray:
    """
    Vectorised Black-Scholes pricer for European call/put options.

    Parameters
    ----------
    S : underlying price
    K : strike price
    T : time to maturity in years
    r : risk-free rate (scalar or array)
    sigma : volatility (scalar or array)
    option_type : "call" or "put" (case-insensitive)

    Returns
    -------
    np.ndarray of theoretical prices
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Numerical safety guards
    T_safe   = np.maximum(T, 1e-8)
    S_safe   = np.maximum(S, 1e-8)
    K_safe   = np.maximum(K, 1e-8)
    sig_safe = np.maximum(sigma, 1e-8)

    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S_safe / K_safe) + (r + 0.5 * sig_safe ** 2) * T_safe) / (sig_safe * sqrt_T)
    d2 = d1 - sig_safe * sqrt_T

    # ndtr is significantly faster than scipy.stats.norm.cdf
    call_price = S_safe * ndtr(d1) - K_safe * np.exp(-r * T_safe) * ndtr(d2)
    put_price  = K_safe * np.exp(-r * T_safe) * ndtr(-d2) - S_safe * ndtr(-d1)

    # Resolve option type
    if isinstance(option_type, pd.Series):
        is_call = option_type.str.lower().values == "call"
    else:
        opt_arr = np.asarray(option_type)
        is_call = np.char.lower(opt_arr.astype(str)) == "call"

    return np.where(is_call, call_price, put_price)
