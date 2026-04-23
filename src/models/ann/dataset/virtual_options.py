"""
virtual_options.py
------------------
Train-only virtual options augmentation.

Currently supports:
  type = "ttm_zero"

    For each module's train split:
      1. Sample real train rows.
      2. Set time_to_maturity = 0.
      3. Recompute option_price as the maturity intrinsic payoff in the
         same BTC-denominated convention used by the rest of the project:
           call : max(S - K, 0) / S
           put  : max(K - S, 0) / S
         where S = underlying_price, K = strike (both in USD).
      4. Append the virtual rows to the train arrays only.

Val and test are never touched.
Scaling must happen AFTER augmentation (caller responsibility).

Feature handling policy (v1, intentional):
------------------------------------------
  MUST change   : time_to_maturity  → 0
  MUST recompute: option_price (target)
  KEPT AS-IS    : all other features (log_moneyness, vol, btc_return, …)
                  These remain from the sampled real row.
                  This is a deliberate v1 simplification: we impose only
                  the boundary condition, not a full market re-simulation.

  Note: log_moneyness is defined as log(S/K), which does NOT depend on TTM,
        so keeping it from the original row is fully consistent.
        Volatility features at TTM=0 are approximate but intentionally
        retained as market-context indicators from the copied row.

Required metadata columns for payoff recomputation:
  - "underlying_price"
  - "strike"
  - "option_type"   (values "call" or "put", case-insensitive)

If any of these are missing from the train container's metadata dict,
augmentation raises a clear ValueError.
"""

import hashlib
import numpy as np
from typing import Dict, Any

from src.models.ann.dataset.container import PreparedTabularData


# ─── Payoff helper ────────────────────────────────────────────────────────────

def _btc_intrinsic_payoff(
    underlying: np.ndarray,
    strike: np.ndarray,
    option_type: np.ndarray,   # array of strings
) -> np.ndarray:
    """
    Compute the maturity intrinsic payoff in BTC (the current project convention).

    BTC-quoted inverse options:
        call payoff = max(S - K, 0) / S   [in BTC]
        put  payoff = max(K - S, 0) / S   [in BTC]

    Both clipped to >= 0.
    """
    S = underlying.astype(np.float64)
    K = strike.astype(np.float64)
    is_call = np.char.lower(option_type.astype(str)) == "call"

    payoff = np.where(
        is_call,
        np.maximum(S - K, 0.0) / S,
        np.maximum(K - S, 0.0) / S,
    )
    return payoff.astype(np.float32)


# ─── Core augmentation ────────────────────────────────────────────────────────

def augment_train_ttm_zero(
    train: PreparedTabularData,
    n_virtual: int,
    rng: np.random.Generator,
) -> PreparedTabularData:
    """
    Augment a single module's train split with TTM=0 virtual rows.

    Parameters
    ----------
    train      : PreparedTabularData  – the real train split (unchanged)
    n_virtual  : int                  – number of virtual rows to generate
    rng        : np.random.Generator  – seeded RNG for reproducibility

    Returns
    -------
    PreparedTabularData with virtual rows appended.
    """
    if n_virtual <= 0:
        return train

    # ── Validate required metadata ──────────────────────────────────────────
    required_meta = ["underlying_price", "strike", "option_type"]
    missing = [m for m in required_meta if m not in train.metadata]
    if missing:
        raise ValueError(
            f"[VirtualOptions] Cannot augment: metadata columns {missing} are required "
            f"for TTM=0 payoff recomputation but are missing from the train container. "
            f"Ensure they are listed in 'metadata_columns' in the config."
        )

    # ── Locate TTM feature index ─────────────────────────────────────────────
    if "time_to_maturity" not in train.feature_columns:
        raise ValueError(
            "[VirtualOptions] 'time_to_maturity' must be a feature column for TTM=0 augmentation."
        )
    ttm_idx = train.feature_columns.index("time_to_maturity")

    n_real = len(train)
    sample_idx = rng.integers(0, n_real, size=n_virtual)

    # ── Build virtual feature matrix ─────────────────────────────────────────
    virt_features = train.features[sample_idx].copy()          # (n_virtual, n_feat)
    virt_features[:, ttm_idx] = 0.0

    # ── Recompute targets ────────────────────────────────────────────────────
    S_samp = train.metadata["underlying_price"][sample_idx]
    K_samp = train.metadata["strike"][sample_idx]
    T_samp = train.metadata["option_type"][sample_idx]

    virt_targets = _btc_intrinsic_payoff(S_samp, K_samp, T_samp).reshape(-1, 1)

    # ── Copy metadata ────────────────────────────────────────────────────────
    virt_meta = {k: v[sample_idx].copy() for k, v in train.metadata.items()}

    # ── Concatenate ──────────────────────────────────────────────────────────
    aug_features = np.concatenate([train.features, virt_features], axis=0)
    aug_targets  = np.concatenate([train.targets,  virt_targets],  axis=0)
    aug_meta     = {k: np.concatenate([train.metadata[k], virt_meta[k]], axis=0)
                    for k in train.metadata}

    return PreparedTabularData(
        features=aug_features,
        targets=aug_targets,
        feature_columns=train.feature_columns,
        target_column=train.target_column,
        metadata=aug_meta,
        module_id=train.module_id,
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

def apply_virtual_options(
    module_split,             # ModuleSplit
    vo_config: Dict[str, Any],
    module_id: str,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """
    Apply virtual options augmentation to a single module's train split.

    Returns a diagnostics dict with before/after row counts.
    Modifies module_split.train in-place (replaces the reference).

    Parameters
    ----------
    module_split : ModuleSplit   – train/val/test containers for one module
    vo_config    : dict          – the "virtual_options" config block
    module_id    : str           – used for logging and seeding
    base_seed    : int           – combined with module hash for per-module RNG

    Config schema
    -------------
    {
        "enabled"     : true,
        "type"        : "ttm_zero",
        "count_mode"  : "ratio" | "fixed",
        "count_value" : 0.25,
        "random_seed" : 42
    }
    """
    diag = {
        "module_id": module_id,
        "n_real_train": len(module_split.train),
        "n_virtual_added": 0,
        "n_augmented_train": len(module_split.train),
        "augmentation_applied": False,
    }

    if not vo_config.get("enabled", False):
        return diag

    vo_type = vo_config.get("type", "ttm_zero")
    if vo_type != "ttm_zero":
        raise ValueError(f"[VirtualOptions] Unsupported type: '{vo_type}'. Only 'ttm_zero' is implemented.")

    count_mode  = vo_config.get("count_mode",  "ratio")
    count_value = vo_config.get("count_value", 0.25)
    seed        = vo_config.get("random_seed", base_seed)

    n_real = len(module_split.train)

    if count_mode == "ratio":
        n_virtual = int(round(n_real * float(count_value)))
    elif count_mode == "fixed":
        n_virtual = int(count_value)
    else:
        raise ValueError(f"[VirtualOptions] Unknown count_mode: '{count_mode}'. Use 'ratio' or 'fixed'.")

    if n_virtual <= 0:
        return diag

    # Per-module deterministic seed via hashlib (stable across Python processes)
    mod_hash = int(hashlib.md5(module_id.encode()).hexdigest(), 16) % 10_000
    mod_seed = seed + mod_hash
    rng = np.random.default_rng(mod_seed)

    module_split.train = augment_train_ttm_zero(module_split.train, n_virtual, rng)

    diag["n_virtual_added"]    = n_virtual
    diag["n_augmented_train"]  = len(module_split.train)
    diag["augmentation_applied"] = True
    return diag
