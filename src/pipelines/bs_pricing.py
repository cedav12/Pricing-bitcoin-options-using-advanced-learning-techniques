"""
Black-Scholes Pricing Pipeline
==============================
Generates theoretical model prices for the entire options dataset.
"""

import os
from typing import Optional
import numpy as np
import pandas as pd

from src.models.black_scholes import black_scholes_price

class BlackScholesPipeline:
    def __init__(self, dataset_path: str = "data/processed/options_dataset.csv"):
        self.dataset_path = dataset_path

    def run(
        self,
        vol_column: str = "rolling_std_24h",
        chunksize: int = 500_000,
        sample_size: Optional[int] = None,
    ):
        """
        Compute BS prices and save to predictions_bs.csv.
        This pipeline does NOT filter option types (Call/Put).
        """
        output_dir = os.path.dirname(self.dataset_path)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "predictions_bs.csv")
        
        print(f"\n[BS Pipeline] Starting prediction pipeline")
        print(f"[BS Pipeline] Input  = {self.dataset_path}")
        print(f"[BS Pipeline] Output = {output_path}")
        print(f"[BS Pipeline] Vol    = {vol_column}")

        if os.path.exists(output_path):
            os.remove(output_path)

        loaded_total = 0
        first_chunk = True

        try:
            reader = pd.read_csv(self.dataset_path, chunksize=chunksize, low_memory=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing dataset at {self.dataset_path}. Run build_dataset first.")

        for chunk in reader:
            if sample_size and (loaded_total >= sample_size):
                break

            # 1. Clean and Prepare
            if "risk_free_rate" not in chunk.columns:
                chunk["risk_free_rate"] = 0.0
            chunk["risk_free_rate"] = chunk["risk_free_rate"].fillna(0.0)

            # Drop missing essential values
            mask = (
                chunk["option_price"].notna()        & (chunk["option_price"]      > 0) &
                chunk["underlying_price"].notna()    & (chunk["underlying_price"]  > 0) &
                chunk["strike"].notna()              & (chunk["strike"]            > 0) &
                chunk["time_to_maturity"].notna()    & (chunk["time_to_maturity"]  > 0) &
                chunk[vol_column].notna()            & (chunk[vol_column]          > 0)
            )
            chunk = chunk.loc[mask].copy()
            
            if chunk.empty:
                continue
                
            # Compute log_moneyness if missing
            if "log_moneyness" not in chunk.columns:
                chunk["log_moneyness"] = np.log( chunk["underlying_price"] / chunk["strike"])

            # Compute Intrinsic Value
            is_call = chunk["option_type"].str.lower() == "call"
            intrinsic_usd = np.where(
                is_call,
                np.maximum(chunk["underlying_price"] - chunk["strike"], 0),
                np.maximum(chunk["strike"] - chunk["underlying_price"], 0)
            )
            # Save it explicitly
            chunk["intrinsic_value"] = intrinsic_usd / chunk["underlying_price"]
            
            # Compute Time Value
            chunk["time_value"] = chunk["option_price"] - chunk["intrinsic_value"]

            # 2. Compute BS Prices (BTC)
            bs_price_usd = black_scholes_price(
                S=chunk["underlying_price"],
                K=chunk["strike"],
                T=chunk["time_to_maturity"],
                r=chunk["risk_free_rate"],
                sigma=chunk[vol_column],
                option_type=chunk["option_type"],
            )
            model_price_btc = bs_price_usd / chunk["underlying_price"]

            # 3. Format Output Schema
            out_df = pd.DataFrame()
            out_df["timestamp"] = chunk["timestamp"]
            out_df["option_type"] = chunk["option_type"]
            out_df["strike"] = chunk["strike"]
            out_df["underlying_price"] = chunk["underlying_price"]
            out_df["time_to_maturity"] = chunk["time_to_maturity"]
            out_df["log_moneyness"] = chunk["log_moneyness"]
            out_df["time_value"] = chunk["time_value"]
            out_df["intrinsic_value"] = chunk["intrinsic_value"]
            
            out_df["market_price"] = chunk["option_price"]
            out_df["model_price"] = model_price_btc
            out_df["model_name"] = "black_scholes"
            out_df["pricing_error"] = out_df["model_price"] - out_df["market_price"]

            # 4. Incremental Save
            out_df.to_csv(output_path, mode='a', index=False, header=first_chunk)
            
            loaded_total += len(chunk)
            first_chunk = False
            print(f"  Processed {loaded_total:,} rows …", end="\r")
            
            if sample_size and (loaded_total >= sample_size):
                break

        if loaded_total > 0:
            print(f"\n[BS Pipeline] Completed. Saved {loaded_total:,} predictions to '{output_path}'.")
        else:
            print("\n[BS Pipeline] No valid rows to process.")
