import torch
import pandas as pd
from typing import Optional, List


class BitcoinOptionsDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataframe: pd.DataFrame,
            feature_columns: List[str],
            target_column: str,
            metadata_columns: Optional[List[str]] = None,
            return_metadata: bool = False,
            dtype: torch.dtype = torch.float32,
    ):
        if not feature_columns:
            raise ValueError("Feature columns list must not be empty.")
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe.")

        self.feature_columns = feature_columns
        self.target_column = target_column
        self.metadata_columns = metadata_columns if metadata_columns else []
        self.return_metadata = return_metadata

        # Safety mappings defensively checking NaN availability explicitly tracking inputs
        if dataframe[self.feature_columns].isna().any().any():
            raise ValueError("NaN values found in feature columns. Please drop them before creating the dataset.")
        if dataframe[self.target_column].isna().any():
            raise ValueError("NaN values found in target column.")

        # Efficiently extract buffers supporting minimal cloning operations natively
        feat_array = dataframe[self.feature_columns].to_numpy(copy=False)
        # Using subset list to enforce 2D arrays naturally extracting single variables targeting modeling directly
        target_array = dataframe[[self.target_column]].to_numpy(copy=False)

        self.features = torch.from_numpy(feat_array).to(dtype)
        self.target = torch.from_numpy(target_array).to(dtype)

        if len(self.features) != len(self.target):
            raise ValueError("Feature tensor and target tensor length mismatch")

        self.metadata_df = None
        if self.return_metadata and self.metadata_columns:
            # Do NOT cast dynamically globally - load metadata reference preserving lightweight boundaries dynamically returning queries locally
            self.metadata_df = dataframe[self.metadata_columns]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.target[idx]

        if self.return_metadata and self.metadata_df is not None:
            # Cast dynamically extracting rows exclusively tracking indexes independently
            meta = self.metadata_df.iloc[idx].to_dict()
            return x, y, meta

        return x, y
