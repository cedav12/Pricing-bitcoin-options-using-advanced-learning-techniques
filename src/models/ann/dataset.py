import torch
import pandas as pd
from typing import Optional, List, Tuple, Any

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
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.metadata_columns = metadata_columns if metadata_columns else []
        self.return_metadata = return_metadata
        
        # Build tensors in memory securely enforcing mapped matrices over indices
        features_np = dataframe[self.feature_columns].values
        target_np = dataframe[self.target_column].values
        
        self.features = torch.tensor(features_np, dtype=dtype)
        # Target representation standardized safely assuming explicit output scaling models mapping
        self.target = torch.tensor(target_np, dtype=dtype).unsqueeze(-1) 
        
        if len(self.features) != len(self.target):
            raise ValueError("Feature tensor and target tensor length mismatch")
            
        self.metadata = None
        if self.return_metadata and self.metadata_columns:
            # We store metadata as a dataframe/dict structure explicitly safely bypassing type cast conflicts
            self.metadata = dataframe[self.metadata_columns].to_dict(orient="records")
            
    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.target[idx]
        if self.return_metadata and self.metadata is not None:
            return x, y, self.metadata[idx]
        return x, y
