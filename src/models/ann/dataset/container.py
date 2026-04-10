import numpy as np
from typing import List, Dict

class PreparedTabularData:
    """
    Lightweight container for prepared tabular data matrices (features, target, metadata).
    """
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_columns: List[str],
        target_column: str,
        metadata: Dict[str, np.ndarray] = None,
        module_id: str = "global"
    ):
        n_samples = len(features)
        if n_samples != len(targets):
            raise ValueError(f"Features length ({n_samples}) and targets length ({len(targets)}) must match.")
            
        self.metadata = metadata if metadata is not None else {}
        for k, v in self.metadata.items():
            if len(v) != n_samples:
                raise ValueError(f"Metadata array '{k}' length ({len(v)}) does not match features length ({n_samples}).")
                
        self.features = features
        self.targets = targets
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.module_id = module_id
        
    def __len__(self) -> int:
        return len(self.features)

    @property
    def shape(self) -> tuple:
        return self.features.shape
