import torch
from torch.utils.data import Dataset
from src.models.ann.dataset.container import PreparedTabularData

class TabularDatasetWrapper(Dataset):
    """
    Generic PyTorch Dataset wrapper for PreparedTabularData.
    """
    def __init__(
        self,
        container: PreparedTabularData,
        return_metadata: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        self.feature_columns = container.feature_columns
        self.target_column = container.target_column
        self.return_metadata = return_metadata
        self.module_id = container.module_id
        self.metadata = container.metadata if return_metadata else None

        self.features = torch.from_numpy(container.features).to(dtype)
        self.targets = torch.from_numpy(container.targets).to(dtype)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.targets[idx]
        
        if self.return_metadata and self.metadata is not None:
            meta = {k: v[idx] for k, v in self.metadata.items()}
            return x, y, meta
            
        return x, y
