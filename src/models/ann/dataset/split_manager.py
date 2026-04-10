import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.models.ann.dataset.container import PreparedTabularData
from src.models.ann.dataset.dataset import TabularDatasetWrapper

class ModuleSplit:
    """Stores train, validation, and test chronological containers for a single module."""
    def __init__(self, train: PreparedTabularData, val: PreparedTabularData, test: PreparedTabularData):
        self.train = train
        self.val = val
        self.test = test
        
    def as_datasets(self, return_metadata: bool = False, dtype=None) -> Tuple[TabularDatasetWrapper, ...]:
        import torch
        dtype = dtype or torch.float32
        return (
            TabularDatasetWrapper(self.train, return_metadata, dtype),
            TabularDatasetWrapper(self.val, return_metadata, dtype),
            TabularDatasetWrapper(self.test, return_metadata, dtype)
        )


class ModularSplitManager:
    """
    Groups data by module_columns, sorts by timestamp, and creates chronological 80/10/10 splits.
    Skips invalid modules and securely initializes ModuleSplit objects.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        module_columns: List[str],
        timestamp_column: str,
        metadata_columns: List[str] = None,
        min_module_size: int = 10
    ):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.module_columns = module_columns
        self.timestamp_column = timestamp_column
        self.metadata_columns = metadata_columns if metadata_columns is not None else []
        
        # min_module_size represents a minimal technical guard to ensure safe array indexing.
        # It is strictly not a recommended statistical or modeling threshold, which should be much higher.
        self.min_module_size = min_module_size
        
        self.skipped_modules = {}
        self.diagnostics_data = []

        all_req = feature_columns + [target_column, timestamp_column] + module_columns + self.metadata_columns
        missing = [c for c in set(all_req) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns inside df: {missing}")
            
        self.modules: Dict[str, ModuleSplit] = self._build_modules(df)
        
    def _build_modules(self, df: pd.DataFrame) -> Dict[str, ModuleSplit]:
        modules = {}
        
        if len(self.module_columns) > 0:
            cluster_groups = df.groupby(self.module_columns)
        else:
            cluster_groups = [("global", df)]
            
        for name, group in cluster_groups:
            if isinstance(name, tuple):
                mod_id = "_".join(str(v) for v in name)
            else:
                mod_id = str(name)
                
            sorted_group = group.sort_values(by=self.timestamp_column).reset_index(drop=True)
            n = len(sorted_group)
            
            if n < self.min_module_size:
                self.skipped_modules[mod_id] = {
                    "reason": f"Size {n} < min_module_size {self.min_module_size}",
                    "total_rows": n
                }
                continue
                
            # Chronological proportions: 80% Train, 10% Val, 10% Test
            idx_train = int(n * 0.8)
            idx_val = int(n * 0.9)
            
            df_train = sorted_group.iloc[:idx_train]
            df_val = sorted_group.iloc[idx_train:idx_val]
            df_test = sorted_group.iloc[idx_val:]
            
            # Explicitly validate non-empty splits
            if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
                self.skipped_modules[mod_id] = {
                    "reason": "Produces an empty train/val/test split.",
                    "total_rows": n
                }
                continue
                
            modules[mod_id] = ModuleSplit(
                train=self._to_container(df_train, mod_id),
                val=self._to_container(df_val, mod_id),
                test=self._to_container(df_test, mod_id)
            )
            
            # Store diagnostics
            self.diagnostics_data.append({
                "module_id": mod_id,
                "status": "active",
                "reason": "",
                "train_rows": len(df_train),
                "val_rows": len(df_val),
                "test_rows": len(df_test),
                "total_rows": n,
                "train_start_ts": df_train[self.timestamp_column].iloc[0],
                "train_end_ts": df_train[self.timestamp_column].iloc[-1],
                "val_start_ts": df_val[self.timestamp_column].iloc[0],
                "val_end_ts": df_val[self.timestamp_column].iloc[-1],
                "test_start_ts": df_test[self.timestamp_column].iloc[0],
                "test_end_ts": df_test[self.timestamp_column].iloc[-1],
            })
            
        for mod_id, result in self.skipped_modules.items():
            self.diagnostics_data.append({
                "module_id": mod_id,
                "status": "skipped",
                "reason": result["reason"],
                "train_rows": 0, "val_rows": 0, "test_rows": 0, "total_rows": result["total_rows"],
                "train_start_ts": None, "train_end_ts": None,
                "val_start_ts": None, "val_end_ts": None,
                "test_start_ts": None, "test_end_ts": None
            })
            
        return modules
        
    def _to_container(self, df: pd.DataFrame, mod_id: str) -> PreparedTabularData:
        features = df[self.feature_columns].to_numpy(copy=False, dtype=np.float32)
        targets = df[[self.target_column]].to_numpy(copy=False, dtype=np.float32)
        
        meta_dict = {}
        if self.timestamp_column not in self.metadata_columns:
            meta_dict[self.timestamp_column] = df[self.timestamp_column].to_numpy(copy=False)
            
        for c in self.metadata_columns:
            meta_dict[c] = df[c].to_numpy(copy=False)
            
        return PreparedTabularData(features, targets, self.feature_columns, self.target_column, meta_dict, mod_id)
        
    def get_diagnostics(self) -> pd.DataFrame:
        """Returns detailed diagnostics including split proportions and timestamp ranges."""
        return pd.DataFrame(self.diagnostics_data)
