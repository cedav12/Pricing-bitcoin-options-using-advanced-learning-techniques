import unittest
import pandas as pd
import numpy as np
import torch

from src.models.ann.dataset.container import PreparedTabularData
from src.models.ann.dataset.dataset import TabularDatasetWrapper
from src.models.ann.dataset.split_manager import ModularSplitManager
from src.models.ann.dataset.dataloaders import build_dataloader

class TestModularANNDataset(unittest.TestCase):
    def setUp(self):
        self.dummy_data = {
            "timestamp": ["2023-01-01 00:00:00", "2023-01-01 01:00:00", "2023-01-01 02:00:00", "2023-01-01 03:00:00"],
            "option_price": [100.0, 150.0, 120.0, 110.0],
            "underlying_price": [15000.0, 16000.0, 17000.0, 18000.0],
            "time_to_maturity": [0.5, 0.4, 0.3, 0.2],
            "log_moneyness": [-0.1, 0.0, 0.1, 0.0],
            "mon_bin": ["ATM", "ATM", "OTM", "OTM"],
            "ttm_bin": ["1M", "1M", "3M", "3M"]
        }
        self.df = pd.DataFrame(self.dummy_data)
        
    def test_missing_column_failure(self):
        with self.assertRaises(ValueError):
            # should fail because missing column `missing_col`
            manager = ModularSplitManager(
                df=self.df,
                feature_columns=["time_to_maturity", "missing_col"],
                target_column="option_price",
                timestamp_column="timestamp",
                module_columns=["mon_bin", "ttm_bin"]
            )
            
    def test_dataset_wrapper_creation_and_shapes(self):
        features = np.array([[0.5, -0.1], [0.4, 0.0]], dtype=np.float32)
        targets = np.array([[100.0], [150.0]], dtype=np.float32)
        
        container = PreparedTabularData(
            features=features,
            targets=targets,
            feature_columns=["time_to_maturity", "log_moneyness"],
            target_column="option_price",
            metadata={"timestamp": np.array(["2023-01-01", "2023-01-02"])}
        )
        
        dataset = TabularDatasetWrapper(
            container=container,
            return_metadata=False,
            dtype=torch.float32
        )
        
        self.assertEqual(len(dataset), 2)
        x, y = dataset[0]
        self.assertEqual(x.shape, (2,))
        self.assertEqual(y.shape, (1,))
        self.assertEqual(x.dtype, torch.float32)
        
    def test_split_manager_routing_and_metadata(self):
        df_clean = self.df.dropna().copy()
        
        manager = ModularSplitManager(
            df=df_clean,
            feature_columns=["time_to_maturity"],
            target_column="option_price",
            module_columns=["mon_bin", "ttm_bin"],
            timestamp_column="timestamp",
            metadata_columns=["underlying_price"]
        )
        
        # We expect two modules: ('ATM', '1M') and ('OTM', '3M')
        self.assertEqual(len(manager.modules), 2)
        
        atm_mod = manager.modules["ATM_1M"]
        self.assertEqual(len(atm_mod.train), 1)  # 2 rows total -> 80% is 1 row
        
        train_ds, val_ds, test_ds = atm_mod.as_datasets(return_metadata=True)
        x, y, meta = train_ds[0]
        
        self.assertIn("timestamp", meta)
        self.assertIn("underlying_price", meta)
        self.assertEqual(meta["timestamp"], "2023-01-01 00:00:00")
        self.assertEqual(meta["underlying_price"], 15000.0)

    def test_dataloader_batching(self):
        features = np.array([[0.5], [0.4]], dtype=np.float32)
        targets = np.array([[100.0], [150.0]], dtype=np.float32)
        
        container = PreparedTabularData(
            features=features,
            targets=targets,
            feature_columns=["time_to_maturity"],
            target_column="option_price",
            metadata={"timestamp": np.array(["2023-01-01", "2023-01-02"])}
        )
        
        dataset = TabularDatasetWrapper(
            container=container,
            return_metadata=True,
            dtype=torch.float32
        )
        
        loader = build_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        self.assertEqual(len(batch), 3)
        self.assertEqual(batch[0].shape, (2, 1))
        self.assertEqual(batch[1].shape, (2, 1))
        self.assertIsInstance(batch[2], tuple)
        self.assertEqual(len(batch[2]), 2)
        self.assertEqual(batch[2][0]["timestamp"], "2023-01-01")

if __name__ == "__main__":
    unittest.main()
