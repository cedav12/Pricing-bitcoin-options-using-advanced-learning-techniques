import unittest
import pandas as pd
import numpy as np
import torch

from src.models.ann.dataset import BitcoinOptionsDataset
from src.models.ann.dataloaders import build_dataloader

class TestANNDataset(unittest.TestCase):
    def setUp(self):
        self.dummy_data = {
            "timestamp": ["2023-01-01 00:00:00", "2023-01-01 01:00:00", "2023-01-01 02:00:00"],
            "option_price": [100.0, 150.0, np.nan],
            "underlying_price": [15000.0, 16000.0, 17000.0],
            "time_to_maturity": [0.5, 0.4, 0.3],
            "log_moneyness": [-0.1, 0.0, 0.1],
            "volume": [10, 20, 30]
        }
        self.df = pd.DataFrame(self.dummy_data)
        
    def test_missing_column_failure(self):
        config = {
            "input_path": "dummy.csv", # Not used directly in this mock flow unless patching read_csv
            "feature_columns": ["time_to_maturity", "missing_col"],
            "target_column": "option_price"
        }
        # Instead of prepare_ann_dataframe which reads csv, we'll manually check the validator
        from src.models.ann.schema import validate_columns
        with self.assertRaises(ValueError):
            validate_columns(self.df, config["feature_columns"])
            
    def test_dataset_creation_and_shapes(self):
        # We manually build the valid dataframe representing prepare_ann_dataframe's output
        df_clean = self.df.dropna().copy()
        features = ["time_to_maturity", "log_moneyness", "volume"]
        target = "option_price"
        
        dataset = BitcoinOptionsDataset(
            dataframe=df_clean,
            feature_columns=features,
            target_column=target,
            return_metadata=False,
            dtype=torch.float32
        )
        
        self.assertEqual(len(dataset), 2)
        x, y = dataset[0]
        self.assertEqual(x.shape, (3,))
        self.assertEqual(y.shape, (1,))
        self.assertEqual(x.dtype, torch.float32)
        
    def test_metadata_return(self):
        df_clean = self.df.dropna().copy()
        
        dataset = BitcoinOptionsDataset(
            dataframe=df_clean,
            feature_columns=["time_to_maturity", "volume"],
            target_column="option_price",
            metadata_columns=["timestamp", "underlying_price"],
            return_metadata=True
        )
        
        x, y, meta = dataset[0]
        self.assertIn("timestamp", meta)
        self.assertIn("underlying_price", meta)
        self.assertEqual(meta["timestamp"], "2023-01-01 00:00:00")
        self.assertEqual(meta["underlying_price"], 15000.0)

    def test_dataloader_batching(self):
        df_clean = self.df.dropna().copy()
        dataset = BitcoinOptionsDataset(
            dataframe=df_clean,
            feature_columns=["time_to_maturity"],
            target_column="option_price",
            metadata_columns=["timestamp"],
            return_metadata=True
        )
        
        loader = build_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        # Batch should be (X_tensor, Y_tensor, Meta_tuple)
        self.assertEqual(len(batch), 3)
        self.assertEqual(batch[0].shape, (2, 1))
        self.assertEqual(batch[1].shape, (2, 1))
        self.assertIsInstance(batch[2], tuple)
        self.assertEqual(len(batch[2]), 2)
        self.assertEqual(batch[2][0]["timestamp"], "2023-01-01 00:00:00")

if __name__ == "__main__":
    unittest.main()
