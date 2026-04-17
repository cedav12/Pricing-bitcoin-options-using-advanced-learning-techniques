import unittest
import pandas as pd
import numpy as np
import torch
import os
import tempfile
import shutil

from src.models.ann.dataset.container import PreparedTabularData
from src.models.ann.dataset.dataset import TabularDatasetWrapper
from src.models.ann.dataset.split_manager import ModularSplitManager
from src.models.ann.dataset.preprocessing import prepare_ann_dataframe
from src.models.ann.dataset.dataloaders import build_dataloader


class TestModularANNDatasetSynthetic(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.synthetic_data_path = os.path.join(self.test_dir, "synthetic.csv")
        self.df = pd.DataFrame({
            "timestamp": [1600000000, 1600003600, 1600007200, 1600010800, 1600014400, 1600018000, 1600021600, 1600025200, 1600028800, 1600032400],
            "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "target": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "mod_a": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A"],
            "mod_b": ["B", "B", "B", "B", "B", "B", "B", "B", "B", "B"],
            "meta_inf": ["info1"] * 10
        })
        self.df.to_csv(self.synthetic_data_path, index=False)
        self.synthetic_config = {
            "input_path": self.synthetic_data_path,
            "feature_columns": ["feature_1"],
            "target_column": "target",
            "timestamp_column": "timestamp",
            "module_columns": ["mod_a", "mod_b"],
            "metadata_columns": ["meta_inf"],
            "dtype": "float32"
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_synthetic_single_global_module(self):
        """Test functionality when there is precisely one single global module."""
        config = self.synthetic_config.copy()
        config["module_columns"] = []
        df = prepare_ann_dataframe(config)
        
        manager = ModularSplitManager(
            df=df,
            feature_columns=config["feature_columns"],
            target_column=config["target_column"],
            timestamp_column=config["timestamp_column"],
            module_columns=config["module_columns"],
            min_module_size=5
        )
        
        self.assertEqual(len(manager.modules), 1)
        self.assertIn("global", manager.modules)
        
        diag = manager.get_diagnostics()
        self.assertEqual(len(diag), 1)
        self.assertEqual(diag.iloc[0]["status"], "active")
        
        mod = manager.modules["global"]
        self.assertEqual(len(mod.train), 8)
        self.assertEqual(len(mod.val), 1)
        self.assertEqual(len(mod.test), 1)

    def test_synthetic_tiny_module_skipped(self):
        """Test that a module below the min_module_size threshold is skipped transparently."""
        df = prepare_ann_dataframe(self.synthetic_config)
        manager = ModularSplitManager(
            df=df,
            feature_columns=self.synthetic_config["feature_columns"],
            target_column=self.synthetic_config["target_column"],
            timestamp_column=self.synthetic_config["timestamp_column"],
            module_columns=self.synthetic_config["module_columns"],
            min_module_size=15
        )
        
        self.assertEqual(len(manager.modules), 0)
        diag = manager.get_diagnostics()
        self.assertEqual(len(diag), 1)
        self.assertEqual(diag.iloc[0]["status"], "skipped")
        self.assertIn("min_module_size", diag.iloc[0]["reason"])
        self.assertEqual(diag.iloc[0]["total_rows"], 10)

    def test_synthetic_missing_timestamps(self):
        """Test behavior if missing timestamps exist."""
        bad_ts_path = os.path.join(self.test_dir, "bad_ts.csv")
        pd.DataFrame({
            "timestamp": [1, np.nan, 3],
            "feature_1": [1.0, 2.0, 3.0],
            "target": [1, 2, 3]
        }).to_csv(bad_ts_path, index=False)
        
        config = {
            "input_path": bad_ts_path,
            "feature_columns": ["feature_1"],
            "target_column": "target",
            "timestamp_column": "timestamp",
        }
        
        with self.assertRaisesRegex(ValueError, "violates chronological splitting"):
            prepare_ann_dataframe(config)

    def test_synthetic_missing_feature_values(self):
        """Test preprocessing correctly handles and drops NaN feature values if configured."""
        bad_feat_path = os.path.join(self.test_dir, "bad_feat.csv")
        pd.DataFrame({
            "timestamp": [1, 2, 3],
            "feature_1": [1.0, np.nan, 3.0],
            "target": [1, 2, 3]
        }).to_csv(bad_feat_path, index=False)
        
        config = {
            "input_path": bad_feat_path,
            "feature_columns": ["feature_1"],
            "target_column": "target",
            "timestamp_column": "timestamp",
            "drop_na": True
        }
        df = prepare_ann_dataframe(config)
        self.assertEqual(len(df), 2)

    def test_synthetic_non_numeric_features(self):
        """Test preprocessing raises an error when feature values cannot be cast to numeric."""
        bad_type_path = os.path.join(self.test_dir, "bad_type.csv")
        pd.DataFrame({
            "timestamp": [1, 2],
            "feature_1": ["A", "B"], 
            "target": [1, 2]
        }).to_csv(bad_type_path, index=False)
        
        config = {
            "input_path": bad_type_path,
            "feature_columns": ["feature_1"],
            "target_column": "target",
            "timestamp_column": "timestamp",
        }
        with self.assertRaisesRegex(TypeError, "Failed to cast"):
            prepare_ann_dataframe(config)

    def test_synthetic_empty_split_module(self):
        """Test a module that is large enough but results in an empty validation or test split due to small dataset side-effects."""
        empty_split_path = os.path.join(self.test_dir, "empty_split.csv")
        pd.DataFrame({
            "timestamp": [1, 2, 3, 4], # Only 4 rows
            "feature_1": [1.0, 2.0, 3.0, 4.0],
            "target": [1, 2, 3, 4],
            "mod_a": ["A", "A", "A", "A"]
        }).to_csv(empty_split_path, index=False)
        
        config = {
            "input_path": empty_split_path,
            "feature_columns": ["feature_1"],
            "target_column": "target",
            "timestamp_column": "timestamp",
            "module_columns": ["mod_a"]
        }
        df = prepare_ann_dataframe(config)
        # Even though min size is 4, 10% of 4 is 0.4 -> 0 length test/val
        manager = ModularSplitManager(
            df=df,
            feature_columns=config["feature_columns"],
            target_column=config["target_column"],
            timestamp_column=config["timestamp_column"],
            module_columns=config["module_columns"],
            min_module_size=4
        )
        self.assertEqual(len(manager.modules), 0)
        diag = manager.get_diagnostics()
        self.assertEqual(diag.iloc[0]["status"], "skipped")
        self.assertIn("empty train/val/test split", diag.iloc[0]["reason"])


class TestModularANNDatasetRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = "data/processed/options_dataset_filtered_v2.csv"
        if not os.path.exists(path):
            alt_path = "data/processed/options_dataset_filtered.csv"
            if os.path.exists(alt_path):
                path = alt_path
                
        cls.config = {
            "input_path": path,
            "feature_columns": ["time_to_maturity", "underlying_price", "strike"],
            "target_column": "option_price",
            "timestamp_column": "timestamp",
            "module_columns": [], 
            "metadata_columns": ["option_type"],
            "dtype": "float32",
            "drop_na": True
        }
        
        if os.path.exists(cls.config["input_path"]):
            sample = pd.read_csv(cls.config["input_path"], nrows=5)
            mods = []
            if "mon_bin" in sample.columns: mods.append("mon_bin")
            if "ttm_bin" in sample.columns: mods.append("ttm_bin")
            
            cls.config["module_columns"] = mods
            cls.df = prepare_ann_dataframe(cls.config)
            
            cls.manager = ModularSplitManager(
                df=cls.df,
                feature_columns=cls.config["feature_columns"],
                target_column=cls.config["target_column"],
                timestamp_column=cls.config["timestamp_column"],
                module_columns=cls.config["module_columns"],
                metadata_columns=cls.config["metadata_columns"],
                min_module_size=100
            )
        else:
            cls.df = None
            cls.manager = None

    def test_real_data_structural_evaluation(self):
        if self.manager is None:
            self.skipTest("Real dataset not found.")
            
        diag = self.manager.get_diagnostics()
        self.assertGreater(len(diag), 0, "No modules evaluated.")
        
        active_count = (diag["status"] == "active").sum()
        self.assertEqual(active_count, len(self.manager.modules), "Diagnostics active count doesn't match modules dictionary")
        
        if active_count > 0:
            active_diag = diag[diag["status"] == "active"].iloc[0]
            self.assertGreater(active_diag["train_rows"], 0)
            self.assertGreater(active_diag["val_rows"], 0)
            self.assertGreater(active_diag["test_rows"], 0)

    def test_real_data_chronological_split_correctness(self):
        if self.manager is None:
            self.skipTest("Real dataset not found.")
        
        if len(self.manager.modules) == 0:
            self.skipTest("No active modules to test chronological split correctness.")
            
        for mod_name, mod in list(self.manager.modules.items())[:3]: # test first 3 for speed
            total = len(mod.train) + len(mod.val) + len(mod.test)
            self.assertLessEqual(abs(len(mod.train) - 0.8 * total), max(3, 0.05 * total), "Train split size is not ~80%")
            self.assertLessEqual(abs(len(mod.val) - 0.1 * total), max(3, 0.05 * total), "Val split size is not ~10%")
            
            max_train_ts = mod.train.metadata["timestamp"].max()
            min_val_ts   = mod.val.metadata["timestamp"].min()
            self.assertLessEqual(max_train_ts, min_val_ts, f"Leakage detected in {mod_name}")
            
            max_val_ts   = mod.val.metadata["timestamp"].max()
            min_test_ts  = mod.test.metadata["timestamp"].min()
            self.assertLessEqual(max_val_ts, min_test_ts, f"Leakage detected in {mod_name}")

            train_ts = mod.train.metadata["timestamp"]
            is_sorted = np.all(train_ts[:-1] <= train_ts[1:])
            self.assertTrue(is_sorted, "Train timestamps not ascending")

    def test_real_data_pytorch_dataloader_correctness(self):
        if self.manager is None:
            self.skipTest("Real dataset not found.")
        
        if len(self.manager.modules) == 0:
            self.skipTest("No active modules to test PyTorch dataloaders.")
            
        first_mod_key = list(self.manager.modules.keys())[0]
        mod = self.manager.modules[first_mod_key]
        
        train_ds, val_ds, test_ds = mod.as_datasets(return_metadata=True)
        feat_count = len(self.config["feature_columns"])
        
        self.assertEqual(len(train_ds), len(mod.train))
        
        x, y, m = train_ds[0]
        self.assertEqual(x.shape, (feat_count,))
        self.assertEqual(y.shape, (1,))
        self.assertIn("timestamp", m)
        
        loader = build_dataloader(train_ds, batch_size=32, shuffle=True)
        batch = next(iter(loader))
        
        bx, by, bm = batch
        batch_actual = bx.size(0)
        self.assertEqual(bx.shape, (batch_actual, feat_count))
        self.assertEqual(by.shape, (batch_actual, 1))
        
        self.assertIsInstance(bm, tuple)
        self.assertIn("timestamp", bm[0])

    def test_real_data_diagnostics_quality(self):
        if self.manager is None:
            self.skipTest("Real dataset not found.")
            
        diag = self.manager.get_diagnostics()
        self.assertIn("module_id", diag.columns)
        self.assertIn("status", diag.columns)
        self.assertIn("total_rows", diag.columns)
        
        skipped = diag[diag["status"] == "skipped"]
        for _, row in skipped.iterrows():
            self.assertTrue(pd.notna(row["reason"]))
            self.assertTrue(pd.isna(row["train_start_ts"]) if "train_start_ts" in row else True)
            self.assertEqual(row["train_rows"], 0)
            
        active = diag[diag["status"] == "active"]
        for _, row in active.iterrows():
            reason_empty = pd.isna(row["reason"]) or str(row["reason"]) == "" or str(row["reason"]) == "nan"
            self.assertTrue(reason_empty)
            self.assertTrue(pd.notna(row["train_start_ts"]) if "train_start_ts" in row else True)
            self.assertEqual(row["total_rows"], row["train_rows"] + row["val_rows"] + row["test_rows"])


if __name__ == "__main__":
    unittest.main()
