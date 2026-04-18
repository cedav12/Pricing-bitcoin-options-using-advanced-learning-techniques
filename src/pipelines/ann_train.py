import os
import json
import torch
import pandas as pd
from typing import Dict, Any

from src.models.ann.dataset.split_manager import ModularSplitManager
from src.models.ann.dataset.dataloaders import build_dataloader
from src.models.ann.dataset.scaling import FeatureScaler
from src.models.ann.models.mlp import create_model_from_config
from src.models.ann.training.trainer import Trainer

class ANNTrainPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_name = config.get("run_name", "ann_v1_modular")
        self.output_dir = os.path.join("output", "ann", self.run_name)
        
        self.input_path = config.get("input_path", "data/processed/options_dataset_filtered.csv")
        self.feature_cols = config.get("feature_columns", [])
        self.target_col = config.get("target_column", "option_price")
        self.time_col = config.get("timestamp_column", "timestamp")
        self.module_cols = config.get("module_columns", [])
        self.meta_cols = config.get("metadata_columns", [])
        self.min_module_size = config.get("min_module_size", 100)
        
        rt = config.get("runtime", {})
        self.seed = rt.get("seed", 42)
        self.threads = rt.get("threads", 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def run(self):
        # 1. Setup deterministic runtime
        import random
        import numpy as np
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        
        print(f"[RUN] ann_train | run_name={self.run_name} | seed={self.seed} | device={self.device.type}")
        
        # 2. Data load
        from src.models.ann.dataset.preprocessing import prepare_ann_dataframe
        print("Loading data via prepare_ann_dataframe...")
        df = prepare_ann_dataframe(self.config)
        
        sample_size = self.config.get("sample_size")
        if sample_size is not None and len(df) > sample_size:
            df = df.iloc[:sample_size]
            
        print(f"[DATA] rows={len(df)}")
        
        # 3. Create modules
        split_manager = ModularSplitManager(
            df=df,
            feature_columns=self.feature_cols,
            target_column=self.target_col,
            module_columns=self.module_cols,
            timestamp_column=self.time_col,
            metadata_columns=self.meta_cols,
            min_module_size=self.min_module_size
        )
        
        diagnostics = split_manager.get_diagnostics()
        active_count = len(diagnostics[diagnostics["status"] == "active"])
        skipped_count = len(diagnostics[diagnostics["status"] == "skipped"])
        print(f"[DATA] active_modules={active_count} | skipped_modules={skipped_count}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        # Save run config
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
            
        train_config = self.config.get("training", {})
        batch_size = train_config.get("batch_size", 256)
        
        # 4. Iterate over active modules
        for idx, (mod_id, module_split) in enumerate(split_manager.modules.items(), 1):
            mod_dir = os.path.join(self.output_dir, mod_id)
            os.makedirs(mod_dir, exist_ok=True)
            
            n_train = len(module_split.train.features)
            n_val = len(module_split.val.features)
            n_test = len(module_split.test.features)
            n_feat = len(self.feature_cols)
            print(f"[MODULE] {idx}/{active_count} | module={mod_id} | train={n_train} | val={n_val} | test={n_test} | features={n_feat}")
            
            # Scale
            if self.config.get("scaling", {}).get("enabled", True):
                scaler = FeatureScaler()
                # Fit strictly on train!
                scaler.fit(module_split.train.features)
                
                # Transform efficiently using the PyTorch dataset structures
                module_split.train.features = scaler.transform(module_split.train.features)
                module_split.val.features = scaler.transform(module_split.val.features)
                module_split.test.features = scaler.transform(module_split.test.features)
                
                scaler.save(os.path.join(mod_dir, "scaler.pkl"))
            
            # Build datasets
            train_ds, val_ds, test_ds = module_split.as_datasets(return_metadata=False)
            train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = build_dataloader(val_ds, batch_size=batch_size, shuffle=False)
            
            # Model
            model_config = self.config.get("model", {})
            model = create_model_from_config(n_feat, model_config)
            
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            hidden_str = str(model_config.get("hidden_dims", []))
            print(f"[MODEL] module={mod_id} | input_dim={n_feat} | hidden={hidden_str} | params={total_params}")
            
            # Train
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=train_config,
                device=self.device,
                module_id=mod_id
            )
            
            ckpt_path = os.path.join(mod_dir, "checkpoint.pt")
            history, _ = trainer.train(target_checkpoint_path=ckpt_path)
            
            with open(os.path.join(mod_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=4)
                
        # Save diagnostics
        diagnostics.to_csv(os.path.join(self.output_dir, "diagnostics.csv"), index=False)
        
        # Save run summary
        run_summary = {
            "run_name": self.run_name,
            "seed": self.seed,
            "device": self.device.type,
            "active_modules": active_count,
            "skipped_modules": skipped_count,
            "total_rows_processed": len(df),
            "scaling_enabled": self.config.get("scaling", {}).get("enabled", True)
        }
        with open(os.path.join(self.output_dir, "run_summary.json"), "w") as f:
            json.dump(run_summary, f, indent=4)
            
        print(f"[RUN] ann_train finished successfully.")
