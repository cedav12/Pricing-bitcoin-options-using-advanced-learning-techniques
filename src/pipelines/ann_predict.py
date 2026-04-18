import os
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

from src.models.ann.dataset.split_manager import ModularSplitManager
from src.models.ann.dataset.dataloaders import build_dataloader
from src.models.ann.dataset.scaling import FeatureScaler
from src.models.ann.models.mlp import create_model_from_config
from src.models.ann.training.metrics import compute_regression_metrics

class ANNPredictPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Original run directory
        self.run_dir = config.get("run_dir", "")
        # Can also supply run_name instead of direct run_dir
        if not self.run_dir:
            run_name = config.get("run_name", "ann_v1_modular")
            self.run_dir = os.path.join("output", "ann", run_name)
            
        if not os.path.exists(self.run_dir):
            raise ValueError(f"Run directory {self.run_dir} does not exist.")
            
        self.split = config.get("prediction", {}).get("split", "test").lower()
        if self.split not in ["val", "test"]:
            raise ValueError("Config prediction layer -> 'split' must be 'val' or 'test'.")
            
        self.batch_size = config.get("prediction", {}).get("batch_size", 512)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.run_dir, f"predict_{self.split}_{timestamp}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        # 1. Load run config
        run_config_path = os.path.join(self.run_dir, "config.json")
        if not os.path.exists(run_config_path):
            raise ValueError(f"Missing config.json in {self.run_dir}")
            
        with open(run_config_path, "r") as f:
            run_config = json.load(f)
            
        input_path = run_config.get("input_path", "data/processed/options_dataset_filtered.csv")
        feature_cols = run_config.get("feature_columns", [])
        target_col = run_config.get("target_column", "option_price")
        time_col = run_config.get("timestamp_column", "timestamp")
        module_cols = run_config.get("module_columns", [])
        meta_cols = run_config.get("metadata_columns", [])
        min_module_size = run_config.get("min_module_size", 100)
        
        print(f"[PREDICT] split={self.split} | run_dir={self.run_dir} | device={self.device.type}")
        
        # 2. Load data
        from src.models.ann.dataset.preprocessing import prepare_ann_dataframe
        print("Loading original data via prepare_ann_dataframe to reconstruct splits...")
        
        # Enforce sample size parity from run_config
        sample_size = run_config.get("sample_size")
        
        df = prepare_ann_dataframe(run_config)
        
        if sample_size is not None and len(df) > sample_size:
            df = df.iloc[:sample_size]
            
        # 3. Create modules (must be identical to training)
        split_manager = ModularSplitManager(
            df=df,
            feature_columns=feature_cols,
            target_column=target_col,
            module_columns=module_cols,
            timestamp_column=time_col,
            metadata_columns=meta_cols,
            min_module_size=min_module_size
        )
        
        scaling_enabled = run_config.get("scaling", {}).get("enabled", True)
        
        all_preds = []
        module_metrics_list = []
        n_feat = len(feature_cols)
        
        os.makedirs(self.output_dir, exist_ok=True)
        active_count = len([m for m in split_manager.modules])
        print(f"[PREDICT] active_modules={active_count}")
        
        # 4. Predict
        used_modules = 0
        missing_modules = 0
        
        for mod_id, module_split in split_manager.modules.items():
            mod_dir = os.path.join(self.run_dir, mod_id)
            if not os.path.exists(mod_dir):
                print(f"[PREDICT_WARNING] Skipping module {mod_id}: Directory not found.")
                missing_modules += 1
                continue
                
            # Pick the split
            target_data = module_split.val if self.split == "val" else module_split.test
            
            n_rows = len(target_data.features)
            # Skip if split is empty
            if n_rows == 0:
                print(f"[PREDICT_WARNING] Skipping module {mod_id}: Zero evaluation rows.")
                missing_modules += 1
                continue
                
            print(f"[MODULE] {mod_id} | rows={n_rows}")
            
            # Load scaler
            if scaling_enabled:
                scaler_path = os.path.join(mod_dir, "scaler.pkl")
                if os.path.exists(scaler_path):
                    scaler = FeatureScaler.load(scaler_path)
                    target_data.features = scaler.transform(target_data.features)
                    
            # Dataset & DataLoader
            import torch.utils.data
            from src.models.ann.dataset.dataset import TabularDatasetWrapper
            ds = TabularDatasetWrapper(target_data, return_metadata=True, dtype=torch.float32)
            loader = build_dataloader(ds, batch_size=self.batch_size, shuffle=False)
            
            # Load model
            model_config = run_config.get("model", {})
            model = create_model_from_config(n_feat, model_config)
            
            ckpt_path = os.path.join(mod_dir, "checkpoint.pt")
            if not os.path.exists(ckpt_path):
                print(f"[PREDICT_WARNING] Skipping module {mod_id}: Checkpoint missing.")
                missing_modules += 1
                continue
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
            model.to(self.device).eval()
            
            used_modules += 1
            
            # Inference
            y_trues = []
            y_preds = []
            
            # Pre-collect metadata mapping
            flat_metas = {k: [] for k in target_data.metadata.keys()}
            
            with torch.inference_mode():
                for batch in loader:
                    x, y, metas = batch
                    x, y = x.to(self.device), y.to(self.device)
                    ypred = model(x)
                    
                    y_trues.append(y.cpu().numpy())
                    y_preds.append(ypred.cpu().numpy())
                    
                    for m in metas:
                        for k in flat_metas.keys():
                            flat_metas[k].append(m[k])
                        
            y_trues = np.concatenate(y_trues)
            y_preds = np.concatenate(y_preds)
            
            # Module metrics
            metrics = compute_regression_metrics(y_trues, y_preds)
            metrics["module_id"] = mod_id
            metrics["rows"] = n_rows
            module_metrics_list.append(metrics)
            
            # Accumulate predictions
            for k in flat_metas.keys():
                flat_metas[k] = np.array(flat_metas[k])
                
            preds_df = pd.DataFrame(flat_metas)
            preds_df["module_id"] = mod_id
            preds_df["actual_price"] = y_trues.flatten()
            preds_df["predicted_price"] = y_preds.flatten()
            all_preds.append(preds_df)

        print(f"[PREDICT] Predictions generated for {used_modules} modules. ({missing_modules} modules skipped).")
        
        # 5. Export
        if not all_preds:
            print("[PREDICT] No predictions generated. Were any models successfully trained?")
            return
            
        final_preds_df = pd.concat(all_preds, ignore_index=True)
        out_csv = os.path.join(self.output_dir, "predictions.csv")
        final_preds_df.to_csv(out_csv, index=False)
        print(f"[EXPORT] predictions={out_csv}")
        
        # Summary metrics
        overall_metrics = compute_regression_metrics(
            final_preds_df["actual_price"].values, 
            final_preds_df["predicted_price"].values
        )
        print(f"[METRICS] mae={overall_metrics['mae']:.5f} | rmse={overall_metrics['rmse']:.5f} | bias={overall_metrics['bias']:.5f}")
        
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(overall_metrics, f, indent=4)
            
        module_metrics_df = pd.DataFrame(module_metrics_list)
        module_metrics_df.to_csv(os.path.join(self.output_dir, "metrics_module.csv"), index=False)
