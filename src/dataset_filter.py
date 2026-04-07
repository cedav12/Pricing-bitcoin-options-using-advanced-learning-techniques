import os
import json
import pandas as pd
from typing import Dict, Any
from src.analysis.dataset_descriptives import compute_time_value

class DatasetFilterPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.input_path = config.get("input_path", "data/processed/options_dataset.csv")
        self.output_path = config.get("output_path", "data/processed/options_dataset_filtered_v2.csv")
        self.summary_output_dir = config.get("summary_output_dir", "output/filter_dataset")
        self.filters = config.get("filters", {})
        
        # Make directories
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        if self.summary_output_dir:
            os.makedirs(self.summary_output_dir, exist_ok=True)

    def run(self):
        print(f"Loading data from {self.input_path}...")
        df = pd.read_csv(self.input_path)
        original_len = len(df)
        
        current_len = original_len
        summary_records = []
        
        time_value_computed = False
        
        def log_step(step_name, new_df):
            nonlocal current_len
            new_len = len(new_df)
            removed = current_len - new_len
            
            summary_records.append({
                'step': step_name,
                'rows_before': current_len,
                'rows_after': new_len,
                'rows_removed': removed,
                'share_removed_from_previous': removed / current_len if current_len > 0 else 0,
                'share_removed_from_original': removed / original_len if original_len > 0 else 0
            })
            current_len = new_len
            return new_df

        # 1. Option Type Filter
        opt_type = self.filters.get("option_type")
        if opt_type and opt_type.lower() in ["call", "put"] and 'option_type' in df.columns:
            target = "C" if opt_type.lower() == "call" else "P"
            df = df[df['option_type'].isin([opt_type.lower(), target])]
            df = log_step(f"filter_option_type_{opt_type}", df)

        # 2. Minimum Trade Count Filter
        min_tc = self.filters.get("min_trade_count")
        if min_tc is not None and 'trade_count' in df.columns:
            df = df[df['trade_count'] >= min_tc]
            df = log_step(f"filter_min_trade_count_{min_tc}", df)

        # 3. Time Value Filter
        remove_neg_tv = self.filters.get("remove_negative_time_value", False)
        tv_threshold = self.filters.get("time_value_threshold", 0.0)
        
        if remove_neg_tv:
            if 'time_value' not in df.columns:
                print("Computing time_value leveraging project utility...")
                df['time_value'] = compute_time_value(df)
                time_value_computed = True
                
            df = df[df['time_value'] >= tv_threshold]
            df = log_step(f"filter_time_value_ge_{tv_threshold}", df)

        # 4. Save Final Dataset
        print(f"Finished filtering: {original_len} -> {current_len} rows.")
        if self.output_path:
            print(f"Saving filtered dataset to {self.output_path}")
            df.to_csv(self.output_path, index=False)

        # 5. Save Summary & Metadata
        summary_df = pd.DataFrame(summary_records)
        if not summary_df.empty:
            summary_csv_path = os.path.join(self.summary_output_dir, "filtering_summary.csv")
            summary_df.to_csv(summary_csv_path, index=False)
            
        metadata = {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "total_rows_loaded": original_len,
            "total_rows_after_all_filters": current_len,
            "active_filters": self.filters,
            "time_value_computed": time_value_computed
        }
        
        meta_json_path = os.path.join(self.summary_output_dir, "metadata.json")
        with open(meta_json_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Saved metadata and summary to -> {self.summary_output_dir}")
