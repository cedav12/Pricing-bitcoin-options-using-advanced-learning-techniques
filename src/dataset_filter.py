import os
import json
import operator
import pandas as pd
from typing import Dict, Any
from src.analysis.dataset_descriptives import compute_time_value, create_buckets

class DatasetFilterPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.input_path = config.get("input_path", "data/processed/options_dataset.csv")
        self.output_path = config.get("output_path", "data/processed/options_dataset_filtered_v2.csv")
        self.summary_output_dir = config.get("summary_output_dir", "output/filter_dataset")
        self.filters = config.get("filters", {})
        self.binning = config.get("binning", {})
        
        # Make directories robustly
        if self.output_path:
            dir_name = os.path.dirname(self.output_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
        if self.summary_output_dir:
            os.makedirs(self.summary_output_dir, exist_ok=True)

    def run(self):
        print(f"Loading data from {self.input_path}...")
        df = pd.read_csv(self.input_path)
        original_len = len(df)
        
        current_len = original_len
        summary_records = []
        
        metadata = {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "total_rows_loaded": original_len,
            "active_filters": self.filters,
            "time_value_computed": False,
            "binning_settings": self.binning,
            "actual_edges": {},
            "outside_bin_stats": {}
        }
        
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

        # 2. Trade Count Operator
        tc_op_str = self.filters.get("trade_count_operator")
        tc_thresh = self.filters.get("trade_count_threshold")
        ops = {
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "=": operator.eq
        }
        if tc_op_str and tc_thresh is not None and 'trade_count' in df.columns:
            if tc_op_str in ops:
                op_func = ops[tc_op_str]
                df = df[op_func(df['trade_count'], tc_thresh)]
                df = log_step(f"filter_trade_count_{tc_op_str}_{tc_thresh}", df)

        # 3. Time Value Filter
        remove_neg_tv = self.filters.get("remove_negative_time_value", False)
        tv_threshold = self.filters.get("time_value_threshold", 0.0)
        
        if remove_neg_tv:
            if 'time_value' not in df.columns:
                print("Computing time_value leveraging project utility...")
                try:
                    df['time_value'] = compute_time_value(df)
                    metadata["time_value_computed"] = True
                except Exception as e:
                    print(f"Warning: could not compute time_value. Missing columns? ({e})")
            
            if 'time_value' in df.columns:
                df = df[df['time_value'] >= tv_threshold]
                df = log_step(f"filter_time_value_ge_{tv_threshold}", df)

        # 4. Moneyness / TTM Range Filters
        mon_fil = self.filters.get("moneyness_filter", {})
        if mon_fil.get("enabled"):
            col = mon_fil.get("column", "log_moneyness")
            mini = mon_fil.get("min")
            maxi = mon_fil.get("max")
            if col in df.columns:
                if mini is not None:
                    df = df[df[col] >= mini]
                    df = log_step(f"filter_{col}_ge_{mini}", df)
                if maxi is not None:
                    df = df[df[col] <= maxi]
                    df = log_step(f"filter_{col}_le_{maxi}", df)

        ttm_fil = self.filters.get("ttm_filter", {})
        if ttm_fil.get("enabled"):
            col = ttm_fil.get("column", "time_to_maturity")
            mini = ttm_fil.get("min")
            maxi = ttm_fil.get("max")
            if col in df.columns:
                if mini is not None:
                    df = df[df[col] >= mini]
                    df = log_step(f"filter_{col}_ge_{mini}", df)
                if maxi is not None:
                    df = df[df[col] <= maxi]
                    df = log_step(f"filter_{col}_le_{maxi}", df)

        # 5. Bin Assignment
        def apply_binning(bin_config, key_name):
            nonlocal df
            if not bin_config.get("enabled"): return
            
            col = bin_config.get("column")
            if col not in df.columns: return
            
            method = bin_config.get("method", "explicit")
            edges = bin_config.get("edges")
            out_col = bin_config.get("output_column", f"{key_name}_bin")
            drop_outside = bin_config.get("drop_rows_outside_bins", False)
            
            labels, actual_edges = create_buckets(series=df[col], method=method, num_bins=3, edges=edges, format_labels=True)
            df[out_col] = labels
            
            metadata["actual_edges"][key_name] = actual_edges
            
            obs = df[col].notna().sum()
            binned = df[out_col].notna().sum()
            outside_ct = obs - binned
            
            metadata["outside_bin_stats"][key_name] = {
                "total_considered": int(obs),
                "assigned_to_bins": int(binned),
                "outside_range": int(outside_ct),
                "share_outside": float(outside_ct / obs if obs > 0 else 0)
            }
            
            df = log_step(f"bin_assignment_{key_name}", df)
            
            if drop_outside:
                df = df[df[out_col].notna()]
                df = log_step(f"drop_outside_bins_{key_name}", df)

        mon_bin = self.binning.get("moneyness", {})
        apply_binning(mon_bin, "moneyness")
        
        ttm_bin = self.binning.get("ttm", {})
        apply_binning(ttm_bin, "ttm")


        # 6. Save Final Dataset & Data
        print(f"Finished filtering: {original_len} -> {current_len} rows.")
        if self.output_path:
            print(f"Saving filtered dataset to {self.output_path}")
            df.to_csv(self.output_path, index=False)

        summary_df = pd.DataFrame(summary_records)
        if not summary_df.empty:
            summary_csv_path = os.path.join(self.summary_output_dir, "filtering_summary.csv")
            summary_df.to_csv(summary_csv_path, index=False)
            
        metadata["total_rows_after_all_filters"] = current_len
        meta_json_path = os.path.join(self.summary_output_dir, "metadata.json")
        with open(meta_json_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Saved metadata and summary to -> {self.summary_output_dir}")
