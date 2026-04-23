import os
import json
import glob
import pandas as pd
import argparse

def aggregate_runs(base_dir="output/ann"):
    records = []
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return
        
    for run_name in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
            
        record = {"run_name": run_name}
        
        # Load training summary
        summary_path = os.path.join(run_dir, "run_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                record.update(summary)
                
        # Find latest prediction directory for this run
        pred_dirs = glob.glob(os.path.join(run_dir, "predict_*"))
        if pred_dirs:
            latest_pred = sorted(pred_dirs)[-1]
            # split indicator
            if "_val_" in os.path.basename(latest_pred):
                split_flag = "val"
            elif "_test_" in os.path.basename(latest_pred):
                split_flag = "test"
            else:
                split_flag = "eval"
                
            metrics_path = os.path.join(latest_pred, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    for k, v in metrics.items():
                        record[f"{split_flag}_{k}"] = round(v, 6)
                        
        records.append(record)
        
    if not records:
        print("No runs found to aggregate.")
        return
        
    df = pd.DataFrame(records)
    out_path = os.path.join(base_dir, "aggregated_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Successfully aggregated {len(df)} runs into {out_path}.")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregates ANN run artifacts into a singular CSV matrix.")
    parser.add_argument("--dir", type=str, default="output/ann", help="Base directory containing ANN runs")
    args = parser.parse_args()
    aggregate_runs(args.dir)
