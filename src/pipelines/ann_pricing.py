from __future__ import annotations

from typing import Any, Dict

import torch

from src.models.ann.dataset.preprocessing import prepare_ann_dataframe
from src.models.ann.dataset.split_manager import ModularSplitManager
from src.models.ann.dataset.dataloaders import build_dataloader


class ANNDatasetPipeline:
    """Pipeline wrapper for Modular ANN dataset validation and initialization.

    This keeps the ANN dataset preparation flow aligned with the structure of
    other project pipelines: configuration is injected once through the
    constructor and execution happens through ``run()``.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def _resolve_dtype(self) -> torch.dtype:
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
        }
        dtype_str = self.config.get("dtype", "float32")
        return dtype_map.get(dtype_str, torch.float32)

    def run(self) -> None:
        print("Starting Modular PyTorch Dataset validation...")

        # 1. Preprocess source dataframe according to config
        df = prepare_ann_dataframe(self.config)

        features = self.config.get("feature_columns", [])
        target = self.config.get("target_column")
        metadata = self.config.get("metadata_columns", [])
        return_metadata = self.config.get("return_metadata", False)
        batch_size = self.config.get("batch_size", 1024)
        dtype = self._resolve_dtype()

        # 2. Chronological PyTorch Dataset Management
        split_manager = ModularSplitManager(
            df=df,
            feature_columns=features,
            target_column=target,
            module_columns=["mon_bin", "ttm_bin"],
            timestamp_column="timestamp",
            metadata_columns=metadata
        )

        print("\n[ Module Diagnostics ]")
        print(split_manager.get_diagnostics())

        # 3. Validation Sampling Output Shape Verification (taking first valid dataset)
        first_mod_id = list(split_manager.modules.keys())[0]
        first_module = split_manager.modules[first_mod_id]

        # Create datasets mapping specific module boundaries dynamically
        train_ds, val_ds, test_ds = first_module.as_datasets(return_metadata=return_metadata, dtype=dtype)

        print(f"\n[ Sample Shapes: {first_mod_id} Train Set ]")
        sample = train_ds[0]
        if return_metadata:
            print(
                f"Single sample shapes: "
                f"X={sample[0].shape}, "
                f"Y={sample[1].shape}, "
                f"Meta dict items={len(sample[2])}"
            )
        else:
            print(f"Single sample shapes: X={sample[0].shape}, Y={sample[1].shape}")

        # 4. Validate batch output structures natively iterating custom sequences 
        loader = build_dataloader(
            dataset=train_ds,
            batch_size=batch_size,
            shuffle=False,
        )

        batch = next(iter(loader))
        if return_metadata:
            print(
                "Single batch shapes: "
                f"X={batch[0].shape}, "
                f"Y={batch[1].shape}, "
                f"Meta batch length={len(batch[2])}"
            )
        else:
            print(f"Single batch shapes: X={batch[0].shape}, Y={batch[1].shape}")

        print("\nModular ANN Dataset successfully verified and initialized cleanly!")
