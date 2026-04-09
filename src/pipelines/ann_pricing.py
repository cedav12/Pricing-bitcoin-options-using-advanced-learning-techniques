from __future__ import annotations

from typing import Any, Dict

import torch

from src.models.ann.preprocessing import prepare_ann_dataframe
from src.models.ann.dataset import BitcoinOptionsDataset
from src.models.ann.dataloaders import build_dataloader


class ANNDatasetPipeline:
    """Pipeline wrapper for ANN dataset validation and initialization.

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
        print("Starting PyTorch ANN Dataset validation...")

        # 1. Preprocess source dataframe according to config
        df = prepare_ann_dataframe(self.config)

        features = self.config.get("feature_columns", [])
        target = self.config.get("target_column")
        metadata = self.config.get("metadata_columns", [])
        return_metadata = self.config.get("return_metadata", False)
        batch_size = self.config.get("batch_size", 1024)
        dtype = self._resolve_dtype()

        # 2. Build dataset instance
        dataset = BitcoinOptionsDataset(
            dataframe=df,
            feature_columns=features,
            target_column=target,
            metadata_columns=metadata,
            return_metadata=return_metadata,
            dtype=dtype,
        )

        print("\n[ Diagnostics ]")
        print(f"Total verified dataset rows: {len(dataset)}")
        print(f"Number of feature columns: {len(features)}")
        print(f"Target column: {target}")
        print(f"Metadata variables included: {'Yes' if return_metadata else 'No'}")

        # 3. Validate single sample output
        sample = dataset[0]
        if return_metadata:
            print(
                "Single sample shapes: "
                f"X={sample[0].shape}, "
                f"Y={sample[1].shape}, "
                f"Meta dict items={len(sample[2])}"
            )
        else:
            print(f"Single sample shapes: X={sample[0].shape}, Y={sample[1].shape}")

        # 4. Validate batch output
        loader = build_dataloader(
            dataset=dataset,
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

        print("\nANN Dataset successfully verified and initialized cleanly!")
