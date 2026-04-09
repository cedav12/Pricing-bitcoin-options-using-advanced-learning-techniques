import torch
from torch.utils.data import DataLoader
from typing import Any

def default_collate_with_metadata(batch: list) -> tuple:
    """
    Custom collate function mapping abstract dictionary returns explicitly grouping inputs/targets independently.
    Default torch collate struggles structurally translating dict instances representing pure string mapping outputs.
    """
    has_meta = len(batch[0]) == 3
    
    if has_meta:
        xs, ys, metas = zip(*batch)
        return torch.stack(xs), torch.stack(ys), metas
    else:
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)

def build_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """
    Builds a customized PyTorch DataLoader for the internal option environments safely checking metadata representations.
    """
    has_meta = getattr(dataset, 'return_metadata', False)
    collate_fn = default_collate_with_metadata if has_meta else None
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
