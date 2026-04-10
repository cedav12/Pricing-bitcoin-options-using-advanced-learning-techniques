import torch
from torch.utils.data import DataLoader


def default_collate_with_metadata(batch: list) -> tuple:
    """
    Custom collate function handling outputs safely collating inputs mapping (features, targets, list(dicts)).
    """
    if len(batch[0]) == 3:
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
    Builds a PyTorch DataLoader directly mapping internally integrated collate implementations securely tracking attributes natively!
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
