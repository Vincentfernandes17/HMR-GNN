import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split

@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: torch.Tensor
    y_bot: torch.Tensor
    y_stance: torch.Tensor

def _load_tensor(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    return torch.tensor(obj)

def load_mgtab(data_dir: str) -> GraphData:
    paths = {
        "x": os.path.join(data_dir, "features.pt"),
        "edge_index": os.path.join(data_dir, "edge_index.pt"),
        "edge_type": os.path.join(data_dir, "edge_type.pt"),
        "edge_weight": os.path.join(data_dir, "edge_weight.pt"),
        "y_bot": os.path.join(data_dir, "labels_bot.pt"),
        "y_stance": os.path.join(data_dir, "labels_stance.pt"),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    x = _load_tensor(paths["x"]).float()
    edge_index = _load_tensor(paths["edge_index"]).long()
    edge_type = _load_tensor(paths["edge_type"]).long().view(-1)
    edge_weight = _load_tensor(paths["edge_weight"]).float().view(-1)
    y_bot = _load_tensor(paths["y_bot"]).long().view(-1)
    y_stance = _load_tensor(paths["y_stance"]).long().view(-1)

    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()

    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

    if edge_index.shape[1] != edge_type.numel():
        raise ValueError("edge_index and edge_type length mismatch")
    if edge_index.shape[1] != edge_weight.numel():
        raise ValueError("edge_index and edge_weight length mismatch")

    return GraphData(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        y_bot=y_bot,
        y_stance=y_stance,
    )

def remap_labels(y: torch.Tensor) -> torch.Tensor:
    uniq = sorted(torch.unique(y).cpu().tolist())
    mapping = {int(v): i for i, v in enumerate(uniq)}
    return torch.tensor([mapping[int(v)] for v in y.cpu().tolist()], dtype=torch.long)

def make_splits(y: torch.Tensor, seed: int, train_ratio: float, val_ratio: float) -> Dict[str, torch.Tensor]:
    idx = np.arange(len(y))
    y_np = y.cpu().numpy()

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=y_np,
    )

    rel_val = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - rel_val),
        random_state=seed,
        stratify=y_np[temp_idx],
    )

    def to_mask(indices):
        mask = torch.zeros(len(y), dtype=torch.bool)
        mask[torch.as_tensor(indices, dtype=torch.long)] = True
        return mask

    return {
        "train_mask": to_mask(train_idx),
        "val_mask": to_mask(val_idx),
        "test_mask": to_mask(test_idx),
    }