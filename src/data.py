import os
from dataclasses import dataclass
from typing import Dict, Tuple

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
    remapped, _ = remap_labels_with_mapping(y)
    return remapped

def remap_labels_with_mapping(y: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, int]]:
    uniq = sorted(torch.unique(y).cpu().tolist())
    mapping = {int(v): i for i, v in enumerate(uniq)}
    remapped = torch.tensor([mapping[int(v)] for v in y.cpu().tolist()], dtype=torch.long)
    return remapped, mapping

def select_task_labels(data: GraphData, task: str) -> Tuple[torch.Tensor, Dict[int, int]]:
    if task == "bot":
        return remap_labels_with_mapping(data.y_bot)
    if task == "stance":
        return remap_labels_with_mapping(data.y_stance)
    raise ValueError(f"Unsupported task '{task}'. Expected 'bot' or 'stance'.")

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

def relation_homophily(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    labels: torch.Tensor,
    num_relations: int,
) -> torch.Tensor:
    """Return same-label edge ratio per relation for homophily-aware gating."""
    src, dst = edge_index.cpu()
    edge_type_cpu = edge_type.cpu().view(-1)
    labels_cpu = labels.cpu().view(-1)
    scores = torch.zeros(num_relations, dtype=torch.float)

    for rel in range(num_relations):
        mask = edge_type_cpu == rel
        if not mask.any():
            scores[rel] = 0.5
            continue
        same = labels_cpu[src[mask]] == labels_cpu[dst[mask]]
        scores[rel] = same.float().mean()

    return scores

def label_distribution(y: torch.Tensor) -> Dict[int, int]:
    values, counts = torch.unique(y.cpu(), return_counts=True)
    return {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}
