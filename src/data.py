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

def apply_edge_attack(
    data: GraphData,
    labels: torch.Tensor,
    fraction: float,
    seed: int,
    target_relations=None,
    attack_type: str = "random",
) -> GraphData:
    """Inject adversarial/spurious edges to test structural robustness.

    We add `fraction * |E|` extra directed edges with relation types drawn from
    `target_relations` (default: all relations). Node features, labels, and splits
    are untouched; only the graph structure is perturbed.

    attack_type:
      - "random" (default): edges between uniformly random node pairs, INDEPENDENT
        of labels. This simulates spurious/noisy adversarial edges (e.g. link
        farming, spam follows) and is the leakage-free setting used for evaluation:
        because the perturbation carries no label information, it cannot leak labels
        into a transductive model, and it directly stresses the edge-gating
        mechanism's ability to suppress uninformative edges.
      - "camouflage": edges between *different* classes. WARNING: in a transductive
        single-graph setting this LEAKS labels (cross-class-only edges plus training
        labels let message passing recover test labels via label propagation, giving
        spuriously near-perfect accuracy). Kept only for analysis/illustration; do
        NOT use it for robustness claims.
    """
    if not fraction or fraction <= 0:
        return data

    edge_index = data.edge_index
    edge_type = data.edge_type
    edge_weight = data.edge_weight
    num_edges = edge_index.size(1)
    n_inject = int(round(float(fraction) * num_edges))
    if n_inject <= 0:
        return data

    generator = torch.Generator().manual_seed(int(seed))
    labels = labels.view(-1)
    num_nodes = labels.size(0)
    num_relations = int(edge_type.max().item()) + 1
    if target_relations is None:
        target_relations = list(range(num_relations))
    target_relations = torch.tensor(target_relations, dtype=torch.long)

    if attack_type == "camouflage":
        # Cross-class edges (LEAKS labels in transductive settings; not for eval).
        collected_u, collected_v = [], []
        remaining = n_inject
        attempts = 0
        while remaining > 0 and attempts < 200:
            batch = max(remaining * 2, 1024)
            cu = torch.randint(num_nodes, (batch,), generator=generator)
            cv = torch.randint(num_nodes, (batch,), generator=generator)
            cross = labels[cu] != labels[cv]
            cu, cv = cu[cross], cv[cross]
            take = min(remaining, cu.numel())
            if take > 0:
                collected_u.append(cu[:take])
                collected_v.append(cv[:take])
                remaining -= take
            attempts += 1
        if not collected_u:
            return data
        u = torch.cat(collected_u)
        v = torch.cat(collected_v)
    else:
        # Leakage-free: uniformly random node pairs, independent of labels.
        u = torch.randint(num_nodes, (n_inject,), generator=generator)
        v = torch.randint(num_nodes, (n_inject,), generator=generator)
        keep = u != v  # drop self-loops
        u, v = u[keep], v[keep]

    rels = target_relations[torch.randint(len(target_relations), (u.numel(),), generator=generator)]

    new_edge_index = torch.cat([edge_index, torch.stack([u, v], dim=0)], dim=1)
    new_edge_type = torch.cat([edge_type, rels])
    new_edge_weight = torch.cat([edge_weight, torch.ones(u.numel(), dtype=edge_weight.dtype)])

    return GraphData(
        x=data.x,
        edge_index=new_edge_index,
        edge_type=new_edge_type,
        edge_weight=new_edge_weight,
        y_bot=data.y_bot,
        y_stance=data.y_stance,
    )


def label_distribution(y: torch.Tensor) -> Dict[int, int]:
    values, counts = torch.unique(y.cpu(), return_counts=True)
    return {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}
