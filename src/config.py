from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    data_dir: str = "./data/MGTAB"
    task: str = "bot"  # "bot" or "stance"
    seed: int = 42
    output_dir: str = "./results"

    hidden_dim: int = 128
    rel_dim: int = 128
    num_layers: int = 2
    num_heads: int = 1
    dropout: float = 0.5

    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 500
    patience: int = 50
    batch_size: int = 0  # 0 means full-batch graph training.

    early_stop_monitor: str = "val_loss"  # "val_loss" or "val_f1"

    train_ratio: float = 0.6
    val_ratio: float = 0.2

    use_class_weights: bool = True
    class_weight_power: float = 1.0

    gate_temperature: float = 1.0
    homophily_gate: bool = False
    homophily_alpha: float = 1.0
    separate_directions: bool = False
    log_gate_scores: bool = False
    gate_sample_edges: int = 50000

    seeds: Optional[List[int]] = None
    tune_trials: int = 12
    tune_seed: int = 2026
