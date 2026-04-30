from dataclasses import dataclass

@dataclass
class Config:
    data_dir: str = "./data/MGTAB"
    task: str = "bot"  # "bot" or "stance"
    seed: int = 42

    hidden_dim: int = 128
    rel_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.3

    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 10
    patience: int = 25

    train_ratio: float = 0.7
    val_ratio: float = 0.1