import copy
import torch
import torch.nn as nn
import os

from config import Config
from data import load_mgtab, remap_labels, make_splits
from eval import evaluate
from model import HMRGNN
from utils import set_seed
from baselines import MLPBaseline, RGCNBaseline

def train(cfg=None, model_name="hmr", return_metrics=False):
    if cfg is None:
        cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_mgtab(cfg.data_dir)

    if cfg.task == "bot":
        y = remap_labels(data.y_bot)
    else:
        y = remap_labels(data.y_stance)

    num_classes = int(torch.unique(y).numel())
    masks = make_splits(y, cfg.seed, cfg.train_ratio, cfg.val_ratio)

    data = data.__class__(
        x=data.x.to(device),
        edge_index=data.edge_index.to(device),
        edge_type=data.edge_type.to(device),
        edge_weight=data.edge_weight.to(device),
        y_bot=data.y_bot.to(device),
        y_stance=data.y_stance.to(device),
    )
    y = y.to(device)

    train_counts = torch.bincount(y[masks["train_mask"]], minlength=num_classes).float()
    class_weights = train_counts.sum() / (num_classes * train_counts.clamp(min=1.0))
    class_weights = class_weights.to(device)

    # "mlp", "rgcn", "hmr"
    # model_name = "mlp"

    if model_name == "mlp":
        model = MLPBaseline(
            in_dim=data.x.size(1),
            hidden_dim=cfg.hidden_dim,
            num_classes=num_classes,
            dropout=cfg.dropout
        )

    elif model_name == "rgcn":
        model = RGCNBaseline(
            in_dim=data.x.size(1),
            hidden_dim=cfg.hidden_dim,
            num_classes=num_classes,
            num_relations=int(data.edge_type.max().item()) + 1,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout
        )

    else:
        model = HMRGNN(
            in_dim=data.x.size(1),
            hidden_dim=cfg.hidden_dim,
            num_classes=num_classes,
            num_relations=int(data.edge_type.max().item()) + 1,
            num_layers=cfg.num_layers,
            rel_dim=cfg.rel_dim,
            dropout=cfg.dropout,
        )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index, data.edge_type, data.edge_weight)
        loss = criterion(logits[masks["train_mask"]], y[masks["train_mask"]])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        metrics_val = evaluate(model, data, y, masks, criterion)
        val_f1 = metrics_val["f1_macro"]

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={loss.item():.4f} | "
            f"val_loss={metrics_val['loss']:.4f} | "
            f"val_acc={metrics_val['accuracy']:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= cfg.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics_test = evaluate(model, data, y, masks, criterion)

    print("\n=== TEST RESULTS ===")
    print(f"Best epoch: {best_epoch}")
    for k, v in metrics_test.items():
        print(f"{k}: {v:.4f}")
    
    if return_metrics:
        return {
            "accuracy": metrics_test["accuracy"],
            "f1_macro": metrics_test["f1_macro"]
        }

if __name__ == "__main__":
    print("Working dir:", os.getcwd())
    print("Check path:", os.path.exists("./data/MGTAB/features.pt"))
    train()