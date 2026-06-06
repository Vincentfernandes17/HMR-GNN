import copy
import os
import time
from dataclasses import asdict

import torch
import torch.nn as nn

from baselines import (
    DirectionalRGCNBaseline,
    GCNBaseline,
    GraphSAGEBaseline,
    MLPBaseline,
    RGCNBaseline,
)
from config import Config
from data import (
    GraphData,
    label_distribution,
    load_mgtab,
    make_splits,
    relation_homophily,
    select_task_labels,
)
from eval import evaluate
from model import HMRGNN
from reporting import ensure_dir, write_csv, write_json
from utils import set_seed


def _to_device(data: GraphData, device) -> GraphData:
    return GraphData(
        x=data.x.to(device),
        edge_index=data.edge_index.to(device),
        edge_type=data.edge_type.to(device),
        edge_weight=data.edge_weight.to(device),
        y_bot=data.y_bot.to(device),
        y_stance=data.y_stance.to(device),
    )


def _model_flags(model_name, cfg):
    use_homophily_gate = cfg.homophily_gate
    separate_directions = cfg.separate_directions

    if model_name == "hmr":
        use_homophily_gate = False
        separate_directions = False
    elif model_name == "hmr_homophily":
        use_homophily_gate = True
        separate_directions = False
    elif model_name == "hmr_directional":
        use_homophily_gate = False
        separate_directions = True
    elif model_name == "hmr_full":
        use_homophily_gate = True
        separate_directions = True

    return use_homophily_gate, separate_directions


def build_model(model_name, cfg, in_dim, num_classes, num_relations):
    if model_name == "mlp":
        return MLPBaseline(in_dim, cfg.hidden_dim, num_classes, cfg.dropout)
    if model_name == "gcn":
        return GCNBaseline(in_dim, cfg.hidden_dim, num_classes, cfg.num_layers, cfg.dropout)
    if model_name == "graphsage":
        return GraphSAGEBaseline(in_dim, cfg.hidden_dim, num_classes, cfg.num_layers, cfg.dropout)
    if model_name == "rgcn":
        return RGCNBaseline(in_dim, cfg.hidden_dim, num_classes, num_relations, cfg.num_layers, cfg.dropout)
    if model_name == "dir_rgcn":
        return DirectionalRGCNBaseline(
            in_dim,
            cfg.hidden_dim,
            num_classes,
            num_relations,
            cfg.num_layers,
            cfg.dropout,
        )
    if model_name in {"hmr", "hmr_homophily", "hmr_directional", "hmr_full"}:
        use_homophily_gate, separate_directions = _model_flags(model_name, cfg)
        return HMRGNN(
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=num_classes,
            num_relations=num_relations,
            num_layers=cfg.num_layers,
            rel_dim=cfg.rel_dim,
            dropout=cfg.dropout,
            num_heads=cfg.num_heads,
            gate_temperature=cfg.gate_temperature,
            use_homophily_gate=use_homophily_gate,
            homophily_alpha=cfg.homophily_alpha,
            separate_directions=separate_directions,
            log_gate_scores=cfg.log_gate_scores,
        )
    raise ValueError(f"Unknown model_name '{model_name}'")


def _class_weights(y, train_mask, num_classes, cfg, device):
    if not cfg.use_class_weights:
        return None
    counts = torch.bincount(y[train_mask], minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp(min=1.0))
    if cfg.class_weight_power != 1.0:
        weights = weights.pow(cfg.class_weight_power)
    return weights.to(device)


def _run_dir(cfg, model_name):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return ensure_dir(os.path.join(cfg.output_dir, cfg.task, model_name, f"seed_{cfg.seed}_{stamp}"))


def train(
    cfg=None,
    model_name="hmr",
    return_metrics=False,
    save_outputs=True,
    output_dir=None,
    verbose=True,
):
    if cfg is None:
        cfg = Config()
    if output_dir is not None:
        cfg.output_dir = output_dir
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data = load_mgtab(cfg.data_dir)
    y_cpu, label_mapping = select_task_labels(raw_data, cfg.task)
    num_classes = int(torch.unique(y_cpu).numel())
    num_relations = int(raw_data.edge_type.max().item()) + 1
    masks_cpu = make_splits(y_cpu, cfg.seed, cfg.train_ratio, cfg.val_ratio)
    rel_homophily_cpu = relation_homophily(raw_data.edge_index, raw_data.edge_type, y_cpu, num_relations)

    data = _to_device(raw_data, device)
    y = y_cpu.to(device)
    masks = {name: mask.to(device) for name, mask in masks_cpu.items()}
    rel_homophily = rel_homophily_cpu.to(device)
    class_names = {mapped: str(original) for original, mapped in label_mapping.items()}
    groups = {
        "stance": raw_data.y_stance.cpu().numpy(),
        "bot": raw_data.y_bot.cpu().numpy(),
    }

    model = build_model(
        model_name=model_name,
        cfg=cfg,
        in_dim=data.x.size(1),
        num_classes=num_classes,
        num_relations=num_relations,
    ).to(device)

    weights = _class_weights(y, masks["train_mask"], num_classes, cfg, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    patience_ctr = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        try:
            logits = model(data.x, data.edge_index, data.edge_type, data.edge_weight, rel_homophily)
        except TypeError:
            logits = model(data.x, data.edge_index, data.edge_type, data.edge_weight)

        loss = criterion(logits[masks["train_mask"]], y[masks["train_mask"]])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        metrics_val = evaluate(
            model,
            data,
            y,
            masks,
            criterion,
            split="val",
            relation_homophily=rel_homophily,
            class_names=class_names,
        )
        val_f1 = metrics_val["f1_macro"]
        row = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "val_loss": metrics_val["loss"],
            "val_accuracy": metrics_val["accuracy"],
            "val_f1_macro": val_f1,
        }
        history.append(row)

        if verbose:
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
            if verbose:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics_test = evaluate(
        model,
        data,
        y,
        masks,
        criterion,
        split="test",
        relation_homophily=rel_homophily,
        groups=groups,
        class_names=class_names,
        return_predictions=True,
    )
    prediction_rows = []
    for idx, (true_label, pred_label) in enumerate(zip(metrics_test["y_true"], metrics_test["y_pred"])):
        row = {
            "sample_index": idx,
            "y_true": true_label,
            "y_pred": pred_label,
        }
        for class_idx, probability in enumerate(metrics_test["probabilities"][idx]):
            row[f"prob_{class_idx}"] = probability
        prediction_rows.append(row)
    report_metrics = {
        key: value
        for key, value in metrics_test.items()
        if key not in {"y_true", "y_pred", "probabilities"}
    }
    gate_stats = model.gate_statistics() if hasattr(model, "gate_statistics") else []

    if verbose:
        print("\n=== TEST RESULTS ===")
        print(f"Best epoch: {best_epoch}")
        for key, value in report_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")

    run_dir = None
    if save_outputs:
        run_dir = _run_dir(cfg, model_name)
        payload = {
            "model": model_name,
            "task": cfg.task,
            "seed": cfg.seed,
            "best_epoch": best_epoch,
            "best_val_f1_macro": best_val_f1,
            "config": asdict(cfg),
            "label_mapping": label_mapping,
            "label_distribution": label_distribution(y_cpu),
            "relation_homophily": rel_homophily_cpu.tolist(),
            "metrics": report_metrics,
        }
        write_json(os.path.join(run_dir, "summary.json"), payload)
        write_csv(os.path.join(run_dir, "history.csv"), history)
        write_csv(os.path.join(run_dir, "classwise_metrics.csv"), report_metrics["classwise"])
        write_csv(os.path.join(run_dir, "subgroup_metrics.csv"), report_metrics["subgroups"])
        write_csv(os.path.join(run_dir, "gate_statistics.csv"), gate_stats)
        write_csv(os.path.join(run_dir, "predictions.csv"), prediction_rows)
        write_csv(
            os.path.join(run_dir, "confusion_matrix.csv"),
            [
                {"true_class": idx, **{f"pred_{j}": value for j, value in enumerate(row)}}
                for idx, row in enumerate(report_metrics["confusion_matrix"])
            ],
        )

    result = {
        "model": model_name,
        "task": cfg.task,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_val_f1,
        "run_dir": run_dir,
        "config": asdict(cfg),
        "metrics": report_metrics,
        "relation_homophily": rel_homophily_cpu.tolist(),
        "gate_statistics": gate_stats,
    }

    if return_metrics:
        return result
    return None


if __name__ == "__main__":
    print("Working dir:", os.getcwd())
    print("Check path:", os.path.exists("./data/MGTAB/features.pt"))
    train()
