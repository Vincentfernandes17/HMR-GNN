import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _forward(model, data, relation_homophily=None):
    try:
        return model(data.x, data.edge_index, data.edge_type, data.edge_weight, relation_homophily)
    except TypeError:
        return model(data.x, data.edge_index, data.edge_type, data.edge_weight)


def _specificity_macro(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specs = []
    total = cm.sum()
    for idx in range(len(labels)):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        specs.append(float(tn / denom) if denom else 0.0)
    return float(np.mean(specs)) if specs else 0.0


def _auc(y_true, prob, labels):
    try:
        if len(labels) == 2:
            return float(roc_auc_score(y_true, prob[:, 1]))
        return float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        return float("nan")


def _overall_metrics(y_true, y_pred, prob, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "sensitivity_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity_macro": _specificity_macro(y_true, y_pred, labels),
        "auc_roc": _auc(y_true, prob, labels),
    }


def _classwise_rows(y_true, y_pred, labels, class_names=None):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    rows = []
    for idx, label in enumerate(labels):
        rows.append({
            "class_id": int(label),
            "class_name": class_names.get(int(label), str(label)) if class_names else str(label),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        })
    return rows


def _subgroup_rows(y_true, y_pred, prob, groups, labels):
    rows = []
    if not groups:
        return rows

    for group_name, group_values in groups.items():
        values = np.asarray(group_values)
        for group_value in sorted(np.unique(values).tolist()):
            mask = values == group_value
            if mask.sum() == 0:
                continue
            metrics = _overall_metrics(y_true[mask], y_pred[mask], prob[mask], labels)
            row = {
                "group": group_name,
                "value": int(group_value) if float(group_value).is_integer() else group_value,
                "support": int(mask.sum()),
            }
            row.update(metrics)
            rows.append(row)
    return rows


@torch.no_grad()
def evaluate(
    model,
    data,
    y,
    masks,
    criterion,
    split="test",
    relation_homophily=None,
    groups=None,
    class_names=None,
    return_predictions=False,
):
    model.eval()
    logits = _forward(model, data, relation_homophily)

    split_mask = masks[f"{split}_mask"]
    loss = criterion(logits[split_mask], y[split_mask])

    prob = torch.softmax(logits, dim=-1).cpu().numpy()
    pred = logits.argmax(dim=-1).cpu().numpy()
    true = y.cpu().numpy()
    eval_idx = split_mask.cpu().numpy().astype(bool)
    labels = list(range(int(torch.unique(y).numel())))

    y_true = true[eval_idx]
    y_pred = pred[eval_idx]
    y_prob = prob[eval_idx]

    result = {"loss": float(loss.item())}
    result.update(_overall_metrics(y_true, y_pred, y_prob, labels))
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist()
    result["classwise"] = _classwise_rows(y_true, y_pred, labels, class_names)

    eval_groups = {}
    if groups:
        for name, values in groups.items():
            eval_groups[name] = np.asarray(values)[eval_idx]
    result["subgroups"] = _subgroup_rows(y_true, y_pred, y_prob, eval_groups, labels)

    if return_predictions:
        result["y_true"] = y_true.tolist()
        result["y_pred"] = y_pred.tolist()
        result["probabilities"] = y_prob.tolist()

    return result
