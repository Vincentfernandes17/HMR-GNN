import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

@torch.no_grad()
def evaluate(model, data, y, masks, criterion):
    model.eval()
    logits = model(data.x, data.edge_index, data.edge_type, data.edge_weight)

    test_mask = masks["test_mask"]
    loss = criterion(logits[test_mask], y[test_mask])

    pred = logits.argmax(dim=-1).cpu().numpy()
    true = y.cpu().numpy()
    test_idx = test_mask.cpu().numpy().astype(bool)

    return {
        "loss": float(loss.item()),
        "accuracy": float(accuracy_score(true[test_idx], pred[test_idx])),
        "f1_macro": float(f1_score(true[test_idx], pred[test_idx], average="macro")),
        "precision_macro": float(precision_score(true[test_idx], pred[test_idx], average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true[test_idx], pred[test_idx], average="macro", zero_division=0)),
    }