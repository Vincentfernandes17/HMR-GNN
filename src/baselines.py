import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================
# BASELINE 1: MLP
# ==========================
class MLPBaseline(nn.Module):
    """
    No graph structure.
    Uses only node features.
    """

    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index=None, edge_type=None, edge_weight=None):
        return self.net(x)


# ==========================
# BASELINE 2: R-GCN
# ==========================
class RGCNLayer(nn.Module):
    """
    Standard relational aggregation
    WITHOUT heterophily gate
    """

    def __init__(self, hidden_dim, num_relations, dropout=0.3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.dropout = dropout

        self.rel_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_relations)
        ])

        self.skip = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        src, dst = edge_index
        num_nodes = x.size(0)
        device = x.device

        out = torch.zeros_like(x)

        for r in range(self.num_relations):
            mask = edge_type == r

            if not mask.any():
                continue

            s = src[mask]
            d = dst[mask]

            msg = self.rel_linears[r](x[s])

            out.index_add_(0, d, msg)

        out = out + self.skip(x)
        out = self.norm(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out


class RGCNBaseline(nn.Module):
    """
    Multi-relational graph baseline
    without heterophily awareness
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        num_relations,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([
            RGCNLayer(hidden_dim, num_relations, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        h = self.input_proj(x)

        for layer in self.layers:
            h = layer(h, edge_index, edge_type, edge_weight)

        return self.classifier(h)