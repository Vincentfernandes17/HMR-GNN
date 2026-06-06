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


class GCNLayer(nn.Module):
    """
    Homogeneous graph convolution baseline. It ignores relation ids but keeps
    the same graph split and node features as the relational models.
    """

    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.skip = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        src, dst = edge_index
        num_nodes = x.size(0)
        device = x.device

        if edge_weight is None:
            edge_weight = torch.ones(src.numel(), device=device, dtype=x.dtype)
        else:
            edge_weight = edge_weight.to(device=device, dtype=x.dtype)

        out = torch.zeros_like(x)
        deg = torch.bincount(dst, minlength=num_nodes).clamp(min=1).to(device=device, dtype=x.dtype)
        msg = self.linear(x[src]) * edge_weight.unsqueeze(-1)
        out.index_add_(0, dst, msg)
        out = out / deg.unsqueeze(-1)
        out = out + self.skip(x)
        out = self.norm(out)
        out = F.relu(out)
        return F.dropout(out, p=self.dropout, training=self.training)


class GCNBaseline(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type, edge_weight)
        return self.classifier(h)


class GraphSAGELayer(nn.Module):
    """
    Mean-neighbor GraphSAGE baseline. Relation ids are ignored by design.
    """

    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        src, dst = edge_index
        num_nodes = x.size(0)
        device = x.device

        if edge_weight is None:
            edge_weight = torch.ones(src.numel(), device=device, dtype=x.dtype)
        else:
            edge_weight = edge_weight.to(device=device, dtype=x.dtype)

        neigh = torch.zeros_like(x)
        deg = torch.bincount(dst, minlength=num_nodes).clamp(min=1).to(device=device, dtype=x.dtype)
        neigh.index_add_(0, dst, x[src] * edge_weight.unsqueeze(-1))
        neigh = neigh / deg.unsqueeze(-1)

        out = self.linear(torch.cat([x, neigh], dim=-1))
        out = self.norm(out)
        out = F.relu(out)
        return F.dropout(out, p=self.dropout, training=self.training)


class GraphSAGEBaseline(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([GraphSAGELayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type, edge_weight)
        return self.classifier(h)


class GATLayer(nn.Module):
    """
    Single graph-attention layer (Velickovic et al., 2018). Relation ids are
    ignored by design; attention coefficients are computed per directed edge and
    normalized with a softmax over each destination node's incoming edges.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_heads = max(1, num_heads)
        self.head_dim = hidden_dim // self.num_heads
        if self.head_dim * self.num_heads != hidden_dim:
            raise ValueError("hidden_dim must be divisible by num_heads for GAT")
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_src = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.skip = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def _edge_softmax(self, scores, dst, num_nodes):
        # scores: [E, heads]; normalize over edges sharing a destination node.
        scores = scores - scores.max()
        exp = scores.exp()
        denom = torch.zeros(num_nodes, self.num_heads, device=scores.device, dtype=scores.dtype)
        denom.index_add_(0, dst, exp)
        return exp / (denom[dst] + 1e-16)

    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        src, dst = edge_index
        num_nodes = x.size(0)
        device = x.device

        h = self.linear(x).view(num_nodes, self.num_heads, self.head_dim)
        if edge_weight is None:
            edge_weight = torch.ones(src.numel(), device=device, dtype=x.dtype)
        else:
            edge_weight = edge_weight.to(device=device, dtype=x.dtype)

        score_src = (h * self.attn_src).sum(-1)  # [N, heads]
        score_dst = (h * self.attn_dst).sum(-1)  # [N, heads]
        edge_scores = self.leaky_relu(score_src[src] + score_dst[dst])  # [E, heads]
        alpha = self._edge_softmax(edge_scores, dst, num_nodes)         # [E, heads]
        alpha = alpha * edge_weight.unsqueeze(-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = h[src] * alpha.unsqueeze(-1)  # [E, heads, head_dim]
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=device, dtype=x.dtype)
        out.index_add_(0, dst, msg)
        out = out.reshape(num_nodes, self.hidden_dim)

        out = out + self.skip(x)
        out = self.norm(out)
        out = F.elu(out)
        return F.dropout(out, p=self.dropout, training=self.training)


class GATBaseline(nn.Module):
    """
    Graph Attention Network baseline. Attention-weighted homogeneous aggregation;
    relation ids are not used, matching the GCN/GraphSAGE baseline protocol.
    """

    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3, num_heads=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            GATLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type, edge_weight)
        return self.classifier(h)


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
        if edge_weight is None:
            edge_weight = torch.ones(src.numel(), device=device, dtype=x.dtype)
        else:
            edge_weight = edge_weight.to(device=device, dtype=x.dtype)

        out = torch.zeros_like(x)

        for r in range(self.num_relations):
            mask = edge_type == r

            if not mask.any():
                continue

            s = src[mask]
            d = dst[mask]

            msg = self.rel_linears[r](x[s]) * edge_weight[mask].unsqueeze(-1)

            out.index_add_(0, d, msg)

        deg = torch.bincount(dst, minlength=num_nodes).clamp(min=1).to(device=device, dtype=x.dtype)
        out = out / deg.unsqueeze(-1)
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


class DirectionalRGCNLayer(nn.Module):
    """
    RGCN-style baseline with separate incoming and outgoing transforms per
    relation, but without heterophily gates.
    """

    def __init__(self, hidden_dim, num_relations, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.dropout = dropout
        self.rel_in = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_relations)
        ])
        self.rel_out = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_relations)
        ])
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.skip = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        src, dst = edge_index
        num_nodes = x.size(0)
        device = x.device

        if edge_weight is None:
            edge_weight = torch.ones(src.numel(), device=device, dtype=x.dtype)
        else:
            edge_weight = edge_weight.to(device=device, dtype=x.dtype)

        out_in = torch.zeros_like(x)
        out_out = torch.zeros_like(x)

        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue
            s = src[mask]
            d = dst[mask]
            w = edge_weight[mask].unsqueeze(-1)
            out_in.index_add_(0, d, self.rel_in[r](x[s]) * w)
            out_out.index_add_(0, s, self.rel_out[r](x[d]) * w)

        deg_in = torch.bincount(dst, minlength=num_nodes).clamp(min=1).to(device=device, dtype=x.dtype)
        deg_out = torch.bincount(src, minlength=num_nodes).clamp(min=1).to(device=device, dtype=x.dtype)
        out = self.combine(torch.cat([out_in / deg_in.unsqueeze(-1), out_out / deg_out.unsqueeze(-1)], dim=-1))
        out = out + self.skip(x)
        out = self.norm(out)
        out = F.relu(out)
        return F.dropout(out, p=self.dropout, training=self.training)


class DirectionalRGCNBaseline(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_relations, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            DirectionalRGCNLayer(hidden_dim, num_relations, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type, edge_weight)
        return self.classifier(h)
