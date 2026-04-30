import torch
import torch.nn as nn
import torch.nn.functional as F

class HeterophilyRelationalLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, rel_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.dropout = dropout

        self.rel_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_relations)
        ])

        self.rel_emb = nn.Embedding(num_relations, rel_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + rel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.rel_attn = nn.Linear(hidden_dim, 1)
        self.skip = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        src, dst = edge_index
        num_nodes = x.size(0)
        device = x.device

        if edge_weight is None:
            edge_weight = torch.ones(edge_type.size(0), device=device, dtype=x.dtype)
        else:
            edge_weight = edge_weight.to(device=device, dtype=x.dtype)

        relation_outputs = []

        for r in range(self.num_relations):
            mask = edge_type == r
            out_r = torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=x.dtype)

            if mask.any():
                s = src[mask]
                d = dst[mask]

                h_s = x[s]
                h_d = x[d]
                rel_vec = self.rel_emb.weight[r].unsqueeze(0).expand(h_s.size(0), -1)

                gate_input = torch.cat([h_s, h_d, rel_vec], dim=-1)
                gate = torch.sigmoid(self.gate_mlp(gate_input))

                msg = self.rel_linears[r](h_s) * gate * edge_weight[mask].unsqueeze(-1)
                out_r.index_add_(0, d, msg)

                deg = torch.bincount(d, minlength=num_nodes).clamp(min=1).to(device=device, dtype=x.dtype)
                out_r = out_r / deg.unsqueeze(-1)

            relation_outputs.append(out_r)

        rel_stack = torch.stack(relation_outputs, dim=1)       # [N, R, H]
        attn_logits = self.rel_attn(rel_stack).squeeze(-1)     # [N, R]
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1) # [N, R, 1]

        out = (rel_stack * attn).sum(dim=1)
        out = out + self.skip(x)
        out = self.norm(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class HMRGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_relations: int,
        num_layers: int = 2,
        rel_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([
            HeterophilyRelationalLayer(
                hidden_dim=hidden_dim,
                num_relations=num_relations,
                rel_dim=rel_dim,
                dropout=dropout,
            )
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