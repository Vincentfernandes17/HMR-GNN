import torch
import torch.nn as nn
import torch.nn.functional as F

class HeterophilyRelationalLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_relations: int,
        rel_dim: int = 32,
        dropout: float = 0.3,
        num_heads: int = 1,
        gate_temperature: float = 1.0,
        use_homophily_gate: bool = False,
        homophily_alpha: float = 1.0,
        separate_directions: bool = False,
        log_gate_scores: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.dropout = dropout
        self.num_heads = max(1, num_heads)
        self.gate_temperature = max(gate_temperature, 1e-6)
        self.use_homophily_gate = use_homophily_gate
        self.homophily_alpha = homophily_alpha
        self.separate_directions = separate_directions
        self.log_gate_scores = log_gate_scores
        self.last_gate_stats = []

        self.rel_linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_relations)
        ])
        self.rel_linears_out = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_relations)
        ])

        self.rel_emb = nn.Embedding(num_relations, rel_dim)

        # Memory-efficient factorized gate. This is algebraically equivalent to a
        # Linear layer applied to [h_src || h_dst || rel_emb (|| homophily)], but it
        # never materializes that wide per-edge concatenation. Each term is a
        # separate projection, and the relation/homophily terms are computed once
        # per relation (broadcast over edges) instead of once per edge.
        self.gate_src = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.gate_dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_rel = nn.Linear(rel_dim, hidden_dim, bias=False)
        self.gate_hom = nn.Linear(1, hidden_dim, bias=False) if use_homophily_gate else None
        self.gate_out = nn.Linear(hidden_dim, 1)
        # Open-gate initialization: start gates near 1 (pass-through) so the model
        # begins as a strong relational GNN and only learns to *suppress* edges if it
        # helps. A zero-initialized gate would output 0.5 and halve every message,
        # handicapping early optimization.
        self.gate_open_bias = 3.0  # sigmoid(3) ~= 0.95
        nn.init.constant_(self.gate_out.bias, self.gate_open_bias)

        self.rel_attn = nn.Linear(hidden_dim, self.num_heads)
        # Explicit bidirectional combine. When directions are separated we pool
        # relations *within* each direction and then fuse the incoming and outgoing
        # node representations with a learned projection, rather than mixing both
        # directions through a single softmax (which dilutes directional signal).
        self.dir_combine = nn.Linear(hidden_dim * 2, hidden_dim) if separate_directions else None
        self.skip = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _gate(self, h_source, h_target, rel_emb_vec, homophily_value):
        # rel_emb_vec is a single relation embedding of shape [rel_dim]; its
        # projection is a [hidden_dim] vector that broadcasts over all edges.
        pre = self.gate_src(h_source) + self.gate_dst(h_target) + self.gate_rel(rel_emb_vec)
        if self.use_homophily_gate:
            hom = h_source.new_tensor([[float(homophily_value)]])  # [1, 1]
            pre = pre + self.gate_hom(hom)  # [1, hidden_dim] broadcast over edges
        gate_logits = self.gate_out(F.relu(pre))
        if self.use_homophily_gate:
            heterophily_bonus = self.homophily_alpha * (1.0 - float(homophily_value))
            gate_logits = gate_logits + heterophily_bonus
        return torch.sigmoid(gate_logits / self.gate_temperature)

    def _collect_gate_stats(self, layer_relation, direction, gate):
        if not self.log_gate_scores or gate.numel() == 0:
            return
        detached = gate.detach().float().cpu().view(-1)
        self.last_gate_stats.append({
            "relation": int(layer_relation),
            "direction": direction,
            "count": int(detached.numel()),
            "gate_mean": float(detached.mean().item()),
            "gate_std": float(detached.std(unbiased=False).item()) if detached.numel() > 1 else 0.0,
            "gate_min": float(detached.min().item()),
            "gate_max": float(detached.max().item()),
        })

    def _relation_pool(self, rel_stack):
        # rel_stack: [N, R, H]. Attention-weighted pool over the relation axis.
        attn_logits = self.rel_attn(rel_stack)                 # [N, R, heads]
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1) # [N, R, heads, 1]
        rel_stack_heads = rel_stack.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        return (rel_stack_heads * attn).sum(dim=1).mean(dim=1)  # [N, H]

    def forward(self, x, edge_index, edge_type, edge_weight=None, relation_homophily=None):
        src, dst = edge_index
        num_nodes = x.size(0)
        device = x.device
        self.last_gate_stats = []

        if edge_weight is None:
            edge_weight = torch.ones(edge_type.size(0), device=device, dtype=x.dtype)
        else:
            edge_weight = edge_weight.to(device=device, dtype=x.dtype)
        if relation_homophily is None:
            relation_homophily = torch.full((self.num_relations,), 0.5, device=device, dtype=x.dtype)
        else:
            relation_homophily = relation_homophily.to(device=device, dtype=x.dtype)

        in_relation_outputs = []
        out_relation_outputs = []

        for r in range(self.num_relations):
            mask = edge_type == r
            out_r = torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=x.dtype)
            out_rev_r = torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=x.dtype)

            if mask.any():
                s = src[mask]
                d = dst[mask]

                h_s = x[s]
                h_d = x[d]
                rel_emb_vec = self.rel_emb.weight[r]

                gate = self._gate(h_s, h_d, rel_emb_vec, relation_homophily[r])
                self._collect_gate_stats(r, "in", gate)

                msg = self.rel_linears[r](h_s) * gate * edge_weight[mask].unsqueeze(-1)
                out_r.index_add_(0, d, msg)

                deg = torch.bincount(d, minlength=num_nodes).clamp(min=1).to(device=device, dtype=x.dtype)
                out_r = out_r / deg.unsqueeze(-1)

                if self.separate_directions:
                    gate_rev = self._gate(h_d, h_s, rel_emb_vec, relation_homophily[r])
                    self._collect_gate_stats(r, "out", gate_rev)
                    msg_rev = self.rel_linears_out[r](h_d) * gate_rev * edge_weight[mask].unsqueeze(-1)
                    out_rev_r.index_add_(0, s, msg_rev)

                    deg_rev = torch.bincount(s, minlength=num_nodes).clamp(min=1).to(device=device, dtype=x.dtype)
                    out_rev_r = out_rev_r / deg_rev.unsqueeze(-1)

            in_relation_outputs.append(out_r)
            if self.separate_directions:
                out_relation_outputs.append(out_rev_r)

        # Pool relations within the incoming direction.
        in_pooled = self._relation_pool(torch.stack(in_relation_outputs, dim=1))  # [N, H]

        if self.separate_directions:
            # Pool relations within the outgoing direction, then fuse both
            # directions with a learned projection (generalizes directional RGCN).
            out_pooled = self._relation_pool(torch.stack(out_relation_outputs, dim=1))
            combined = self.dir_combine(torch.cat([in_pooled, out_pooled], dim=-1))
        else:
            combined = in_pooled

        out = combined + self.skip(x)
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
        num_heads: int = 1,
        gate_temperature: float = 1.0,
        use_homophily_gate: bool = False,
        homophily_alpha: float = 1.0,
        separate_directions: bool = False,
        log_gate_scores: bool = False,
    ):
        super().__init__()
        self.log_gate_scores = log_gate_scores
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
                num_heads=num_heads,
                gate_temperature=gate_temperature,
                use_homophily_gate=use_homophily_gate,
                homophily_alpha=homophily_alpha,
                separate_directions=separate_directions,
                log_gate_scores=log_gate_scores,
            )
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, edge_type, edge_weight=None, relation_homophily=None):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type, edge_weight, relation_homophily)
        return self.classifier(h)

    def gate_statistics(self):
        stats = []
        for layer_idx, layer in enumerate(self.layers):
            for item in layer.last_gate_stats:
                row = dict(item)
                row["layer"] = layer_idx
                stats.append(row)
        return stats
