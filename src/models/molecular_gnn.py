import torch
import torch.nn as nn


class MolecularGNN(nn.Module):
    """Small message-passing molecular encoder implemented with native PyTorch."""

    def __init__(self, node_in_dim=29, edge_in_dim=6, hidden_dim=64, out_dim=128):
        super().__init__()
        self.node_projection = nn.Linear(node_in_dim, hidden_dim)
        self.message_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(3)
        )
        self.update_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(3)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size):
        h = self.node_projection(x)
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            aggregate = torch.zeros_like(h)
            degree = h.new_zeros((h.shape[0], 1))
            if edge_index.numel():
                source, target = edge_index
                messages = message_layer(
                    torch.cat((h[source], h[target], edge_attr), dim=-1)
                )
                aggregate.index_add_(0, target, messages)
                degree.index_add_(0, target, h.new_ones((target.shape[0], 1)))
                aggregate = aggregate / degree.clamp_min(1.0)
            h = update_layer(torch.cat((h, aggregate), dim=-1))

        pooled = h.new_zeros((batch_size, h.shape[-1]))
        counts = h.new_zeros((batch_size, 1))
        if h.shape[0]:
            pooled.index_add_(0, batch, h)
            counts.index_add_(0, batch, h.new_ones((batch.shape[0], 1)))
            pooled = pooled / counts.clamp_min(1.0)
        return self.output_projection(pooled)
