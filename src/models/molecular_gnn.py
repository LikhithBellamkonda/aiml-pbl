import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class MolecularGNN(nn.Module):
    def __init__(self, node_in_dim=4, hidden_dim=64, desc_dim=6, out_dim=128):
        super(MolecularGNN, self).__init__()

        # GNN layers for spatial graph
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # MLP for 2D Descriptors
        self.desc_mlp = nn.Sequential(
            nn.Linear(desc_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Combine GNN readout and Descriptors
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim + 32, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, batch, mol_desc):
        # 1. Message Passing
        h = self.conv1(x, edge_index)
        h = F.relu(h)

        h = self.conv2(h, edge_index)
        h = F.relu(h)

        h = self.conv3(h, edge_index)
        h = F.relu(h)

        # 2. Graph Readout (Global pooling)
        h_graph = global_mean_pool(h, batch) # [batch_size, hidden_dim]

        # 3. Descriptor processing
        h_desc = self.desc_mlp(mol_desc) # [batch_size, 32]

        # 4. Concatenate and Project
        h_concat = torch.cat([h_graph, h_desc], dim=1) # [batch_size, hidden_dim + 32]
        h_mol = self.proj(h_concat) # [batch_size, out_dim]

        return h_mol
