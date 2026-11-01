import torch.nn as nn


class ADRBranch(nn.Module):
    def __init__(self, in_dim=20, hidden_dim=64, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, features):
        return self.encoder(features)


class ChemicalDescriptorBranch(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=32, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, features):
        return self.encoder(features)


class EvidenceBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, features):
        return self.encoder(features)
