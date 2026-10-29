import torch
import torch.nn as nn

class ADRBranch(nn.Module):
    def __init__(self, in_dim=20, hidden_dim=64, out_dim=128):
        super(ADRBranch, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)


class ToxBranch(nn.Module):
    # Notice: we might not use the Tox branch as an input in early iterations since it's the target.
    # However, FR-4 mentions "The system shall incorporate available developmental and reproductive toxicity evidence".
    # And Module 4 mentions "Train a multi-task neural network to predict relevant developmental or reproductive toxicity endpoints".
    # Therefore, ToxBranch could act as the prediction HEAD from shared representations, OR as a feature branch if labels are used as inputs for testing.
    # For now, we define a small network that can either encode toxicity labels if available, or act as a task head.
    # According to the PRD Module 4: "Train a multi-task neural network to predict relevant developmental or reproductive toxicity endpoints using suitable labeled data."
    # So it should be a head. But let's follow the standard fusion architecture where we encode if present.

    def __init__(self, in_dim=3, hidden_dim=32, out_dim=128):
        super(ToxBranch, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)
