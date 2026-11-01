import torch
import torch.nn as nn


class SourceGatedFusion(nn.Module):
    """Fuse source encodings using explicit availability and quality inputs."""

    def __init__(self, rep_dim=128, num_sources=6):
        super().__init__()
        self.num_sources = num_sources
        self.gates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(rep_dim + 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_sources)
        )
        self.projections = nn.ModuleList(
            nn.Linear(rep_dim, rep_dim) for _ in range(num_sources)
        )

    def forward(self, representations, availability_mask, evidence_quality):
        if len(representations) != self.num_sources:
            raise ValueError(f"Expected {self.num_sources} source representations")
        if availability_mask.shape[1] != self.num_sources:
            raise ValueError(f"Expected {self.num_sources} availability values")
        if evidence_quality.shape != availability_mask.shape:
            raise ValueError("Evidence quality must have the same shape as availability")

        gated = []
        gates = []
        for index, (representation, gate, projection) in enumerate(
            zip(representations, self.gates, self.projections)
        ):
            available = availability_mask[:, index : index + 1]
            quality = evidence_quality[:, index : index + 1]
            weight = gate(torch.cat((representation, available, quality), dim=-1))
            weight = weight * available
            gates.append(weight)
            gated.append(weight * projection(representation))

        total_gate = torch.stack(gates, dim=0).sum(dim=0)
        fused = torch.stack(gated, dim=0).sum(dim=0)
        return fused / total_gate.clamp_min(torch.finfo(fused.dtype).eps)
