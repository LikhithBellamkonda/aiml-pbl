import torch
import torch.nn as nn

from .adr_tox_models import ADRBranch, ChemicalDescriptorBranch, EvidenceBranch
from .fusion import SourceGatedFusion
from .molecular_gnn import MolecularGNN


def rank_outcomes(probabilities, outcome_names):
    values = probabilities.flatten().tolist()
    if len(values) != len(outcome_names):
        raise ValueError("The number of scores must match the outcome names")
    return sorted(
        zip(outcome_names, values),
        key=lambda item: item[1],
        reverse=True,
    )


class MultimodalPregnancyADRModel(nn.Module):
    """Outcome-specific research model; labels and evidence remain distinct."""

    def __init__(self, num_outcomes=4, rep_dim=128):
        super().__init__()
        self.adr_branch = ADRBranch(out_dim=rep_dim)
        self.descriptor_branch = ChemicalDescriptorBranch(out_dim=rep_dim)
        self.graph_branch = MolecularGNN(out_dim=rep_dim)
        self.pregnancy_branch = EvidenceBranch(in_dim=4, out_dim=rep_dim)
        self.faers_branch = EvidenceBranch(in_dim=1, out_dim=rep_dim)
        self.fusion = SourceGatedFusion(rep_dim=rep_dim, num_sources=5)
        self.pred_head = nn.Sequential(
            nn.Linear(rep_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_outcomes),
        )

    def forward(self, mol_data):
        representations = [
            self.adr_branch(mol_data.adr_features),
            self.descriptor_branch(mol_data.mol_desc),
            self.graph_branch(
                mol_data.x,
                mol_data.edge_index,
                mol_data.edge_attr,
                mol_data.batch,
                mol_data.adr_features.shape[0],
            ),
            self.pregnancy_branch(mol_data.pregnancy_features),
            self.faers_branch(mol_data.faers_features),
        ]
        fused = self.fusion(
            representations,
            mol_data.availability_mask,
            mol_data.evidence_quality,
        )
        return torch.sigmoid(self.pred_head(fused))

    def predict_with_uncertainty(self, mol_data, num_samples=20):
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2 for uncertainty estimation")
        was_training = self.training
        self.train()
        try:
            with torch.no_grad():
                predictions = torch.stack(
                    [self.forward(mol_data) for _ in range(num_samples)], dim=0
                )
            return predictions.mean(dim=0), predictions.var(dim=0, unbiased=False)
        finally:
            self.train(was_training)

    @staticmethod
    def has_predictive_evidence(mol_data):
        return mol_data.availability_mask.sum(dim=1) > 0
