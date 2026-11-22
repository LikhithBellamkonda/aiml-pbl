import torch
import torch.nn as nn

from .adr_tox_models import ADRBranch, ChemicalDescriptorBranch, EvidenceBranch
from .animal_adr_encoder import AnimalADRBranch
from .fusion import MaskAwareCrossAttentionGating, SourceGatedFusion
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
    """Outcome-specific research model; labels and evidence remain distinct.

    ``include_animal_adr`` adds a sixth, independent evidence source: reported
    animal spontaneous adverse events.  It is off by default so the existing
    five-source architecture and its trained artefacts keep working.  When it
    is enabled the fusion gate receives the animal availability mask, so a drug
    without animal evidence contributes nothing rather than a zero-risk value.
    """

    def __init__(self, num_outcomes=4, rep_dim=128, include_animal_adr=False):
        super().__init__()
        self.include_animal_adr = include_animal_adr
        self.adr_branch = ADRBranch(out_dim=rep_dim)
        self.descriptor_branch = ChemicalDescriptorBranch(out_dim=rep_dim)
        self.graph_branch = MolecularGNN(out_dim=rep_dim)
        self.pregnancy_branch = EvidenceBranch(in_dim=4, out_dim=rep_dim)
        self.faers_branch = EvidenceBranch(in_dim=1, out_dim=rep_dim)
        if include_animal_adr:
            self.animal_adr_branch = AnimalADRBranch(out_dim=rep_dim)
            num_sources = 6
        else:
            self.animal_adr_branch = None
            num_sources = 5
        self.fusion = SourceGatedFusion(rep_dim=rep_dim, num_sources=num_sources)
        self.pred_head = nn.Sequential(
            nn.Linear(rep_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_outcomes),
        )

    def _representations(self, mol_data):
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
        if self.include_animal_adr:
            animal_features = getattr(mol_data, "animal_features", None)
            if animal_features is None:
                raise ValueError(
                    "include_animal_adr=True requires mol_data.animal_features "
                    "and mol_data.animal_availability (use PregnancyDataset with "
                    "animal_adr_path)."
                )
            representations.append(
                self.animal_adr_branch(
                    animal_features,
                    getattr(mol_data, "animal_species_idx", None),
                    getattr(mol_data, "animal_category_idx", None),
                    getattr(mol_data, "animal_availability", None),
                )
            )
        return representations

    def _fuse(self, mol_data):
        return self.fusion(
            self._representations(mol_data),
            mol_data.availability_mask,
            mol_data.evidence_quality,
        )

    def forward(self, mol_data):
        return torch.sigmoid(self.pred_head(self._fuse(mol_data)))

    def predict_with_evidence(self, mol_data):
        """Return probabilities together with per-source fusion weights."""
        details = MaskAwareCrossAttentionGating.forward(
            self.fusion,
            self._representations(mol_data),
            mol_data.availability_mask,
            mol_data.evidence_quality,
        )
        probabilities = torch.sigmoid(self.pred_head(details["fused"]))
        return {
            "probabilities": probabilities,
            "source_weights": details["source_weights"],
            "attention": details["attention"],
            "sources": (
                ("adr", "descriptors", "graph", "pregnancy", "faers", "animal_adr")
                if self.include_animal_adr
                else ("adr", "descriptors", "graph", "pregnancy", "faers")
            ),
        }

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
