import torch
import torch.nn as nn
from .molecular_gnn import MolecularGNN
from .adr_tox_models import ADRBranch, ToxBranch
from .fusion import SourceGatedFusion

class MultimodalPregnancyADRModel(nn.Module):
    def __init__(self, num_outcomes=1, rep_dim=128):
        super(MultimodalPregnancyADRModel, self).__init__()

        # 1. Branches
        self.mol_branch = MolecularGNN(out_dim=rep_dim)
        self.adr_branch = ADRBranch(out_dim=rep_dim)
        self.tox_branch = ToxBranch(out_dim=rep_dim)

        # 2. Fusion
        self.fusion = SourceGatedFusion(rep_dim=rep_dim)

        # 3. Outcome Prediction Head (Multi-task)
        # Using Dropout for Epistemic Uncertainty via Monte Carlo Dropout
        self.pred_head = nn.Sequential(
            nn.Linear(rep_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2), # Standard MC Dropout layer
            nn.Linear(64, num_outcomes),
            nn.Sigmoid() # Calibrated probabilities
        )

    def forward(self, mol_data):
        """
        mol_data is a PyTorch Geometric Batch object containing custom attributes.
        """
        # 1. Molecular Representation
        h_mol = self.mol_branch(mol_data.x, mol_data.edge_index, mol_data.batch, mol_data.mol_desc)

        # 2. ADR Representation
        h_adr = self.adr_branch(mol_data.adr_features)

        # 3. Toxicity Proxy Representation
        h_tox = self.tox_branch(mol_data.tox_features)

        # 4. Fusion
        z_fused = self.fusion(h_adr, h_mol, h_tox, mol_data.availability_mask)

        # 5. Prediction
        out = self.pred_head(z_fused)

        return out

    def predict_with_uncertainty(self, mol_data, num_samples=10):
        """
        Estimates epistemic uncertainty using Monte Carlo Dropout.
        Model should be in train() mode for dropout to be active, or we force dropout.
        """
        self.train() # Enable dropout

        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds.append(self.forward(mol_data))

        preds_stack = torch.stack(preds) # [num_samples, batch_size, num_outcomes]

        # Mean probability
        mean_pred = preds_stack.mean(dim=0)

        # Variance (Uncertainty)
        variance_pred = preds_stack.var(dim=0)

        return mean_pred, variance_pred
