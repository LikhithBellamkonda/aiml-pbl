import torch
import torch.nn as nn

class SourceGatedFusion(nn.Module):
    def __init__(self, rep_dim=128):
        super(SourceGatedFusion, self).__init__()

        # Each source gets a gating network that takes [representation, availability_mask]
        # and outputs a gating scalar (or vector). We'll output a scalar weight per source.
        self.gate_adr = nn.Sequential(
            nn.Linear(rep_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.gate_mol = nn.Sequential(
            nn.Linear(rep_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.gate_tox = nn.Sequential(
            nn.Linear(rep_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Transformations for representations
        self.W_adr = nn.Linear(rep_dim, rep_dim)
        self.W_mol = nn.Linear(rep_dim, rep_dim)
        self.W_tox = nn.Linear(rep_dim, rep_dim)

    def forward(self, h_adr, h_mol, h_tox, availability_mask):
        """
        availability_mask: [batch_size, 3] where cols are [has_adr, has_mol, has_tox]
        """
        # Masks
        m_adr = availability_mask[:, 0].unsqueeze(-1) # [batch_size, 1]
        m_mol = availability_mask[:, 1].unsqueeze(-1)
        m_tox = availability_mask[:, 2].unsqueeze(-1)

        # Gates: g_i = sigmoid(MLP([h_i, m_i]))
        # Note: we multiply by m_i at the end to hard-mask missing sources to 0 weight
        g_adr = self.gate_adr(torch.cat([h_adr, m_adr], dim=-1)) * m_adr
        g_mol = self.gate_mol(torch.cat([h_mol, m_mol], dim=-1)) * m_mol
        g_tox = self.gate_tox(torch.cat([h_tox, m_tox], dim=-1)) * m_tox

        # Gated representations: z_i = g_i * W_i(h_i)
        z_adr = g_adr * self.W_adr(h_adr)
        z_mol = g_mol * self.W_mol(h_mol)
        z_tox = g_tox * self.W_tox(h_tox)

        # Normalize sum of gates to avoid magnitude scaling based on number of available sources
        # We add epsilon to prevent division by zero
        total_gate = g_adr + g_mol + g_tox + 1e-8

        # Fused representation
        z_fused = (z_adr + z_mol + z_tox) / total_gate

        return z_fused
