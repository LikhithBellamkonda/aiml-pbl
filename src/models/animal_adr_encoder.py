import torch
import torch.nn as nn

from src.animal_adr.features import NUMERIC_FEATURES


class AnimalADRBranch(nn.Module):
    def __init__(
        self,
        in_dim: int = len(NUMERIC_FEATURES),
        hidden_dim: int = 64,
        out_dim: int = 128,
        species_count: int = 8,
        category_count: int = 8,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.species_embedding = nn.Embedding(species_count, embedding_dim)
        self.category_embedding = nn.Embedding(category_count, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim + (embedding_dim * 2), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        numeric_features: torch.Tensor,
        species_idx: torch.Tensor,
        category_idx: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat(
            (
                numeric_features,
                self.species_embedding(species_idx),
                self.category_embedding(category_idx),
            ),
            dim=-1,
        )
        output = self.encoder(combined)
        mask = availability.to(dtype=output.dtype).view(-1, 1)
        return output * mask
