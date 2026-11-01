import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskAwareCrossAttentionGating(nn.Module):
    """Mask-Aware Cross-Attention Gating for human evidence fusion.

    Fuses per-source evidence representations conditioned on the
    Missing-Evidence Availability Mask (M) and Quality Vector (Q).
    Sources that are unavailable (M=0) are masked out; low-quality
    sources (Q near 0) receive attenuated attention weights.

    The gating produces a convex combination of source projections,
    ensuring the model never attends to missing evidence.
    """

    def __init__(self, rep_dim: int = 128, num_sources: int = 5, num_heads: int = 4):
        super().__init__()
        self.rep_dim = rep_dim
        self.num_sources = num_sources
        self.num_heads = num_heads
        self.head_dim = rep_dim // num_heads

        # Per-source projection: maps each source's raw features to rep_dim
        self.source_projections = nn.ModuleList(
            nn.Linear(rep_dim, rep_dim) for _ in range(num_sources)
        )

        # Cross-attention: query from each source attends to all others
        self.q_proj = nn.Linear(rep_dim, rep_dim)
        self.k_proj = nn.Linear(rep_dim, rep_dim)
        self.v_proj = nn.Linear(rep_dim, rep_dim)
        self.out_proj = nn.Linear(rep_dim, rep_dim)

        # Mask-aware quality gate: combines M (availability) and Q (quality)
        self.mq_gate = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Final fusion weight per source (learned, but bounded by M*Q)
        self.source_weight = nn.Linear(rep_dim, 1)

        self.layer_norm = nn.LayerNorm(rep_dim)

    def forward(
        self,
        source_representations: list[torch.Tensor],
        availability_mask: torch.Tensor,   # [batch, num_sources] M in {0,1}
        evidence_quality: torch.Tensor,    # [batch, num_sources] Q in [0,1]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            source_representations: list of num_sources tensors, each [batch, rep_dim]
            availability_mask: [batch, num_sources], 1=source present, 0=missing
            evidence_quality: [batch, num_sources], quality score in [0,1]

        Returns:
            dict with "fused" [batch, rep_dim], "attention" [batch, num_sources],
            "source_weights" [batch, num_sources]
        """
        batch_size = source_representations[0].shape[0]
        device = source_representations[0].device

        if len(source_representations) != self.num_sources:
            raise ValueError(
                f"Expected {self.num_sources} source representations, "
                f"got {len(source_representations)}"
            )

        # Project each source into the shared representation space
        projected = [
            proj(rep)
            for proj, rep in zip(self.source_projections, source_representations)
        ]
        stacked = torch.stack(projected, dim=1)  # [batch, num_sources, rep_dim]

        # Cross-attention: each source attends to all sources
        q = self.q_proj(stacked).view(
            batch_size, self.num_sources, self.num_heads, self.head_dim
        ).transpose(1, 2)   # [batch, heads, num_sources, head_dim]
        k = self.k_proj(stacked).view(
            batch_size, self.num_sources, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(stacked).view(
            batch_size, self.num_sources, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # [batch, heads, num_sources, num_sources]

        # Mask unavailable sources in attention (they cannot be keys/values)
        # M_j = 0 means source j is unavailable; mask its column
        mask_expanded = availability_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, num_sources]
        scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # [batch, heads, num_sources, num_sources]
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Apply attention to values
        attended = torch.matmul(attn_weights, v)  # [batch, heads, num_sources, head_dim]
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, self.num_sources, self.rep_dim)
        )
        attended = self.out_proj(attended)  # [batch, num_sources, rep_dim]

        # Mask-aware quality gating: M * Q modulates each source's contribution
        mq = torch.stack([availability_mask, evidence_quality], dim=-1)  # [batch, num_sources, 2]
        mq_gate = self.mq_gate(mq).squeeze(-1)  # [batch, num_sources]

        # Source-specific importance weight (learned)
        source_logits = self.source_weight(stacked).squeeze(-1)  # [batch, num_sources]
        source_weights_raw = torch.sigmoid(source_logits)

        # Final per-source weight: learned * M * Q gate
        source_weights = source_weights_raw * availability_mask * mq_gate
        # If no source is available for any batch entry, return zeros
        weight_sum = source_weights.sum(dim=1, keepdim=True)
        all_missing = (weight_sum.squeeze(-1) <= 0).all()
        if all_missing:
            return {
                "fused": torch.zeros_like(stacked.mean(dim=1)),
                "attention": torch.zeros(batch_size, self.num_sources, device=device),
                "source_weights": torch.zeros(batch_size, self.num_sources, device=device),
            }
        # Normalize to convex combination (avoid division by zero)
        weight_sum = weight_sum.clamp_min(torch.finfo(source_weights.dtype).eps)
        source_weights = source_weights / weight_sum

        # Weighted sum of attended representations
        fused = (attended * source_weights.unsqueeze(-1)).sum(dim=1)  # [batch, rep_dim]
        fused = self.layer_norm(fused)  # norm without residual when masking active

        # Aggregate attention for interpretability: mean over heads and query dims
        attention_summary = attn_weights.mean(dim=(1, 2))  # [batch, num_sources]

        return {
            "fused": fused,
            "attention": attention_summary,
            "source_weights": source_weights,
        }


class SourceGatedFusion(MaskAwareCrossAttentionGating):
    """Backward-compatible wrapper: exposes the same interface as the
    original SourceGatedFusion while implementing Mask-Aware Cross-Attention Gating."""

    def forward(
        self,
        representations: list[torch.Tensor],
        availability_mask: torch.Tensor,
        evidence_quality: torch.Tensor,
    ) -> torch.Tensor:
        result = super().forward(representations, availability_mask, evidence_quality)
        return result["fused"]
